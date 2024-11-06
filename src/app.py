#!flask/bin/python
from flask import Flask, request, jsonify, url_for
from flask_caching import Cache
from celery import Celery
import picture_filter
import json
import numpy as np


app = Flask(__name__)
app.config.from_pyfile('config.py')

celery = Celery(app.name, broker=app.config['CELERY_CONFIG']['broker_url'])
celery.conf.update(app.config.get("CELERY_CONFIG", {}))
cache = Cache(config={'CACHE_TYPE': 'SimpleCache'})
cache.init_app(app)

@celery.task(bind=True)
def _get_dataset(self):
    return picture_filter.get_dataset()


@app.route('/dataset', methods=['GET'])
@cache.cached(timeout=604800)
def get_dataset():
    dataset_getter = _get_dataset.apply_async()
    image_data = dataset_getter.get()

    result = json.dumps(
        {'status': 'OK', 'image_data': image_data}, default=default)

    return result, 200


@celery.task(bind=True)
def filter(self, mask_path, filter_criteria):

    image_data = picture_filter.get_filter_results(mask_path, filter_criteria)

    result = {'image_data': image_data}

    return result


@app.route('/status/<task_id>', methods=['GET'])
def get_task_status(task_id):

    task = filter.AsyncResult(task_id)

    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'result': 'Pending...'
        }
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'result': task.result,
        }
    else:
        response = {
            'state': task.state,
            'result': str(task.result),
        }
    return jsonify(response), 200


@app.route('/filter', methods=['POST'])
def start_filter():

    if request.json:
        request_data = request.json
    else:
        return jsonify({'error': 'missing required input_image volume'}), 400

    if 'input_image' in request_data:
        input_image = request_data['input_image']
    else:
        return jsonify({'error': 'missing required input_image volume'}), 400

    if 'filter_criteria' in request_data:
        filter_criteria = request_data['filter_criteria']
    else:
        filter_criteria = {}

    if not isinstance(filter_criteria, dict):
        return jsonify({'error': 'not a valid set of filter criteria'}), 400
    else:
        supported_filter_criteria = picture_filter.get_supported_filter_criteria()

        for k, v in filter_criteria.items():
            if k not in supported_filter_criteria:
                return jsonify(
                    {'error': 'not a valid set of filter criteria'}), 400

    if input_image == 'test':
        mask_path = '/data/transforms_binary_map.nii'
    else:
        mask_path = picture_filter.save_volume_as_image(input_image)

    task = filter.delay(mask_path, filter_criteria)
    response = {'location': url_for('get_task_status', task_id=task.id)}

    return jsonify(response), 202


def default(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError('Unknown type:', type(obj))


@app.route('/filter_options', methods=['GET'])
def filter_criteria():
    filter_options = picture_filter.get_supported_filter_criteria()

    result = json.dumps(
        {'status': 'OK', 'filter_options': filter_options}, default=default)
    return result, 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
    
