
# PICTURE Filter

Microservice to filter the PICTURE dataset based on an input image and return a filtered probability map and sum tumors map. The filtering is an asynchronous process that is started with an API request and can be monitored through another endpoint.

## Introduction within the PICTURE application
In the context of PICTURE, the filtering service is an essential component for filtering the dataset based on specific clinical variables. This filter, named `vumc-picture-filter`, operates between data acquisition and analysis phases to improve the accuracy of patient data and provide advanced medical insights.

The Filter Service uses clinical variables to filter and segment medical image data, improving prognostic evaluations. These variables are selectable via the API.

## Requirements
- Docker
- Dataset with clinical variables and NIfTI images
- GPU (optional, but recommended for large datasets)

## Installation
1. Add the dataset to `./data` and set environment variables in `.env`.
2. Build the Docker image:
   ```bash
   docker compose build
   ```
3. Create the filtering network if it doesn’t exist:
   ```bash
   docker network create
   ```
4. Start the service:
   ```bash
   docker compose up -d
   ```
5. Preload the `/dataset` endpoint response (see caching section).

## Usage
1. Prepare the input image as a base64-encoded string from an MHA volume. Volumes can be created using the Python `SimpleITK` package. See `./src/picture_filter.py` for examples.
2. Start the filtering process by sending an HTTP POST request to `http://localhost:5000/filter`:
   ```json
   {
       "input_image": "mha_volume_string",
       "filter_criteria": {}
   }
   ```
   - Request filter criteria from `http://localhost:5000/filter_options`.
3. The `/filter` endpoint returns a JSON string containing a "location" key.
4. Append the "location" value to `http://localhost:5000` to check status and results.

## Output
The output is a JSON string with a key "location" pointing to the URL for checking status and results. The result includes clinical variables and base64-encoded MHA volume strings for the probability map and sum tumors map.

## Clinical Variables for Filtering
The filterable clinical variables (found in `picture_filter.py`) may include:
- **Age (age):** Key prognostic indicator based on patient population.
- **Karnofsky Performance Status (kpspre):** Assesses patient functional capacity.
- **Surgery Type (surgeryextend):** Indicates surgical intervention extent.
- **Tumor Grade (grade):** Crucial for prognosis.
- **5-ALA Score (5alascore):** Aids in surgical planning based on tumor fluorescence.

## Extension
This filter uses the current PICTURE dataset’s data model. Column mappings, such as `"birthyear"` to `"BirthYear"`, are defined in `./src/custom_mappings.py`. New dataset columns must be added to this mapping dictionary.

## Caching
When using the `/dataset` endpoint, results are cached for a week. Caching is crucial since initial dataset requests are computationally expensive and may result in timeouts. Cache warming requests should be sent after restarting the filter container.

Run cache warming:
```bash
sudo docker exec -it vumc-picture-filter-filter-1 curl http://localhost:5000/dataset
```

Add a crontab entry for periodic cache warming:
```cron
0 2 * * * cd /data/volume_2/picture-webapp/vumc-picture-filter && sudo docker exec -it vumc-picture-filter-filter-1 curl http://localhost:5000/dataset >> /dev/null 2>&1
```

## Technical Components
### Languages and Frameworks
- Python
- Flask

### Libraries
- SimpleITK
- numpy
- pandas
- torch (GPU/CPU tensor computation)
- torchvision
- numba (JIT compilation)
- tqdm (progress bars)
- torchmetrics (distance metrics)

### Task Queue
- Celery (asynchronous task management)
- Redis (message broker)

### Containerization
- Docker
- docker-compose

## Files and Configuration

### Dockerfile
Defines the environment for the Docker container.

```dockerfile
FROM nvcr.io/nvidia/pytorch:21.12-py3
WORKDIR /src
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ .
COPY ./config/supervisord.conf /etc/supervisord.conf
COPY ./entrypoint.sh /entrypoint.sh
COPY generate_docker_compose.py /src/generate_docker_compose.py
RUN python /src/generate_docker_compose.py
CMD /entrypoint.sh
```

### supervisord.conf
Supervisord configuration for managing processes.

```ini
[supervisord]
logfile=/var/log/supervisord.log
logfile_maxbytes=10MB
nodaemon=false

[program:celery-worker]
command=celery -A app.celery worker --loglevel=info --concurrency=1
stdout_logfile=/var/log/celery-worker.log
stderr_logfile=/var/log/celery-worker.err
autostart=true

[program:app]
command=python app.py
stdout_logfile=/var/log/app.log
stderr_logfile=/var/log/app.err
autostart=true
```

### entrypoint.sh
Entrypoint script that starts supervisord and Celery worker.

```bash
#!/bin/bash
echo "Starting entrypoint script..."
if command -v nvidia-smi > /dev/null 2>&1; then
  export USE_GPU=true
else
  export USE_GPU=false
fi
/usr/bin/supervisord -c /etc/supervisord.conf
supervisorctl start celery-worker:*
cd /src && python app.py & tail -f /var/log/celery-worker.log
```

### requirements.txt
Python package dependencies.

```plaintext
torch
torchvision
celery
Flask
numpy
pandas
SimpleITK
numba
redis
torchmetrics
tqdm
```

### docker-compose.yml
Docker Compose configuration for services.

```yaml
version: "3.3"
services:
  filter:
    build: .
    ports:
      - "5001:5000"
    volumes:
      - ./data:/data
    env_file:
      - .env
    networks:
      - filtering
      - internal
  redis:
    image: "redis:alpine"
  flower:
    image: mher/flower:0.9.7
    ports:
      - "5554:5555"
networks:
  filtering:
    external: true
  internal:
    external: false
```

## Known Issues
### Dependencies
- Conflicts between different package versions listed in `requirements.txt` can cause compatibility issues.

### Performance
- Image processing functions, especially distance matrix calculations and aggregations, may be slow for large datasets without GPU acceleration.

### Containerization
- GPU detection issues might arise, and Docker network configuration may require tuning depending on the deployment environment.

### Logging and Debugging
- Ensure logs are regularly reviewed, especially from `celery-worker` and `app.py`, for errors in task execution.

### Configuration Management
- Manage separate configurations for development, testing, and production environments to avoid configuration conflicts.
