# set base image (host OS)
FROM nvcr.io/nvidia/pytorch:21.12-py3

# add supervisor
RUN apt-get update && apt-get install -y supervisor cron

# set the working directory in the container
WORKDIR /src

# copy the dependencies file to the working directory
COPY requirements.txt .

# install dependencies
RUN pip install -r requirements.txt

# copy the content of the local src directory to the working directory
COPY src/ .

# copy the data directory to the appropriate location
#COPY data /data

# copy the supervisor config and start celery script
COPY ./config/supervisord.conf /etc/supervisord.conf
COPY ./entrypoint.sh /entrypoint.sh

# copy the script to generate docker-compose file
#COPY generate_docker_compose.py /src/generate_docker_compose.py

# create the log file to be able to run tail
RUN touch /var/log/cron.log

# run the generate script
#RUN python /src/generate_docker_compose.py

# script to run when starting the container
CMD /entrypoint.sh
