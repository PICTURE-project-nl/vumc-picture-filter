#!/bin/bash

# Log the start of the entrypoint script
echo "Starting entrypoint script..."

# Check for GPU
if command -v nvidia-smi > /dev/null 2>&1; then
  echo "GPU detected"
  export USE_GPU=true
else
  echo "No GPU detected"
  export USE_GPU=false
fi

# Log the GPU status
echo "USE_GPU is set to $USE_GPU"

# Start supervisord
echo "Starting supervisord..."
/usr/bin/supervisord -c /etc/supervisord.conf

# Start Celery worker
echo "Starting Celery worker..."
supervisorctl -c /etc/supervisord.conf start celery-worker:*

# Log the start of the application
echo "Starting the application..."

# Start the Python application
cd /src && python app.py & tail -f /var/log/celery-worker.log