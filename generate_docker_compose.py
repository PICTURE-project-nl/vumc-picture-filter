import os
import subprocess

# Check for NETWORK_PREFIX or set default
network_prefix = os.getenv('NETWORK_PREFIX', 'default_prefix')

# Function to check if NVIDIA GPU is available
def is_gpu_available():
    try:
        result = subprocess.run(["nvidia-smi"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("DEBUG: NVIDIA GPU detected.")
        return True
    except subprocess.CalledProcessError:
        print("DEBUG: No NVIDIA GPU detected, using CPU mode.")
        return False

# Set GPU availability based on detection
gpu_available = '1' if is_gpu_available() else '0'

# Define the GPU deploy section if a GPU is detected
gpu_deploy_section = """
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]
""" if gpu_available == '1' else ""

# Docker Compose template for the filter service
docker_compose_template = f"""
version: "3.8"
services:
  filter:
    build: .
    ports:
      - "5001:5000"
    volumes:
      - ./data:/data
    env_file:
      - .env
    environment:
      - USE_GPU={gpu_available}
    networks:
      - filtering
      - internal
    {gpu_deploy_section}
  redis:
    image: "redis:alpine"
    networks:
      - internal
  flower:
    image: mher/flower:0.9.7
    command: ['flower', '--broker=redis://redis:6379/0', '--port=5555']
    ports:
      - 5554:5555
    networks:
      - internal
    links:
      - redis
    depends_on:
      - redis
    restart: always

networks:
  filtering:
    external: true
    name: "{network_prefix}_filtering"
  internal:
    external: false
"""

# Write the generated docker-compose.yml
output_path = os.path.join(os.getcwd(), "docker-compose.generated.yml")
with open(output_path, "w") as f:
    f.write(docker_compose_template)

# Confirm if the file was created successfully
if os.path.exists(output_path):
    print(f"docker-compose.generated.yml generated successfully at {output_path}")
else:
    print("Error: Failed to generate docker-compose.generated.yml.")