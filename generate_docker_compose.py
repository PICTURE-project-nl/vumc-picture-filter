import os
import subprocess
import sys

# Debug logging function
def log_debug(message):
    print(f"DEBUG: {message}")

# Check for NETWORK_PREFIX or set default
network_prefix = os.getenv('NETWORK_PREFIX', 'default_prefix')
log_debug(f"Network prefix set to: {network_prefix}")

# Function to check if NVIDIA GPU is available
def is_gpu_available():
    try:
        # Attempt to run the nvidia-smi command
        result = subprocess.run(["nvidia-smi"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        log_debug("NVIDIA GPU detected.")
        return True
    except FileNotFoundError:
        log_debug("nvidia-smi command not found; assuming no GPU available.")
        return False
    except subprocess.CalledProcessError as e:
        log_debug(f"nvidia-smi command failed with error: {e}; assuming no GPU available.")
        return False
    except Exception as e:
        log_debug(f"Unexpected error while checking for GPU: {e}; defaulting to CPU mode.")
        return False

# Set GPU availability based on detection
gpu_available = '1' if is_gpu_available() else '0'
log_debug(f"GPU available: {gpu_available} (1 for yes, 0 for no)")

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

# Write the generated docker-compose.yml file
def write_docker_compose_file(output_path, content):
    try:
        with open(output_path, "w") as f:
            f.write(content)
        log_debug(f"docker-compose.generated.yml created successfully at {output_path}")
    except Exception as e:
        log_debug(f"Error writing docker-compose.generated.yml: {e}")
        sys.exit(1)

# Main execution
def main():
    output_path = os.path.join(os.getcwd(), "docker-compose.generated.yml")
    write_docker_compose_file(output_path, docker_compose_template)

    # Confirm if the file was successfully created
    if os.path.exists(output_path):
        log_debug(f"File successfully created at {output_path}")
    else:
        log_debug("Error: docker-compose.generated.yml not found after writing. Check permissions and paths.")
        sys.exit(1)

if __name__ == "__main__":
    main()