#!/bin/bash
# Detect OS type and set HOST_IP appropriately

# Create workspace directory if it doesn't exist
if [ ! -d "${PWD}/workspace" ]; then
    mkdir ${PWD}/workspace
    echo "Created workspace directory"
fi


if [[ "$OSTYPE" == "darwin"* ]]; then
  # macOS
  export HOST_IP=$(ipconfig getifaddr en0)
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
  # Linux
  export HOST_IP=$(hostname -I | awk '{print $1}')
else
  echo "Unsupported OS type: $OSTYPE"
  export HOST_IP="localhost"
fi

echo "Using HOST_IP: $HOST_IP"
export PUBLIC_DOMAIN=${HOST_IP}.nip.io

export PROXY_SERVER_PORT=9000
export FRONTEND_PORT=3000
export BACKEND_PORT=8000
export WORKSPACE_PATH=${PWD}/workspace


# IF YOU ARE USING VERTEX AI, SET THESE VARIABLES ELSE PUT ANTHROPIC_API_KEY in .env
export GOOGLE_APPLICATION_CREDENTIALS=
export PROJECT_ID=
export REGION=

# Start docker-compose with the HOST_IP variable
docker-compose -f docker/docker-compose.yaml up "$@"