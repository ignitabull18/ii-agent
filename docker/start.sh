#!/bin/bash

# Detect OS type and set HOST_IP appropriately
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

export PROXY_SERVER_PORT=9000
export FRONTEND_PORT=3000
export BACKEND_PORT=8000
export WORKSPACE_PATH=../workspace

# Start docker-compose with the HOST_IP variable
docker-compose up "$@"

