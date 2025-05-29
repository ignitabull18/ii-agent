#!/bin/bash
# Create workspace directory if it doesn't exist
if [ ! -d "${PWD}/workspace" ]; then
  mkdir ${PWD}/workspace
  echo "Created workspace directory"
fi

#BACKEND ENVIRONMENT VARIABLES
export FRONTEND_PORT=3000
export BACKEND_PORT=8000
export WORKSPACE_PATH=${PWD}/workspace

# Start docker-compose with the HOST_IP variable
COMPOSE_PROJECT_NAME=agent docker compose -f docker/docker-compose.yaml up "$@"
