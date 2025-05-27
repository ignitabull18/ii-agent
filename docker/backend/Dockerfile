# Backend Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies and Docker
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release \
    && curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/debian $(lsb_release -cs) stable" > /etc/apt/sources.list.d/docker.list \
    && apt-get update \
    && apt-get install -y docker-ce docker-ce-cli containerd.io \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire codebase for editable install
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install -e .

# Create workspace directory
RUN mkdir -p workspace

RUN playwright install
RUN playwright install-deps

# Set environment variables
ENV PYTHONUNBUFFERED=1

COPY docker/backend/run.sh /app/run.sh

RUN chmod +x /app/run.sh

# Expose port for WebSocket server
EXPOSE 8000

ENTRYPOINT ["/app/run.sh"]