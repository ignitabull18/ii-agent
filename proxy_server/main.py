from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import httpx
import datetime
from typing import Dict, Any
from logging import getLogger
import os
from collections import defaultdict

logger = getLogger(__name__)

app = FastAPI(title="Agent Proxy API")
# Dictionary to store registered services: {container_name: {service_name: {"port": service_port, "registered_at": registered_at}}}
registered_services: Dict[str, Dict[str, Any]] = defaultdict(lambda: defaultdict(dict))

MAIN_APP_HOST = os.environ.get("MAIN_APP_HOST", "localhost")
MAIN_APP_PORT = os.environ.get("MAIN_APP_PORT", "9000")


@app.get("/ping")
async def ping():
    """Simple health check endpoint to test API availability.

    Returns:
        A simple JSON response indicating the API is up
    """
    return {"status": "ok", "message": "pong"}


@app.api_route(
    "/agent/{container_name}/{service_path:path}",
    methods=["GET", "POST", "PUT", "DELETE"],
)
async def proxy(container_name: str, service_path: str, request: Request):
    """Proxy requests to agent containers within the Docker network.

    Args:
        container_name: The ID/name of the agent container in the Docker network
        service_path: The path to the service within the container
        request: The incoming request to proxy

    Returns:
        The response from the target service
    """
    # Extract service name from path
    service_name = service_path.split("/")[0]

    if container_name not in registered_services:
        return JSONResponse(
            status_code=404,
            content={"error": f"Container '{container_name}' not found"},
        )

    if service_name not in registered_services[container_name]:
        return JSONResponse(
            status_code=404, content={"error": f"Service '{service_name}' not found"}
        )

    service_info = registered_services[container_name][service_name]
    port = service_info["port"]

    # Get remaining path after service name
    remaining_path = "/".join(service_path.split("/")[1:])

    # Construct target URL within Docker network
    target_url = f"http://{container_name}:{port}/{remaining_path}"
    print(remaining_path)
    print("--------------------------------")
    print(f"Proxying request to {target_url}")
    print("--------------------------------")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.request(
                method=request.method,
                url=target_url,
                headers=request.headers.raw,
                content=await request.body(),
                timeout=30.0,  # Add timeout to prevent hanging requests
            )

        # Return the raw response content
        from fastapi.responses import Response

        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=dict(response.headers),
        )
    except httpx.RequestError as e:
        logger.error(f"Error proxying request to {target_url}: {str(e)}")
        return JSONResponse(
            status_code=502,
            content={"error": f"Failed to connect to agent service: {str(e)}"},
        )


@app.post("/api/register")
async def register_service(request: Request):
    """Register an external service with the WebSocket server.

    External services can register their name, container name, and port for later discovery
    and communication.

    Args:
        request: The request containing the service details

    Returns:
        JSON response confirming registration
    """
    try:
        data = await request.json()
        service_name = data.get("service_name")
        container_name = data.get("container_name")

        # Validate required fields
        if not service_name:
            return JSONResponse(
                status_code=400, content={"error": "Service name is required"}
            )

        if not container_name:
            return JSONResponse(
                status_code=400, content={"error": "Container name is required"}
            )

        new_service = {
            "port": 8000,
            "registered_at": datetime.datetime.now().isoformat(),
        }

        # Register a service within a container
        registered_services[container_name][service_name] = new_service

        return JSONResponse(
            status_code=200,
            content={
                "status": "ok",
                "message": f"Service '{service_name}' of container '{container_name}' running on port '8000'",
                "service": registered_services[container_name][service_name],
            },
        )

    except Exception as e:
        logger.error(f"Error registering service: {str(e)}")
        return JSONResponse(
            status_code=500, content={"error": f"Failed to register service: {str(e)}"}
        )
