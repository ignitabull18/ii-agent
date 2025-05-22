from typing import Any, Optional
import os

from ii_agent.tools.base import (
    ToolImplOutput,
    LLMTool,
)
from ii_agent.llm.message_history import MessageHistory
from ii_agent.utils import WorkspaceManager


class RegisterDeploymentTool(LLMTool):
    """Tool for registering deployments"""

    name = "register_deployment"
    description = "Register a deployment and get the public url as well as the port that you can deploy your service on."

    input_schema = {
        "type": "object",
        "properties": {
            "service_name": {
                "type": "string",
                "description": "Name of the service to deploy and used in the public url, this must be unique",
            },
        },
        "required": ["service_name"],
    }

    def __init__(self, workspace_manager: WorkspaceManager):
        super().__init__()
        self.workspace_manager = workspace_manager
        # TODO: this is a hack to get the base URL for the static files
        # TODO: we should use a proper URL for the static files
        default_base_url = f"file://{workspace_manager.root.parent.parent.absolute()}"
        self.base_url = os.getenv("STATIC_FILE_BASE_URL", default_base_url)

    def run_impl(
        self,
        tool_input: dict[str, Any],
        message_history: Optional[MessageHistory] = None,
    ) -> ToolImplOutput:
        service_name = tool_input["service_name"]
        # Make request to register service
        import httpx

        client = httpx.Client()
        response = client.post(
            "http://localhost:9000/api/register",
            json={
                "service_name": service_name,
                "container_name": self.workspace_manager.root.name,
            },
        )

        if response.status_code != 200:
            return ToolImplOutput(
                f"Failed to register service: {response.text}",
                f"Failed to register service: {response.text}",
            )

        # Get the UUID from the workspace path
        connection_uuid = self.workspace_manager.root.name

        # Construct the public URL using the base URL and connection UUID
        public_url = f"/agent/{connection_uuid}/{service_name}"

        return ToolImplOutput(
            public_url,
            f"Registering successfully. Public url/base path to access the service: {public_url}.  service port: 8000",
        )
