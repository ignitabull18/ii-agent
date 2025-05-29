from typing import Any, Optional
import subprocess
import os
from ii_agent.llm.message_history import MessageHistory
from ii_agent.tools.base import LLMTool, ToolImplOutput
from ii_agent.utils.workspace_manager import WorkspaceManager


class SlideDeckInitTool(LLMTool):
    name = "slide_deck_init"
    description = "This tool initializes a presentation environment by cloning the reveal.js framework and setting up all necessary dependencies. It creates a presentation directory structure, downloads the reveal.js HTML presentation framework from GitHub, and installs all required npm packages to enable slide deck creation and presentation capabilities."
    input_schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def __init__(self, workspace_manager: WorkspaceManager) -> None:
        super().__init__()
        self.workspace_manager = workspace_manager

    def run_impl(
        self,
        tool_input: dict[str, Any],
        message_history: Optional[MessageHistory] = None,
    ) -> ToolImplOutput:
        self.history = MessageHistory()

        try:
            # Create the presentation directory if it doesn't exist
            presentation_dir = f"{self.workspace_manager.root}/presentation"
            os.makedirs(presentation_dir, exist_ok=True)

            # Clone the reveal.js repository to the specified path
            clone_command = f"git clone https://github.com/khoangothe/reveal.js.git {self.workspace_manager.root}/presentation/reveal.js"
            
            # Execute the clone command
            clone_result = subprocess.run(
                clone_command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=self.workspace_manager.root
            )
            
            if clone_result.returncode != 0:
                return ToolImplOutput(
                    f"Failed to clone reveal.js repository: {clone_result.stderr}",
                    f"Failed to clone reveal.js repository",
                    auxiliary_data={"success": False, "error": clone_result.stderr},
                )

            # Install dependencies
            install_command = "npm install"
            
            # Execute the install command in the reveal.js directory
            install_result = subprocess.run(
                install_command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=f"{self.workspace_manager.root}/presentation/reveal.js"
            )
            
            if install_result.returncode != 0:
                return ToolImplOutput(
                    f"Failed to install dependencies: {install_result.stderr}",
                    f"Failed to install dependencies",
                    auxiliary_data={"success": False, "error": install_result.stderr},
                )

            return ToolImplOutput(
                f"Successfully initialized slide deck. Repository cloned into `./presentation/reveal.js` and dependencies installed (npm install).",
                f"Successfully initialized slide deck",
                auxiliary_data={"success": True, "clone_output": clone_result.stdout, "install_output": install_result.stdout},
            )
            
        except Exception as e:
            return ToolImplOutput(
                f"Error initializing slide deck: {str(e)}",
                f"Error initializing slide deck",
                auxiliary_data={"success": False, "error": str(e)},
            )