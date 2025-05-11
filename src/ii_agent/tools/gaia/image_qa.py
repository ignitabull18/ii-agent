import mimetypes
from pathlib import Path
from typing import Any, Optional

from ii_agent.llm.message_history import MessageHistory
from ii_agent.tools.base import (
    LLMTool,
    ToolImplOutput,
)
from ii_agent.llm.base import LLMClient, TextResult
from ..utils import encode_image
from ii_agent.utils import WorkspaceManager


class ImageQATool(LLMTool):
    name = "image_qa"
    description = "A tool that answers questions about images by analyzing their content using a vision-language model."
    input_schema = {
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "The path to the image to analyze. This should be a local path to downloaded image.",
            },
            "question": {
                "type": "string",
                "description": "The question to ask about the image content.",
            },
        },
        "required": ["image_path", "question"],
    }

    def __init__(self, workspace_manager: WorkspaceManager | None, client: LLMClient):
        if workspace_manager is None:
            workspace_manager = WorkspaceManager(Path("."))
        self.workspace_manager = workspace_manager
        self.client = client

    def run_impl(
        self,
        tool_input: dict[str, Any],
        message_history: Optional[MessageHistory] = None,
    ) -> ToolImplOutput:
        image_path = tool_input["image_path"]
        question = tool_input["question"]

        if not isinstance(image_path, str):
            return ToolImplOutput(
                tool_output="Error: image_path must be a string",
                tool_result_message="Error: image_path must be a string"
            )
        
        if not isinstance(question, str):
            return ToolImplOutput(
                tool_output="Error: question must be a string",
                tool_result_message="Error: question must be a string"
            )

        try:
            # Convert relative path to absolute path using workspace_manager
            import ipdb; ipdb.set_trace()
            abs_path = str(self.workspace_manager.workspace_path(image_path))
            
            # Get mime type and encode image
            mime_type, _ = mimetypes.guess_type(abs_path)
            if not mime_type:
                mime_type = "image/png"  # Default to PNG if type cannot be determined
            
            base64_image = encode_image(abs_path)

            # Prepare the single message with image and question
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type,
                                "data": base64_image,
                            },
                        },
                        {
                            "type": "text",
                            "text": question
                        }
                    ]
                }
            ]

            # Generate response from the vision-language model
            model_response, _ = self.client.generate(
                messages=messages,
                max_tokens=1024,  # Adjust based on your needs
                tools=[],  # No tools needed for this interaction
            )

            # Extract the text response from the model output
            text_results = [item for item in model_response if isinstance(item, TextResult)]
            if not text_results:
                return ToolImplOutput(
                    tool_output="Error: No text response received from the model",
                    tool_result_message="Error: No text response received from the model"
                )

            answer = text_results[0].text

            return ToolImplOutput(
                tool_output=answer,
                tool_result_message=f"Successfully analyzed image and answered question: {question}"
            )

        except Exception as e:
            error_msg = f"Failed to process image or answer question: {str(e)}"
            return ToolImplOutput(
                tool_output=error_msg,
                tool_result_message=error_msg
            )
