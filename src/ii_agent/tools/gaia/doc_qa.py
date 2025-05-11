from pathlib import Path
from typing import Any, Optional

from ii_agent.llm.message_history import MessageHistory
from ii_agent.tools.base import (
    LLMTool,
    ToolImplOutput,
)
from ii_agent.llm.base import LLMClient, TextResult
from ii_agent.utils import WorkspaceManager
from ii_agent.tools.markdown_converter import MarkdownConverter


class DocQATool(LLMTool):
    name = "doc_qa"
    description = "A tool that answers questions about documents by analyzing their content using a language model."
    input_schema = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "The path to the document to analyze. This should be a local path to a text-based file (PDF, DOCX, etc.).",
            },
            "question": {
                "type": "string",
                "description": "The question to ask about the document content.",
            },
        },
        "required": ["file_path", "question"],
    }

    def __init__(self, workspace_manager: WorkspaceManager | None, client: LLMClient, text_limit: int = 100000):
        if workspace_manager is None:
            workspace_manager = WorkspaceManager(Path("."))
        self.workspace_manager = workspace_manager
        self.client = client
        self.text_limit = text_limit
        self.md_converter = MarkdownConverter()

    def run_impl(
        self,
        tool_input: dict[str, Any],
        message_history: Optional[MessageHistory] = None,
    ) -> ToolImplOutput:
        file_path = tool_input["file_path"]
        question = tool_input["question"]

        try:
            # Convert relative path to absolute path using workspace_manager
            abs_path = str(self.workspace_manager.workspace_path(file_path))
            
            # Check for image files
            if file_path.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
                return ToolImplOutput(
                    tool_output="Error: Cannot use doc_qa tool with images. Use image_qa tool instead.",
                    tool_result_message="Error: Cannot use doc_qa tool with images"
                )

            # Convert document to text
            result = self.md_converter.convert(abs_path)
            
            # For zip files, just return the content
            if ".zip" in file_path:
                return ToolImplOutput(
                    tool_output=result.text_content,
                    tool_result_message=f"Successfully extracted content from zip file: {file_path}"
                )

            # For small documents, return content directly
            if len(result.text_content) < 4000:
                return ToolImplOutput(
                    tool_output=f"Document content: {result.text_content}",
                    tool_result_message=f"Successfully extracted content from document: {file_path}"
                )

            # Prepare messages for the model
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Here is a document:\n### {result.title}\n\n{result.text_content[:self.text_limit]}\n\nQuestion: {question}"
                        }
                    ]
                }
            ]

            # Generate response from the language model
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
                tool_result_message=f"Successfully analyzed document and answered question: {question}"
            )

        except Exception as e:
            error_msg = f"Failed to process document or answer question: {str(e)}"
            return ToolImplOutput(
                tool_output=error_msg,
                tool_result_message=error_msg
            )
