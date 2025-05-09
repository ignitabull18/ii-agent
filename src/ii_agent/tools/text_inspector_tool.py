from typing import Any, Optional
from ii_agent.llm.message_history import MessageHistory
from ii_agent.tools.base import (
    LLMTool,
    ToolImplOutput,
)
from .markdown_converter import MarkdownConverter
from ii_agent.utils import WorkspaceManager


class TextInspectorTool(LLMTool):
    name = "inspect_file_as_text"
    description = """
You cannot load files yourself: instead call this tool to read a file as markdown text and ask questions about it.
This tool handles the following file extensions: [".html", ".htm", ".xlsx", ".pptx", ".wav", ".mp3", ".m4a", ".flac", ".pdf", ".docx"], and all other types of text files. IT DOES NOT HANDLE IMAGES."""

    input_schema = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "The path to the file you want to read as text. Must be a '.something' file, like '.pdf'. If it is an image, use the visualizer tool instead! DO NOT use this tool for an HTML webpage: use the web_search tool instead!",
            },
        },
        "required": ["file_path"],
    }

    def __init__(self, workspace_manager: WorkspaceManager, text_limit: int = 100000):
        self.text_limit = text_limit
        self.md_converter = MarkdownConverter()
        self.workspace_manager = workspace_manager

    def forward(self, file_path: str) -> str:
        # Convert relative path to absolute path using workspace_manager
        abs_path = str(self.workspace_manager.workspace_path(file_path))
        result = self.md_converter.convert(abs_path)

        if file_path[-4:] in [".png", ".jpg"]:
            raise Exception("Cannot use inspect_file_as_text tool with images: use visualizer instead!")

        if ".zip" in file_path:
            return result.text_content

        return result.text_content


    def run_impl(
        self,
        tool_input: dict[str, Any],
        message_history: Optional[MessageHistory] = None,
    ) -> ToolImplOutput:
        file_path = tool_input["file_path"]

        try:
            output = self.forward(file_path)
            return ToolImplOutput(
                output,
                f"Successfully inspected file {file_path}",
                auxiliary_data={"success": True},
            )
        except Exception as e:
            return ToolImplOutput(
                f"Error inspecting file: {str(e)}",
                f"Failed to inspect file {file_path}",
                auxiliary_data={"success": False},
            )
