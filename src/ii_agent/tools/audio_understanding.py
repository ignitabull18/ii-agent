import os
from ii_agent.tools.base import (
    LLMTool,
    ToolImplOutput,
)
from typing import Any, Optional
from ii_agent.llm.message_history import MessageHistory
from google import genai
from google.genai import types


class AudioUnderstandingTool(LLMTool):
    name = "audio_understanding"
    description = """This tool is used to understand an audio file (supported formats: .mp3, .wav, .m4a). Use this tool to:
- Describe, summarize, or answer questions about audio content.
- Provide a transcription of the audio.
- Analyze specific segments of the audio."""

    input_schema = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Local audio file path",
            },
            "prompt": {
                "type": "string",
                "description": "Prompt for the audio understanding",
            }
        },
        "required": ["file_path", "prompt"],
    }
    output_type = "string"

    def __init__(self, workspace_manager):
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        self.client = genai.Client(
            api_key=api_key
        )
        self.workspace_manager = workspace_manager

    
    def run_impl(
        self,
        tool_input: dict[str, Any],
        message_history: Optional[MessageHistory] = None,
    ) -> ToolImplOutput:
        file_path = tool_input["file_path"]
        prompt = tool_input["prompt"]
        model = "gemini-2.5-pro-preview-05-06"
        abs_path = str(self.workspace_manager.workspace_path(file_path))
        with open(abs_path, 'rb') as f:
            audio_bytes = f.read()
        try:
            response = self.client.models.generate_content(
                model=model,
                contents=types.Content(
                parts=[
                    types.Part(text=prompt),
                    types.Part.from_bytes(
                        data=audio_bytes,
                        mime_type='audio/mp3',
                        )
                    ]
                )
            )
            output = response.text
        except Exception as e:
            output = f"Error analyzing the audio file, try again later."
            print(e)

        return ToolImplOutput(
            output,
            output
        )
