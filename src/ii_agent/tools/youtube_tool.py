import os
from ii_agent.tools.base import (
    LLMTool,
    ToolImplOutput,
)
from typing import Any, Optional
from ii_agent.llm.message_history import MessageHistory
from google import genai
from google.genai import types


class YoutubeVideoUnderstandingTool(LLMTool):
    name = "youtube_video_understanding"
    description = """This tool is used to understand a Youtube video. Use this tool to:
- Describe, segment, and extract information from videos
- Answer questions about video content
- Refer to specific timestamps within a video"""

    input_schema = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Youtube Video URL",
            },
            "prompt": {
                "type": "string",
                "description": "Prompt for the video understanding",
            }
        },
        "required": ["url", "prompt"],
    }
    output_type = "string"

    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        self.client = genai.Client(
            api_key=api_key
        )

    
    def run_impl(
        self,
        tool_input: dict[str, Any],
        message_history: Optional[MessageHistory] = None,
    ) -> ToolImplOutput:
        url = tool_input["url"]
        prompt = tool_input["prompt"]
        model = "gemini-2.5-pro-preview-05-06"
        
        try:
            response = self.client.models.generate_content(
                model=model,
                contents=types.Content(
                parts=[
                    types.Part(
                        file_data=types.FileData(file_uri=url)
                    ),
                    types.Part(text=prompt)
                ]
                )
            )
            output = response.candidates[0].content.parts[0].text
        except Exception as e:
            output = f"Error analyzing the Youtube video, try again later."
            print(e)

        return ToolImplOutput(
            output,
            output
        )
