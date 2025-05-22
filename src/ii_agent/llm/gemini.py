import os

from typing import Any, Tuple
from google import genai
from google.genai import types
from ii_agent.llm.base import (
    LLMClient,
    AssistantContentBlock,
    ToolParam,
    TextPrompt,
    ToolCall,
    TextResult,
    LLMMessages,
    ToolFormattedResult,
    recursively_remove_invoke_tag,
    ImageBlock,
)


class GeminiDirectClient(LLMClient):
    """Use Gemini models via first party API."""

    def __init__(self, model_name: str):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set")
        self.model_name = model_name
        self.client = genai.Client(api_key=api_key)


    def generate(
        self,
        messages: LLMMessages,
        max_tokens: int,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        tools: list[ToolParam] = [],
        tool_choice: dict[str, str] | None = None,
        thinking_tokens: int | None = None,
    ) -> Tuple[list[AssistantContentBlock], dict[str, Any]]:
        
        gemini_messages = []
        for idx, message_list in enumerate(messages):
            role = "user" if idx % 2 == 0 else "model"
            message_content_list = []
            for message in message_list:
                if str(type(message)) == str(TextPrompt):
                    message_content = types.Part(text=message.text)
                elif str(type(message)) == str(ImageBlock):
                    message_content = types.Part.from_bytes(
                            data=message.source["data"],
                            mime_type=message.source["media_type"],
                        )
                elif str(type(message)) == str(TextResult):
                    message_content = types.Part(text=message.text)
                elif str(type(message)) == str(ToolCall):
                    message_content = types.Part.from_function_call(
                        name=message.tool_name,
                        args=message.tool_input,
                    )
                elif str(type(message)) == str(ToolFormattedResult):
                    if isinstance(message.tool_output, str):
                        message_content = types.Part.from_function_response(
                            name=message.tool_name,
                            response={"result": message.tool_output}
                        )
                    elif isinstance(message.tool_output, list):
                        message_content = []
                        for item in message.tool_output:
                            if item['type'] == 'text':
                                message_content.append(types.Part(text=item['text']))
                            elif item['type'] == 'image':
                                message_content.append(types.Part.from_bytes(
                                    data=item['source']['data'],
                                    mime_type=item['source']['media_type']
                                ))
                else:
                    raise ValueError(f"Unknown message type: {type(message)}")
                
                if isinstance(message_content, list):
                    message_content_list.extend(message_content)
                else:
                    message_content_list.append(message_content)
            
            message_content_fmt = types.Content(
                role=role,
                parts=message_content_list
            )
            gemini_messages.append(message_content_fmt)
        
        tool_params = []
        for tool in tools:
            tool_params.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                }
            )
        tool_params = types.Tool(function_declarations=tool_params)

        response = self.client.models.generate_content(
            model=self.model_name,
            config=types.GenerateContentConfig(
                tools=[tool_params],
                system_instruction=system_prompt,
                temperature=temperature,
                max_output_tokens=max_tokens,
                tool_config=types.FunctionCallingConfig(mode="ANY")
                ),
            contents=gemini_messages,
        )

        internal_messages = []
        if response.function_calls:
            for fn_call in response.function_calls:
                response_message_content = ToolCall(
                    tool_call_id=fn_call.id,
                    tool_name=fn_call.name,
                    tool_input=fn_call.args,
                )
                internal_messages.append(response_message_content)

        if response.text:
            internal_messages.append(TextResult(text=response.text))

        if len(internal_messages) == 0:
            raise ValueError("No response from Gemini")

        message_metadata = {
            "raw_response": response,
            "input_tokens": response.usage_metadata.prompt_token_count,
            "output_tokens": response.usage_metadata.candidates_token_count,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }
        
        return internal_messages, message_metadata