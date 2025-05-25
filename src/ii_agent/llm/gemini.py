import os
import time
import logging

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

logger = logging.getLogger(__name__)


class GeminiDirectClient(LLMClient):
    """Use Gemini models via first party API."""

    def __init__(
        self, 
        model_name: str, 
        max_retries: int = 2, 
        retry_delay: float = 1.0,
        project_id: None | str = None,
        region: None | str = None,
    ):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set")
        self.model_name = model_name
        if project_id and region:
            self.client = genai.Client(
                vertexai=True,
                project=project_id,
                location=region,
            )
            print(f"====== Using Vertex AI API with project_id: {project_id} and region: {region} ======")
        else:
            self.client = genai.Client(api_key=api_key)
            print(f"====== Using Gemini API ======")
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def _process_gemini_response(self, response: types.GenerateContentResponse) -> list[AssistantContentBlock]:
        """Process the Gemini API response into internal message format.
        
        Args:
            response: The raw response from the Gemini API
            
        Returns:
            List of processed message blocks
            
        Raises:
            ValueError: If no valid response is found
        """
        internal_messages = []
        
        # Process all parts in the response to avoid warnings
        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            
            # Check for malformed function call
            if hasattr(candidate, 'finish_message') and candidate.finish_message:
                if 'Malformed function call' in candidate.finish_message:
                    logger.error(f"Gemini returned malformed function call: {candidate.finish_message}")
                    # Extract the error message and return it as text to help debug
                    error_msg = f"Gemini encountered an error with the function call: {candidate.finish_message[:500]}..."
                    internal_messages.append(TextResult(text=error_msg))
                    return internal_messages
            
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if part.text:
                        internal_messages.append(TextResult(text=part.text))
                    elif part.function_call:
                        fn_call = part.function_call
                        response_message_content = ToolCall(
                            tool_call_id=fn_call.id if hasattr(fn_call, 'id') else None,
                            tool_name=fn_call.name,
                            tool_input=fn_call.args,
                        )
                        internal_messages.append(response_message_content)

        if len(internal_messages) == 0:
            # Log more details for debugging
            logger.error(f"No valid response from Gemini. Response: {response}")
            if response.candidates:
                for i, candidate in enumerate(response.candidates):
                    logger.error(f"Candidate {i}: finish_reason={getattr(candidate, 'finish_reason', 'N/A')}, "
                               f"finish_message={getattr(candidate, 'finish_message', 'N/A')}")
            raise ValueError("No response from Gemini")
            
        return internal_messages

    def generate(
        self,
        messages: LLMMessages,
        max_tokens: int,
        system_prompt: str | None = None,
        tools: list[ToolParam] = [],
    ) -> Tuple[list[AssistantContentBlock], dict[str, Any]]:
        
        gemini_messages = []
        for idx, message_list in enumerate(messages):
            role = "user" if idx % 2 == 0 else "model"
            message_content_list = []
            for message in message_list:
                if isinstance(message, TextPrompt):
                    message_content = types.Part(text=message.text)
                elif isinstance(message, ImageBlock):
                    message_content = types.Part.from_bytes(
                            data=message.source["data"],
                            mime_type=message.source["media_type"],
                        )
                elif isinstance(message, TextResult):
                    message_content = types.Part(text=message.text)
                elif isinstance(message, ToolCall):
                    message_content = types.Part.from_function_call(
                        name=message.tool_name,
                        args=message.tool_input,
                    )
                elif isinstance(message, ToolFormattedResult):
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
                    raise ValueError(f"Unknown message type: {type(message).__name__}")
                
                if isinstance(message_content, list):
                    message_content_list.extend(message_content)
                else:
                    message_content_list.append(message_content)
            
            message_content_fmt = types.Content(
                role=role,
                parts=message_content_list
            )
            gemini_messages.append(message_content_fmt)
        
        # Only create tools if there are any
        if tools:
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
            config = types.GenerateContentConfig(
                tools=[tool_params],
                system_instruction=system_prompt,
                max_output_tokens=max_tokens,
                tool_config={'function_calling_config': {'mode': 'ANY'}},
            )
        else:
            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=max_tokens,
            )
        # Retry logic with exponential backoff
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    config=config,
                    contents=gemini_messages,
                )
                
                # Check if we got a malformed function call
                if (response.candidates and len(response.candidates) > 0 and 
                    hasattr(response.candidates[0], 'finish_message') and 
                    response.candidates[0].finish_message and
                    'Malformed function call' in response.candidates[0].finish_message):
                    
                    logger.warning(f"Received malformed function call on attempt {attempt + 1}/{self.max_retries}")
                    
                    # If we have tools and this is not the last attempt, retry without problematic tools
                    if tools and attempt < self.max_retries - 1:
                        # Remove tools that commonly cause issues
                        problematic_tools = ["SequentialThinkingTool", "str_replace_editor"]
                        filtered_tools = [tool for tool in tools if tool.name not in problematic_tools]
                        
                        if filtered_tools:
                            filtered_tool_params = []
                            for tool in filtered_tools:
                                filtered_tool_params.append(
                                    {
                                        "name": tool.name,
                                        "description": tool.description,
                                        "parameters": tool.input_schema,
                                    }
                                )
                            filtered_tool_params = types.Tool(function_declarations=filtered_tool_params)
                            config = types.GenerateContentConfig(
                                tools=[filtered_tool_params],
                                system_instruction=system_prompt,
                                max_output_tokens=max_tokens,
                                tool_config={'function_calling_config': {'mode': 'ANY'}},
                            )
                        else:
                            # If no tools left, try without tools
                            config = types.GenerateContentConfig(
                                system_instruction=system_prompt,
                                max_output_tokens=max_tokens,
                            )
                        
                        wait_time = self.retry_delay * (2 ** attempt)
                        logger.info(f"Retrying without problematic tools in {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                
                internal_messages = self._process_gemini_response(response)
                return internal_messages, {
                    "raw_response": response,
                    "input_tokens": response.usage_metadata.prompt_token_count,
                    "output_tokens": response.usage_metadata.candidates_token_count,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                }
                
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(
                        f"Gemini API call failed (attempt {attempt + 1}/{self.max_retries}): {e}. "
                        f"Retrying in {wait_time} seconds..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"Gemini API call failed after {self.max_retries} attempts: {e}")
                    raise
        
        # This should not be reached, but just in case
        if last_error:
            raise last_error
        else:
            raise ValueError("Failed to get response from Gemini after all retries")