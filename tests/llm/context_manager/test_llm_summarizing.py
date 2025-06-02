import logging
from unittest.mock import Mock
import re

from ii_agent.llm.base import (
    ImageBlock,
    TextPrompt,
    TextResult,
    LLMClient,
    ToolCall,
    ToolFormattedResult,
)
from ii_agent.llm.context_manager.llm_summarizing import LLMSummarizingContextManager
from ii_agent.llm.token_counter import TokenCounter


def test_llm_summarizing_context_manager():
    mock_logger = Mock(spec=logging.Logger)
    mock_llm_client = Mock(spec=LLMClient)

    # Mock the generate method to return a summary response
    def mock_generate(messages, max_tokens=None):
        return [TextResult(text="Generated summary of conversation events.")], None

    mock_llm_client.generate.side_effect = mock_generate
    token_counter = TokenCounter()

    context_manager = LLMSummarizingContextManager(
        client=mock_llm_client,
        token_counter=token_counter,
        logger=mock_logger,
        token_budget=1000,
        max_size=10,
        keep_first=2,
    )

    for num_messages in range(9, 13):
        message_lists = []
        for j in range(num_messages):
            if j % 2 == 0:
                message_lists.append([TextPrompt(text=f"Turn {j // 2}")])
            else:
                message_lists.append([TextResult(text=f"Turn {j // 2}")])
        result = context_manager.apply_truncation_if_needed(message_lists)
        print(result)

        # Add assertions based on expected behavior
        if num_messages <= 10:  # No truncation needed (9 and 10 messages)
            assert len(result) == num_messages
            assert result == message_lists
        else:  # Truncation needed (11 and 12 messages)
            assert len(result) == 5  # target_size = max_size // 2 = 5
            # First 2 messages should be kept (keep_first=2)
            assert result[0] == message_lists[0]
            assert result[1] == message_lists[1]
            # Third message should be the summary
            assert isinstance(result[2][0], TextPrompt)
            assert "Conversation Summary:" in result[2][0].text
            # Last 2 messages should be from the tail
            assert result[3] == message_lists[-2]
            assert result[4] == message_lists[-1]


def test_llm_calls_during_summarization():
    """Test that captures and inspects the actual LLM calls made during summarization."""

    # Create a spy that captures all LLM calls
    llm_calls = []

    def spy_generate(messages, max_tokens=None, **kwargs):
        # Capture the call details
        call_info = {
            "messages": messages,
            "max_tokens": max_tokens,
            "kwargs": kwargs,
            "prompt_text": messages[0][0].text
            if messages and messages[0] and hasattr(messages[0][0], "text")
            else None,
        }
        llm_calls.append(call_info)

        # Return a mock summary response
        return [
            TextResult(
                text="USER_CONTEXT: Flask web application development\nCOMPLETED: Read config.py and main.py files, added error handling\nPENDING: Add logging and additional improvements\nCODE_STATE: main.py with Flask app and error handling\nCHANGES: Added try-catch block to hello() route"
            )
        ], None

    mock_logger = Mock(spec=logging.Logger)
    mock_llm_client = Mock(spec=LLMClient)
    mock_llm_client.generate.side_effect = spy_generate
    token_counter = TokenCounter()

    context_manager = LLMSummarizingContextManager(
        client=mock_llm_client,
        token_counter=token_counter,
        logger=mock_logger,
        token_budget=1000,
        max_size=8,  # Smaller size to trigger summarization
        keep_first=2,
    )

    # Create a conversation with tool calls that will trigger summarization
    conversation = [
        [TextPrompt(text="Can you read the contents of config.py?")],
        [
            ToolCall(
                tool_call_id="call_123",
                tool_name="read_file",
                tool_input={"file_path": "config.py"},
            )
        ],
        [
            ToolFormattedResult(
                tool_call_id="call_123",
                tool_name="read_file",
                tool_output="DEBUG = True\nDATABASE_URL = 'sqlite:///app.db'",
            )
        ],
        [
            TextResult(
                text="I can see the config.py file contains debug settings and database configuration."
            )
        ],
        [TextPrompt(text="Now check the main.py file")],
        [
            ToolCall(
                tool_call_id="call_456",
                tool_name="read_file",
                tool_input={"file_path": "main.py"},
            )
        ],
        [
            ToolFormattedResult(
                tool_call_id="call_456",
                tool_name="read_file",
                tool_output="from flask import Flask\napp = Flask(__name__)\n@app.route('/')\ndef hello():\n    return 'Hello World!'",
            )
        ],
        [TextResult(text="The main.py file contains a simple Flask application.")],
        [TextPrompt(text="Add error handling to the Flask app")],
        [
            ToolCall(
                tool_call_id="call_789",
                tool_name="edit_file",
                tool_input={
                    "file_path": "main.py",
                    "new_content": "from flask import Flask\napp = Flask(__name__)\n@app.route('/')\ndef hello():\n    try:\n        return 'Hello World!'\n    except Exception as e:\n        return f'Error: {str(e)}', 500",
                },
            )
        ],
        [
            ToolFormattedResult(
                tool_call_id="call_789",
                tool_name="edit_file",
                tool_output="File successfully modified",
            )
        ],
        [TextResult(text="I've added error handling to the Flask application.")],
    ]

    print(f"Original conversation length: {len(conversation)}")

    # Apply truncation - this should trigger LLM summarization
    result = context_manager.apply_truncation_if_needed(conversation)

    print(f"Truncated conversation length: {len(result)}")
    print(f"Number of LLM calls made: {len(llm_calls)}")

    # Assertions about the LLM calls
    assert len(llm_calls) == 1, f"Expected 1 LLM call, but got {len(llm_calls)}"

    # Inspect the first (and only) LLM call
    first_call = llm_calls[0]

    # Check that max_tokens was set
    assert first_call["max_tokens"] == 4000, (
        f"Expected max_tokens=4000, got {first_call['max_tokens']}"
    )

    # Check that the prompt contains the expected structure
    prompt_text = first_call["prompt_text"]
    assert prompt_text is not None, "Prompt text should not be None"

    # Verify the prompt contains the expected sections
    assert "You are maintaining a context-aware state summary" in prompt_text
    assert "USER_CONTEXT:" in prompt_text
    assert "COMPLETED:" in prompt_text
    assert "PENDING:" in prompt_text
    assert "CODE_STATE:" in prompt_text
    assert "<PREVIOUS SUMMARY>" in prompt_text
    assert "<EVENT id=" in prompt_text
    assert "Now summarize the events using the rules above." in prompt_text

    # Check that events are properly formatted in the prompt
    assert "ToolCall:" in prompt_text or "ToolFormattedResult:" in prompt_text
    assert "read_file" in prompt_text
    assert "edit_file" in prompt_text

    # Verify the result structure
    assert len(result) == 4, (
        f"Expected 4 message lists after truncation, got {len(result)}"
    )

    # First 2 should be kept (keep_first=2)
    assert result[0] == conversation[0]
    assert result[1] == conversation[1]

    # Third should be the summary
    assert isinstance(result[2][0], TextPrompt)
    assert result[2][0].text.startswith("Conversation Summary:")

    # Last should be from the tail
    assert result[3] == conversation[-1]

    print("✅ LLM call inspection test passed!")
    print(f"📝 Prompt preview: {prompt_text[:200]}...")


def test_multiple_summarization_rounds_with_llm_spy():
    """Test multiple rounds of summarization and inspect all LLM calls."""

    llm_calls = []

    def spy_generate(messages, max_tokens=None, **kwargs):
        call_info = {
            "messages": messages,
            "max_tokens": max_tokens,
            "prompt_text": messages[0][0].text
            if messages and messages[0] and hasattr(messages[0][0], "text")
            else None,
            "call_number": len(llm_calls) + 1,
        }
        llm_calls.append(call_info)

        # Return different summaries for different calls
        if len(llm_calls) == 1:
            summary = "USER_CONTEXT: Flask development\nCOMPLETED: Read files, added error handling\nPENDING: Add logging\nCODE_STATE: main.py with Flask app"
        else:
            summary = "USER_CONTEXT: Flask development with testing\nCOMPLETED: Added error handling, logging, tests\nPENDING: Deploy application\nCODE_STATE: main.py, test_main.py, requirements.txt"

        return [TextResult(text=summary)], None

    mock_logger = Mock(spec=logging.Logger)
    mock_llm_client = Mock(spec=LLMClient)
    mock_llm_client.generate.side_effect = spy_generate
    token_counter = TokenCounter()

    context_manager = LLMSummarizingContextManager(
        client=mock_llm_client,
        token_counter=token_counter,
        logger=mock_logger,
        token_budget=1000,
        max_size=6,  # Small size to trigger multiple summarizations
        keep_first=1,
    )

    # First conversation that will trigger first summarization
    first_conversation = [
        [TextPrompt(text="Read config.py")],
        [
            ToolCall(
                tool_call_id="call_1",
                tool_name="read_file",
                tool_input={"file_path": "config.py"},
            )
        ],
        [
            ToolFormattedResult(
                tool_call_id="call_1", tool_name="read_file", tool_output="DEBUG = True"
            )
        ],
        [TextResult(text="Config file read successfully.")],
        [TextPrompt(text="Add error handling")],
        [
            ToolCall(
                tool_call_id="call_2",
                tool_name="edit_file",
                tool_input={"file_path": "main.py", "content": "try/catch added"},
            )
        ],
        [
            ToolFormattedResult(
                tool_call_id="call_2",
                tool_name="edit_file",
                tool_output="File modified",
            )
        ],
        [TextResult(text="Error handling added.")],
    ]

    # First truncation
    first_result = context_manager.apply_truncation_if_needed(first_conversation)

    # Add more conversation to trigger second summarization
    additional_conversation = [
        [TextPrompt(text="Add logging")],
        [
            ToolCall(
                tool_call_id="call_3",
                tool_name="edit_file",
                tool_input={"file_path": "main.py", "content": "logging added"},
            )
        ],
        [
            ToolFormattedResult(
                tool_call_id="call_3",
                tool_name="edit_file",
                tool_output="Logging added",
            )
        ],
        [TextResult(text="Logging functionality added.")],
        [TextPrompt(text="Create tests")],
        [
            ToolCall(
                tool_call_id="call_4",
                tool_name="create_file",
                tool_input={"file_path": "test_main.py", "content": "unit tests"},
            )
        ],
        [
            ToolFormattedResult(
                tool_call_id="call_4",
                tool_name="create_file",
                tool_output="Test file created",
            )
        ],
        [TextResult(text="Unit tests created.")],
    ]

    # Extend and apply second truncation
    extended_conversation = first_result + additional_conversation
    final_result = context_manager.apply_truncation_if_needed(extended_conversation)

    print(f"Total LLM calls made: {len(llm_calls)}")

    # Should have made 2 LLM calls (one for each summarization round)
    assert len(llm_calls) == 2, f"Expected 2 LLM calls, got {len(llm_calls)}"

    # Inspect first call
    first_call = llm_calls[0]
    assert "<PREVIOUS SUMMARY>" in first_call["prompt_text"]
    assert first_call["call_number"] == 1

    # Inspect second call
    second_call = llm_calls[1]
    assert "<PREVIOUS SUMMARY>" in second_call["prompt_text"]
    print(second_call["prompt_text"])
    previous_summary_match = re.search(
        r"<PREVIOUS SUMMARY>(.*?)</PREVIOUS SUMMARY>",
        second_call["prompt_text"],
        re.DOTALL,
    )
    assert previous_summary_match is not None, (
        "Could not find PREVIOUS SUMMARY section in prompt"
    )
    previous_summary = previous_summary_match.group(1)
    assert "Flask development" in previous_summary
    assert second_call["call_number"] == 2
    # Second call should include the previous summary
    assert "Flask development" in second_call["prompt_text"]  # From first summary

    print("✅ Multiple summarization rounds test passed!")
    print(f"📝 First call prompt preview: {first_call['prompt_text'][:100]}...")
    print(f"📝 Second call prompt preview: {second_call['prompt_text'][:100]}...")


def test_summarization_with_images():
    """Test summarization behavior with image-related content in the conversation."""
    
    llm_calls = []

    def spy_generate(messages, max_tokens=None, **kwargs):
        call_info = {
            "messages": messages,
            "max_tokens": max_tokens,
            "prompt_text": messages[0][0].text if messages and messages[0] and hasattr(messages[0][0], "text") else None,
            "call_number": len(llm_calls) + 1,
        }
        llm_calls.append(call_info)

        # Return a summary that includes image-related context
        summary = """USER_CONTEXT: Image processing and analysis task
COMPLETED: Loaded image 'sample.jpg', applied Gaussian blur filter
PENDING: Generate image report
CODE_STATE: Image processing pipeline with OpenCV
CHANGES: Processed images with blur filter"""

        return [TextResult(text=summary)], None

    mock_logger = Mock(spec=logging.Logger)
    mock_llm_client = Mock(spec=LLMClient)
    mock_llm_client.generate.side_effect = spy_generate
    token_counter = TokenCounter()

    context_manager = LLMSummarizingContextManager(
        client=mock_llm_client,
        token_counter=token_counter,
        logger=mock_logger,
        token_budget=1000,
        max_size=6,
        keep_first=1,
    )

    # Test image data
    test_image_data = {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": "fake_base64_data_for_testing_purposes_only"
        }
    }

    # Create a conversation with image-related operations
    conversation = [
        [TextPrompt(text="Can you analyze the image in sample.jpg?")],
        [
            ToolCall(
                tool_call_id="img_1",
                tool_name="read_image",
                tool_input={"file_path": "sample.jpg"},
            )
        ],
        [
            ToolFormattedResult(
                tool_call_id="img_1",
                tool_name="read_image",
                tool_output=[test_image_data],
            )
        ],
        [TextResult(text="I can see the image. It's an RGB image of size 800x600.")],
        [
            TextPrompt(text="Apply a Gaussian blur filter"),
            ImageBlock(**test_image_data)
        ],
        [
            ToolCall(
                tool_call_id="img_2",
                tool_name="process_image",
                tool_input={"operation": "gaussian_blur", "sigma": 1.5},
            )
        ],
        [
            ToolFormattedResult(
                tool_call_id="img_2",
                tool_name="process_image",
                tool_output=[test_image_data],
            )
        ],
        [TextResult(text="The blur filter has been applied to reduce noise.")],
    ]

    # Apply truncation to trigger summarization
    result = context_manager.apply_truncation_if_needed(conversation)

    # Verify the summarization occurred
    assert len(llm_calls) == 1, "Expected one LLM call for summarization"
    
    # Check the structure of the summarized conversation
    assert len(result) == 3, "Expected 3 parts: first message, summary, and last message"
    
    # Verify the first message is kept
    assert result[0] == conversation[0], "First message should be preserved"
    
    # Verify the summary is inserted
    assert isinstance(result[1][0], TextPrompt)
    assert result[1][0].text.startswith("Conversation Summary:")
    
    # Verify the last message is kept
    assert result[2] == conversation[-1], "Last message should be preserved"
    
    # Check that the summary includes image-related context
    summary_prompt = llm_calls[0]["prompt_text"]
    assert "sample.jpg" in summary_prompt, "Summary should mention the image file"
    assert "gaussian_blur" in summary_prompt, "Summary should mention the image processing operation"
    
    print("--------------------------------")
    print(summary_prompt)
    print("--------------------------------")
    
    print("✅ Image summarization test passed!")
