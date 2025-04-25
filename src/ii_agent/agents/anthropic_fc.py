import asyncio
import logging
from typing import Optional
from fastapi import WebSocket
from ii_agent.agents.base import BaseAgent
from ii_agent.llm.base import LLMClient
from ii_agent.prompts.system_prompt import SYSTEM_PROMPT
from ii_agent.tools import (
    CompleteTool,
    create_bash_tool,
    create_docker_bash_tool,
    StrReplaceEditorTool,
    SequentialThinkingTool,
    TavilySearchTool,
    TavilyVisitWebpageTool,
    FileWriteTool,
    StaticDeployTool,
)
from ii_agent.tools.base import DialogMessages
from ii_agent.utils import WorkspaceManager


class AnthropicFC(BaseAgent):
    name = "general_agent"
    description = """\
A general agent that can accomplish tasks and answer questions.

If you are faced with a task that involves more than a few steps, or if the task is complex, or if the instructions are very long,
try breaking down the task into smaller steps. After call this tool to update or create a plan, use write_file or str_replace_tool to update the plan to todo.md
"""
    input_schema = {
        "type": "object",
        "properties": {
            "instruction": {
                "type": "string",
                "description": "The instruction to the agent.",
            },
        },
        "required": ["instruction"],
    }

    def _get_system_prompt(self):
        """Get the system prompt, including any pending messages.

        Returns:
            The system prompt with messages prepended if any
        """
        return SYSTEM_PROMPT.format(
            workspace_root=self.workspace_manager.root,
        )

    def __init__(
        self,
        client: LLMClient,
        workspace_manager: WorkspaceManager,
        logger_for_agent_logs: logging.Logger,
        max_output_tokens_per_turn: int = 8192,
        max_turns: int = 10,
        use_prompt_budgeting: bool = True,
        ask_user_permission: bool = False,
        docker_container_id: Optional[str] = None,
        websocket: Optional[WebSocket] = None,
        file_server_port: int = 8088,
    ):
        """Initialize the agent.

        Args:
            client: The LLM client to use
            max_output_tokens_per_turn: Maximum tokens per turn
            max_turns: Maximum number of turns
            workspace_manager: Optional workspace manager for taking snapshots
        """
        super().__init__()
        self.client = client
        self.logger_for_agent_logs = logger_for_agent_logs
        self.max_output_tokens = max_output_tokens_per_turn
        self.max_turns = max_turns
        self.workspace_manager = workspace_manager
        self.interrupted = False
        self.dialog = DialogMessages(
            logger_for_agent_logs=logger_for_agent_logs,
            use_prompt_budgeting=use_prompt_budgeting,
        )

        # Create and store the complete tool
        self.complete_tool = CompleteTool()

        if docker_container_id is not None:
            self.logger_for_agent_logs.info(
                f"Enabling docker bash tool with container {docker_container_id}"
            )
            bash_tool = create_docker_bash_tool(
                container=docker_container_id,
                ask_user_permission=ask_user_permission,
            )
        else:
            bash_tool = create_bash_tool(
                ask_user_permission=ask_user_permission,
                cwd=workspace_manager.root,
            )

        self.message_queue = asyncio.Queue()
        # Start file server
        self.file_server_port = file_server_port

        # Initialize tools with file server port
        self.tools = [
            bash_tool,
            StrReplaceEditorTool(workspace_manager=workspace_manager),
            SequentialThinkingTool(),
            TavilySearchTool(),
            TavilyVisitWebpageTool(),
            # BrowserUse(message_queue=self.message_queue),
            self.complete_tool,
            FileWriteTool(),
            StaticDeployTool(
                workspace_manager=workspace_manager,
                file_server_port=self.file_server_port,
            ),
        ]
        self.websocket = websocket
