import asyncio

from asyncio import Queue
from typing import Optional
from ii_agent.tools.base import (
    LLMTool,
    ToolImplOutput,
)
from ii_agent.browser.browser import Browser, BrowserState
from ii_agent.llm.message_history import MessageHistory
from ii_agent.core.event import EventType, RealtimeEvent
from typing import Any, Optional


def get_event_loop():
    try:
        # Try to get the existing event loop
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # If no event loop exists, create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def format_screenshot_tool_output(screenshot: str, msg: str) -> ToolImplOutput:
    return ToolImplOutput(
        tool_output=[
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": screenshot,
                },
            },
            {"type": "text", "text": msg},
        ],
        tool_result_message=msg,
    )


class BrowserTool(LLMTool):
    def __init__(self, browser: Browser, message_queue: Optional[Queue] = None):
        self.browser = browser
        self.message_queue = message_queue

    async def _run(
        self,
        tool_input: dict[str, Any],
        message_history: Optional[MessageHistory] = None,
    ) -> ToolImplOutput:
        raise NotImplementedError("Subclasses must implement this method")

    def run_impl(
        self,
        tool_input: dict[str, Any],
        message_history: Optional[MessageHistory] = None,
    ) -> ToolImplOutput:
        loop = get_event_loop()
        return loop.run_until_complete(self._run(tool_input, message_history))

    def log_browser_state(self, state: BrowserState):
        if self.message_queue:
            self.message_queue.put_nowait(
                RealtimeEvent(
                    type=EventType.BROWSER_USE,
                    content={
                        "url": state.url,
                        "screenshot": state.screenshot,
                        "screenshot_with_highlights": state.screenshot_with_highlights,
                    },
                )
            )