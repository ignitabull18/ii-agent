import asyncio

from asyncio import Queue
from typing import Any, Optional
from ii_agent.browser.browser import Browser
from ii_agent.tools.base import ToolImplOutput
from ii_agent.tools.browser_tools import BrowserTool, utils
from ii_agent.llm.message_history import MessageHistory


class BrowserWaitTool(BrowserTool):
    name = "browser_wait"
    description = "Wait for the page to load"
    input_schema = {"type": "object", "properties": {}, "required": []}

    def __init__(self, browser: Browser, message_queue: Optional[Queue] = None):
        super().__init__(browser, message_queue)

    async def _run(
        self,
        tool_input: dict[str, Any],
        message_history: Optional[MessageHistory] = None,
    ) -> ToolImplOutput:
        await asyncio.sleep(1)
        state = await self.browser.update_state()
        state = await self.browser.handle_pdf_url_navigation()
        self.log_browser_state(state)
        msg = "Waited for page"

        return utils.format_screenshot_tool_output(state.screenshot, msg)
