import asyncio

from asyncio import Queue
from typing import Any, Optional
from ii_agent.browser.browser import Browser
from ii_agent.tools.base import ToolImplOutput
from ii_agent.tools.browser_tools import BrowserTool, utils
from ii_agent.llm.message_history import MessageHistory


class BrowserPressKeyTool(BrowserTool):
    name = "browser_press_key"
    description = "Simulate key press in the current browser page"
    input_schema = {
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": "Key name to simulate (e.g., Enter, Tab, ArrowUp), supports key combinations (e.g., Control+Enter).",
            }
        },
        "required": ["key"],
    }

    def __init__(self, browser: Browser, message_queue: Optional[Queue] = None):
        super().__init__(browser, message_queue)

    async def _run(
        self,
        tool_input: dict[str, Any],
        message_history: Optional[MessageHistory] = None,
    ) -> ToolImplOutput:
        key = tool_input["key"]
        page = await self.browser.get_current_page()
        await page.keyboard.press(key)
        await asyncio.sleep(0.5)

        msg = f'Pressed "{key}" on the keyboard.'
        state = await self.browser.update_state()
        self.log_browser_state(state)
        return utils.format_screenshot_tool_output(state.screenshot, msg)
