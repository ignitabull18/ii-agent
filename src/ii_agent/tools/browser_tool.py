import json
import time
from ii_agent.tools.base import (
    DialogMessages,
    LLMTool,
    ToolImplOutput,
)
from ii_agent.browser.browser import Browser
from typing import Any, Optional
import os


class BrowserNavigationTool(LLMTool):
    name = "browser_navigation"
    description = "Navigate browser to specified URL."
    input_schema = {
        "type": "object",
        "properties": {
          "url": {
            "type": "string",
            "description": "Complete URL to visit. Must include protocol prefix."
          }
        },
        "required": ["url"]
      }
    output_type = "string"

    def __init__(self, browser: Browser):
        self.browser = browser

    def run_impl(
        self,
        tool_input: dict[str, Any],
        dialog_messages: Optional[DialogMessages] = None,
    ) -> ToolImplOutput:
        url = tool_input["url"]
        
        page = self.browser.get_current_page()
        page.goto(url, wait_until="domcontentloaded")
        time.sleep(1.5)
        msg = f"Navigated to {url}"
        return ToolImplOutput(msg, msg)


class BrowserViewTool(LLMTool):
    name = "browser_view"
    description = """\
Capture the current view of the browser page.

Use this tool to inspect the latest content of the currently loaded web page. It is useful for reviewing either interactive or static pages depending on the context.

- Set `return_screenshot_with_interactive_elements` to `true` when the task involves interacting with elements on the page (e.g., clicking buttons, filling forms, selecting dropdown options). This will return a screenshot with interactive elements visually highlighted and listed with their metadata.

- Set `return_screenshot_with_interactive_elements` to `false` when you only need to view the page content (e.g., scrolling through a blog post, reading a PDF, or skimming for information). This will return a clean screenshot without overlays.

If you've just navigated to a new page and are unsure of the next step, start with `return_screenshot_with_interactive_elements` set to `false` to get a general view of the page before deciding whether interaction is necessary.\
"""
    input_schema = {
        "type": "object",
        "properties": {
        "return_screenshot_with_interactive_elements": {
            "type": "boolean",
            "description": "If true, includes a screenshot with interactive elements visually highlighted and listed. If false, returns a plain screenshot of the current page."
        }
        },
        "required": ["return_screenshot_with_interactive_elements"]
    }

    output_type = "string"

    def __init__(self, browser: Browser):
        self.browser = browser

    def run_impl(
        self,
        tool_input: dict[str, Any],
        dialog_messages: Optional[DialogMessages] = None,
    ) -> ToolImplOutput:
        url = tool_input["url"]
        
        page = self.browser.get_current_page()
        page.goto(url, wait_until="domcontentloaded")
        time.sleep(1.5)
        msg = f"Navigated to {url}"
        return ToolImplOutput(msg, msg)