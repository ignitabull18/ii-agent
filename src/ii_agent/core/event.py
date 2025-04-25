from pydantic import BaseModel
from typing import Literal, Any


class RealtimeEvent(BaseModel):
    type: Literal["make_response", "tool_call", "tool_result"]
    raw_message: dict[str, Any]
