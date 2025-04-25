from pydantic import BaseModel
from typing import Literal, Any, Dict


class RealtimeEvent(BaseModel):
    type: Literal["make_response", "tool_call", "tool_result"]
    content: dict[str, Any]


# WebSocket message models
class ClientMessage(BaseModel):
    type: str
    content: Dict[str, Any]


class ServerMessage(BaseModel):
    type: str
    content: Dict[str, Any]
