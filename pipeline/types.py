"""Shared type definitions for the LLM progress monitor pipeline."""

from typing import TypedDict, Optional
from pydantic import BaseModel, Field


class PromptData(BaseModel):
    """Type definition for input prompt data from JSON files with runtime validation."""
    instruction: str = Field(..., description="The instruction/prompt text")
    category: Optional[str] = Field(None, description="Optional category label")


class ChatMessage(TypedDict):
    """Type definition for a chat message."""
    role: str
    content: str


class RolloutResponse(BaseModel):
    """Type definition for rollout response data with runtime validation."""
    instruction: str = Field(..., description="The instruction/prompt text")
    response: str = Field(..., description="The model's generated response")
    char_length: int = Field(..., description="Character length of response")
    tokens_length: int = Field(..., description="Token length of response")


class FormattedResponse(TypedDict):
    """Type definition for a formatted response with chat template."""
    instruction: str
    response: str
    chat_formatted: str
    char_length: int
    tokens_length: int

