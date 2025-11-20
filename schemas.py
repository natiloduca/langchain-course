from typing import List

from pydantic import BaseModel, Field


class Source(BaseModel):
    """Schema for a source used by an agent."""

    url: str = Field(description="The URL of the source.")


class AgentResponse(BaseModel):
    """Schema for an agent's response."""

    answer: str = Field(description="The answer provided by the agent.")
    sources: List[Source] = Field(
        default_factory=list,
        description="A list of sources used by the agent to generate the answer.",
    )
