"""Provider-facing motion instructions.

Adapters receive this object. They do not interpret the raw motion-plan JSON.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TimelineCue:
    start_seconds: float
    end_seconds: float
    instruction: str


@dataclass(frozen=True)
class MotionInstructions:
    prompt: str
    negative_prompt: str
    identity_lock: str
    camera_lock: str
    timeline: tuple[TimelineCue, ...]
    warnings: tuple[str, ...]
