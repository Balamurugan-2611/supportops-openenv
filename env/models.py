from pydantic import BaseModel
from typing import List, Optional

class Observation(BaseModel):
    ticket_text: str
    history: List[str]
    step_count: int
    status: str
    detected_category: str

class Action(BaseModel):
    classification: str
    action_type: str
    message: str

class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Optional[dict] = {}