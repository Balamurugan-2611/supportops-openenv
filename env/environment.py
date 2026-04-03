import random
from .models import Observation, Action, StepResult
from .tasks import TASKS
from .graders import compute_reward

class SupportOpsEnv:

    def __init__(self):
        self.state_data = {}
        self.current_task = None
        self.max_steps = 4

    def reset(self):
        self.current_task = random.choice(TASKS)
        self.state_data = {
            "history": [],
            "step": 0,
            "done": False
        }

        return self._get_obs()

    def step(self, action: Action):
        if self.state_data["done"]:
            return StepResult(
                observation=self._get_obs(),
                reward=0.0,
                done=True
            )

        self.state_data["step"] += 1

        reward = compute_reward(
            self.current_task,
            action,
            self.state_data["step"],
            self.state_data["history"]
        )

        self.state_data["history"].append(action.message)

        done = (
            action.action_type == self.current_task["correct_action"]
            or self.state_data["step"] >= self.max_steps
        )

        self.state_data["done"] = done

        return StepResult(
            observation=self._get_obs(),
            reward=reward,
            done=done,
            info={"task_id": self.current_task["id"]}
        )

    def state(self):
        return self.state_data

    def _get_obs(self):
        return Observation(
            ticket_text=self.current_task["ticket"],
            history=self.state_data["history"],
            step_count=self.state_data["step"],
            status="resolved" if self.state_data["done"] else "ongoing",
            detected_category=self.current_task["category"]
        )