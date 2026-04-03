from fastapi import FastAPI
from env.environment import SupportOpsEnv
from env.models import Action

app = FastAPI()
env = SupportOpsEnv()

@app.get("/")
def home():
    return {"status": "running"}

@app.get("/reset")
@app.post("/reset")
def reset():
    obs = env.reset()
    return obs.dict()

@app.get("/step")
def step_get():
    act = Action(
        classification="delivery",
        action_type="check_status",
        message="Checking your order"
    )
    result = env.step(act)
    return {
        "observation": result.observation.dict(),
        "reward": result.reward,
        "done": result.done,
        "info": result.info
    }
