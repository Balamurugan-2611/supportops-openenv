from fastapi import FastAPI
from env.environment import SupportOpsEnv
from env.models import Action

app = FastAPI()
env = SupportOpsEnv()

@app.get("/")
def home():
    return {"status": "running"}

@app.post("/reset")
def reset():
    obs = env.reset()
    return obs.dict()

@app.post("/step")
def step(action: dict):
    act = Action(**action)
    result = env.step(act)
    return {
        "observation": result.observation.dict(),
        "reward": result.reward,
        "done": result.done,
        "info": result.info
    }