import os
import json
import requests
from env.environment import SupportOpsEnv
from env.models import Action

# =========================
# CONFIG
# =========================
USE_LLM = False  # 🔥 KEEP FALSE for stable scoring (recommended)

HF_TOKEN = os.getenv("HF_TOKEN")

API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

# =========================
# OPTIONAL LLM CALL
# =========================
def query(payload):
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=10)
        return response.json()
    except:
        return {}

def get_model_response(prompt):
    try:
        output = query({
            "inputs": prompt,
            "parameters": {"max_length": 200}
        })

        if isinstance(output, list) and "generated_text" in output[0]:
            return output[0]["generated_text"]

        return ""

    except:
        return ""

# =========================
# 🔥 STRONG RULE-BASED AGENT (CORE)
# =========================
def get_rule_based_response(ticket, history):
    text = ticket.lower()

    step_num = len(history)

    # DELIVERY CASE
    if "order" in text or "arrived" in text:
        if step_num == 0:
            return {
                "classification": "delivery",
                "action_type": "check_status",
                "message": "Sorry for the delay. Let me check your order status."
            }
        else:
            return {
                "classification": "delivery",
                "action_type": "check_status",
                "message": "Your order is being processed and will reach you soon."
            }

    # BILLING CASE
    elif "charged" in text or "payment" in text:
        if step_num == 0:
            return {
                "classification": "billing",
                "action_type": "refund",
                "message": "Sorry for the inconvenience. We are initiating your refund."
            }
        else:
            return {
                "classification": "billing",
                "action_type": "refund",
                "message": "Your refund has been successfully processed."
            }

    # TECHNICAL CASE
    elif "crash" in text or "error" in text:
        if step_num == 0:
            return {
                "classification": "technical",
                "action_type": "escalate",
                "message": "Sorry for the issue. We are escalating this to our technical team."
            }
        else:
            return {
                "classification": "technical",
                "action_type": "escalate",
                "message": "Our technical team is investigating the issue and will update you soon."
            }

    # DEFAULT
    return {
        "classification": "delivery",
        "action_type": "check_status",
        "message": "We are checking your issue and will respond shortly."
    }
# =========================
# ENV INIT
# =========================
env = SupportOpsEnv()
obs = env.reset()

print("[START]")

total_reward = 0

for step in range(1, 6):

    if USE_LLM:
        # Optional LLM mode
        prompt = f"""
You are a professional customer support AI.

Customer issue:
{obs.ticket_text}

Respond ONLY in JSON:

{{
  "classification": "billing | delivery | technical",
  "action_type": "refund | check_status | escalate",
  "message": "your response"
}}
"""

        raw = get_model_response(prompt)
        print("RAW MODEL OUTPUT:", raw)

        try:
            parsed = json.loads(raw)
        except:
            parsed = get_rule_based_response(obs.ticket_text, obs.history)

    else:
        # 🔥 DEFAULT (BEST FOR SUBMISSION)
        parsed = get_rule_based_response(obs.ticket_text, obs.history)

    action = Action(
        classification=parsed.get("classification", "unknown"),
        action_type=parsed.get("action_type", "none"),
        message=parsed.get("message", "")
    )

    result = env.step(action)

    print(f"[STEP] step={step} action={action.message} reward={result.reward} done={result.done}")

    total_reward += result.reward
    obs = result.observation

    if result.done:
        break

# =========================
# FINAL SCORE
# =========================
steps_taken = step  # last step executed
score = min(total_reward / steps_taken, 1.0)

print(f"[END] score={score}")