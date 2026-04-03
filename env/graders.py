def compute_reward(task, action, step, history):
    score = 0.0

    # classification accuracy
    if action.classification == task["category"]:
        score += 0.3

    # correct action
    if action.action_type == task["correct_action"]:
        score += 0.4

    msg = action.message.lower()

    # politeness
    if any(word in msg for word in ["sorry", "apologize", "apology"]):
        score += 0.1

    # meaningful response
    if len(msg) > 20:
        score += 0.1

    # repetition penalty
    if action.message in history:
        score -= 0.2

    # invalid action penalty
    if action.action_type not in ["refund", "check_status", "escalate"]:
        score -= 0.2

    return max(0.0, min(score, 1.0))