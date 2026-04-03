---
title: SupportOps OpenEnv
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---

# SupportOps Pro - OpenEnv

## Overview
A realistic customer support simulation environment where AI agents must classify, decide, and respond.

## Workflow
1. Understand issue
2. Classify category
3. Choose correct action
4. Respond professionally

## Tasks
- Easy: Delivery delay
- Medium: Double charge
- Hard: Technical crash scenarios

## Reward Design
- Classification accuracy
- Correct decision making
- Response quality
- Penalties for repetition and invalid actions

## Action Space
Structured JSON:
- classification
- action_type
- message

## Observation Space
- ticket_text
- history
- step_count
- status
- detected_category

## Run

## Example Episode

**Input Ticket:**
"My order hasn't arrived yet."

**Step 1:**
- Classification: delivery  
- Action: check_status  
- Response: "Sorry for the delay. Let me check your order status."

**Reward:** 0.9  
**Done:** True

---

## Design Highlights

- Multi-step decision-making environment
- Structured action space (classification + action + response)
- Real-world simulation of customer support workflows
- Deterministic grading for reproducibility
- Reward shaping for partial progress

---

## Why This Matters

Customer support is a critical real-world task involving:
- intent understanding
- decision making
- communication quality

This environment enables training and evaluation of AI agents in realistic support workflows.