⭐ If you find this repository helpful, please consider giving it a ⭐ here on GitHub (click the star button in the top right corner) 
It's a quick way to show support for this openly available code. ⭐

![OReilly_logo_rgb.png](resources%2FOReilly_logo_rgb.png)

Here is the improved **README.md** with your headline, your structure, and your new folder organization for **demos**, **hands-on exercises**, and **example solutions**. Everything is clean, consistent, and ready for a real repository layout.

---

# Develop Self-Improving AI Agents with Reinforcement Learning

# Reinforcement Learning for LLMs – Code Repository

This repository provides code examples, notebooks, and reference implementations used throughout the workshop on reinforcement learning for large language models. It covers policy rollouts, reward modeling, trajectory generation, optimization methods, and tool-use training for agentic systems.

## Repository Structure

```
.
├── demos/                # Notebooks shown in live demonstrations  
├── hands_on/             # Participant exercises  
├── solutions/            # Reference solutions for hands-on work   
└── README.md
```

---

## Contents

### Section 1 • Foundations of RL for LLMs

Core concepts:

* Policies, rollouts, trajectories, rewards
* Minimal reinforcement loop for LLMs
* Relative feedback integration

Included:
* `demos/01_rl_basic_loop.ipynb` – minimal RL loop implementation
* `hands_on/01_frozen_lake_exercise.ipynb`
* `solutions/01_frozen_lake_solution.ipynb`

---

### Section 2 • Reward Modeling and Relative Scoring

Core concepts:

* Reward shaping for LLM behavior
* Relative ranking vs absolute scoring
* RULER: Relative Universal LLM-Elicited Rewards

Included:

* `demos/02_ruler_relative_scoring_demo.ipynb`
* `hands_on/02_reward_modeling_exercise.ipynb`
* `solutions/02_reward_modeling_solution.ipynb`

---

### Section 3 • Agent Trajectories and Rollout Design

Core concepts:

* Agent Reinforcement Trainer (ART) rollout generation
* Message sequences, actions, and feedback
* Multi-trajectory sampling for optimization

Included:

* `demos/03_art_rollout_generation.ipynb`
* `hands_on/03_rollout_design_exercise.ipynb`
* `solutions/03_rollout_design_solution.ipynb`

---

### Section 4 • Optimization and Continuous Improvement

Core concepts:

* GRPO: Group Relative Policy Optimization
* GSPO: Group Sequence Policy Optimization
* Stabilizing RL for large models including MoE architectures

Included:

* `demos/04_grpo_gspo_optimization.ipynb`
* `hands_on/04_optimization_exercise.ipynb`
* `solutions/04_optimization_solution.ipynb`

---

### Section 5 • Teaching Agents Tool Mastery (MCP Integration)

Core concepts:

* Automatic tool-use training via structured task sets
* Using RULER for tool-use reward evaluation
* MCP-based agent to agent communication

Included:

* `demos/05_mcp_tool_training.ipynb`
* `hands_on/05_mcp_training_exercise.ipynb`
* `solutions/05_mcp_training_solution.ipynb`

---

## Requirements

* Python 3.10
* Dependencies listed in `requirements.txt`
* Optional: MCP server for tool-use training

---


## Running the Notebooks

Every notebook contains buttons so that the notebook can be opened and run on Google Colab like this:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]()   
