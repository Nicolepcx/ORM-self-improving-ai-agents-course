# MCP + MCTS Coding Agents

This repository contains the code and Docker setup for the **Demo** session for **Develop Self-Improving AI Agents with Reinforcement Learning**

---

## Demo Overview

This code shows how **autonomous agents can use MCP not just to connect with tools but to communicate with each other**.  

We run a **multi-agent coding system** where agents collaborate by:

- generating Python solutions,  
- testing and benchmarking them inside a **sandbox**,  
- refining based on structured feedback,  
- and evolving code together toward higher quality.  

MCP acts as the **shared protocol** between reasoning agents, enabling **structured coordination, reflection, and continuous innovation**.  
This is a step toward **self-improving AI systems**.


## Why MCP?

Traditional local execution (like a Python REPL) comes with risks and limitations:

- **Security risks**: Executing arbitrary generated code on the host machine can lead to data leaks or system damage.  
- **Lack of isolation**: Mistakes (e.g. infinite loops, recursion errors) can hang the process.  
- **Difficult scaling**: Running code on multiple agents across nodes requires orchestration and sandboxing.  

**MCP (Model Context Protocol)** solves these problems:

- Runs Python in **Pyodide + Deno**, sandboxed from the host OS.  
- Automatically **installs packages** into the sandbox when needed.  
- Captures **stdout, stderr, and return values** cleanly.  
- Works **across nodes and clusters** (e.g. Kubernetes with gVisor).  
- Provides a **shared protocol for inter-agent communication**, not just tool calls.  

This makes MCP an excellent fit for **multi-agent systems** where reasoning agents collaborate safely.

---



## Prerequisites

- **Docker** installed (version 24+ recommended).
- An OpenAI API key stored in a `.env` file.

Your `.env` file should look like this:

```bash
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o
```

---

## Step 1: Build Docker image

The provided Dockerfile builds a container with:
- Python 3.11 slim
- Deno (for MCP Pyodide backend)
- mcp-run-python, treequest, langgraph, langchain-openai, and other dependencies

Build the image:

```bash
docker build -t mcp-run-python:latest .

```

## Step 2: Start the MCP server (stdio mode)

This runs the Python MCP server inside the container:

```bash
docker run --rm -it mcp-run-python:latest /usr/local/bin/entrypoint.sh stdio

```

## Step 3: Run the Tree Search + MCP Demo

From the project root, run:

```bash
docker run --rm -it \
  --env-file .env \
  -v "$PWD":/app -w /app \
  mcp-run-python:latest \
  python treesearch_fib.py

```

NOTE: Make sure your API_key isn't surrounded by `"` because this will give you a key error, as this won't map correctly. 

This executes the LangGraph + TreeQuest + MCP sandboxed agents pipeline.
The system will:

- Iteratively generate Fibonacci implementations,
- Run unit tests and benchmarks in the sandbox,
- Refine the code based on structured feedback,
- Track performance improvements step by step,
- And finally print the best evolved solution with a score.

## Example Output

```plaintext

[Step 1] score=0.905 tests_ok=True rt=1.509 contract=sequence growth=1 note=all tests passed
[Step 2] score=0.905 tests_ok=True rt=2.875 contract=sequence growth=1 note=all tests passed
[Step 3] score=0.905 tests_ok=True rt=2.031 contract=sequence growth=1 note=all tests passed
[Step 4] score=0.942 tests_ok=True rt=1.337 contract=sequence growth=1 note=all tests passed
[Step 5] score=0.942 tests_ok=True rt=2.282 contract=sequence growth=1 note=all tests passed
Final Best Answer score=0.942

Best answer

```python
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b
```


---

## Key Idea

- Each agent **generates or refines code** using the LLM.  
- MCP provides a **safe execution sandbox** for correctness + performance checks.  
- Agents communicate through MCP, sharing results and corrections.  
- Over iterations, the system **self-improves** its code automatically.  

---

## How to Expand This Demo

This repo is a **boilerplate** — the intention is to move away from the idea that MCP is *just* a tool connector.  
Instead, think of MCP as a **shared infrastructure for agent collaboration**.

Here are some ways to extend this demo:

### 1. Sandbox Variations
- Run MCP servers on **Kubernetes with gVisor** for secure multi-node isolation.  
- Swap in **E2B sandboxes** for cloud-based ephemeral compute.  
- Add specialized MCP servers (databases, APIs, file systems).

### 2. Search Algorithms
- Replace **TreeQuest AB-MCTS** with:
  - Classic MCTS  
  - Beam search  
  - Best-of-N sampling  
  - Hybrid evolutionary search.  

Each will shape how agents explore and refine code.

### 3. Benchmarks
- Extend beyond runtime + memory:
  - Add **time complexity inference** from LLMs.  
  - Include **robustness benchmarks** (edge cases, random inputs).  
  - Add **energy efficiency or cost metrics** for sustainability-aware code.  

### 4. Unit Tests
- Provide **custom unit test harnesses** for different problem domains (sorting, graph algorithms, data structures).  
- Introduce an **agent that writes new unit tests on the fly** to challenge other agents.  

### 5. Multi-Agent Roles
- Add agents with **distinct roles**:
  - *Coder* agent: writes code.  
  - *Tester* agent: builds unit tests dynamically.  
  - *Benchmark agent*: runs stress tests.  
  - *Reviewer agent*: evaluates readability and style.  

These can coordinate via MCP to continuously improve solutions.

---

## License

This demo is provided as part of **O’Reilly Demo Day** educational content.  
It is licensed under the [MIT License](LICENSE).

