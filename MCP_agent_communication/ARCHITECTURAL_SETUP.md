# Architectural Setup: Agent-to-Agent Communication via MCP

This document describes the architecture of the multi-agent coding system and how **MCP (Model Context Protocol)** enables secure, structured agent-to-agent communication.

---

## System Overview

The system uses a **hierarchical architecture** combining:
- **MCTS (Monte Carlo Tree Search)** for exploration and optimization
- **LangGraph** for agent workflow orchestration
- **MCP Sandbox** for secure code execution and inter-agent communication

```mermaid
%%{init: {'theme': 'dark'}}%%
graph TB

  %% Cluster panel style (white background, solid border, fully opaque)
  classDef panel fill:#FFFFFF,stroke:#111827,color:#111827,stroke-width:2px,fill-opacity:1,stroke-opacity:1,opacity:1;

  %% Node styles (vivid, readable on white panels)
  classDef blueNode fill:#DBEAFE,stroke:#2563EB,color:#111827,stroke-width:2px,fill-opacity:1,opacity:1;
  classDef orangeNode fill:#FFEDD5,stroke:#F97316,color:#111827,stroke-width:2px,fill-opacity:1,opacity:1;
  classDef greenNode fill:#DCFCE7,stroke:#16A34A,color:#111827,stroke-width:2px,fill-opacity:1,opacity:1;
  classDef purpleNode fill:#F3E8FF,stroke:#A855F7,color:#111827,stroke-width:2px,fill-opacity:1,opacity:1;

  %% --- Top Level cluster (named so we can style it) ---
  subgraph TL["Top Level: MCTS Orchestrator"]
    MCTS[MCTS Search Tree]:::blueNode
    Actions[Agent Actions]:::blueNode
    MCTS -->|Explores| Actions
  end
  class TL panel;

  %% --- Agent cluster ---
  subgraph AG["Agent Subgraph: LangGraph Workflow"]
    START([START]):::blueNode --> Coder[Coder Agent]:::blueNode
    Coder -->|Code| Tester[Tester Agent]:::orangeNode
    Tester -->|Test Results| Reviewer[Reviewer Agent]:::greenNode
    Reviewer --> END([END]):::blueNode
  end
  class AG panel;

  %% --- Sandbox cluster ---
  subgraph SB["MCP Sandbox Layer"]
    Sandbox[MCP Sandbox: Pyodide + Deno]:::purpleNode
    Code[Python Code]:::purpleNode
    Results[Test Results]:::purpleNode
    Sandbox -->|Executes| Code
    Sandbox -->|Returns| Results
  end
  class SB panel;

  %% Cross edges
  Actions -->|Invokes| START
  END -->|Returns| NodeState[NodeState]:::blueNode

  Coder -.->|Executes via| Sandbox
  Tester -.->|Executes via| Sandbox
  Reviewer -.->|Executes via| Sandbox


```

---

## Agent Communication Flow

### How Agents Communicate

Agents communicate through **two primary mechanisms**:

1. **Shared State (LangGraph)**: Agents pass structured data through the workflow
2. **MCP Protocol**: Agents share execution results, test outcomes, and benchmarks via the sandbox

```mermaid
sequenceDiagram
    participant MCTS as MCTS Orchestrator
    participant Coder as Coder Agent
    participant Tester as Tester Agent
    participant Reviewer as Reviewer Agent
    participant MCP as MCP Sandbox
    
    MCTS->>Coder: Invoke with parent state
    Note over Coder: Generates/refines code<br/>using LLM
    
    Coder->>MCP: Execute code in sandbox
    MCP-->>Coder: Execution result
    Coder->>Coder: Create NodeState<br/>with code & score
    
    Coder->>Tester: Pass NodeState<br/>(llm_answer, score)
    
    Tester->>MCP: Run unit tests
    MCP-->>Tester: Test results (pass/fail)
    Tester->>MCP: Run benchmarks
    MCP-->>Tester: Runtime, memory, contract
    
    Tester->>Tester: Evaluate score<br/>Update NodeState
    Tester->>Reviewer: Pass NodeState<br/>(with test results)
    
    Reviewer->>Reviewer: Improve API/readability<br/>using LLM
    Reviewer->>MCP: Execute improved code
    MCP-->>Reviewer: Execution result
    
    Reviewer->>MCTS: Return final NodeState<br/>(with messages trace)
    
    Note over MCTS: Selects best candidate<br/>for next iteration
```

---

## Detailed Component Architecture

### 1. Agent Subgraph (LangGraph)

The agent subgraph defines a **sequential workflow** where each agent builds upon the previous agent's work:

```mermaid
stateDiagram-v2
    [*] --> Coder: Initial generation<br/>or refinement
    
    state Coder {
        [*] --> GenerateCode
        GenerateCode --> ExecuteSandbox
        ExecuteSandbox --> CreateState
        CreateState --> [*]
    }
    
    Coder --> Tester: NodeState with code
    
    state Tester {
        [*] --> ExtractCode
        ExtractCode --> RunUnitTests
        RunUnitTests --> RunBenchmarks
        RunBenchmarks --> EvaluateScore
        EvaluateScore --> [*]
    }
    
    Tester --> Reviewer: NodeState with tests
    
    state Reviewer {
        [*] --> ReviewCode
        ReviewCode --> ImproveAPI
        ImproveAPI --> ExecuteSandbox
        ExecuteSandbox --> FinalScore
        FinalScore --> [*]
    }
    
    Reviewer --> [*]: Final NodeState
```

**Key Communication Points:**
- **Coder → Tester**: Passes `llm_answer` (code) and initial `score`
- **Tester → Reviewer**: Passes test results, benchmarks, and updated `score`
- **Reviewer → MCTS**: Returns final `NodeState` with complete `messages` trace

### 2. MCP Sandbox Integration

The MCP sandbox serves as the **shared execution environment** and **communication medium**:

```mermaid
%%{init: {'theme': 'dark'}}%%
graph LR
  %% White subgraph panels
  classDef panel fill:#FFFFFF,stroke:#111827,color:#111827,stroke-width:2px,fill-opacity:1,opacity:1;

  %% Vivid light nodes that pop on white panels
  classDef blueNode fill:#DBEAFE,stroke:#2563EB,color:#111827,stroke-width:2px,fill-opacity:1,opacity:1;
  classDef orangeNode fill:#FFEDD5,stroke:#F97316,color:#111827,stroke-width:2px,fill-opacity:1,opacity:1;
  classDef greenNode fill:#DCFCE7,stroke:#16A34A,color:#111827,stroke-width:2px,fill-opacity:1,opacity:1;
  classDef purpleNode fill:#F3E8FF,stroke:#A855F7,color:#111827,stroke-width:2px,fill-opacity:1,opacity:1;

  subgraph AL["Agent Layer"]
    A1[Coder Agent]:::orangeNode
    A2[Tester Agent]:::blueNode
    A3[Reviewer Agent]:::greenNode
  end
  class AL panel;

  subgraph CL["MCP Sandbox Client"]
    SC[SandboxClient<br/>Synchronous Facade]:::purpleNode
  end
  class CL panel;

  subgraph PL["MCP Protocol Layer"]
    MCP[MCP Server<br/>stdio/HTTP]:::purpleNode
  end
  class PL panel;

  subgraph EX["Execution Environment"]
    Deno[Deno Runtime]:::blueNode
    Pyodide[Pyodide<br/>Python in WASM]:::orangeNode
  end
  class EX panel;

  A1 -->|eval code| SC
  A2 -->|run tests| SC
  A3 -->|execute| SC

  SC -->|MCP Protocol| MCP
  MCP -->|spawns| Deno
  Deno -->|runs| Pyodide

  Pyodide -->|results| Deno
  Deno -->|MCP response| MCP
  MCP -->|structured data| SC
  SC -->|NodeState updates| A1
  SC -->|NodeState updates| A2
  SC -->|NodeState updates| A3

    style Pyodide fill:#fff9c4
```

**MCP as Communication Protocol:**
- **Structured I/O**: All agents receive standardized responses (status, stdout, stderr, return_value)
- **Shared Context**: The sandbox maintains state across agent invocations
- **Isolation**: Each execution is isolated, preventing agent interference
- **Protocol Standardization**: MCP provides a common interface regardless of execution backend

### 3. Data Flow: NodeState Communication

Agents communicate through the `NodeState` dataclass, which carries:

```mermaid
classDiagram
    class NodeState {
        +string llm_answer
        +float score
        +bool tests_ok
        +map bench
        +string note
        +string[] messages
    }

    class AgentState {
        +NodeState parent
        +int step_idx
        +NodeState out
    }

    class LGState {
        +int iterations
        +string best_answer
        +float best_score
        +string trace
        +string[] best_messages
    }

    Coder --> NodeState
    Tester --> NodeState
    Reviewer --> NodeState
    NodeState --> AgentState
    AgentState --> LGState

    note for NodeState "Shared state object\\npassed between agents"

```

**Message Trace Example:**
```
[coder] produced initial code (score=0.901)
[tester] tests_ok=True rt=2.303584000000747
[reviewer] adjusted API/readability (score=0.350)
```

---

## MCTS Integration: Multi-Agent Exploration

The MCTS algorithm orchestrates multiple agent runs in parallel:

```mermaid
%%{init: {'theme': 'dark'}}%%
graph TB
  %% White subgraph panels (fully opaque)
  classDef panel fill:#FFFFFF,stroke:#111827,color:#111827,stroke-width:2px,fill-opacity:1,opacity:1;

  %% Vivid light nodes that pop on white panels
  classDef blueNode fill:#DBEAFE,stroke:#2563EB,color:#111827,stroke-width:2px,fill-opacity:1,opacity:1;
  classDef orangeNode fill:#FFEDD5,stroke:#F97316,color:#111827,stroke-width:2px,fill-opacity:1,opacity:1;
  classDef greenNode fill:#DCFCE7,stroke:#16A34A,color:#111827,stroke-width:2px,fill-opacity:1,opacity:1;
  classDef purpleNode fill:#F3E8FF,stroke:#A855F7,color:#111827,stroke-width:2px,fill-opacity:1,opacity:1;

  subgraph IT["MCTS Iteration"]
    Root[Root Node]:::blueNode
    Root -->|Action A| AgentRun1[Agent Run #1]:::orangeNode
    Root -->|Action B| AgentRun2[Agent Run #2]:::orangeNode
    Root -->|Action C| AgentRun3[Agent Run #3]:::orangeNode
  end
  class IT panel;

  subgraph RUN["Each Agent Run"]
    AgentRun1 --> Subgraph1[Agent Subgraph]:::blueNode
    AgentRun2 --> Subgraph2[Agent Subgraph]:::blueNode
    AgentRun3 --> Subgraph3[Agent Subgraph]:::blueNode
  end
  class RUN panel;

  subgraph SH["Shared MCP Sandbox"]
    Sandbox[MCP Sandbox]:::purpleNode
  end
  class SH panel;

  Subgraph1 -.->|uses| Sandbox
  Subgraph2 -.->|uses| Sandbox
  Subgraph3 -.->|uses| Sandbox

  Subgraph1 -->|NodeState| Score1[Score: 0.901]:::greenNode
  Subgraph2 -->|NodeState| Score2[Score: 0.850]:::greenNode
  Subgraph3 -->|NodeState| Score3[Score: 0.920]:::greenNode

  Score1 --> Best[Select Best]:::greenNode
  Score2 --> Best
  Score3 --> Best

  Best --> NextIter[Next MCTS Iteration]:::blueNode

```

**Key Points:**
- Multiple agent subgraphs run **concurrently** (via MCTS actions)
- All agents share the **same MCP sandbox instance** (thread-safe)
- MCTS selects the **best NodeState** based on score
- The best state becomes the **parent** for the next iteration

---

## Why MCP for Agent-to-Agent Communication?

### Traditional Approach (Without MCP)
```
Agent 1 → Direct Python exec() → Host OS
Agent 2 → Direct Python exec() → Host OS
❌ Security risks
❌ No isolation
❌ Difficult to coordinate
❌ No standardized protocol
```

### MCP-Enabled Approach
```
Agent 1 → MCP Protocol → Sandbox (isolated)
Agent 2 → MCP Protocol → Sandbox (isolated)
Agent 3 → MCP Protocol → Sandbox (isolated)
✅ Secure execution
✅ Complete isolation
✅ Shared protocol
✅ Structured communication
✅ Scalable across nodes
```

### Benefits for Multi-Agent Systems

1. **Protocol Standardization**
   - All agents use the same MCP interface
   - Consistent response format (status, stdout, stderr, return_value)
   - Easy to add new agents without changing communication layer

2. **Shared Execution Context**
   - Sandbox maintains state across agent invocations
   - Agents can build upon each other's work
   - Dependencies installed once, reused by all agents

3. **Isolation & Safety**
   - Each agent's code runs in isolation
   - No interference between concurrent agent runs
   - Timeout protection prevents hanging

4. **Distributed Communication**
   - MCP works over stdio, HTTP, or WebSocket
   - Agents can run on different nodes
   - Enables Kubernetes-based multi-agent systems

5. **Observability**
   - All agent interactions logged via MCP protocol
   - Message traces show agent handoffs
   - Easy to debug multi-agent workflows

---

## Execution Flow: Complete Example

Here's how a single MCTS iteration flows through the system:

```mermaid
%%{init: {
  "theme": "dark",
  "themeCSS": "
    /* node labels */
    .nodeLabel { font-size: 24px !important; font-weight: 600; }

    /* edge labels like |runs| */
    .edgeLabel { font-size: 16px !important; font-weight: 600; }

    /* subgraph titles */
    .cluster-label text { font-size: 18px !important; font-weight: 700; }

    /* general text fallback */
    text { font-size: 18px !important; }
  "
}}%%
flowchart TD
  classDef blueNode fill:#DBEAFE,stroke:#2563EB,color:#111827,stroke-width:2px,fill-opacity:1,opacity:1;
  classDef orangeNode fill:#FFEDD5,stroke:#F97316,color:#111827,stroke-width:2px,fill-opacity:1,opacity:1;
  classDef greenNode fill:#DCFCE7,stroke:#16A34A,color:#111827,stroke-width:2px,fill-opacity:1,opacity:1;
  classDef purpleNode fill:#F3E8FF,stroke:#A855F7,color:#111827,stroke-width:2px,fill-opacity:1,opacity:1;

  Start([MCTS Step i]):::blueNode --> MCTSSelect[MCTS selects parent node]:::blueNode
  MCTSSelect --> CreateActions[Create 3 agent actions]:::blueNode
  
  CreateActions --> RunAgent1[Run Agent Subgraph #1]:::orangeNode
  CreateActions --> RunAgent2[Run Agent Subgraph #2]:::orangeNode
  CreateActions --> RunAgent3[Run Agent Subgraph #3]:::orangeNode
  
  RunAgent1 --> Coder1[Coder generates code]:::orangeNode
  Coder1 --> MCP1[Execute via MCP]:::purpleNode
  MCP1 --> Tester1[Tester runs tests]:::orangeNode
  Tester1 --> MCP1
  MCP1 --> Reviewer1[Reviewer improves]:::orangeNode
  Reviewer1 --> MCP1
  MCP1 --> State1[NodeState #1]:::greenNode
  
  RunAgent2 --> Coder2[Coder generates code]:::orangeNode
  Coder2 --> MCP2[Execute via MCP]:::purpleNode
  MCP2 --> Tester2[Tester runs tests]:::orangeNode
  Tester2 --> MCP2
  MCP2 --> Reviewer2[Reviewer improves]:::orangeNode
  Reviewer2 --> MCP2
  MCP2 --> State2[NodeState #2]:::greenNode
  
  RunAgent3 --> Coder3[Coder generates code]:::orangeNode
  Coder3 --> MCP3[Execute via MCP]:::purpleNode
  MCP3 --> Tester3[Tester runs tests]:::orangeNode
  Tester3 --> MCP3
  MCP3 --> Reviewer3[Reviewer improves]:::orangeNode
  Reviewer3 --> MCP3
  MCP3 --> State3[NodeState #3]:::greenNode
  
  State1 --> Compare[Compare scores]:::greenNode
  State2 --> Compare
  State3 --> Compare
  
  Compare --> Best[Select best NodeState]:::greenNode
  Best --> UpdateTree[Update MCTS tree]:::blueNode
  UpdateTree --> NextIter([Next iteration]):::blueNode

```

---

## Key Design Patterns

### 1. **Sandbox as Shared Resource**
- Single `SandboxClient` instance shared across all agents
- Thread-safe execution via async event loop
- Reuses dependencies and context

### 2. **State-Based Communication**
- Agents communicate through `NodeState` objects
- Immutable state updates prevent race conditions
- Message traces provide audit trail

### 3. **Protocol Abstraction**
- Agents don't know about MCP internals
- `SandboxClient` provides clean synchronous interface
- Easy to swap MCP backends (stdio, HTTP, WebSocket)

### 4. **Hierarchical Orchestration**
- MCTS at top level (exploration strategy)
- LangGraph at agent level (workflow orchestration)
- MCP at execution level (protocol & sandboxing)

---

## Extending the Architecture

### Adding New Agents

To add a new agent (e.g., "Optimizer"):

1. **Create agent function:**
```python
def role_optimizer(sb: SandboxClient, parent: Optional[NodeState]) -> NodeState:
    # Use MCP sandbox to analyze and optimize code
    code = extract_python_block(parent.llm_answer)
    # ... optimization logic using sb.eval() ...
    return NodeState(...)
```

2. **Add to agent subgraph:**
```python
g.add_node("optimizer_ag", _optimizer)
g.add_edge("reviewer_ag", "optimizer_ag")
```

3. **Agents automatically communicate via:**
   - Shared `NodeState` (LangGraph)
   - MCP sandbox for execution (MCP Protocol)

### Distributed Agents

To run agents on different nodes:

1. **Use MCP over HTTP/WebSocket:**
```python
# Instead of stdio, use HTTP MCP server
mcp_client = MCPClientHTTP("http://agent-node-1:8080")
```

2. **Agents still communicate via:**
   - MCP protocol (standardized)
   - Shared state (via message queue or database)

---

## Summary

This architecture demonstrates how **MCP enables agent-to-agent communication** by:

1. **Providing a shared protocol** for code execution and results
2. **Enabling secure isolation** so agents don't interfere
3. **Standardizing communication** through structured I/O
4. **Supporting distributed execution** across nodes
5. **Maintaining observability** through message traces

MCP is not just a tool connector—it's a **foundation for multi-agent collaboration** where agents can safely execute code, share results, and build upon each other's work.

 