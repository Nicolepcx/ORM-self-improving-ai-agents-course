import os, asyncio
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio

load_dotenv()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "openai:gpt-5")

server = MCPServerStdio(
    # Launch in stdio mode
    "docker",
    args=["run","--rm","-i","mcp-run-python:latest","/usr/local/bin/entrypoint.sh","stdio"],
)

agent = Agent(OPENAI_MODEL, toolsets=[server])

async def main():
    async with agent:
        r = await agent.run("Compute 2**10 and print it")
    print(r.output)

if __name__ == "__main__":
    asyncio.run(main())
