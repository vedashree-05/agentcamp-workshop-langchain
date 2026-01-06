"""
Phase 6: MCP (Model Context Protocol) Integration
Run with: chainlit run app.py -w

This phase builds on Phase 5 by adding MCP (Model Context Protocol) tools.
MCP is a standard protocol for connecting AI agents to external tools and services.

Key Concepts Introduced:
- MCP (Model Context Protocol): A standard for tool integration
- MultiServerMCPClient: Connect to multiple MCP servers
- Dynamic tool discovery: Tools from MCP servers are fetched at runtime
- Combining local tools with MCP tools

Building on Previous Phases:
- Phase 2: Same LLM configuration (ChatOpenAI with GitHub Models)
- Phase 3: Same Chainlit patterns (@cl.on_chat_start, @cl.on_message)
- Phase 3: Same session management (cl.user_session)
- Phase 4: Same agent creation pattern (create_agent)
- Phase 5: Same tool handling and visualization (cl.Step)
- Phase 5: Same streaming with ["messages", "updates"] modes

What's New (compared to Phase 5):
- get_mcp_tools(): Fetches tools from remote MCP servers
- tools=[*TOOLS, *mcp_tools]: Combines local tools with MCP tools
- langchain_mcp_adapters for MCP integration

Prerequisites:
- Phase 5 completed (Agent with local tools working)
- langchain-mcp-adapters package installed
- Network access to MCP servers (e.g., https://docs.langchain.com/mcp)
"""

import os
from datetime import date
import chainlit as cl
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.messages.tool import ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient  # NEW: MCP adapter
from tools import TOOLS  # Local tools from Phase 5

# Load environment variables (same as all previous phases)
load_dotenv()

# System prompt for the agent
# Updated from Phase 5 to mention both local tools AND MCP tools
SYSTEM_PROMPT = f"""
You are a helpful AI assistant named Aria.

You have access to multiple tools:
- Local tools: get_weather for weather queries
- MCP tools: LangChain documentation search and other external services

Guidelines for tool usage:
- For weather queries, use the get_weather tool
- For LangChain documentation questions, use the appropriate MCP tool
- When using tools, briefly explain what you're doing
- If a tool returns an error, explain the issue to the user

You have the following traits:
- Friendly and conversational tone
- Concise but thorough answers
- You admit when you don't know something
- You can help with coding, writing, analysis, and general questions

Current date: {date.today().strftime("%B %d, %Y")}
"""


def get_llm():
    """
    Create and return the LLM client.

    Same configuration as all previous phases:
    - ChatOpenAI pointed at GitHub Models endpoint
    """
    return ChatOpenAI(
        model="openai/gpt-4.1-nano",  # Same model as previous phases
        api_key=os.getenv("GITHUB_TOKEN"),  # Same auth as previous phases
        base_url="https://models.github.ai/inference",  # Same endpoint
        temperature=0.7,
    )


async def get_mcp_tools():
    """
    Fetch tools from MCP (Model Context Protocol) servers.

    NEW in Phase 6!

    MCP is a standard protocol that allows AI agents to connect to external
    tools and services. Instead of defining tools locally (like in Phase 5),
    we can fetch them from remote MCP servers.

    Benefits of MCP:
    - Tools are maintained by external services
    - Automatic updates when services add new capabilities
    - Standardized interface across different tool providers
    """
    # Create a client that can connect to multiple MCP servers
    mcp_client = MultiServerMCPClient(
        {
            # Each key is a name for the server, value is connection config
            "langchain_docs": {
                "transport": "http",  # Can be "http" or "stdio"
                "url": "https://docs.langchain.com/mcp",  # MCP server URL
            }
            # Add more MCP servers here as needed!
        }
    )
    # Fetch all tools from all configured servers
    return await mcp_client.get_tools()


async def create_assistant_agent():
    """
    Create a LangChain agent with BOTH local tools AND MCP tools.

    This is the key difference from Phase 5:
    - Phase 5: tools=TOOLS (only local tools)
    - Phase 6: tools=[*TOOLS, *mcp_tools] (local + MCP tools!)

    The agent can now:
    - Use local tools (get_weather from tools.py)
    - Use remote MCP tools (LangChain docs search, etc.)
    - Reason about which tool to use based on the query
    """
    llm = get_llm()

    # NEW: Fetch tools from MCP servers
    mcp_tools = await get_mcp_tools()

    # Create agent with COMBINED tools (local + MCP)
    # The * operator unpacks both lists into a single list
    agent = create_agent(
        model=llm,
        tools=[*TOOLS, *mcp_tools],  # CHANGED from just TOOLS to include MCP tools!
        system_prompt=SYSTEM_PROMPT,
    )

    return agent


@cl.on_chat_start
async def start():
    """
    Initialize the chat session.

    Same pattern as Phase 3, 4, and 5.
    The agent creation is async because we need to fetch MCP tools.
    """
    # Create the agent (async because of MCP tool fetching)
    agent = await create_assistant_agent()

    # Store agent in session (same pattern as previous phases)
    cl.user_session.set("agent", agent)
    cl.user_session.set("chat_history", [])

    # Updated welcome message mentioning expanded capabilities
    await cl.Message(
        content="👋 Hi! I'm Aria, your AI assistant. I can check weather, search LangChain docs, and more! How can I help you today?"
    ).send()


@cl.on_message
async def main(message: cl.Message):
    """
    Handle incoming messages using the agent.

    This is identical to Phase 5!
    The tool handling works the same whether tools are local or from MCP.
    That's the beauty of LangChain's abstraction - the agent doesn't care
    where tools come from, it just uses them.

    The streaming and visualization code handles:
    - Local tools (get_weather from tools.py)
    - MCP tools (LangChain docs search, etc.)
    - Any combination of tool calls
    """
    # Retrieve agent and history from session (same pattern as Phase 3, 4, 5)
    agent = cl.user_session.get("agent")
    chat_history = cl.user_session.get("chat_history")

    # Add user message to history (same pattern as Phase 4 and 5)
    chat_history.append({"role": "user", "content": message.content})

    # Stream the response from the agent
    response_message = cl.Message(content="")
    full_response = ""

    # Track tool call steps for visualization (same as Phase 5)
    steps = {}

    # Stream with both modes to handle tool calls and text (same as Phase 5)
    async for stream_mode, data in agent.astream(
        {"messages": chat_history}, stream_mode=["messages", "updates"]
    ):

        # Handle "updates" - tool calls and results (same as Phase 5)
        if stream_mode == "updates":
            for source, update in data.items():
                if source in ("model", "tools"):
                    msg = update["messages"][-1]

                    # Detect tool calls (works for both local and MCP tools!)
                    if isinstance(msg, AIMessage) and msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            step = cl.Step(f"🔧 {tool_call['name']}", type="tool")
                            step.input = tool_call["args"]
                            await step.send()
                            steps[tool_call["id"]] = step

                    # Handle tool results (works for both local and MCP tools!)
                    if isinstance(msg, ToolMessage):
                        tool_call_id = msg.tool_call_id
                        step = steps.get(tool_call_id)
                        if step:
                            step.output = msg.content
                            await step.update()

        # Handle "messages" - streaming text tokens (same as Phase 5)
        if stream_mode == "messages":
            token, _ = data
            if isinstance(token, AIMessageChunk):
                full_response += token.content
                await response_message.stream_token(token.content)

    await response_message.send()

    # Update message history (same pattern as all previous phases)
    chat_history.append({"role": "assistant", "content": full_response})
    cl.user_session.set("chat_history", chat_history)
