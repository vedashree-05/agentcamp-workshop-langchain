# Phase 6: MCP Integration

> ⏱️ **Time to complete**: 15 minutes

In this final phase, we'll add **MCP (Model Context Protocol)** tools to our agent. MCP lets you connect to external tool servers!

---

## 🎯 Learning Objectives

By the end of this phase, you will:
- Understand what MCP is
- Connect your agent to MCP servers
- Combine local tools with MCP tools
- Use LangChain's MCP adapter

---

## �� What is MCP?

**Model Context Protocol (MCP)** is an open standard for connecting AI agents to tools.

### Without MCP
Each AI app defines its own tools locally.

### With MCP
Tools are provided by external **MCP servers** that any AI app can connect to.

```
┌──────────────────────────────────────────────┐
│          Your Agent                           │
│                                               │
│   ┌─────────────┐    ┌──────────────┐        │
│   │ Local Tools │    │  MCP Client  │        │
│   │ (get_weather)│    │              │        │
│   └─────────────┘    └──────┬───────┘        │
└──────────────────────────────┼────────────────┘
                               │
                    ┌──────────▼───────────┐
                    │   MCP Protocol       │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
      ┌───────▼──────┐ ┌───────▼──────┐ ┌───────▼──────┐
      │ LangChain    │ │ Database     │ │ Your Own     │
      │ Docs Server  │ │ Server       │ │ MCP Server   │
      └──────────────┘ └──────────────┘ └──────────────┘
```

### Why MCP?
| Benefit | Description |
|---------|-------------|
| **Reusable** | One server, many apps |
| **Modular** | Add tools without code changes |
| **Standard** | Works with Claude, ChatGPT, your agent |

---

## 📁 Step 1: Create Your Project Folder

```bash
mkdir -p phase-06
cd phase-06
```

---

## 📦 Step 2: Install MCP Adapter

```bash
pip install langchain-mcp-adapters
```

---

## 📋 Step 3: Copy Files from Phase 5

Start with Phase 5's working code:

```bash
cp ../phase-05/app.py .
cp ../phase-05/tools.py .
cp ../phase-05/chainlit.md .
```

---

## 🔧 Step 4: Import the MCP Adapter

At the top of `app.py`, add:

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
```

---

## 🌐 Step 5: Create the MCP Tool Fetcher

Add this function after `get_llm()`:

```python
async def get_mcp_tools():
    """
    Fetch tools from MCP servers.
    
    MCP servers provide tools remotely - we don't define them locally!
    """
    mcp_client = MultiServerMCPClient(
        {
            "langchain_docs": {
                "transport": "http",
                "url": "https://docs.langchain.com/mcp",
            }
            # Add more MCP servers here as needed!
        }
    )
    return await mcp_client.get_tools()
```

**What's happening:**
- `MultiServerMCPClient` connects to MCP servers
- Each server has a name and URL
- `get_tools()` fetches all available tools

---

## 🤖 Step 6: Update the Agent to Use MCP Tools

Change `create_assistant_agent()` to be async and fetch MCP tools:

```python
async def create_assistant_agent():
    """Create agent with BOTH local tools AND MCP tools."""
    llm = get_llm()
    
    # Fetch tools from MCP servers
    mcp_tools = await get_mcp_tools()
    
    # Combine local tools (TOOLS) with MCP tools!
    agent = create_agent(
        model=llm,
        tools=[*TOOLS, *mcp_tools],  # Local + MCP tools!
        system_prompt=SYSTEM_PROMPT,
    )
    
    return agent
```

**Key change:** `tools=[*TOOLS, *mcp_tools]` combines both!

---

## 📋 Step 7: Update the System Prompt

Tell the agent about its expanded capabilities:

```python
SYSTEM_PROMPT = f"""You are a helpful AI assistant named Aria.

You have access to multiple tools:
- Local tools: get_weather for weather queries
- MCP tools: LangChain documentation search and other services

Guidelines:
- For weather, use get_weather
- For LangChain questions, use the docs search tool
- Be helpful and explain what you're doing

Current date: {date.today().strftime("%B %d, %Y")}
"""
```

---

## 🚀 Step 8: Make on_chat_start Async

Update `start()` to await the agent creation:

```python
@cl.on_chat_start
async def start():
    agent = await create_assistant_agent()  # Note: await!
    
    cl.user_session.set("agent", agent)
    cl.user_session.set("chat_history", [])
    
    await cl.Message(
        content="👋 Hi! I'm Aria. I can check weather, search LangChain docs, and more!"
    ).send()
```

---

## 💬 Step 9: Keep the Message Handler (Almost Same)

The `main()` function is almost identical to Phase 5! The only difference is handling tool calls from both sources.

Here's the updated version with better tool visualization:

```python
@cl.on_message
async def main(message: cl.Message):
    agent = cl.user_session.get("agent")
    chat_history = cl.user_session.get("chat_history")
    
    chat_history.append({"role": "user", "content": message.content})
    
    msg = cl.Message(content="")
    full_response = ""
    steps = {}  # Track tool call steps

    async for stream_mode, data in agent.astream(
        {"messages": chat_history}, 
        stream_mode=["messages", "updates"]
    ):
        # Handle tool calls and results
        if stream_mode == "updates":
            for source, update in data.items():
                if source in ("model", "tools"):
                    last_msg = update["messages"][-1]
                    
                    # Show tool being called
                    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
                        for tool_call in last_msg.tool_calls:
                            step = cl.Step(f"🔧 {tool_call['name']}", type="tool")
                            step.input = tool_call["args"]
                            await step.send()
                            steps[tool_call["id"]] = step
                    
                    # Show tool result
                    if isinstance(last_msg, ToolMessage):
                        step = steps.get(last_msg.tool_call_id)
                        if step:
                            step.output = last_msg.content
                            await step.update()
        
        # Handle streaming text
        if stream_mode == "messages":
            token, _ = data
            if isinstance(token, AIMessageChunk):
                full_response += token.content
                await msg.stream_token(token.content)

    await msg.send()
    
    chat_history.append({"role": "assistant", "content": full_response})
    cl.user_session.set("chat_history", chat_history)
```

**Note:** Add this import if you don't have it:
```python
from langchain_core.messages import AIMessage, AIMessageChunk
```

---

## 📋 Your Complete app.py

```python
import os
from datetime import date
import chainlit as cl
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.messages.tool import ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from tools import TOOLS

load_dotenv()

SYSTEM_PROMPT = f"""You are a helpful AI assistant named Aria.

You have access to multiple tools:
- Local tools: get_weather for weather queries
- MCP tools: LangChain documentation search and other services

Guidelines:
- For weather, use get_weather
- For LangChain questions, use the docs search tool
- Be helpful and explain what you're doing

Current date: {date.today().strftime("%B %d, %Y")}
"""

def get_llm():
    return ChatOpenAI(
        model="openai/gpt-4.1-nano",
        api_key=os.getenv("GITHUB_TOKEN"),
        base_url="https://models.github.ai/inference",
    )

async def get_mcp_tools():
    mcp_client = MultiServerMCPClient(
        {
            "langchain_docs": {
                "transport": "http",
                "url": "https://docs.langchain.com/mcp",
            }
        }
    )
    return await mcp_client.get_tools()

async def create_assistant_agent():
    llm = get_llm()
    mcp_tools = await get_mcp_tools()
    
    agent = create_agent(
        model=llm,
        tools=[*TOOLS, *mcp_tools],
        system_prompt=SYSTEM_PROMPT,
    )
    
    return agent

@cl.on_chat_start
async def start():
    agent = await create_assistant_agent()
    
    cl.user_session.set("agent", agent)
    cl.user_session.set("chat_history", [])
    
    await cl.Message(
        content="👋 Hi! I'm Aria. I can check weather, search LangChain docs, and more!"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    agent = cl.user_session.get("agent")
    chat_history = cl.user_session.get("chat_history")
    
    chat_history.append({"role": "user", "content": message.content})
    
    msg = cl.Message(content="")
    full_response = ""
    steps = {}

    async for stream_mode, data in agent.astream(
        {"messages": chat_history}, 
        stream_mode=["messages", "updates"]
    ):
        if stream_mode == "updates":
            for source, update in data.items():
                if source in ("model", "tools"):
                    last_msg = update["messages"][-1]
                    
                    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
                        for tool_call in last_msg.tool_calls:
                            step = cl.Step(f"🔧 {tool_call['name']}", type="tool")
                            step.input = tool_call["args"]
                            await step.send()
                            steps[tool_call["id"]] = step
                    
                    if isinstance(last_msg, ToolMessage):
                        step = steps.get(last_msg.tool_call_id)
                        if step:
                            step.output = last_msg.content
                            await step.update()
        
        if stream_mode == "messages":
            token, _ = data
            if isinstance(token, AIMessageChunk):
                full_response += token.content
                await msg.stream_token(token.content)

    await msg.send()
    
    chat_history.append({"role": "assistant", "content": full_response})
    cl.user_session.set("chat_history", chat_history)
```

---

## ▶️ Step 10: Run and Test

```bash
chainlit run app.py -w
```

### Test Scenarios

**Test 1: Local tool (weather)**
```
You: What's the weather in London?
[get_weather step appears]
Aria: It's 7°C in London...
```

**Test 2: MCP tool (LangChain docs)**
```
You: How do I use LangChain agents?
[langchain_docs step appears]
Aria: Based on the docs, agents are...
```

**Test 3: Both tools**
```
You: What's the weather in Paris, and how do I create a LangChain tool?
[Both tool steps appear]
Aria: In Paris it's 12°C. For tools, you...
```

---

## 🗂️ Project Structure

```
phase-06/
├── app.py          # Agent with local + MCP tools
├── tools.py        # Local tool definitions
└── chainlit.md     # Welcome page
```

---

## 💡 Key Takeaways

You now have an agent that:
1. Uses **local tools** (weather from tools.py)
2. Uses **remote MCP tools** (LangChain docs server)
3. **Combines** both seamlessly!

### Adding More MCP Servers

```python
mcp_client = MultiServerMCPClient(
    {
        "langchain_docs": {...},
        "another_server": {
            "transport": "http",
            "url": "https://example.com/mcp",
        },
        # Add as many as you want!
    }
)
```

---

## ✅ Checkpoint

| Check | Status |
|-------|--------|
| `langchain-mcp-adapters` installed | ☐ |
| Weather queries still work | ☐ |
| LangChain docs queries work | ☐ |
| Both show tool steps in UI | ☐ |

### 🎉 Congratulations!

You've completed the workshop! Your agent now:
- ✅ Has a chat UI with streaming
- ✅ Remembers conversation history
- ✅ Uses local tools (weather)
- ✅ Uses remote MCP tools (docs search)
- ✅ Visualizes tool calls

---

## ❓ Common Issues

### "langchain_mcp_adapters not found"
```bash
pip install langchain-mcp-adapters
```

### MCP connection errors
Check your network can reach `https://docs.langchain.com/mcp`

### "await outside async function"
Make sure `create_assistant_agent()` is defined with `async def`

### Tool not appearing
- Check the MCP server URL is correct
- Make sure `get_mcp_tools()` is awaited
- Verify tools are combined: `[*TOOLS, *mcp_tools]`
