# Phase 5: Adding Tools to Your Agent

> ⏱️ **Time to complete**: 20 minutes

In this phase, we'll give our agent the ability to use **tools**! We'll create a weather tool that fetches real-time data.

---

## 🎯 Learning Objectives

By the end of this phase, you will:
- Create a tool using the `@tool` decorator
- Add tools to your agent
- Handle tool calls in the UI
- See real-time weather data in your chat

---

## 🔧 What is Tool Calling?

Tool calling lets the LLM:
1. **Recognize** when it needs external data
2. **Request** a function call with arguments
3. **Use** the result to formulate a response

```
User: "What's the weather in Tokyo?"
         │
         ▼
   Agent THINKS: "I need to call get_weather"
         │
         ▼
   Agent ACTS: get_weather("Tokyo")
         │
         ▼
   Tool returns: "Tokyo: 8°C, Cloudy"
         │
         ▼
   Agent responds: "It's 8°C and cloudy in Tokyo!"
```

---

## 📁 Step 1: Create Your Project Folder

```bash
mkdir -p phase-05
cd phase-05
```

---

## 🔑 Step 2: Get a Weather API Key

Add your WeatherAPI key to `.env`:

```bash
# Edit your .env file - add this line:
WEATHER_API_KEY=your_key_here
```

> 💡 Get a free key at [weatherapi.com](https://www.weatherapi.com/) - takes 2 minutes!

---

## 🛠️ Step 3: Create the Tool File

Create a new file `tools.py`:

```python
import os
import httpx
from langchain_core.tools import tool
```

Now let's build the weather tool step by step.

---

## 🌤️ Step 4: Define the Weather Tool

Add this to `tools.py`:

```python
@tool
def get_weather(city: str) -> str:
    """
    Get the current weather for a city.
    
    Args:
        city: The name of the city (e.g., "London", "Tokyo")
    
    Returns:
        Current weather conditions.
    """
    api_key = os.getenv("WEATHER_API_KEY")
    
    if not api_key:
        return "Error: WEATHER_API_KEY not set in .env"
    
    try:
        response = httpx.get(
            "http://api.weatherapi.com/v1/current.json",
            params={"key": api_key, "q": city},
            timeout=10.0
        )
        response.raise_for_status()
        data = response.json()
        
        location = data["location"]["name"]
        country = data["location"]["country"]
        temp_c = data["current"]["temp_c"]
        condition = data["current"]["condition"]["text"]
        humidity = data["current"]["humidity"]
        
        return f"""Weather for {location}, {country}:
🌡️ Temperature: {temp_c}°C
☁️ Condition: {condition}
💧 Humidity: {humidity}%"""
        
    except httpx.HTTPStatusError:
        return f"Could not find weather for '{city}'"
    except Exception as e:
        return f"Error: {e}"
```

**Understanding the `@tool` decorator:**
| Part | Purpose |
|------|---------|
| `@tool` | Tells LangChain this is a tool |
| Docstring | LLM reads this to know when to use it |
| `city: str` | LLM knows it needs a city name |
| Return string | What the agent receives back |

---

## 📋 Step 5: Export the Tools

Add this at the end of `tools.py`:

```python
# List of tools to give to the agent
TOOLS = [get_weather]
```

Your complete `tools.py`:

```python
import os
import httpx
from langchain_core.tools import tool


@tool
def get_weather(city: str) -> str:
    """
    Get the current weather for a city.
    
    Args:
        city: The name of the city (e.g., "London", "Tokyo")
    
    Returns:
        Current weather conditions.
    """
    api_key = os.getenv("WEATHER_API_KEY")
    
    if not api_key:
        return "Error: WEATHER_API_KEY not set in .env"
    
    try:
        response = httpx.get(
            "http://api.weatherapi.com/v1/current.json",
            params={"key": api_key, "q": city},
            timeout=10.0
        )
        response.raise_for_status()
        data = response.json()
        
        location = data["location"]["name"]
        country = data["location"]["country"]
        temp_c = data["current"]["temp_c"]
        condition = data["current"]["condition"]["text"]
        humidity = data["current"]["humidity"]
        
        return f"""Weather for {location}, {country}:
🌡️ Temperature: {temp_c}°C
☁️ Condition: {condition}
💧 Humidity: {humidity}%"""
        
    except httpx.HTTPStatusError:
        return f"Could not find weather for '{city}'"
    except Exception as e:
        return f"Error: {e}"


TOOLS = [get_weather]
```

---

## 📝 Step 6: Copy app.py from Phase 4

Start with Phase 4's code:

```bash
cp ../phase-04/app.py .
```

Now we'll modify it to use tools.

---

## 🔧 Step 7: Import the Tools

At the top of `app.py`, add:

```python
from langchain_core.messages import AIMessageChunk, ToolMessage
from tools import TOOLS
```

---

## 📋 Step 8: Update the System Prompt

Update `SYSTEM_PROMPT` to tell the agent about its tools:

```python
SYSTEM_PROMPT = f"""You are a helpful AI assistant named Aria.
You have access to tools that let you fetch real-time information.

Available tools:
- get_weather: Get current weather for any city

When users ask about weather, USE the get_weather tool. Don't make up weather data.
For other questions, answer from your knowledge.

Current date: {date.today().strftime("%B %d, %Y")}
"""
```

---

## 🤖 Step 9: Add Tools to the Agent

Change `create_assistant_agent()` to include tools:

```python
def create_assistant_agent():
    llm = get_llm()
    
    agent = create_agent(
        model=llm,
        tools=TOOLS,  # Changed from [] to TOOLS!
        system_prompt=SYSTEM_PROMPT,
    )
    
    return agent
```

**The key change:** `tools=TOOLS` instead of `tools=[]`

---

## 💬 Step 10: Handle Tool Calls in Messages

Replace the `main()` function to handle both text AND tool calls:

```python
@cl.on_message
async def main(message: cl.Message):
    agent = cl.user_session.get("agent")
    chat_history = cl.user_session.get("chat_history")
    
    chat_history.append({"role": "user", "content": message.content})
    
    msg = cl.Message(content="")
    full_response = ""

    # Use BOTH "messages" and "updates" stream modes
    async for data, kind in agent.astream(
        {"messages": chat_history}, 
        stream_mode=["messages", "updates"]  # Changed!
    ):
        # Handle text streaming
        if kind == "messages":
            if isinstance(data, AIMessageChunk):
                chunks = data.content_blocks
                if len(chunks) > 0:
                    chunk = chunks[-1]["text"]
                    full_response += chunk
                    await msg.stream_token(chunk)
        
        # Handle tool calls
        elif kind == "updates":
            for key, value in data.items():
                if "messages" in key and isinstance(value, list):
                    for tool_msg in value:
                        if isinstance(tool_msg, ToolMessage):
                            # Show tool result in UI
                            async with cl.Step(name=tool_msg.name) as step:
                                step.output = tool_msg.content

    await msg.send()

    chat_history.append({"role": "assistant", "content": full_response})
    cl.user_session.set("chat_history", chat_history)
```

**What's new:**
| Phase 4 | Phase 5 |
|---------|---------|
| `stream_mode="messages"` | `stream_mode=["messages", "updates"]` |
| Only text | Text + tool calls |
| No `cl.Step` | Show tool results in collapsible step |

---

## 📋 Your Complete app.py

```python
import os
from datetime import date
import chainlit as cl
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import AIMessageChunk, ToolMessage

from tools import TOOLS

load_dotenv()

SYSTEM_PROMPT = f"""You are a helpful AI assistant named Aria.
You have access to tools that let you fetch real-time information.

Available tools:
- get_weather: Get current weather for any city

When users ask about weather, USE the get_weather tool. Don't make up weather data.
For other questions, answer from your knowledge.

Current date: {date.today().strftime("%B %d, %Y")}
"""

def get_llm():
    return ChatOpenAI(
        model="openai/gpt-4.1-nano",
        api_key=os.getenv("GITHUB_TOKEN"),
        base_url="https://models.github.ai/inference",
    )

def create_assistant_agent():
    llm = get_llm()
    
    agent = create_agent(
        model=llm,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
    )
    
    return agent

@cl.on_chat_start
async def start():
    agent = create_assistant_agent()
    
    cl.user_session.set("agent", agent)
    cl.user_session.set("chat_history", [])
    
    await cl.Message(
        content="👋 Hi! I'm Aria. I can check the weather for you! Try: 'What's the weather in Paris?'"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    agent = cl.user_session.get("agent")
    chat_history = cl.user_session.get("chat_history")
    
    chat_history.append({"role": "user", "content": message.content})
    
    msg = cl.Message(content="")
    full_response = ""

    async for data, kind in agent.astream(
        {"messages": chat_history}, 
        stream_mode=["messages", "updates"]
    ):
        if kind == "messages":
            if isinstance(data, AIMessageChunk):
                chunks = data.content_blocks
                if len(chunks) > 0:
                    chunk = chunks[-1]["text"]
                    full_response += chunk
                    await msg.stream_token(chunk)
        
        elif kind == "updates":
            for key, value in data.items():
                if "messages" in key and isinstance(value, list):
                    for tool_msg in value:
                        if isinstance(tool_msg, ToolMessage):
                            async with cl.Step(name=tool_msg.name) as step:
                                step.output = tool_msg.content

    await msg.send()

    chat_history.append({"role": "assistant", "content": full_response})
    cl.user_session.set("chat_history", chat_history)
```

---

## 📄 Step 11: Copy Chainlit Markdown

```bash
cp ../phase-04/chainlit.md .
```

---

## ▶️ Step 12: Run and Test

```bash
chainlit run app.py -w
```

### Test Scenarios

**Test 1: Weather (uses tool)**
```
You: What's the weather in London?
[Collapsible "get_weather" step appears]
Aria: The weather in London is 7°C with cloudy skies...
```

**Test 2: Non-weather (no tool)**
```
You: What is Python?
Aria: Python is a programming language...
(No tool called)
```

**Test 3: Multiple cities**
```
You: Compare weather in Tokyo and Sydney
[Two tool calls shown]
Aria: Tokyo is 8°C while Sydney is 22°C...
```

---

## 🗂️ Project Structure

```
phase-05/
├── app.py          # Agent with tools
├── tools.py        # Tool definitions
└── chainlit.md     # Welcome page
```

---

## 💡 Key Takeaways

The agent now:
1. **Decides** when to use a tool
2. **Calls** the tool with the right arguments
3. **Uses** the result to respond naturally

You can add more tools by:
1. Define function with `@tool` decorator
2. Add it to the `TOOLS` list
3. Update the system prompt

---

## ✅ Checkpoint

| Check | Status |
|-------|--------|
| `tools.py` created | ☐ |
| `WEATHER_API_KEY` in .env | ☐ |
| Weather query shows tool step | ☐ |
| Returns real weather data | ☐ |
| Non-weather questions work | ☐ |

### 🎉 Tools Working?

Your agent can now interact with the real world!

👉 **Next: [Phase 6: MCP Integration](06-mcp-integration.md)**

---

## ❓ Common Issues

### "WEATHER_API_KEY not set"
Add it to your `.env` file and restart Chainlit.

### Tool never gets called
- Check `tools=TOOLS` in `create_agent()`
- Make sure system prompt mentions the tool
- Try asking more directly: "Use the weather tool for Paris"

### "ToolMessage" not defined
Add import: `from langchain_core.messages import AIMessageChunk, ToolMessage`

### Tool step not showing in UI
Make sure you're using `stream_mode=["messages", "updates"]` (with the list!)
