"""
Phase 5: LangChain Agent with Tools
Run with: chainlit run app.py -w

This phase builds on Phase 4 by adding TOOLS to the agent.
Tools allow the agent to take actions beyond just generating text.

Key Concepts Introduced:
- Tools: Functions the agent can call to get external data or perform actions
- Tool calling: The agent decides when to use tools based on user queries
- Tool visualization: Showing tool calls and results in the Chainlit UI
- @tool decorator: How to define tools in LangChain

Building on Previous Phases:
- Phase 2: Same LLM configuration (ChatOpenAI with GitHub Models)
- Phase 3: Same Chainlit patterns (@cl.on_chat_start, @cl.on_message)
- Phase 3: Same session management (cl.user_session)
- Phase 4: Same agent creation pattern (create_agent)
- Phase 4: Same streaming approach (agent.astream)

What's New (compared to Phase 4):
- tools=TOOLS instead of tools=[] - the agent now has capabilities!
- Tool visualization with cl.Step() to show what the agent is doing
- Handling of AIMessage with tool_calls and ToolMessage responses
- Updated system prompt mentioning available tools

Prerequisites:
- Phase 4 completed (Agent without tools working)
- tools.py file with tool definitions
- WEATHER_API_KEY in .env (for the weather tool)
"""

import os
from datetime import date
import chainlit as cl
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.messages.tool import ToolMessage
from tools import TOOLS  # NEW: Import tools from tools.py

# Load environment variables (same as all previous phases)
load_dotenv()

# System prompt for the agent
# Updated from Phase 4 to mention the available tools
# This helps the agent understand when to use tools
SYSTEM_PROMPT = f"""
You are a helpful AI assistant named Aria.

You have access to tools that let you take actions:
- get_weather: Get current weather for any city

Guidelines for tool usage:
- For weather queries, ALWAYS use the get_weather tool rather than making up information
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
        model="openai/gpt-4.1-nano",       # Same model as previous phases
        api_key=os.getenv("GITHUB_TOKEN"), # Same auth as previous phases
        base_url="https://models.github.ai/inference",  # Same endpoint
        temperature=0.7,
    )


async def create_assistant_agent():
    """
    Create a LangChain agent WITH tools.
    
    This is the key difference from Phase 4:
    - Phase 4: tools=[]     (no tools, just chat)
    - Phase 5: tools=TOOLS  (can take actions!)
    
    The agent can now:
    - Reason about tasks (same as Phase 4)
    - Decide WHEN to use tools based on user queries
    - Execute tools and incorporate results into responses
    - Iterate towards solutions using multiple tool calls if needed
    """
    llm = get_llm()
    
    # Create agent WITH tools (the key difference from Phase 4!)
    agent = create_agent(
        model=llm,
        tools=TOOLS,  # CHANGED from [] to TOOLS!
        system_prompt=SYSTEM_PROMPT,
    )
    
    return agent


@cl.on_chat_start
async def start():
    """
    Initialize the chat session.
    
    Same pattern as Phase 3 and 4, creating the agent and storing in session.
    The welcome message now mentions the tool capabilities!
    """
    agent = await create_assistant_agent()
    
    # Store agent in session (same pattern as Phase 3 and 4)
    cl.user_session.set("agent", agent)
    cl.user_session.set("chat_history", [])
    
    # Updated welcome message to mention tool capabilities
    await cl.Message(
        content="👋 Hi! I'm Aria, your AI assistant. I can now check the weather for you! Try asking: 'What's the weather in Paris?'"
    ).send()


@cl.on_message
async def main(message: cl.Message):
    """
    Handle incoming messages using the agent.
    
    This is more complex than Phase 4 because we need to:
    1. Stream the agent's responses (same as Phase 4)
    2. Detect and display tool calls as they happen (NEW!)
    3. Show tool results before the final response (NEW!)
    
    We use stream_mode=["messages", "updates"] to get both:
    - "messages": The streaming text tokens (for real-time output)
    - "updates": Tool calls and results (for visualization)
    """
    # Retrieve agent and history from session (same pattern as Phase 3 and 4)
    agent = cl.user_session.get("agent")
    chat_history = cl.user_session.get("chat_history")
    
    # Add user message to history (same pattern as Phase 4)
    chat_history.append({"role": "user", "content": message.content})
    
    # Stream the response from the agent
    response_message = cl.Message(content="")
    full_response = ""

    # Track tool call steps so we can update them with results
    steps = {}

    # Use agent.astream() with BOTH stream modes to get messages AND updates
    # This is the key difference from Phase 4 - we handle tool calls!
    async for stream_mode, data in agent.astream({"messages": chat_history}, stream_mode=["messages", "updates"]):
        
        # Handle "updates" - these contain tool calls and results
        if stream_mode == "updates":
            for source, update in data.items():
                if source in ("model", "tools"):
                    msg = update["messages"][-1]
                    
                    # NEW: Detect when the agent decides to call a tool
                    if isinstance(msg, AIMessage) and msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            # Create a visual step in Chainlit to show the tool call
                            step = cl.Step(f"🔧 {tool_call['name']}", type="tool")
                            step.input = tool_call["args"]
                            await step.send()
                            steps[tool_call["id"]] = step
                    
                    # NEW: Handle tool results
                    if isinstance(msg, ToolMessage):
                        tool_call_id = msg.tool_call_id
                        step = steps.get(tool_call_id)
                        if step:
                            step.output = msg.content
                            await step.update()

        # Handle "messages" - these contain the streaming text tokens
        # (Same pattern as Phase 4)
        if stream_mode == "messages":
            token, _ = data
            if isinstance(token, AIMessageChunk):
                full_response += token.content
                await response_message.stream_token(token.content)

    await response_message.send()

    # Update message history with assistant response (same pattern as Phase 4)
    chat_history.append({"role": "assistant", "content": full_response})
    cl.user_session.set("chat_history", chat_history)
