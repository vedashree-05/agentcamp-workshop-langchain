"""
Phase 4: LangChain Agents (without tools)
Run with: chainlit run app.py -w

This phase introduces LangChain agents - a higher-level abstraction built on
LangGraph that can reason about tasks, use tools, and iterate towards solutions.

In this phase, we create an agent WITHOUT tools to understand the concept.
In Phase 5, we'll add tools to give the agent the ability to take actions.

Key Concepts Introduced:
- LangChain Agents via create_agent()
- Agent vs Chain: Agents can reason and decide what to do next
- Agent streaming with stream_mode for real-time output
- Simplified message format (dict-based instead of LangChain message objects)

Building on Previous Phases:
- Phase 2: Same LLM configuration (ChatOpenAI with GitHub Models)
- Phase 3: Same Chainlit patterns (@cl.on_chat_start, @cl.on_message)
- Phase 3: Same session management (cl.user_session)
- Phase 3: Same streaming pattern for real-time responses

What's New:
- create_agent() replaces direct LLM calls
- Agent handles conversation flow automatically
- Prepared for tool integration in Phase 5

Prerequisites:
- Phase 3 completed (Chainlit chat working)
- langchain package installed (provides create_agent)
"""

import os
from datetime import date
import chainlit as cl
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

# Load environment variables (same as all previous phases)
load_dotenv()

# System prompt for the agent
# Similar to Phase 3, but now passed to create_agent() instead of as a SystemMessage
# The agent uses this to understand its role and behavior
SYSTEM_PROMPT = f"""You are a helpful AI assistant named Aria. You have the following traits:
- Friendly and conversational tone
- Concise but thorough answers
- You admit when you don't know something
- You can help with coding, writing, analysis, and general questions

Current date: {date.today().strftime("%B %d, %Y")}
"""


def get_llm():
    """
    Create and return the LLM client.
    
    Same configuration as Phase 2 and 3:
    - ChatOpenAI pointed at GitHub Models endpoint
    - Note: streaming is handled by the agent, not the LLM directly
    """
    return ChatOpenAI(
        model="openai/gpt-4.1-nano",       # Same model as previous phases
        api_key=os.getenv("GITHUB_TOKEN"), # Same auth as previous phases
        base_url="https://models.github.ai/inference",  # Same endpoint
        temperature=0.7,
    )


def create_assistant_agent():
    """
    Create a LangChain agent without tools.
    
    An agent is a higher-level abstraction that can:
    - Reason about tasks
    - Decide which tools to use (when available)
    - Iterate towards solutions
    
    In this phase, we create an agent without tools.
    The agent will simply respond using the LLM.
    In the next phase, we'll add tools to extend its capabilities.
    """
    llm = get_llm()
    
    # Create agent with no tools (empty list)
    # This creates a simple agent that just uses the LLM
    agent = create_agent(
        model=llm,
        tools=[],  # No tools yet - we'll add them in phase 5!
        system_prompt=SYSTEM_PROMPT,
    )
    
    return agent


@cl.on_chat_start
async def start():
    """
    Initialize the chat session.
    
    Same pattern as Phase 3, but now we create an agent instead of using LLM directly.
    The agent is stored in the session for use in message handling.
    """
    # Create the agent (replaces direct LLM usage from Phase 3)
    agent = create_assistant_agent()
    
    # Store agent in session (same pattern as Phase 3)
    cl.user_session.set("agent", agent)
    cl.user_session.set("chat_history", [])
    
    await cl.Message(
        content="👋 Hi! I'm Aria, your AI assistant. How can I help you today?"
    ).send()


@cl.on_message
async def main(message: cl.Message):
    """
    Handle incoming messages using the agent.
    
    Similar to Phase 3, but uses agent.astream() instead of llm.astream().
    The agent handles the conversation flow and can use tools (in Phase 5).
    """
    # Retrieve agent and history from session (same pattern as Phase 3)
    agent = cl.user_session.get("agent")
    chat_history = cl.user_session.get("chat_history")
    
    # Add user message to history
    # Note: Using dict format instead of HumanMessage (simpler for agents)
    chat_history.append({"role": "user", "content": message.content})
    
    # Stream the response from the agent (similar to Phase 3 streaming)
    msg = cl.Message(content="")
    full_response = ""

    # Use agent.astream() to get streaming responses
    # stream_mode="messages" gives us the message chunks as they're generated
    async for data, _ in agent.astream({"messages": chat_history}, stream_mode="messages"):
        chunks = data.content_blocks
        if len(chunks) == 0:
            continue
        chunk = chunks[-1]["text"]
        full_response += chunk
        await msg.stream_token(chunk)

    await msg.send()

    # Update message history with assistant response (same pattern as Phase 3)
    chat_history.append({"role": "assistant", "content": full_response})
    cl.user_session.set("chat_history", chat_history)
