"""
Phase 3: Basic Chainlit Chat with Conversation Memory
Run with: chainlit run app.py -w

This phase builds on Phase 2 by adding a web-based chat interface using Chainlit.
We also introduce conversation memory so the assistant remembers previous messages.

Key Concepts Introduced:
- Chainlit for building chat UIs (@cl.on_chat_start, @cl.on_message)
- Session management (cl.user_session) to store data per user
- Streaming responses for better UX (llm.astream)
- LangChain message types (SystemMessage, HumanMessage, AIMessage)
- Conversation history to maintain context

Building on Phase 2:
- Same LLM configuration (ChatOpenAI with GitHub Models)
- Same environment variable loading pattern
- Added: streaming=True for real-time token output

Prerequisites:
- Phase 2 completed (GitHub Models connection verified)
- chainlit package installed
"""

import os
import chainlit as cl
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Load environment variables (same as Phase 2)
load_dotenv()


def get_llm():
    """
    Create and return the LLM client.
    
    This is the same configuration as Phase 2, with one addition:
    - streaming=True: Enables token-by-token streaming for real-time output
    """
    return ChatOpenAI(
        model="openai/gpt-4.1-nano",       # Same model as Phase 2
        api_key=os.getenv("GITHUB_TOKEN"), # Same auth as Phase 2
        base_url="https://models.github.ai/inference",  # Same endpoint
        temperature=0.7,
        streaming=True,  # NEW: Enable streaming for real-time responses
    )


# System prompt that defines the assistant's behavior
# This is sent at the start of every conversation to set the AI's personality
SYSTEM_PROMPT = """You are a helpful AI assistant. You are friendly, concise, and informative.
When you don't know something, you say so honestly."""


@cl.on_chat_start
async def start():
    """
    Initialize the chat session.
    
    Chainlit calls this function when a new user connects.
    We use it to set up the LLM and initialize conversation history.
    """
    # Store the LLM in the user's session (each user gets their own)
    cl.user_session.set("llm", get_llm())
    
    # Initialize message history with system prompt
    # The system message sets the AI's behavior for the entire conversation
    cl.user_session.set("messages", [
        SystemMessage(content=SYSTEM_PROMPT)
    ])
    
    # Send a welcome message to the user
    await cl.Message(
        content="👋 Hello! I'm an AI assistant powered by GitHub Models. I can remember our conversation. How can I help you today?"
    ).send()


@cl.on_message
async def main(message: cl.Message):
    """
    Handle incoming messages with conversation history.
    
    Chainlit calls this function whenever the user sends a message.
    We maintain conversation history so the AI remembers context.
    """
    # Retrieve LLM and message history from the user's session
    llm = cl.user_session.get("llm")
    messages = cl.user_session.get("messages")
    
    # Add the user's message to history (HumanMessage = user input)
    messages.append(HumanMessage(content=message.content))
    
    # Create an empty message that we'll stream tokens into
    msg = cl.Message(content="")
    full_response = ""
    
    # Stream the response token by token using astream
    # This gives a better UX as users see the response as it's generated
    async for chunk in llm.astream(messages):
        if chunk.content:
            full_response += chunk.content
            await msg.stream_token(chunk.content)  # Send each token to the UI
    
    # Finalize the message
    await msg.send()
    
    # Add assistant's response to history (AIMessage = assistant output)
    # This allows the AI to remember what it said in future turns
    messages.append(AIMessage(content=full_response))
    cl.user_session.set("messages", messages)
