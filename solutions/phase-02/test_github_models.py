"""
Phase 2: Test GitHub Models Connection
Run with: python test_github_models.py

This script verifies that your GitHub Models connection is working correctly.
It's the foundation for all subsequent phases.

Key Concepts Introduced:
- Loading environment variables with dotenv
- Creating a ChatOpenAI client configured for GitHub Models
- Making a simple LLM request

Prerequisites:
- .env file with GITHUB_TOKEN set to your GitHub personal access token
- langchain-openai and python-dotenv packages installed
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
# This reads the GITHUB_TOKEN from your .env file
load_dotenv()


def main():
    # Get the token from environment
    github_token = os.getenv("GITHUB_TOKEN")
    
    if not github_token or github_token == "your_github_token_here":
        print("❌ Error: GITHUB_TOKEN not set in .env file")
        print("   Please add your GitHub token to the .env file")
        return
    
    print("🔄 Testing connection to GitHub Models...")
    
    # Create the LLM client using LangChain's ChatOpenAI
    # GitHub Models uses an OpenAI-compatible API, so we can use ChatOpenAI
    # but point it to GitHub's endpoint instead of OpenAI's
    llm = ChatOpenAI(
        model="openai/gpt-4.1-nano",  # Model format: provider/model-name
        api_key=github_token,          # Your GitHub PAT for authentication
        base_url="https://models.github.ai/inference",  # GitHub's inference endpoint
        temperature=0.7,               # Controls randomness (0=deterministic, 1=creative)
    )
    
    try:
        # Send a simple test message
        response = llm.invoke("Say 'Hello, Workshop!' and nothing else.")
        print(f"✅ Success! Model responded: {response.content}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTroubleshooting:")
        print("- Make sure your GitHub token is valid")
        print("- Check your internet connection")
        print("- Verify the token hasn't expired")

if __name__ == "__main__":
    main()