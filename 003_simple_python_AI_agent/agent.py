"""
agent.py - The core stock research agent using OpenAI GPT-4o-mini and tool use.

This module implements an agentic loop where:
1. User sends a query
2. GPT-4o-mini decides which tools to call (if any)
3. Tools are executed and results returned to the model
4. The model synthesizes a final response
5. All steps are traced with W&B Weave
"""

import os
import json
from openai import OpenAI
import weave
from dotenv import load_dotenv
from tools import TOOLS, TOOL_FUNCTIONS

# Load environment variables
load_dotenv()

# System prompt defines the agent's personality and behavior
SYSTEM_PROMPT = """You are an expert financial research assistant with access to real-time market data. Your role is to help users understand stocks, analyze investments, and provide data-driven insights.

## Your Capabilities
You have access to the following tools:
- get_stock_price: Fetch current price, volume, and market cap
- get_company_financials: Get P/E ratios, margins, growth metrics, and analyst recommendations
- get_price_history: Retrieve historical prices and calculate returns

## Your Approach
1. **Always use tools to get real data** - Never make up numbers or prices
2. **Be thorough** - When analyzing a stock, fetch both current price AND financials
3. **Compare when asked** - If comparing stocks, gather data for all of them before analysis
4. **Cite your data** - Reference the specific metrics you retrieved
5. **Be balanced** - Present both bullish and bearish factors

## Your Constraints
- You provide financial INFORMATION, not financial ADVICE
- Always remind users to do their own research and consult financial advisors
- Acknowledge limitations in the data (e.g., analyst estimates may vary)
- If a stock symbol is invalid or data is unavailable, clearly state this

## Response Format
- Start with a brief summary of what you found
- Present key metrics in a clear, organized way
- End with balanced observations, not recommendations
- Use specific numbers from your tool calls"""


class StockResearchAgent:
    """
    An agentic stock research assistant that uses OpenAI GPT-4o-mini and financial data tools.
    
    The agent implements an agentic loop:
    1. Receive user query
    2. Model decides on tool calls
    3. Execute tools and collect results
    4. Model synthesizes final response
    """
    
    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Initialize the agent with OpenAI client and model.
        
        Args:
            model: The OpenAI model to use. Default is gpt-4o-mini for
                   good balance of capability and cost.
        """
        self.client = OpenAI()
        self.model = model
        self.conversation_history = []
    
    @weave.op()
    def execute_tool(self, tool_name: str, tool_input: dict) -> str:
        """
        Execute a tool and return its result as a string.
        
        Args:
            tool_name: Name of the tool to execute
            tool_input: Dictionary of input parameters
            
        Returns:
            JSON string of the tool result
        """
        if tool_name not in TOOL_FUNCTIONS:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        
        tool_fn = TOOL_FUNCTIONS[tool_name]
        result = tool_fn(**tool_input)
        return json.dumps(result, indent=2)
    
    @weave.op()
    def process_tool_calls(self, tool_calls: list) -> list:
        """
        Process multiple tool calls and return results.
        
        Args:
            tool_calls: List of tool call objects from OpenAI
            
        Returns:
            List of tool result messages
        """
        results = []
        for tool_call in tool_calls:
            tool_input = json.loads(tool_call.function.arguments)
            tool_result = self.execute_tool(
                tool_name=tool_call.function.name,
                tool_input=tool_input
            )
            results.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result
            })
        return results
    
    @weave.op()
    def chat(self, user_message: str) -> str:
        """
        Process a user message through the agentic loop.
        
        This is the main entry point for the agent. It:
        1. Adds the user message to history
        2. Sends to GPT-4o-mini with available tools
        3. If the model wants to use tools, executes them
        4. Continues the loop until the model provides a final response
        
        Args:
            user_message: The user's query
            
        Returns:
            The agent's final text response
        """
        # Add user message to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Build messages with system prompt
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.conversation_history
        
        # Agentic loop - continues until we get a final response
        while True:
            # Call OpenAI with the current conversation
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto"
            )
            
            assistant_message = response.choices[0].message
            
            # Check if the model wants to use tools
            if assistant_message.tool_calls:
                # Add assistant's response (with tool calls) to messages
                messages.append({
                    "role": "assistant",
                    "content": assistant_message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in assistant_message.tool_calls
                    ]
                })
                
                # Execute tools and get results
                tool_results = self.process_tool_calls(assistant_message.tool_calls)
                
                # Add tool results to messages
                messages.extend(tool_results)
                
                # Continue the loop - model will process tool results
                continue
            
            # No more tool calls - we have the final response
            final_response = assistant_message.content or ""
            
            # Add assistant's final response to conversation history
            self.conversation_history.append({
                "role": "assistant",
                "content": final_response
            })
            
            return final_response
    
    def reset_conversation(self):
        """Clear conversation history for a new session."""
        self.conversation_history = []


def main():
    """Interactive CLI for the stock research agent."""
    # Initialize Weave for tracing
    weave.init("stock-research-agent")
    
    # Create the agent
    agent = StockResearchAgent()
    
    print("=" * 60)
    print("📈 Stock Research Agent (GPT-4o-mini)")
    print("=" * 60)
    print("I can help you research stocks with real-time data.")
    print("Try asking me to:")
    print("  - 'What's the current price of AAPL?'")
    print("  - 'Analyze Tesla's financials'")
    print("  - 'Compare MSFT and GOOGL for investment'")
    print("\nType 'quit' to exit, 'reset' to start a new conversation")
    print("=" * 60)
    
    while True:
        try:
            user_input = input("\n🧑 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                print("Goodbye! Remember: always do your own research. 📊")
                break
            
            if user_input.lower() == 'reset':
                agent.reset_conversation()
                print("🔄 Conversation reset. How can I help you?")
                continue
            
            print("\n🤖 Agent: ", end="", flush=True)
            response = agent.chat(user_input)
            print(response)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break


if __name__ == "__main__":
    main()
