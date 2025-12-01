"""
tools.py - Financial data tools for the stock research agent.

Each tool is a function that the agent can call to retrieve real-world data.
Tools should have clear inputs, outputs, and error handling.
"""

import yfinance as yf
from datetime import datetime
import weave


@weave.op()
def get_stock_price(symbol: str) -> dict:
    """
    Get the current stock price and basic trading information.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
    
    Returns:
        Dictionary with current price, daily change, and volume
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Handle case where stock doesn't exist
        if not info or 'regularMarketPrice' not in info:
            return {"error": f"Could not find stock with symbol '{symbol}'"}
        
        return {
            "symbol": symbol.upper(),
            "current_price": info.get('regularMarketPrice'),
            "previous_close": info.get('previousClose'),
            "day_change": info.get('regularMarketChange'),
            "day_change_percent": info.get('regularMarketChangePercent'),
            "volume": info.get('volume'),
            "market_cap": info.get('marketCap'),
            "currency": info.get('currency', 'USD'),
            "exchange": info.get('exchange'),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": f"Failed to fetch price for {symbol}: {str(e)}"}


@weave.op()
def get_company_financials(symbol: str) -> dict:
    """
    Get key financial metrics for fundamental analysis.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
    
    Returns:
        Dictionary with P/E ratio, revenue, profit margins, and other metrics
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        if not info or 'symbol' not in info:
            return {"error": f"Could not find financials for symbol '{symbol}'"}
        
        return {
            "symbol": symbol.upper(),
            "company_name": info.get('longName'),
            "sector": info.get('sector'),
            "industry": info.get('industry'),
            # Valuation metrics
            "pe_ratio": info.get('trailingPE'),
            "forward_pe": info.get('forwardPE'),
            "peg_ratio": info.get('pegRatio'),
            "price_to_book": info.get('priceToBook'),
            # Profitability
            "profit_margin": info.get('profitMargins'),
            "operating_margin": info.get('operatingMargins'),
            "return_on_equity": info.get('returnOnEquity'),
            # Growth & Revenue
            "revenue": info.get('totalRevenue'),
            "revenue_growth": info.get('revenueGrowth'),
            "earnings_growth": info.get('earningsGrowth'),
            # Dividends
            "dividend_yield": info.get('dividendYield'),
            "payout_ratio": info.get('payoutRatio'),
            # Analyst info
            "target_mean_price": info.get('targetMeanPrice'),
            "recommendation": info.get('recommendationKey'),
            "number_of_analysts": info.get('numberOfAnalystOpinions'),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": f"Failed to fetch financials for {symbol}: {str(e)}"}


@weave.op()
def get_price_history(symbol: str, period: str = "1mo") -> dict:
    """
    Get historical price data for trend analysis.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        period: Time period - '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'
    
    Returns:
        Dictionary with price history and basic statistics
    """
    valid_periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max']
    if period not in valid_periods:
        return {"error": f"Invalid period '{period}'. Must be one of: {valid_periods}"}
    
    try:
        ticker = yf.Ticker(symbol)
        history = ticker.history(period=period)
        
        if history.empty:
            return {"error": f"No historical data found for symbol '{symbol}'"}
        
        # Calculate statistics
        prices = history['Close'].tolist()
        
        return {
            "symbol": symbol.upper(),
            "period": period,
            "data_points": len(prices),
            "start_date": history.index[0].strftime('%Y-%m-%d'),
            "end_date": history.index[-1].strftime('%Y-%m-%d'),
            "start_price": round(prices[0], 2),
            "end_price": round(prices[-1], 2),
            "high": round(max(prices), 2),
            "low": round(min(prices), 2),
            "average": round(sum(prices) / len(prices), 2),
            "total_return_percent": round(((prices[-1] - prices[0]) / prices[0]) * 100, 2),
            # Recent prices (last 5 data points)
            "recent_prices": [
                {"date": history.index[i].strftime('%Y-%m-%d'), "close": round(prices[i], 2)}
                for i in range(-min(5, len(prices)), 0)
            ],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": f"Failed to fetch history for {symbol}: {str(e)}"}


# Tool definitions for OpenAI's function calling format
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the current stock price, daily change, volume, and market cap for a given stock symbol. Use this when you need real-time price information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "The stock ticker symbol (e.g., 'AAPL' for Apple, 'MSFT' for Microsoft, 'GOOGL' for Alphabet)"
                    }
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_company_financials",
            "description": "Get detailed financial metrics including P/E ratio, revenue, profit margins, growth rates, and analyst recommendations. Use this for fundamental analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "The stock ticker symbol (e.g., 'AAPL' for Apple, 'MSFT' for Microsoft)"
                    }
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_price_history",
            "description": "Get historical price data and statistics for trend analysis. Returns start/end prices, highs, lows, and total return over the period.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "The stock ticker symbol"
                    },
                    "period": {
                        "type": "string",
                        "description": "Time period for historical data. Options: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'. Default is '1mo'.",
                        "enum": ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"]
                    }
                },
                "required": ["symbol"]
            }
        }
    }
]

# Map tool names to functions for easy lookup
TOOL_FUNCTIONS = {
    "get_stock_price": get_stock_price,
    "get_company_financials": get_company_financials,
    "get_price_history": get_price_history
}
