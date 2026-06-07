import logging
import yfinance as yf
from commands.registry import command

logger = logging.getLogger(__name__)

_TICKERS = {
    "apple": "AAPL", "microsoft": "MSFT", "google": "GOOGL", "alphabet": "GOOGL",
    "amazon": "AMZN", "meta": "META", "facebook": "META", "tesla": "TSLA",
    "nvidia": "NVDA", "netflix": "NFLX", "infosys": "INFY", "tcs": "TCS.NS",
    "reliance": "RELIANCE.NS", "bitcoin": "BTC-USD", "ethereum": "ETH-USD",
    "dogecoin": "DOGE-USD", "bnb": "BNB-USD", "solana": "SOL-USD",
    "sp500": "^GSPC", "s&p 500": "^GSPC", "dow jones": "^DJI", "nasdaq": "^IXIC",
}


def _find_ticker(text: str) -> tuple[str, str] | None:
    """Find a ticker symbol from the user's spoken text. Returns (name, symbol) or None."""
    for name, symbol in _TICKERS.items():
        if name in text:
            return name, symbol
    return None


@command(keywords=["stock", "share price", "crypto", "bitcoin", "ethereum", "price of", "how is", "trading at"])
def get_price(text: str) -> str:
    """Fetch and report the current price of a stock or cryptocurrency."""
    match = _find_ticker(text)

    if not match:
        return "I don't recognise that stock or crypto. Try saying 'what's the price of Apple' or 'how is Bitcoin doing'."

    name, symbol = match

    try:
        ticker = yf.Ticker(symbol)
        info = ticker.fast_info

        current = info.last_price
        prev_close = info.previous_close

        if current is None:
            return f"I couldn't get a price for {name} right now."

        current = round(current, 2)
        currency = "₹" if symbol.endswith(".NS") else "$" if "USD" in symbol or "USD" not in symbol else ""

        if prev_close and prev_close > 0:
            change_pct = round((current - prev_close) / prev_close * 100, 2)
            direction = "up" if change_pct >= 0 else "down"
            change_str = f", {direction} {abs(change_pct)}% today"
        else:
            change_str = ""

        return f"{name.title()} is trading at {currency}{current}{change_str}."

    except Exception as e:
        logger.error(f"Stock lookup failed for {symbol}: {e}")
        return f"I couldn't fetch the price for {name} right now. The market data service may be unavailable."
