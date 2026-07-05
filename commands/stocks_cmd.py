import logging
import yfinance as yf
from commands.registry import command

logger = logging.getLogger(__name__)

_TICKERS = {
    # US tech
    "apple": "AAPL", "microsoft": "MSFT", "google": "GOOGL", "alphabet": "GOOGL",
    "amazon": "AMZN", "meta": "META", "facebook": "META", "tesla": "TSLA",
    "nvidia": "NVDA", "netflix": "NFLX",
    # Indian stocks
    "infosys": "INFY", "tcs": "TCS.NS", "reliance": "RELIANCE.NS",
    "wipro": "WIPRO.NS", "hdfc": "HDFCBANK.NS", "icici": "ICICIBANK.NS",
    "bajaj": "BAJFINANCE.NS", "sbi": "SBIN.NS", "bharti": "BHARTIARTL.NS",
    # Indian indices
    "nifty 50": "^NSEI", "nifty": "^NSEI", "sensex": "^BSESN",
    "bank nifty": "^NSEBANK", "nse": "^NSEI", "bse": "^BSESN",
    # Crypto
    "bitcoin": "BTC-USD", "ethereum": "ETH-USD", "dogecoin": "DOGE-USD",
    "bnb": "BNB-USD", "solana": "SOL-USD",
    # US indices
    "sp500": "^GSPC", "s&p 500": "^GSPC", "dow jones": "^DJI", "nasdaq": "^IXIC",
}

# Keywords that indicate the user means the Indian equity market overall
_INDIA_MARKET_WORDS = ("stock market", "share market", "equity market", "market index")


def _currency_symbol(symbol: str) -> str:
    """Pick the currency prefix for a ticker's price.

    - '.NS' suffix  → Indian equities, quoted in rupees.
    - '^' prefix    → an index (Nifty, Sensex, S&P 500...). These are point
                      levels, not money, so they get no currency symbol.
    - everything else → US equities and '*-USD' crypto pairs, quoted in dollars.

    (The previous inline expression `'$' if 'USD' in symbol or 'USD' not in symbol`
    was a tautology — always '$' — so index levels were wrongly read as dollars.)
    """
    if symbol.endswith(".NS"):
        return "₹"
    if symbol.startswith("^"):
        return ""
    return "$"


def _find_ticker(text: str) -> tuple[str, str] | None:
    """Find a ticker symbol from the user's spoken text. Returns (name, symbol) or None."""
    # Longest names first so "nifty 50" beats "nifty"
    for name in sorted(_TICKERS, key=len, reverse=True):
        if name in text:
            return name, _TICKERS[name]
    # Generic market query ("how is the stock market today") → Nifty 50.
    # Without this default such queries fell through to the LLM, which
    # confidently invented an index level. Explicit US phrasing gets the
    # S&P 500 instead (named US indices already matched above).
    if any(phrase in text for phrase in _INDIA_MARKET_WORDS):
        if any(w in text for w in ("us ", "u.s.", "american", "wall street")):
            return "s&p 500", "^GSPC"
        return "nifty 50", "^NSEI"
    return None


# "sensex"/"nifty" etc. are keywords too: "how is the sensex doing today"
# previously matched nothing and took the direct-LLM path (hallucinated price).
@command(keywords=["stock", "stocks", "share price", "crypto", "bitcoin", "ethereum",
                   "price of", "trading at", "sensex", "nifty", "nasdaq", "dow jones",
                   "stock market", "share market"])
def get_price(text: str) -> str | None:
    """Fetch and report the current price of a stock or cryptocurrency."""
    match = _find_ticker(text)

    if not match:
        # No known ticker in the query — decline so other handlers / the LLM
        # get a chance ("price of onions" is not a stocks question).
        return None

    name, symbol = match

    try:
        ticker = yf.Ticker(symbol)
        info = ticker.fast_info

        current = info.last_price
        prev_close = info.previous_close

        if current is None:
            return f"I couldn't get a price for {name} right now."

        current = round(current, 2)
        currency = _currency_symbol(symbol)

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
