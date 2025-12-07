# cross market example proof of concept (what price diffs would actually look like)
# currently sort of nonfunctional, backtesting was vibecoded and probably needs to be entirely rewritten

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

# Config (go to polygon/massive and get a free API key) -----------------------

POLYGON_API_KEY = "REDACTED"

ADR_TICKER = "TM"             # NYSE ADR
ORD_TICKER = "TOYOF"          # OTC Ordinary Share Proxy aka foreign share

FX_PAIR = "C:GBPUSD"          # Currency conversion
ADR_RATIO = 1.0               # 1 ADR = 1 ordinary share

BAR_TIMESCALE = "day"         # daily price bars
LOOKBACK_DAYS = 180
TRADE_SIZE_USD = 10_000       # each simulated share is 10k
MIN_PROFIT_USD = 0.00         # min arbitrage profit to trigger a trade

# execution costs that reduce arbitrage opps vvv
SPREAD_ADR = 0.002
SPREAD_ORD = 0.002
COMMISSION_ADR = 0.0005
COMMISSION_ORD = 0.0005
STAMP_DUTY_UK = 0.005

# Requesting info for ADR, OTC, and FX in pd dataframes -------------------------

def polygon_us(ticker, start, end, timescale="day"):
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/{timescale}/{start}/{end}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": POLYGON_API_KEY,
    }

    r = requests.get(url, params=params)
    data = r.json()

    if "results" not in data:
        raise RuntimeError(f"Polygon US Error: {data}")

    df = pd.DataFrame(data["results"])
    df["timestamp"] = pd.to_datetime(df["t"], unit="ms")
    df.set_index("timestamp", inplace=True)

    return df.rename(columns={"c": "close", "v": "volume"})[["close", "volume"]]


def polygon_otc(ticker, start, end, timescale="day"):
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/{timescale}/{start}/{end}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": POLYGON_API_KEY,
    }

    r = requests.get(url, params=params)
    data = r.json()

    if "results" not in data:
        raise RuntimeError(f"Polygon OTC Error: {data}")

    df = pd.DataFrame(data["results"])
    df["timestamp"] = pd.to_datetime(df["t"], unit="ms")
    df.set_index("timestamp", inplace=True)

    return df.rename(columns={"c": "close", "v": "volume"})[["close", "volume"]]


def polygon_fx(start, end, timescale="day"):
    url = f"https://api.polygon.io/v2/aggs/ticker/{FX_PAIR}/range/1/{timescale}/{start}/{end}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": POLYGON_API_KEY,
    }

    r = requests.get(url, params=params)
    data = r.json()

    if "results" not in data:
        raise RuntimeError(f"Polygon FX Error: {data}")

    df = pd.DataFrame(data["results"])
    df["timestamp"] = pd.to_datetime(df["t"], unit="ms")
    df.set_index("timestamp", inplace=True)

    return df.rename(columns={"c": "close"})[["close"]]


# Computing implied vs actual pricing and detecting arbitrage ------------------

# OTC GBP -> USD and adjusts for the adr ratio 
# and ADR ratio doesn't matter for these bc it's 1:1
def implied_adr_price(ord_gbp, fx, adr_ratio):
    return (ord_gbp * fx) / adr_ratio


def apply_costs(price, spread, commission, other=0):
    return price * (1 + spread + commission + other)


def detect_arbitrage(df):
    df["implied_adr"] = implied_adr_price(
        df["ord_close"],
        df["fx"],
        ADR_RATIO
    )

    df["adj_ord"] = apply_costs(
        df["implied_adr"],
        SPREAD_ORD,
        COMMISSION_ORD,
        STAMP_DUTY_UK
    )

    df["adj_adr"] = apply_costs(
        df["adr_close"],
        SPREAD_ADR,
        COMMISSION_ADR
    )

    # diff between actual and implied ADR price
    df["gross_spread"] = df["adr_close"] - df["implied_adr"]  
    
    # diff after costs
    df["net_spread"] = df["adj_adr"] - df["adj_ord"]

    # positive net spread means adr is expensive, buy ordinary, sell adr
    # negative net spread means adr is cheap, buy adr, sell ordinary

    # so these trigger depending on the net spread (if profit exceeds min)
    df["buy_ord_sell_adr"] = df["net_spread"] > MIN_PROFIT_USD
    df["buy_adr_sell_ord"] = df["net_spread"] < -MIN_PROFIT_USD

    return df


# Backtesting (needs to be fixed) -----------------------------------------------

def backtest(df):
    trades = []

    for i in range(len(df) - 1):
        row = df.iloc[i]
        nxt = df.iloc[i + 1]

        # BUY ORD / SELL ADR
        if row["buy_ord_sell_adr"]:
            units = TRADE_SIZE_USD / row["adj_ord"]

            pnl = units * (
                (row["adj_adr"] - nxt["adr_close"]) +
                (nxt["implied_adr"] - row["adj_ord"])
            )

            trades.append(pnl)

        # BUY ADR / SELL ORD
        if row["buy_adr_sell_ord"]:
            units = TRADE_SIZE_USD / row["adj_adr"]

            pnl = units * (
                (nxt["implied_adr"] - row["adj_ord"]) +
                (row["adj_adr"] - nxt["adr_close"])
            )

            trades.append(pnl)

    trades = np.array(trades)

    return {
        "trades": len(trades),
        "total_pnl_usd": round(trades.sum(), 2) if len(trades) else 0,
        "avg_pnl": round(trades.mean(), 2) if len(trades) else 0,
        "max_win": round(trades.max(), 2) if len(trades) else 0,
        "max_loss": round(trades.min(), 2) if len(trades) else 0,
        "win_rate": round((trades > 0).mean(), 3) if len(trades) else 0,
    }


if __name__ == "__main__":
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=LOOKBACK_DAYS)

    ord_df = polygon_otc(ORD_TICKER, start, end, BAR_TIMESCALE)
    adr_df = polygon_us(ADR_TICKER, start, end, BAR_TIMESCALE)
    fx_df = polygon_fx(start, end, BAR_TIMESCALE)

    # only keeping rows where all three prices exist for comp
    df = pd.concat([ 
        ord_df["close"].rename("ord_close"),
        adr_df["close"].rename("adr_close"),
        fx_df["close"].rename("fx"),
    ], axis=1).dropna()


    arb = detect_arbitrage(df)
    results = backtest(arb)

    print("\nAribtrage Backtest Summary\n")

    for k, v in results.items():
        print(f"{k}: {v}")

    print("\nLast 10 signals:")
    print(
        arb.tail(10)[[
            "ord_close",
            "adr_close",
            "fx",
            "implied_adr",
            "net_spread",
            "buy_ord_sell_adr",
            "buy_adr_sell_ord"
        ]]
    )
