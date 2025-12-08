import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import requests
from datetime import datetime

api_key = 'redacted'

nyse_ticker = 'AAPL'  # Apple
lse_ticker = 'TSCDY'  # Tesco


def get_polygon_data(ticker, start_date, end_date, api_key):
    url = f'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}?apiKey={api_key}'
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        # print(data) debugging
        if "results" not in data or len(data["results"]) == 0:
            print(f"No data returned for ticker '{ticker}'.")
            return pd.Series(dtype=float)

        df = pd.DataFrame(data["results"])
        df["t"] = pd.to_datetime(df["t"], unit="ms")
        df.set_index("t", inplace=True)
        return df["c"]

    else:
        raise Exception(f"Error fetching data: {response.status_code} - {response.text}")


# date range for historical data
start_date = '2023-01-01'
end_date = '2025-01-01'

aapl_data = get_polygon_data(nyse_ticker, start_date, end_date, api_key)
tsco_data = get_polygon_data(lse_ticker, start_date, end_date, api_key)

# align dates to avoid mismatched lengths
combined = pd.concat([aapl_data, tsco_data], axis=1, join="inner")
combined.columns = ["AAPL", "TSCDY"]
combined.dropna(inplace=True)

aapl_data = combined["AAPL"]
tsco_data = combined["TSCDY"]

# checking for cointegration
# cointegration basically means the two stocks long run behavior is the same
# so as they fluctuate independently their difference stays relatively stable
score, p_value, _ = coint(aapl_data, tsco_data)
print(f'Cointegration p-value: {p_value}')

# a low p value means the stocks are more likely to have this stable difference
if p_value < 0.05:
    print("Cointegrated, proceeding with pairs trading strategy.")
else:
    print("Not cointegrated, skip this pair.")

# compute spread (difference) between the two stock prices
# linear regression to find the hedge ratio (beta)
# linear relationship between the two -> hedge ratio tells you how much TSCDY moves when AAPL moves
X = sm.add_constant(tsco_data)
model = sm.OLS(aapl_data, X).fit()
beta = model.params[tsco_data.name]  # hedge ratio (slope)

# so the spread shows how out of balance the pair is
# ex: high means AAPL is expensive compared to TSCDY
spread = aapl_data - beta * tsco_data 

# mean and std of the spread
spread_mean = spread.rolling(window=30).mean()
spread_std = spread.rolling(window=30).std()

# trading signals
# buy pair (long AAPL, short TSCDY) when the spread is more than 1 std dev below the mean
# sell pair (short AAPL, long TSCDY) when the spread is more than 1 std dev above the mean

upper_threshold = spread_mean + spread_std
lower_threshold = spread_mean - spread_std

# 1 = long, -1 = short, 0 = no position
long_signal = spread < lower_threshold
short_signal = spread > upper_threshold

# positions based on the above signals
positions = pd.DataFrame(index=spread.index)
positions['long'] = np.where(long_signal, 1, 0)
positions['short'] = np.where(short_signal, -1, 0)

# calculate returns assuming equal allocation to the two stocks
returns = pd.DataFrame(index=spread.index)
returns['aapl_returns'] = aapl_data.pct_change().shift(-1)  # daily % returns for AAPL
returns['tsco_returns'] = tsco_data.pct_change().shift(-1)  # same for TSCDY

# portfolio return is based on the positions in both stocks
portfolio_returns = positions['long'] * returns['aapl_returns'] - positions['short'] * returns['tsco_returns']

# plotting
cumulative_returns = (1 + portfolio_returns).cumprod() - 1

plt.figure(figsize=(10, 6))
plt.plot(cumulative_returns, label='Pairs Trading Strategy')
plt.title('Cumulative Returns of Pairs Trading Strategy')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.show()
