import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

# Step 1: Fetch AAPL Stock Data
ticker = 'AAPL'
data = yf.download(ticker, start="2023-01-01", end="2023-08-01")
data['Close'].plot(title='AAPL Stock Price')

# Step 2: Detrend the Stock Data using a Moving Average
data['Moving_Avg'] = data['Close'].rolling(window=20).mean()  # 20-day moving average
data['Detrended_Close'] = data['Close'] - data['Moving_Avg']

# Remove NaN values for Gaussian fitting
detrended_data = data['Detrended_Close'].dropna()

# Step 3: Plot the Histogram and Gaussian Fit
plt.figure(figsize=(10, 6))

# Plot Histogram
n, bins, patches = plt.hist(detrended_data, bins=20, density=True, alpha=0.6, color='skyblue', label='Detrended Histogram')

# Fit a Gaussian (Normal) Distribution
mu, std = norm.fit(detrended_data)  # Mean and standard deviation of the detrended data

# Plot the Gaussian Curve
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'r--', linewidth=2, label=f'Gaussian Fit: $\mu$ = {mu:.2f}, $\sigma$ = {std:.2f}')

plt.title('Histogram of Detrended AAPL Stock Prices with Gaussian Fit')
plt.xlabel('Detrended Closing Price')
plt.ylabel('Density')
plt.legend()
plt.show()
