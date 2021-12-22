import yfinance as yf
from pandas_datareader import data as pdr
import numpy as np
import aiinpy as ai

model = ai.model(7, 100, [
  ai.nn(outshape=100, activation=ai.relu(), learningrate=0.1),
  ai.nn(outshape=100, activation=ai.relu(), learningrate=0.1),
  ai.nn(outshape=100, activation=ai.relu(), learningrate=0.1),
  ai.nn(outshape=100, activation=ai.sigmoid(), learningrate=0.1)
])
 
stonk = pdr.get_data_yahoo("AAPL", start="2019-01-01", end="2021-12-21")

High = np.array([])
Low = np.array([])
AveragePrice = np.array([])
StandardDiviation = np.array([])
Open = np.array([])
Close = np.array([])
Volume = np.array([])

for ind in stonk.index:
  Open = np.append(Open, stonk['Open'][ind])
  Close = np.append(Close, stonk['Close'][ind])
  AveragePrice = np.append(AveragePrice, (Open[-1] + Close[-1]) / 2)
  High = np.append(High, stonk['High'][ind] / AveragePrice[-1])
  Low = np.append(Low, stonk['Low'][ind] / AveragePrice[-1])
  StandardDiviation = np.append(StandardDiviation, High[-1] - Low[-1])
  Volume = np.append(Volume, stonk['Volume'][ind])

input = np.zeros((len(Open) - 28, 7))
output = np.zeros((len(Open) - 28, 100))

for date in range(len(Open) - 28):
  input[date] = [
    np.min(Low[date : date + 28]), 
    np.max(High[date : date + 28]), 
    np.mean(StandardDiviation[date : date + 28]),
    (Close[date + 28] - Open[date]),
    Open[date],
    Close[date + 28],
    np.mean(Volume[date : date + 28])
    ]
  k = np.zeros(100)
  k[int(np.min(Low[date + 28 : date + 56]) * 100)] = 1
  output[date] = k

model.train((input, output), 1000)
print(model.test((input, output)))

'''
input:
- low
- high
- standard diviation
- growth/loss
- open
- close

output:
- low in the next four weeks
'''