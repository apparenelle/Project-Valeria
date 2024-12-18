import yfinance as yf
import math
import numpy as np

class stock_data:
  def __init__(self, symbol:str):
    self.symbol = symbol
    self.stock = yf.Ticker(symbol)
    self.info = self.stock.info


  def getHistory(self, period, interval):
    self.history = self.stock.history(period=period, interval=interval).reset_index()
    return self.history


  def calc_future_close_indicator(self, daysAhead=1):
    history = self.history

    if history.empty:
      print('History is empty')
      print('Please call getHistory()')
      return None

    else:
      # future_close column added to history
      history['Future_Close_Indicator'] = 0

      future_close_shifted = history['Close'].shift(-daysAhead)

      history['Future_Close_Indicator'] = (future_close_shifted > history['Close']).astype(int)
      self.history = history

      return history



  def calcVolumePercentilesFromHist(self):
    history = self.history

    # retrieves 70th, 80th, and 90th, percentile
    if history.empty is False:
      percentiles = {
                      '70': math.ceil(np.percentile(self.history['Volume'], 70)),
                      '80': math.ceil(np.percentile(self.history['Volume'], 80)),
                      '90': math.ceil(np.percentile(self.history['Volume'], 90))
                    }
      self.vol_percentiles = percentiles
      return percentiles
    else:
      print('History is empty')
      print('Please call getHistory()')
      return None

  


  def findLastStockSplitIndex(self):
    history = self.history

    if not history.empty:
      for index, value in history['Stock Splits'][::-1].items():
        if value > 0.0:
          return index
      return 'No stock splits present.'



  def movingAverages(self):
    #Grab history
    history = self.history

    # Add columns for 5-Day, 10-Day, and 20-Day moving averages
    history['5-Day_Moving_Avg'] = history['Close'].rolling(window=5, min_periods=1).mean()
    history['10-Day_Moving_Avg'] = history['Close'].rolling(window=10, min_periods=1).mean()
    history['20-Day_Moving_Avg'] = history['Close'].rolling(window=20, min_periods=1).mean()

    #Save to object df
    self.history = history

  
  
  # removes time from stock history for cleaner code
  def removeTime(self):
    history = self.history

    if history.empty is False:
      self.history['Date'] = self.history['Date'].apply(lambda a: a.date())
      return self.history
    
    else:
      print('History never was saved')
      print('Please recover history.')
      return None