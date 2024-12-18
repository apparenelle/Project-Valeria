import pandas as pd

class data_pre_processor:
  def __init__(self, data: pd.DataFrame , stock):
    self.data = data
    self.stock = stock

  def display_info(self):
    print(self.data)

  def clean(self):
    dataframe = self.data
    stock = self.stock

    #all yf data already adjusted for splits
    dataframe = dataframe.drop(columns=['Stock Splits', 'Dividends'], axis=1)

    # stock percentiles or 'low' make out of 0 - 4
    def rankingVolume(a, ranking):
      if a <= ranking['70']:
        return 0
      elif a > ranking['70'] and a <= ranking['80']:
        return 1
      elif a > ranking['80'] and a <= ranking['90']:
        return 2
      elif a >= ranking['90']:
        return 3
      else:
        print('Something is off w/ the volume')
        return -1

    # Adds volume ranking to df
    dataframe['Volume_Ranking'] = dataframe['Volume'].apply(lambda x: rankingVolume(x, stock.vol_percentiles))

    def priceMovement(row):
      low = row['Low']
      high = row['High']
      open_ = row['Open']  # Renamed to open_ to avoid conflict with Python's open function
      close = row['Close']

      if close >= open_:
        return high - low
      else:
        return (high - low) * -1

    #Price change from open to close
    dataframe['Price_Movement'] = dataframe.apply(priceMovement, axis=1)

    #add a day column for an iterator called days
    dataframe['Day'] = range(1, len(dataframe)+1)

    # X Attributes
    # building time-series for open
    dataframe['open_lag_1'] = dataframe["Open"].shift(1)
    dataframe['open_lag_2'] = dataframe["Open"].shift(2)
    dataframe['open_lag_3'] = dataframe["Open"].shift(3)

    #building time-series for open
    dataframe['close_lag_1'] = dataframe["Close"].shift(1)
    dataframe['close_lag_2'] = dataframe["Close"].shift(2)
    dataframe['close_lag_3'] = dataframe["Close"].shift(3)
    
    dataframe['high_lag_1'] = dataframe["High"].shift(1)
    dataframe['high_lag_2'] = dataframe["High"].shift(2)
    dataframe['high_lag_3'] = dataframe["High"].shift(3)

    dataframe['low_lag_1'] = dataframe["Low"].shift(1)
    dataframe['low_lag_2'] = dataframe["Low"].shift(2)
    dataframe['low_lag_3'] = dataframe["Low"].shift(3)
    
    dataframe['volume_lag_1'] = dataframe["Volume"].shift(1)
    dataframe['volume_lag_2'] = dataframe["Volume"].shift(2)
    dataframe['volume_lag_3'] = dataframe["Volume"].shift(3)

    dataframe['volume_ranking_lag_1'] = dataframe["Volume_Ranking"].shift(1)
    dataframe['volume_ranking_lag_2'] = dataframe["Volume_Ranking"].shift(2)
    dataframe['volume_ranking_lag_3'] = dataframe["Volume_Ranking"].shift(3)

    dataframe["5-moving_avg_lag_1"] = dataframe["5-Day_Moving_Avg"].shift(1)
    dataframe["5-moving_avg_lag_2"] = dataframe["5-Day_Moving_Avg"].shift(2)
    dataframe["5-moving_avg_lag_3"] = dataframe["5-Day_Moving_Avg"].shift(3)

    dataframe["10-moving_avg_lag_1"] = dataframe["10-Day_Moving_Avg"].shift(1)
    dataframe["10-moving_avg_lag_2"] = dataframe["10-Day_Moving_Avg"].shift(2)
    dataframe["10-moving_avg_lag_3"] = dataframe["10-Day_Moving_Avg"].shift(3)

    dataframe["20-moving_avg_lag_1"] = dataframe["20-Day_Moving_Avg"].shift(1)
    dataframe["20-moving_avg_lag_2"] = dataframe["20-Day_Moving_Avg"].shift(2)
    dataframe["20-moving_avg_lag_3"] = dataframe["20-Day_Moving_Avg"].shift(3)

    
    # Y Attributes
    dataframe['target_open_1'] = dataframe["Open"].shift(-1)
    dataframe['target_open_2'] = dataframe["Open"].shift(-2)
    dataframe['target_open_3'] = dataframe["Open"].shift(-3)

    #building time-series for open
    dataframe['target_close_1'] = dataframe["Close"].shift(-1)
    dataframe['target_close_2'] = dataframe["Close"].shift(-2)
    dataframe['target_close_3'] = dataframe["Close"].shift(-3)
    
    dataframe['target_high_1'] = dataframe["High"].shift(-1)
    dataframe['target_high_2'] = dataframe["High"].shift(-2)
    dataframe['target_high_3'] = dataframe["High"].shift(-3)

    dataframe['target_low_1'] = dataframe["Low"].shift(-1)
    dataframe['target_low_2'] = dataframe["Low"].shift(-2)
    dataframe['target_low_3'] = dataframe["Low"].shift(-3)
    
    dataframe['target_volume_1'] = dataframe["Volume"].shift(-1)
    dataframe['target_volume_2'] = dataframe["Volume"].shift(-2)
    dataframe['target_volume_3'] = dataframe["Volume"].shift(-3)
    
    dataframe = dataframe.dropna()

    #saves changes
    self.data = dataframe



  def remove_last(self, amount=1):
      self.data = self.data.iloc[:-amount]
      print('Last Records Removed.')
      return self.data



  def toCSV(self, fileName, sep):
    data = self.data
    data.to_csv(fileName, sep=sep, index=False, encoding='utf-8')