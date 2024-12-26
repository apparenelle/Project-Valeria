# -*- coding: utf-8 -*-
"""
Project Valeria (Algo Trading Bot)

WIP:
-
"""
import yfinance as yf
import numpy as np
import math
import pandas as pd
from stock_data import stock_data
from data_pre_processor import data_pre_processor

# AFFECTS LENGTH OF DATAFRAME PRINT STMTS
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 50)

"""MAIN FX's"""

# Goal: Prep and Clean the dataset
def buildStockDataframe() -> pd.DataFrame:
  stock = stock_data('nvda')

  # ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
  stock.getHistory(period='10y', interval='1d')
  stock.removeTime()


  stock.movingAverages()
  stock.calcVolumePercentilesFromHist()
  stock.calc_future_close_indicator()

  #cleaner must be used after stock attributes have been configured
  data = data_pre_processor(stock.history, stock)
  
  # build ranking series
  data.create_Price_Movement_Ranking()
  data.create_Volume_Ranking()
  data.clean()

  # cleaner.reOrder()
  data.display_info()

  data.toCSV(fileName='./stock_data.csv', sep=',')


  # Relevant Fields
  # ['Date', 'Open', 
  # 'High', 'Low', 
  # 'Close', 'Volume', 
  # 'Dividends', 'Stock Split', 
  # 'Volume Ranking', 'Price Movement', 
  # '5-Day_Moving_Avg', '10-Day_Moving_Avg', 
  # '20-Day_Moving_Avg']

  # 70th Percentile         87 million shares a day  --- AAPL
  # 80th Percentile         96 million shares a day  --- AAPL
  # 90th Percentile         115 million shares a day --- AAPL

  # RETURNING processed dataset
  data.remove_last()

  return data.data

def main2():
  dataframe = buildStockDataframe()

  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import LSTM, Dense
  from sklearn.metrics import mean_squared_error, mean_absolute_error
  from tensorflow.keras.optimizers import RMSprop

  # Train-test split
  train_df = dataframe.iloc[:int(len(dataframe) * 0.8)]
  test_df = dataframe.iloc[int(len(dataframe) * 0.8):]

  # Feature selection
  X_train = train_df[
      [
        'open_lag_1', 'open_lag_2', 'open_lag_3',
        'close_lag_1', 'close_lag_2', 'close_lag_3',
        'high_lag_1', 'high_lag_2', 'high_lag_3',
        'low_lag_1', 'low_lag_2', 'low_lag_3',
        'volume_lag_1', 'volume_lag_2', 'volume_lag_3',
        'volume_ranking_lag_1', 'volume_ranking_lag_2', 'volume_ranking_lag_3',
        '5-moving_avg_lag_1', '5-moving_avg_lag_2', '5-moving_avg_lag_3',
        '10-moving_avg_lag_1', '10-moving_avg_lag_2', '10-moving_avg_lag_3',
        '20-moving_avg_lag_1', '20-moving_avg_lag_2', '20-moving_avg_lag_3'
      ]
  ].values

  # X_train = train_df[
  #     [
  #         'open_lag_1', 'open_lag_2', 'open_lag_3',
  #         'close_lag_1', 'close_lag_2', 'close_lag_3',
  #         'high_lag_1', 'high_lag_2', 'high_lag_3',
  #         'low_lag_1', 'low_lag_2', 'low_lag_3',
  #         'volume_lag_1', 'volume_lag_2', 'volume_lag_3',
  #         'volume_ranking_lag_1', 'volume_ranking_lag_2', 'volume_ranking_lag_3',
  #         '5-moving_avg_lag_1', '5-moving_avg_lag_2', '5-moving_avg_lag_3',
  #         '10-moving_avg_lag_1', '10-moving_avg_lag_2', '10-moving_avg_lag_3',
  #         '20-moving_avg_lag_1', '20-moving_avg_lag_2', '20-moving_avg_lag_3'
  #     ]
  # ].values

  # Target selection
  y_train = train_df[[
    'Future_Close_Indicator']].values

  # y_train = train_df[[
  #   'Open', 
  #   'Close', 
  #   'High', 
  #   'Low']].values

  # Reshape X_train for LSTM
  X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))  # [samples, timesteps, features]

  # Define LSTM model for multi-output prediction
  model = Sequential([
      LSTM(2000, activation='tanh', input_shape=(X_train.shape[1], 1)),
      Dense(y_train.shape[1])  # Number of outputs matches the number of targets
  ])
  
  model.compile(optimizer= RMSprop(learning_rate=0.001, rho=0.9), loss='mse')

  # Train the model
  model.fit(X_train, y_train, epochs=7, batch_size=10)
#  45 - epochs is perfect
  # Test dataset preparation
  X_test = test_df[
      [
        'open_lag_1', 'open_lag_2', 'open_lag_3',
        'close_lag_1', 'close_lag_2', 'close_lag_3',
        'high_lag_1', 'high_lag_2', 'high_lag_3',
        'low_lag_1', 'low_lag_2', 'low_lag_3',
        'volume_lag_1', 'volume_lag_2', 'volume_lag_3',
        'volume_ranking_lag_1', 'volume_ranking_lag_2', 'volume_ranking_lag_3',
        '5-moving_avg_lag_1', '5-moving_avg_lag_2', '5-moving_avg_lag_3',
        '10-moving_avg_lag_1', '10-moving_avg_lag_2', '10-moving_avg_lag_3',
        '20-moving_avg_lag_1', '20-moving_avg_lag_2', '20-moving_avg_lag_3'
      ]
  ].values
  
  y_test = test_df[['Future_Close_Indicator']].values  # Ground truth for X-days ahead
  # y_test = test_df[['Open', 'Close', 'High', 'Low']].values  # Ground truth for X-days ahead
  # y_test = test_df[['Open', 'Close', 'High', 'Low']].values  # Ground truth for X-days ahead

  X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

  # Make predictions
 # Make predictions
  predictions = model.predict(X_test)

  # Print raw predictions (continuous probabilities)
  print("\n\n\nRaw Predictions:\n", predictions)

  # Convert predictions to binary using a threshold of 0.5
  predictions_binary = (predictions > 0.6).astype(int)

  # Evaluate classification metrics
  from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

  # Calculate metrics
  accuracy = accuracy_score(y_test, predictions_binary)
  precision = precision_score(y_test, predictions_binary)
  recall = recall_score(y_test, predictions_binary)
  f1 = f1_score(y_test, predictions_binary)
  cm = confusion_matrix(y_test, predictions_binary)

  # Print metrics
  print(f"\nAccuracy: {accuracy}")
  print(f"Precision: {precision}")
  print(f"Recall: {recall}")
  print(f"F1-Score: {f1}")
  print("Confusion Matrix:\n", cm)


    
    
  # mse = mean_squared_error(y_test, predictions)
  # mae = mean_absolute_error(y_test, predictions)
  
  # epsilon = 1e-8
  # mape = np.mean(np.abs((y_test - predictions) / (y_test + epsilon))) * 100

  
  #mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100  # Mean Absolute Percentage Error

  # print(f"Mean Squared Error (MSE): {mse}")
  # print(f"Mean Absolute Error (MAE): {mae}")
  # print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")



# main2()
buildStockDataframe()

"""Good News: The data is auto adjusted by yahoo to account for stock splits no work necessary and you can train on as much data as I please!

# Test - shows rel. days.

**Libraries**
- yfinance - provides stock price history | Open Close High Low Volume Stock_Splits
- datetime - standard datetime obj.
- numpy - statistical analysis | percentile
- math - cleaning numerics | ceil
tensorflow - training LSTM
"""
def main3():
  # Find days where volume is higher than TARGET_PERCENTILE ( 87m shares )
  def findDaysOfHighVolume_ARR(date, volume, target_percentile):
    arr = list()
    for x,y in zip(date, volume):
      if (y >= target_percentile):
        arr.append([x,y])
    return arr
  
  stock = stock_data("AAPL")
  stock.getHistory('10y', '1d')
  print(findDaysOfHighVolume_ARR(aapl_date_vol_df['Date'], aapl_date_vol_df['Volume'], stock_AT_80_percentile))
