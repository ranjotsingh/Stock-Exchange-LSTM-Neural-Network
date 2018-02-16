from sys import argv
if __name__ == '__main__':
    if len(argv) != 4:
        FILENAME = argv[0]
        print('Usage: python3', FILENAME, '[ticker] [startdate=MM/DD/YYYY] [enddate=MM/DD/YYYY]')
        print('(e.g. "python3', FILENAME, 'AAPL 01/17/2014 02/11/2018")')
        exit()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt
from time import time
from datetime import timedelta
from urllib.error import HTTPError
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM

def get_stock_data(ticker, startdate, enddate, normalized=0):
    """ Returns the 'Open', 'High', and 'Close' stock data for ticker from startdate to enddate. """
    url = "http://finance.google.com/finance/historical?q=" + ticker + "&startdate=" + startdate + "&enddate=" + enddate + "&num=30&output=csv"
    col_names = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    try:
        stocks = pd.read_csv(url, header=0, names=col_names)
        data = pd.DataFrame(stocks)
        # Only keeping Open, High, Close columns
        data.drop(data.columns[[0, 3, 5]], axis=1, inplace=True)
        return data
    except HTTPError as e:
        print("Error: Invalid ticker and/or network connection.")
        exit()

def date_check(dt):
    """ Returns given date if formatted correctly else exits. """
    result = dt.split('/')
    if len(result) != 3:
        print("Error: Invalid date or date format. Use MM/DD/YYYY")
        exit()
    month, day, year = result
    if len(month) != 2 or len(day) != 2 or len(year) != 4 or not(1 <= int(month) <= 12) or not(1 <= int(day) <= 31) or not(1900 <= int(year) <= 2019):
        print("Error: Invalid date or date format. Use MM/DD/YYYY")
        exit()
    return dt

def split_data(data, seq_len):
    """ Returns data after splitting into x_train, y_train, x_test, y_test. """
    num_features = len(data.columns)
    data = data.as_matrix()
    seq_len += 1
    result = []
    for index in range(len(data) - seq_len):
        result.append(data[index: index + seq_len])
    result = np.array(result)
    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    x_train = train[:, :-1]
    y_train = train[:, -1][:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1][:, -1]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], num_features))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], num_features))
    return x_train, y_train, x_test, y_test

def make_model(shape):
    """ Returns LSTM model. """
    model = Sequential([
        LSTM(128, input_shape=shape, return_sequences=True),
        Dropout(0.2),
        LSTM(64, input_shape=shape, return_sequences=False),
        Dropout(0.2),
        Dense(16, kernel_initializer='uniform', activation='relu'),
        Dense(1, kernel_initializer='uniform', activation='relu'),
    ])
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model

def main():
    start_time = time()
    ticker, startdate, enddate = argv[1:]

    data = get_stock_data(ticker, date_check(startdate), date_check(enddate), 0)
    data.tail()
    file_name = ticker + '_stock.csv'
    data.to_csv(file_name)
    data /= 1000
    window = 5
    x_train, y_train, x_test, y_test = split_data(data[::-1], window)
    model = make_model((window, 3))
    model.fit(x_train, y_train, batch_size=512, epochs=500, validation_split=0.1, verbose=0)
    trainScore = model.evaluate(x_train, y_train, verbose=0)
    print('Train MSE: %.6f (%.6f RMSE)' % (trainScore[0], sqrt(trainScore[0])))
    testScore = model.evaluate(x_test, y_test, verbose=0)
    print('Test  MSE: %.6f (%.6f RMSE)' % (testScore[0], sqrt(testScore[0])))
    p = model.predict(x_test)
    for i in range(len(y_test)):
        y_test[i] *= 1000
        p[i] *= 1000

    timeItTook = int(time() - start_time)
    print('Time:', str(timedelta(seconds=timeItTook)))

    plt.plot(p, color='red', label='prediction')
    plt.plot(y_test, color='blue', label='actual')
    plt.legend(loc='upper left')
    title = '{} Stock'.format(ticker.upper())
    plt.title(title)
    plt.ylabel('Closing Price')
    plt.show()

if __name__ == '__main__':
    main()
