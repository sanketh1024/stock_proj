#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 20:14:32 2017

@author: sanketh
"""

#stock project

import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
path = './DataSets/'
#print os.listdir(path)
#print glob.glob("./DataSets/*.csv")
#for filename in glob.glob(os.path.join(path, '*.csv')):
#    print filename




def get_max_close(symbol):
    print "here",symbol
    df = pd.read_csv("./DataSets/{symbol}".format(symbol=symbol))
#    df[['Close', 'Adj Close']].plot()
#    plt.show()
    df = df.iloc[::-1]
    df[['Close']].plot()
    plt.show()
    return df['Close'].max()
    

def normalized_plot(df1):
    df1 = (df1/df1.ix[0,:])[['ICICIBANK_LIMITED_IBN.csv' + 'Close', 'ICICIBANK.NS.csv' + 'Close']].plot()
    plt.show()    
    
def test_run():
    start_date = '2012-03-23'
    end_date = '2017-03-23'
    dates = pd.date_range(start_date, end_date)
    df1 = pd.DataFrame(index = dates)
    for filename in ['ICICIBANK_LIMITED_IBN.csv', 'ICICIBANK.NS.csv']:#glob.glob(os.path.join(path, '*.csv')):
#        each = open(filename, 'r').read()
        print "Max close",
        print filename, get_max_close(filename)
        temp_df = pd.read_csv('./DataSets/{}'.format(filename), index_col = "Date", parse_dates = True, usecols = ['Date','Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'], na_values = ['nan'])
        temp_df = temp_df.rename(columns = {'Adj Close': filename + 'Adj Close', 'Open' : filename + 'Open', 'High' : filename + 'High', 'Low' : filename + 'Low', 'Close' : filename + 'Close', 'Volume' : filename + 'Volume'})
        df1 = df1.join(temp_df)
    tempdf = pd.read_csv('./DataSets/shuffled_predicted.csv', index_col = "Date", parse_dates = True, usecols = ['Date', 'Close'], na_values = ['nan'])
    ax = df1['ICICIBANK.NS.csvClose'].plot(title = "Close rolling mean", label = 'Close')
#    rm_ICICI_NSE = pd.rolling_mean(df1['ICICIBANK.NS.csvClose'], window = 30)
#    rm_ICICI_NSE.plot(label = 'Rolling mean', ax = ax)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc = 'best')
    plt.show()

    df1 = df1.dropna()
    df1['rolling mean NSEClose'] = df1['ICICIBANK.NS.csvClose'].rolling(5).mean()
    df1['rolling mean NYSEClose'] = df1['ICICIBANK_LIMITED_IBN.csvClose'].rolling(5).mean()
    df1['daily return NYSE'] = df1['ICICIBANK_LIMITED_IBN.csvClose'].div(df1['ICICIBANK_LIMITED_IBN.csvClose'].shift(1))-1
    df1['daily return NSE'] = df1['ICICIBANK.NS.csvClose'].div(df1['ICICIBANK.NS.csvClose'].shift(1))-1

    print df1['ICICIBANK.NS.csvClose']
    print df1['ICICIBANK.NS.csvClose'].shift(1)
    df1 = df1.join(tempdf)
    df1.Close = df1.Close.shift(-1)
    df1 = df1.dropna()
    df1.to_csv('test.csv', encoding = 'utf-8')
#    normalized_plot(df1)
        
        
if __name__ == "__main__":
    test_run()
    pass
        