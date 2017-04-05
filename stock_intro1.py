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
    df = pd.read_csv("{symbol}".format(symbol=symbol))
#    df[['Close', 'Adj Close']].plot()
#    plt.show()
    df[['Close']].plot()
    plt.show()
    return df['Close'].max()
    

def test_run():
    start_date = '2016-12-22'
    end_date = '2017-01-20'
    dates = pd.date_range(start_date, end_date)
    df1 = pd.DataFrame(index = dates)
    for filename in glob.glob(os.path.join(path, '*.csv')):
#        each = open(filename, 'r').read()
        print "Max close",
        print filename, get_max_close(filename)
        temp_df = pd.read_csv('{}'.format(filename), index_col = "Date", parse_dates = True, usecols = ['Date', 'Adj Close'], na_values = ['nan'])
        temp_df = temp_df.rename(columns = {'Adj Close': filename})
        df1 = df1.join(temp_df)
    df1 = df1.dropna()
    print df1 
        
        
if __name__ == "__main__":
    test_run()
    pass
        