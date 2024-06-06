import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import os
import requests
from time import mktime
import pytz
import yfinance as yf
'''
Function to retrieve the option chain listed for specified stock from Nasdaq exchange 
'''
def yFinance_retrieve_options_chain(yticker):
    tk = yf.Ticker(yticker)
    exps = tk.options # Expiration dates

    options_yf_pddf = pd.DataFrame() # Get options for each expiration
    for e in exps:
        opt = tk.option_chain(e)
        opt = pd.concat([opt.calls, opt.puts], axis=0)
        opt['expiryDate'] = e
        options_yf_pddf = pd.concat([options_yf_pddf, opt], ignore_index=True, axis=0)
    
    options_yf_pddf['isCall'] = options_yf_pddf['contractSymbol'].str[4:].apply(lambda x: "C" in x) # Boolean column if the option is a Call
    options_yf_pddf['mid'] = (options_yf_pddf['bid'] + options_yf_pddf['ask']) / 2 # Calculate the midpoint of the bid-ask
    options_yf_pddf = options_yf_pddf.drop(columns = ['contractSize', 'currency', 'change', 'inTheMoney' , 'percentChange']) # Drop unnecessary and meaningless columns
    options_yf_pddf["yFinance_dividend_yield"] = tk.dividends.tail(10).mean()

    return options_yf_pddf


'''
Function to retrieve the option chain live listed for specified stock from Nasdaq exchange 
'''
def yFinance_collect_live_listed_options_information(yticker, time, dividend_yield=0, cutoff_hours_lag_from_now=None, log=False):
    options_yf_pddf = yFinance_retrieve_options_chain(yticker)
    if log: print("Number of options retrieved :", len(options_yf_pddf))

    spots = yf.download( yticker, period="5d",interval="1m", progress=False)

    #if no cutoff_hours we keep options traded between today 00:00 and tomorrow 00:00, else we keep from (now-cutoff_hours_lag_from_now) and (now_plus_one_day-cutoff_hours_lag_from_now
    now = time
    ref_time = datetime.min.time() if cutoff_hours_lag_from_now is None else (now+timedelta(hours=-cutoff_hours_lag_from_now)).time()
    cutoff = datetime.combine(now, ref_time, pytz.UTC)
    cutoff_plus_one = datetime.combine(now+ timedelta(days=+1), ref_time, pytz.UTC)
    print(yticker)
    #Retrieve reference spot for last trade date
    options_yf_pddf["referenceSpot"]=-1.0
    for i in range(len(options_yf_pddf)):
        lastTradeDate = pd.to_datetime(options_yf_pddf.loc[i, "lastTradeDate"])
        if lastTradeDate>cutoff and lastTradeDate<cutoff_plus_one :
            iloc_idx = spots.index.get_indexer([lastTradeDate], method='nearest')
            options_yf_pddf.loc[i, "referenceSpot"]=round(spots.iloc[iloc_idx]["Adj Close"].item(),4)
    options_yf_pddf = options_yf_pddf.loc[options_yf_pddf["referenceSpot"]>0]
    options_yf_pddf["last close"] = round(spots["Close"][-1],4)
    options_yf_pddf["yFinance_dividend_yield"] = options_yf_pddf["yFinance_dividend_yield"].div(options_yf_pddf["last close"])
    options_yf_pddf["ticker"] = yticker
    options_yf_pddf["mid"] = (options_yf_pddf['ask'].add(options_yf_pddf['bid'])) / 2
    options_yf_pddf.reset_index(inplace=True)
    if log: print("close:",round(spots["Close"][-1],4))
    if log: print("Number of options kept (reference spot available):", len(options_yf_pddf))
    options_yf_pddf = options_yf_pddf[['contractSymbol','ticker', 'lastTradeDate', 'expiryDate', 'strike', 'lastPrice', 'bid', 'ask', 'mid',
       'volume', 'openInterest', 'impliedVolatility', 'last close', 'isCall',  'yFinance_dividend_yield']]
    options_yf_pddf['isCall'].loc[options_yf_pddf['isCall'] == True] = 'calls'
    options_yf_pddf['isCall'].loc[options_yf_pddf['isCall'] == False] = 'puts'
    options_yf_pddf.columns = ['contractSymbol','ticker', 'lastTradeDate', 'expiryDate', 'strike', 'lastPrice', 'bid', 'ask', 'mid',
       'volume', 'openInterest', 'impliedVolatility', 'last close', 'optionType',  'yFinance_dividend_yield']
    
    for i in range(len(options_yf_pddf)):
        options_yf_pddf["lastTradeDate"][i] = pd.to_datetime(options_yf_pddf.loc[i, "lastTradeDate"]).tz_localize(None)
        
    options_yf_pddf["lastTradeDate"] = options_yf_pddf["lastTradeDate"].astype(str)
    return options_yf_pddf

'''
Function to retrieve the option chain live listed for specified stock from Nasdaq exchange 
Making file with option Data in your current folder 
'''

def get_data_about_option(ticker, time, directory,save_data = False):
    data_options = yFinance_collect_live_listed_options_information(ticker, time)
    
    if save_data:
        data_folder = directory + '//' + ticker + ".csv"
        data_options.to_csv(data_folder, index= False )
    
    return data_options

'''
Function to retrieve the option chain listed for specified crypto currency from Derebit exchange 
Making file with option Data in your current folder 
Currency: ETH, BTC,
'''

def get_data_about_crypto_options(currency):
    #scrapping data about curenct active option instruments
    url = 'https://history.deribit.com/api/v2/public/get_instruments?currency=' + currency + '&kind=option&expired=false'
    api_response = (requests.get(url)).json()
    data_options = pd.DataFrame()
    #scrapping data about curenct currency price in USD
    url_spot = 'https://deribit.com/api/v2/public/get_index?currency=' + currency
    spot_price = (requests.get(url_spot)).json()['result']['edp']

    #scrapping data about crypto options
    for k in api_response['result']:
        strike = k['strike']
        option_type = k['option_type']
        time1 = datetime.fromtimestamp(k['expiration_timestamp'] / 1000)
        expiration =  str(time1.year) + '-' + str(time1.month) + '-' + str(time1.day)
        time1 = datetime.fromtimestamp(k['creation_timestamp'] / 1000)
        last_trade_day =  str(time1.year) + '-' + str(time1.month) + '-' + str(time1.day) + ' ' + str(time1.hour) + ':' + str(time1.minute) + ':' + str(time1.second)

        instrument_name = k['instrument_name']
        # scrapping data about given order book
        instuments_information_url = 'https://deribit.com/api/v2/public/get_order_book?instrument_name=' + instrument_name
        instuments_information = (requests.get(instuments_information_url)).json()
        bid_price = instuments_information['result']['best_bid_price']
        ask_price = instuments_information['result']['best_ask_price']
        mid_price = (ask_price + bid_price) / 2
        volume = instuments_information['result']['stats']['volume']
        open_interest = instuments_information['result']['open_interest']
        DerebitIV = (instuments_information['result']['bid_iv'] + instuments_information['result']['ask_iv']) / 2
        data_options = pd.concat([data_options, pd.DataFrame([instrument_name, last_trade_day, expiration, strike, option_type, spot_price, bid_price,ask_price, mid_price, volume, open_interest, DerebitIV]).T])
    
    #making necessary adjustment
    data_options.columns = ['instrumentName', 'lastTradeDate', 'expiryDate', 'strike','optionType','last close', 'bid', 'ask', 'mid', 'volume', 'openInterest', 'DerebitIV']
    data_options['yFinance_dividend_yield'] = 0
    data_options['optionType'].loc[data_options['optionType'] == 'call'] = 'calls'
    data_options['optionType'].loc[data_options['optionType'] == 'put'] = 'puts'
    data_options['ticker'] = currency
    data_options['bid'] = data_options['bid'] * data_options['last close'].unique()[0]
    data_options['ask'] = data_options['ask'] * data_options['last close'].unique()[0]
    data_options['mid'] = data_options['mid'] * data_options['last close'].unique()[0]
    data_options = data_options.loc[data_options['bid'] != 0]
    data_options = data_options.loc[data_options['ask'] != 0]

    data_options.to_csv('DATA//Crypto_currencies//' + currency + '//' + str(datetime.now().date()) + '-' + str(datetime.now().hour) + '-' + str(datetime.now().minute) + '.csv', index=False)

def get_funding_rate(currency):
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)
    start_timestamp = int(mktime(datetime.strptime(str(yesterday), "%Y-%m-%d").timetuple())) * 1000
    end_timestamp = int(mktime(datetime.strptime(str(today), "%Y-%m-%d").timetuple())) * 1000
    url = 'https://deribit.com/api/v2/public/get_funding_rate_history?instrument_name=' + currency + '-PERPETUAL&start_timestamp=' + str(start_timestamp) +'&end_timestamp=' + str(end_timestamp)
    api_response = requests.get(url).json()

    funding_rate = pd.DataFrame()
    for k in api_response['result']:
        time = datetime.fromtimestamp(k['timestamp'] / 1000)
        Date = str(time.year) + '-' + str(time.month) + '-' + str(time.day)
        rate = k['interest_1h']
    funding_rate = pd.concat([funding_rate, pd.DataFrame([Date,rate]).T])
    funding_rate.columns = ['Date', 'rate']
    funding_rate = funding_rate.set_index('Date')
    rate = funding_rate.mean() * 365
    
    return rate[0]