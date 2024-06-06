import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np


def lines_3D(ticker, data_dict):
    plt.figure(figsize = (8,8))
    ax = plt.axes(projection ="3d")

    for k in data_dict["implied_volatility_surface"]:
        ax.plot(np.array(k['strikes'])/k['reference_forward'], k['mid_implied_volatilities'], zs=k['expiry_date_in_act365_year_fraction'],zdir='x')
        ax.scatter( k['expiry_date_in_act365_year_fraction'], np.array(k['strikes'])/k['reference_forward'], k['mid_implied_volatilities'], s=10)

    ax.set_title(ticker)
    ax.set_xlabel(r'Maturity (year fraction)')
    ax.axes.set_xlim3d(left=0, right=2.0)
    ax.set_ylabel(r'Strike (fwd moneyness)')
    ax.axes.set_ylim3d(bottom=0.4, top=2)
    ax.set_zlabel(r'Implied Volatility')
    ax.axes.set_zlim3d(bottom=0.0, top=1) 
    ax.set_zticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    ax.grid()
    ax.view_init(20, 40)
    plt.show()

    
def lines_2D_SVI(ticker, data_dict, model, expiry_date):
    fig = plt.figure(figsize=(13, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    k = data_dict["implied_volatility_surface"][expiry_date]
    grid = np.array(k['strikes'])/k['reference_forward']
    ax1.scatter(grid, k['mid_implied_volatilities'], s = 5, color='y')
    ax1.plot(grid, k['mid_implied_volatilities'], label= 'mid IV from BM', color='y', linewidth=0.7)
    
    ax1.scatter(grid, k['bid_implied_volatilities'], s = 5, color='g')
    ax1.plot(grid, k['bid_implied_volatilities'], label= 'bid IV from BM', color='g', linewidth=0.7)
    
    ax1.scatter(grid, k['ask_implied_volatilities'], s = 5, color='r')
    ax1.plot(grid, k['ask_implied_volatilities'], label= 'ask IV from BM', color='r', linewidth=0.7)
    
    ax1.scatter(grid, k[model], s = 5, color='b')
    ax1.plot(grid, k[model], label= model, color='b', linewidth=1.5)
    
#     max_error = k[model] - np.array(k['mid_implied_volatilities'])
#     max_relative_error = max_error / np.array(k['mid_implied_volatilities'])
    max_relative_error = [0] * len(k[model])
    for j in range(0,len(k[model])):
        if (k[model][j] >= k['bid_implied_volatilities'][j]) & (k[model][j] <= k['ask_implied_volatilities'][j]):
            max_relative_error[j] = 0
        else:
            if k[model][j] > k['ask_implied_volatilities'][j]:
                max_relative_error[j] = k[model][j] - k['ask_implied_volatilities'][j]
            else:
                max_relative_error[j] = k['bid_implied_volatilities'][j] - k[model][j]
    
    
    ax2.plot(grid, max_relative_error, label= 'error', color='grey', linewidth=3, linestyle='--')

    
    
    ax1.set_title(ticker + ' '+ k['expiry_date'] + ' ' + model)
    ax1.set_xlabel(r'strike (fwd moneyness)')
    ax1.set_ylabel(r'Implied Volatility')
    ax1.legend()
    ax1.grid()
    ax2.set_title(ticker + ' '+ k['expiry_date'] +' error graph' + ' ' + model)
    ax2.set_xlabel(r'strike (fwd moneyness)')
    ax2.set_ylabel(r'Implied Volatility')
    ax2.legend()
    ax2.grid()
    plt.show()
'''
    -------------------------------------------------------
    -------------------------------------------------------
    -------------------------------------------------------
    -------------------------------------------------------
    -------------------------------------------------------
    -------------------------------------------------------
    -------------------------------------------------------
    -------------------------------------------------------
    -------------------------------------------------------
    -------------------------------------------------------
    -------------------------------------------------------
    -------------------------------------------------------
    -------------------------------------------------------
    -------------------------------------------------------
    -------------------------------------------------------
'''
def scatter_3D(ticker, data_dict):
    plt.figure(figsize = (8,8))
    ax = plt.axes(projection ="3d")

    for k in data_dict["implied_volatility_surface"]:
        ax.scatter( k['expiry_date_in_act365_year_fraction'], np.array(k['strikes'])/k['reference_forward'], k['mid_implied_volatilities'], s=10)

    ax.set_title(ticker)
    ax.set_xlabel(r'Maturity (year fraction)')
    ax.set_ylabel(r'Strike (fwd moneyness)')
    ax.set_zlabel(r'Implied Volatility')
    ax.grid()
    ax.view_init(20, 40)
    plt.show()


    
def lines_3D_SVI_raw(ticker, data_dict, low_limit, high_limit, N, with_IV_from_BlackModel=False):
    plt.figure(figsize = (8,8))
    ax = plt.axes(projection ="3d")
    if with_IV_from_BlackModel:
        for k in data_dict["implied_volatility_surface"]:
            x_grid = np.array(k['strikes'])/k['reference_forward']
            ax.plot(x_grid, k['SVI_implied_volatilities'], zs=k['expiry_date_in_act365_year_fraction'],zdir='x', color='r')
            ax.plot(np.array(k['strikes'])/k['reference_forward'], k['mid_implied_volatilities'], zs=k['expiry_date_in_act365_year_fraction'],zdir='x', color ='b')
    else:    
        for k in data_dict["implied_volatility_surface"]:
            x_grid = np.array(k['strikes'])/k['reference_forward']
            ax.plot(x_grid, k['SVI_implied_volatilities'], zs=k['expiry_date_in_act365_year_fraction'],zdir='x')

    ax.set_title(ticker)
    ax.set_xlabel(r'Maturity (year fraction)')
    ax.axes.set_xlim3d(left=0, right=3.5)
    ax.set_ylabel(r'Strike (fwd moneyness)')
    ax.axes.set_ylim3d(bottom=0.1, top=3)
    ax.set_zlabel(r'Implied Volatility')
    ax.axes.set_zlim3d(bottom=0.0, top=2.5) 
    ax.set_zticks(np.linspace(0.1,3,30))
    ax.grid()
    ax.view_init(20, 40)
    plt.show()
    
def lines_2D_SVI_raw(ticker, data_dict, expire_date, low_limit, high_limit, N, with_IV_from_BlackModel=False):
    plt.figure(figsize = (8,3))
    ax = plt.gca()
    k = data_dict["implied_volatility_surface"][expire_date]
    x_grid = np.array(k['strikes'])/k['reference_forward']
    ax.plot(x_grid, k['SVI_implied_volatilities'], label = 'IV from SVI', color='r')
    ax.plot(np.array(k['strikes'])/k['reference_forward'], k['mid_implied_volatilities'], label= 'IV from Black model', color ='b')
    ax.set_title(ticker + '  ' + k['expiry_date'])
    ax.set_xlabel(r'strike (fwd moneyness)')
    ax.set_ylabel(r'Implied Volatility')
    ax.legend()
    ax.grid()
    plt.show()

def lines_2D(ticker, data_dict):
    plt.figure(figsize = (8,8))
    ax = plt.gca()

    for k in data_dict["implied_volatility_surface"]:
        ax.scatter(np.array(k['strikes'])/k['reference_forward'], k['mid_implied_volatilities'], label= k['expiry_date'], s=10)
        plt.plot(np.array(k['strikes'])/k['reference_forward'],k['mid_implied_volatilities'])

    ax.set_title(ticker)
    ax.set_xlabel(r'strike (fwd moneyness)')
    ax.set_ylabel(r'Implied Volatility')
    ax.legend()
    ax.grid()
    plt.show()
    
def lines_3D_data_options(ticker, data_dict, type_option='ITM'):
    plt.figure(figsize = (8,8))
    ax = plt.axes(projection ="3d")
    
    if type_option == 'OTM':
        for k in data_dict:
            #     keep only ATM options
            k = k.loc[((k["x_grid"] >= 1) & (k["optionType"] == "calls")) | ((k["x_grid"] < 1) & (k["optionType"] == 'puts'))]
            k = k.sort_values(by=['x_grid'])
            ax.plot(k['x_grid'], k['value_option'], zs=k['expiry_date_in_act365_year_fraction'],zdir='x')
            ax.scatter( k['expiry_date_in_act365_year_fraction'], k['x_grid'], k['value_option'], s=10)
    
    if type_option == 'ITM':
        for k in data_dict:
            #     keep only ATM options
            k = k.loc[((k["x_grid"] <= 1) & (k["optionType"] == "calls")) | ((k["x_grid"] > 1) & (k["optionType"] == 'puts'))]
            k = k.sort_values(by=['x_grid'])
            ax.plot(k['x_grid'], k['value_option'], zs=k['expiry_date_in_act365_year_fraction'],zdir='x')
            ax.scatter( k['expiry_date_in_act365_year_fraction'], k['x_grid'], k['value_option'], s=10)
    
    

    ax.set_title(ticker + ' ' + type_option)
    ax.set_xlabel(r'Maturity (year fraction)')
    ax.axes.set_xlim3d(left=0, right=2.5)
    ax.set_ylabel(r'Strike (fwd moneyness)')
    ax.axes.set_ylim3d(bottom=0.4, top=2)
    ax.set_zlabel(r'Implied Volatility')
    ax.axes.set_zlim3d(bottom=0.0, top=200) 
    ax.grid()
    ax.view_init(20, 40)
    plt.show()

def error_graph_3D(data_dict, ticker, data_date, type_option=None):
    plt.figure(figsize = (8,8))
    ax = plt.axes(projection ="3d")

    if type_option == None:
        for expiry_date in data_dict['expiry_date'].unique():
            
            k = data_dict.loc[data_dict['expiry_date'] == expiry_date]
            inside = k.loc[k['correct_price'] == 'inside']
            below = k.loc[k['correct_price'] == 'below']
            above = k.loc[k['correct_price'] == 'above']
            ax.scatter(inside['expiry_date_in_act365_year_fraction'], inside['x_grid'], inside['value_option'], s=5, color='silver')
            ax.scatter(below['expiry_date_in_act365_year_fraction'], below['x_grid'], below['value_option'], s=5, color='b')
            ax.scatter(above['expiry_date_in_act365_year_fraction'], above['x_grid'], above['value_option'], s=5, color='r')

    if type_option == 'OTM':
        for expiry_date in data_dict['expiry_date'].unique():
            k = data_dict.loc[data_dict['expiry_date'] == expiry_date]
            #     keep only ATM options
            k = k.loc[((k["x_grid"] >= 1) & (k["optionType"] == "calls")) | ((k["x_grid"] < 1) & (k["optionType"] == 'puts'))]
            k = k.sort_values(by=['x_grid'])
            inside = k.loc[k['correct_price'] == 'inside']
            below = k.loc[k['correct_price'] == 'below']
            above = k.loc[k['correct_price'] == 'above']
            ax.scatter( inside['expiry_date_in_act365_year_fraction'], inside['x_grid'], inside['value_option'], s=5, color='silver')
            ax.scatter( below['expiry_date_in_act365_year_fraction'], below['x_grid'], below['value_option'], s=5, color='b')
            ax.scatter( above['expiry_date_in_act365_year_fraction'], above['x_grid'], above['value_option'], s=5, color='r')

    if type_option == 'ITM':
        for expiry_date in data_dict['expiry_date'].unique():
            k = data_dict.loc[data_dict['expiry_date'] == expiry_date]
            #     keep only ATM options
            k = k.loc[((k["x_grid"] <= 1) & (k["optionType"] == "calls")) | ((k["x_grid"] > 1) & (k["optionType"] == 'puts'))]
            k = k.sort_values(by=['x_grid'])
            inside = k.loc[k['correct_price'] == 'inside']
            below = k.loc[k['correct_price'] == 'below']
            above = k.loc[k['correct_price'] == 'above']
            ax.scatter( inside['expiry_date_in_act365_year_fraction'], inside['x_grid'], inside['value_option'], s=5, color='silver')
            ax.scatter( below['expiry_date_in_act365_year_fraction'], below['x_grid'], below['value_option'], s=5, color='b')
            ax.scatter( above['expiry_date_in_act365_year_fraction'], above['x_grid'], above['value_option'], s=5, color='r')
    
    ax.scatter(inside['expiry_date_in_act365_year_fraction'], inside['x_grid'], inside['value_option'], s=5, color='silver', label ='inside = ' + str(len(data_dict.loc[data_dict['correct_price'] == 'inside']) / len(data_dict) * 100) + '%')
    ax.scatter(below['expiry_date_in_act365_year_fraction'], below['x_grid'], below['value_option'], s=5, color='b', label ='below = ' + str(len(data_dict.loc[data_dict['correct_price'] == 'below']) / len(data_dict) * 100) + '%')
    ax.scatter(above['expiry_date_in_act365_year_fraction'], above['x_grid'], above['value_option'], s=5, color='r', label = 'above = ' + str(len(data_dict.loc[data_dict['correct_price'] == 'above']) / len(data_dict) * 100) + '%')       
    
    
    ax.set_title('Error graph for ' + ticker + ' data scrapped on ' + data_date + ' ' + 'IV from SVI')
    ax.set_xlabel(r'Maturity (year fraction)')
    ax.axes.set_xlim3d(left=0, right=2.5)
    ax.set_ylabel(r'Strike (fwd moneyness)')
    ax.axes.set_ylim3d(bottom=data_dict['strike'].min(), top=data_dict['strike'].max())
    ax.set_zlabel(r'Implied Volatility')
    ax.axes.set_zlim3d(bottom=0.0, top=200)
    ax.legend()
    ax.grid()
    ax.view_init(20, 40)
    plt.show()
    

    
def plot_discount_curve(discount_curve, data_date):
    plt.figure(figsize = (12, 4))
    plt.scatter(discount_curve['Date'], discount_curve['3 Month Term SOFR Forward Curve'] * 100, s=5, color='cornflowerblue')
    plt.plot(discount_curve['Date'], discount_curve['3 Month Term SOFR Forward Curve'] * 100, color='cornflowerblue', label ='3 month')
    plt.plot(discount_curve['Date'], discount_curve['1 Month Term SOFR Forward Curve'] * 100, label='1 month', color='r')
    plt.title('SOFR Forward Curve from Chatham Finance on ' + data_date)
    # reference: https://www.chathamfinancial.com/technology/us-forward-curves
    plt.grid()
    plt.legend()
    plt.show()

def lines_2D_total_varience(ticker, data_dict, end_number, interpolation=False):
    fig = plt.figure(figsize=(13, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
        
    if interpolation:  
        x_grid = np.linspace(0.1,2.5,1000)

        for j in range(0, len(data_dict["implied_volatility_surface"]) - end_number):
            k = data_dict["implied_volatility_surface"][j]
            ax1.scatter(np.array(x_grid), k['w_SVI_total_variance'], label= k['expiry_date'], s=1)
            ax1.plot(np.array(x_grid),k['w_SVI_total_variance'], linewidth = 0.5)

        for j in range(len(data_dict["implied_volatility_surface"]) - end_number, len(data_dict["implied_volatility_surface"])):
            k = data_dict["implied_volatility_surface"][j]
            ax2.scatter(np.array(x_grid), k['w_SVI_total_variance'], label= k['expiry_date'], s=1)
            ax2.plot(np.array(x_grid),k['w_SVI_total_variance'], linewidth = 0.5)
    else:
        for j in range(0, len(data_dict["implied_volatility_surface"]) - end_number):
            k = data_dict["implied_volatility_surface"][j]
            w_SVI_total_variance = np.power(k['IV_from_SVI'],2) * k['expiry_date_in_act365_year_fraction']
            ax1.scatter(np.array(k['strikes']), w_SVI_total_variance, label= k['expiry_date'], s=10)
            ax1.plot(np.array(k['strikes']),w_SVI_total_variance)
        
        for j in range(len(data_dict["implied_volatility_surface"]) - end_number, len(data_dict["implied_volatility_surface"])):
            k = data_dict["implied_volatility_surface"][j]
            w_SVI_total_variance = np.power(k['IV_from_SVI'],2) * k['expiry_date_in_act365_year_fraction']
            ax2.scatter(np.array(k['strikes']), w_SVI_total_variance, label= k['expiry_date'], s=10)
            ax2.plot(np.array(k['strikes']),w_SVI_total_variance)

    ax1.set_title(ticker + ' total variance')
    ax2.set_title(ticker + ' total variance')
    ax1.set_xlabel(r'strike (fwd moneyness)')
    ax2.set_xlabel(r'strike (fwd moneyness)')
    ax1.set_ylabel(r'Implied Volatility')
    ax2.set_ylabel(r'Implied Volatility')
    ax1.legend()
    ax2.legend()
    ax1.grid()
    ax2.grid()
    plt.show()
