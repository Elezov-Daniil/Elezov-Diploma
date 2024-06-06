import pandas as pd
import numpy as np
from datetime import datetime, timedelta
# from Functions_for_extrapolation import get_jump_wing_param, get_curvature, get_skew, get_SVI_extrapolation, lines_2D_SVI_example, get_grids_for_volatility_surface, lines_2D_SVI_for_interpolate, get_w_SVI_raw
from SVI_calibrator import SVI

import matplotlib.pyplot as plt
from scipy.integrate import odeint

#-------------------------------------------------------------------
'''
Function checks 3 types of arbitrage: calendar, butterflies and asymptotics behaviour for large and small strikes
Function prints if some kind of arbitrage is exist
'''
def check_static_arbitrage(value_options, data_dict):
    calendar_arbitrage = check_calendar_arbitrage(value_options)
    butterfly_arbitrage = check_butterflies_arbitrage_with_formula(data_dict)
    limit_price = check_limit_price(data_dict)
    print()
    if len(calendar_arbitrage) + len(butterfly_arbitrage) + len(limit_price) == 0:
        print('static_arbitrage is absence')
    else:
        
        if len(calendar_arbitrage) != 0:
            print('calandar arbitrage exists')
        if len(butterfly_arbitrage) != 0:
            print('butterfly arbitrage exists')
        if len(limit_price) != 0:
            print('limit price arbitrage exists')
    

#--------------------------------------------------------
'''
Function checks calender arbitrage, based on article "Arbitrage-free SVI volatility surfaces" by
Jim Gatheral and Antoine Jacquiery. We have to compare value option in term of bid for previous expiration with value option in term of ask for further expiration.
Function returns expirations, when calendar arbitrage exists.
'''

def check_calendar_arbitrage(value_options):
    value_option_check = value_options
    strike = value_option_check['strike'].unique()
    calendar_arbitrage = pd.DataFrame()
    for k in strike:
        Calls = value_option_check.loc[(value_option_check['strike'] == k) & (value_option_check['optionType'] == 'calls')] # take calls for specified strike
        Puts = value_option_check.loc[(value_option_check['strike'] == k) & (value_option_check['optionType'] == 'puts')] # take puts for specified strike
        Calls = Calls.sort_values(by='expiry_date')
        Puts = Puts.sort_values(by='expiry_date')
             # compare value options between different expiration and find those, which value less than in previous expiration in term of bid ask spread
        c_0 = 0
        spread_size_last = 0
        for i in range(0, len(Calls)):
            spread_size = (Calls['ask'].iloc[i] - Calls['bid'].iloc[i]) / 2
            c_1 = Calls['value_option'].iloc[i] + spread_size # find value option in term of ask
            if c_0 < c_1:
                c_0 = c_1 - spread_size * 2 #find value option in term of bid
                spread_size_last = spread_size
            else:
                calendar_arbitrage = pd.concat([calendar_arbitrage, Calls.loc[(Calls['value_option'] == c_0 + spread_size_last) | (Calls['value_option'] == c_1 - spread_size)]])
                c_0 = c_1 - spread_size * 2
                spread_size_last = spread_size

        spread_size_last = 0
        p_0 = 0
        for i in range(0, len(Puts)):
            spread_size = (Puts['ask'].iloc[i] - Puts['bid'].iloc[i]) / 2
            p_1 = Puts['value_option'].iloc[i] + spread_size # find value option in term of ask
            if p_0 < p_1:
                p_0 = p_1 - spread_size * 2 #find value option in term of bid
                spread_size_last = spread_size
            else:
                calendar_arbitrage = pd.concat([calendar_arbitrage, Puts.loc[(Puts['value_option'] == p_0 + spread_size_last) | (Puts['value_option'] == p_1 - spread_size)]])
                p_0 = p_1 - spread_size * 2
                spread_size_last = spread_size

    return calendar_arbitrage
#-------------------------------------------------------------
'''
Function checks butterflies arbitrage, based on article "Robust Calibration For SVI Model Arbitrage Free" by Tahar Ferhati.
    (a - m*b*(rho + 1))*(4 - a + m*b*(rho + 1)) / b^2*(rho + 1)^2 > 1
    (a - m*b*(rho - 1))*(4 - a + m*b*(rho - 1)) / b^2*(rho - 1)^2 > 1
    0 < b^2*(rho - 1)^2 < 4
    0 < b^2*(rho + 1)^2 < 4
Function returns expirations, when butterflies arbitrage exists.
'''
def check_butterflies_arbitrage_with_formula(data_dict):    
    butterflies_arbitrage = []
    for k in data_dict["implied_volatility_surface"]:
        set_param = k['set_param_raw']
        if not (((set_param.a - set_param.m * set_param.b * (set_param.rho + 1)) * (4 - set_param.a + set_param.m * set_param.b * (set_param.rho + 1))) / (set_param.b ** 2 * (set_param.rho + 1) ** 2) > 1) & (((set_param.a - set_param.m * set_param.b * (set_param.rho - 1)) * (4 - set_param.a + set_param.m * set_param.b * (set_param.rho - 1))) / (set_param.b ** 2 * (set_param.rho - 1) ** 2) > 1) & (set_param.b ** 2 * (set_param.rho - 1) ** 2 < 4) & (set_param.b ** 2 * (set_param.rho - 1) ** 2 > 0) & (set_param.b ** 2 * (set_param.rho + 1) ** 2 < 4) & (set_param.b ** 2 * (set_param.rho + 1) ** 2 > 0):
            butterflies_arbitrage.append(k['expiry_date'])
            
    return butterflies_arbitrage
#-----------------------------------------------------------------------
'''
Function checks the asymptotics behaviour for large and small strikes (1 + |rho| * b ⩽ 2).
Function returns expirations, when arbitrage exists.
'''

def check_limit_price(data_dict):
    limit_price = []
    for k in data_dict["implied_volatility_surface"]:
        set_param = k['set_param_raw']
        if not (set_param.b * (1 + set_param.rho) < 2):
            limit_price.append(k['expiry_date'])
    return limit_price

#-----------------------------------------------------------------------
'''
Function computes g-function from article "Arbitrage-free SVI volatility surfaces" by
Jim Gatheral and Antoine Jacquiery for given expiration and calibrated parameters.
g(x) := (1 - xw'(x)/2w(x))^2 - w'(x)^2/4 * (1/w(x) + 1/4) + w''(x)/2

Function returns g-function value and plots graph.
'''

def get_g_function_for_butterfly_arbitrage(X, set_param_raw, expiry_date, coef, t, graph=None, N=1000):
    w_old = []
    for i in X:
        w_old.append(get_SVI_raw(set_param_raw, i))
    New_w = get_but_free_arb_curve(set_param_raw, X)
    
    if graph:
        fig = plt.figure(figsize=(13, 6))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

#         ax1.plot(X,np.sqrt(New_w / t), label='new_volatility')
        ax1.plot(X,np.sqrt(np.array(w_old) / t), label='old_volatility')
        ax1.set_title(expiry_date + ' total implied variance from SVI')
        ax1.set_xlabel(r'log moneyness')
        ax1.set_ylabel(r'total implied variance')
        ax1.grid()
#         ax1.legend()


#         ax2.plot(*compute_g(X, New_w), label='new_volatility')
        ax2.plot(*compute_g(X, np.array(w_old)), label='old_volatility')
        ax2.set_title(expiry_date + ' g-function from SVI')
        ax2.set_xlabel(r'log moneyness')
        ax2.set_ylabel(r'function g')
        ax2.grid()
#         ax2.legend()

        plt.show()
    return w_old, New_w

def get_SVI_raw(self, x):
    return self.a + self.b*(self.rho*(x-self.m) +np.sqrt((x-self.m)**2 + self.sigma**2))

def durrleman_function(self, x):
    w = get_SVI_raw(self, x)
    wp = self.b*(self.rho + (x-self.m) / np.sqrt(
        (x-self.m)**2 + self.sigma**2))
    wpp = self.b*(1/np.sqrt((x-self.m)**2 + self.sigma**2) -
                  (x-self.m)**2/((x-self.m)**2 + self.sigma**2)**1.5)
    return (1-0.5*x*wp/w)**2 - 0.25*wp**2*(1/w+0.25) + 0.5*wpp
    
def get_but_free_arb_curve(S, X):    
    def f(y, x):
        w, w_first = y
        return [w_first, 2*max(durrleman_function(S, x), 0) + 0.5*(w_first)**2*(1/w+0.25) - 2*(1-0.5*x*w_first/w)**2]
    y0 = [get_SVI_raw(S,X[0]), (get_SVI_raw(S,X[1])-get_SVI_raw(S,X[0]))/(X[1]-X[0])]
    sol = odeint(f, y0, X)
    return sol[:,0]

def compute_g(X, W):
    wp = np.diff(W)/np.diff(X)
    wpp = np.diff(wp)/np.diff(X[1:])
    w = W[2:]
    wp = wp[1:]
    x = X[2:]
    return x, (1-0.5*x*wp/w)**2 - 0.25*(wp)**2*(1/w+0.25) + 0.5*wpp
