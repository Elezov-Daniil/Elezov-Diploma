import numpy as np
import pandas as pd
from blackscholes import BlackScholes, call_price, call_iv
from cev import CEV
from heston import Heston
from sabr import SABR
from svi import SVI

def get_IV_from_stochastic_model(data_dict, log=False):
    for i in range(0, len(data_dict['implied_volatility_surface'])):
        data_initial = data_dict['implied_volatility_surface'][i]
        time = data_initial['expiry_date_in_act365_year_fraction']
        F = data_initial['reference_forward']
        DF = data_initial['reference_discount_factor']
        strikes = np.array(data_initial['strikes'])
        mid_IV = np.array(data_initial['mid_implied_volatilities'])
        s= data_dict['reference_spot']
        if log:
            print('Expiration =', data_initial['expiry_date'])
            
        # implied volatility via SVI 
        w_SVI = 'Fall'
        SVI_IV = SVI(0.5, 0.1, -0.5, 0.1, 0.1)
        x_array = np.log(np.array(strikes/F)) # log-moneyness
        w_array = np.power(mid_IV,2) * time #total varience
        while w_SVI == 'Fall':
            try:
                w_SVI = SVI_IV.calibrate(x_array, w_array)
                data_dict['implied_volatility_surface'][i]['set_param_raw'] = w_SVI
                data_dict['implied_volatility_surface'][i]['IV_from_SVI'] = w_SVI.iv(k = x_array, t = time)
            except Exception:
                w_SVI = 'Fall'
        if log:
            print('SVI_iv have computed')
     
        # implied volatility via CEV 
        CEV_IV = CEV(s = s, sigma=0.3, beta=0.8, r = (1/DF-1))
        try:
            CEV_IV = CEV_IV.calibrate(t = time, k = strikes, iv = mid_IV, s = s)
            data_dict['implied_volatility_surface'][i]['IV_from_CEV'] = CEV_IV.iv(t = time, k = strikes)
            if log:
                print('CEV_iv have computed')
        except Exception:
            data_dict['implied_volatility_surface'][i]['IV_from_CEV'] = [0] * len(mid_IV)

       
        # implied volatility via Dupire 


        # implied volatility via Heston 
        Heston_IV = Heston(s=s,v=0.25,kappa=0.1, theta=0.2, sigma=0.3, rho=0.4, r = (1/DF-1))
        Heston_IV = Heston_IV.calibrate(t=time,k = strikes,iv = mid_IV, s = s)
        data_dict['implied_volatility_surface'][i]['IV_from_Heston'] = Heston_IV.iv(t = time, k = strikes)
        if log:
            print('Heston_iv have computed')
        # implied volatility via SABR 
        SABR_IV = SABR(s=s,alpha=0.1, beta = 0.1, rho = 0.1, nu = 0.1)
        SABR_IV = SABR_IV.calibrate(t=time,k = strikes,iv = mid_IV, s = s)
        data_dict['implied_volatility_surface'][i]['IV_from_SABR'] = SABR_IV.iv(t=time,k = strikes)
        if log:
            print('SABR_iv have computed')

    return data_dict