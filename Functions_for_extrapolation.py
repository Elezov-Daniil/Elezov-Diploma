import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
from Computing_IV import black_delta_formula
from scipy.stats import norm, linregress
from SVI_calibrator import SVI
def get_jump_wing_param(data_dict, number_droped_maturities_in_begin = 0, number_droped_maturities_in_end = 0):
    w = []
    v = []
    psi = []
    p = []
    c = []
    v_tilde = []
    for i in range(number_droped_maturities_in_begin, len(data_dict['implied_volatility_surface']) - number_droped_maturities_in_end):
        k = data_dict['implied_volatility_surface'][i]
        t = k['expiry_date_in_act365_year_fraction']
        self = k['set_param_raw']
        w.append(self.a + self.b*(-self.rho*self.m + np.sqrt(self.m**2+self.sigma**2)))
        w1 = self.a + self.b*(-self.rho*self.m + np.sqrt(self.m**2+self.sigma**2))
        v.append(w1/t)
        psi.append(self.b/np.sqrt(w1)/2 * (-self.m/np.sqrt(self.m**2+self.sigma**2) + self.rho))
        p.append(self.b*(1-self.rho)/np.sqrt(w1))
        c.append(self.b*(1+self.rho)/np.sqrt(w1))
        v_tilde.append((self.a + self.b*self.sigma*np.sqrt(1-self.rho**2)) / t)
        
    return w, v, psi, p, c, v_tilde

def get_skew(data_dict):
    vol = []
    vol_min = []
    skew_with_div= []
    skew_without_div = []
    for k in data_dict['implied_volatility_surface']:
        T = k['expiry_date_in_act365_year_fraction']
        set_param_raw = k['set_param_raw']
        F = k['reference_forward']
        w = set_param_raw.a + set_param_raw.b*(set_param_raw.rho*(0-set_param_raw.m) + np.sqrt((0-set_param_raw.m)**2 + set_param_raw.sigma**2))
        vol.append(np.sqrt(w / T))
        x = np.log(np.linspace(0.5, 20, 1000))
        w11 = np.linspace(0,0, 1000)
        for ind, i in enumerate(x):
            w11[ind] = set_param_raw.a + set_param_raw.b*(set_param_raw.rho*(i-set_param_raw.m) + np.sqrt((i-set_param_raw.m)**2 + set_param_raw.sigma**2))
        w12 = np.sqrt(w11 / T)
#         vol_min.append(k['SVI_implied_volatilities'].min())
        vol_min.append(w12.min())
    
        x_25_put = 0
        x_25_call = 0
        x_50 = 0
        delta25_put = -50
        delta25_call = 50
        sigma25_put = 0
        sigma25_call = 0
        sigma50 = 0
        while (delta25_put < -25):
            strike = np.exp(x_25_put) * F
            sigma = set_param_raw.a + set_param_raw.b*(set_param_raw.rho*(x_25_put-set_param_raw.m) + np.sqrt((x_25_put-set_param_raw.m)**2 + set_param_raw.sigma**2))
            sigma = np.sqrt(sigma / T)
            delta25_put = black_delta_formula(F, strike , T, sigma, 'puts') * 100
            x_25_put -= 0.001
        sigma25_put = set_param_raw.a + set_param_raw.b*(set_param_raw.rho*(x_25_put-set_param_raw.m) + np.sqrt((x_25_put-set_param_raw.m)**2 + set_param_raw.sigma**2))
        sigma25_put = np.sqrt(sigma25_put / T)
        while (delta25_call > 25):
            strike = np.exp(x_25_call) * F
            sigma = set_param_raw.a + set_param_raw.b*(set_param_raw.rho*(x_25_call-set_param_raw.m) + np.sqrt((x_25_call-set_param_raw.m)**2 + set_param_raw.sigma**2))
            sigma = np.sqrt(sigma / T)
            delta25_call = black_delta_formula(F, strike , T, sigma, 'calls') * 100
            x_25_call += 0.001
        sigma25_call = set_param_raw.a + set_param_raw.b*(set_param_raw.rho*(x_25_call-set_param_raw.m) + np.sqrt((x_25_call-set_param_raw.m)**2 + set_param_raw.sigma**2))
        sigma25_call = np.sqrt(sigma25_call / T)
        sigma50 = set_param_raw.a + set_param_raw.b*(set_param_raw.rho*(0-set_param_raw.m) + np.sqrt((0-set_param_raw.m)**2 + set_param_raw.sigma**2))
        sigma50 = np.sqrt(sigma50 / T)

        skew_with_div.append((sigma25_call - sigma25_put) / sigma50 )
        skew_without_div.append(sigma25_call - sigma25_put)

    return vol_min, vol, skew_without_div, skew_with_div

def get_curvature(data_dict):
    curvature_put = []
    curvature_call = []
    for k in data_dict['implied_volatility_surface']:
        t = k['expiry_date_in_act365_year_fraction']
        set_param_raw = k['set_param_raw']
        x = np.log(np.linspace(0.2, 1, 1000))
        w1 = np.linspace(0,0, 1000)
        w_k_first = np.linspace(0,0,len(w1)) # first derivatives by the strike 
        w_k_second = np.linspace(0,0,len(w1))# second derivatives by the strike 
        curvature1 = np.linspace(0,0,len(w1))
        for ind, i in enumerate(x):
            w1[ind] = set_param_raw.a + set_param_raw.b*(set_param_raw.rho*(i-set_param_raw.m) + np.sqrt((i-set_param_raw.m)**2 + set_param_raw.sigma**2))
        w11 = np.sqrt(w1 / t)
        for i in range(0, len(w11) - 2):
            w_k_first[i+1] = (w11[i+2] - w11[i]) / (x[i+2] - x[i])
            w_k_second[i+1] = (w11[i+2] - 2 * w11[i+1] + w11[i]) / (x[i+2] - x[i]) ** 2
        w_k_first[0] = (w11[1] - w11[0]) / (x[1] - x[0])
        w_k_second[0] = (w11[1] - w11[0]) / (x[1] - x[0])
        w_k_first[len(w11)-1] = (w11[len(w11)-1] - w11[len(w11)-2]) / (x[len(w11)-1] - x[len(w11)-2])
        w_k_second[len(w11)-1] = (w11[len(w11)-1] - w11[len(w11)-2]) / (x[len(w11)-1] - x[len(w11)-2])
        for i in range(0, len(w11) - 2):
            curvature1[i+1] = abs(w_k_second[i+1]) / (1 + w_k_first[i+1] ** 2) ** 1.5
        curvature_put.append(curvature1.mean())
        
        x = np.log(np.linspace(1, 2.3, 1000))
        w1 = np.linspace(0,0, 1000)
        w_k_first = np.linspace(0,0,len(w1)) # first derivatives by the strike 
        w_k_second = np.linspace(0,0,len(w1))# second derivatives by the strike 
        curvature1 = np.linspace(0,0,len(w1))
        for ind, i in enumerate(x):
            w1[ind] = set_param_raw.a + set_param_raw.b*(set_param_raw.rho*(i-set_param_raw.m) + np.sqrt((i-set_param_raw.m)**2 + set_param_raw.sigma**2))
        w11 = np.sqrt(w1 / t)
        for i in range(0, len(w11) - 2):
            w_k_first[i+1] = (w11[i+2] - w11[i]) / (x[i+2] - x[i])
            w_k_second[i+1] = (w11[i+2] - 2 * w11[i+1] + w11[i]) / (x[i+2] - x[i]) ** 2
        w_k_first[0] = (w11[1] - w11[0]) / (x[1] - x[0])
        w_k_second[0] = (w11[1] - w11[0]) / (x[1] - x[0])
        w_k_first[len(w11)-1] = (w11[len(w11)-1] - w11[len(w11)-2]) / (x[len(w11)-1] - x[len(w11)-2])
        w_k_second[len(w11)-1] = (w11[len(w11)-1] - w11[len(w11)-2]) / (x[len(w11)-1] - x[len(w11)-2])
        for i in range(0, len(w11) - 2):
            curvature1[i+1] = abs(w_k_second[i+1]) / (1 + w_k_first[i+1] ** 2) ** 1.5
        curvature_call.append(curvature1.mean())
        
    return curvature_put, curvature_call
    

def graph_for_polynomial_regression(data_dict, v, v_tilda, psi, c,p, time_fraction, expire_time, start_v, start_v_tilda, end_psi, start_c, start_p):
    ticker = data_dict['underlying_ticker']
    data_date = str(data_dict['data_date'].date())
    fig = plt.figure(figsize=(15,15))
    fig.suptitle(ticker + ' on ' + data_date)
    ax1 = fig.add_subplot(321)
    ax2 = fig.add_subplot(322)
    ax3 = fig.add_subplot(323)
    ax4 = fig.add_subplot(324)
    ax5 = fig.add_subplot(325)
    fig.tight_layout(h_pad=3)
    
    #compute v
    v1 = []
    time = []
    for i in range(start_v, len(v)):
        v1.append(v[i])
        time.append(time_fraction[i])
    time = np.array(time)
    model_coef = np.poly1d(np.polyfit(np.log(time), v1, 2))
    ax1.scatter(time, v1, s=50)
    ax1.plot(np.append(time, expire_time), model_coef(np.log(np.append(time, expire_time))), 'r')

    #compute v_tilda
    v1 = []
    time = []
    for i in range(start_v_tilda, len(v_tilda)):
        v1.append(v_tilda[i])
        time.append(time_fraction[i])
    time = np.array(time)
    model_coef = np.poly1d(np.polyfit(np.log(time), v1, 2))
    ax2.scatter(time, v1, s=50)
    ax2.plot(np.append(time, expire_time), model_coef(np.log(np.append(time, expire_time))), 'r')
    

    #compute psi
    psi1 = []
    time = []
    for i in range(len(psi)-end_psi, len(psi)):
        psi1.append(psi[i])
        time.append(time_fraction[i])
    time = np.array(time)
    model_coef = np.poly1d(np.polyfit(np.log(time), psi1, 1))
    ax3.scatter(time,psi1, s=50)
    ax3.plot(np.append(time, expire_time), model_coef(np.log(np.append(time, expire_time))), 'r')

    #compute c
    c1 = []
    time = []
    for i in range(start_c, len(c)):
        c1.append(c[i])
        time.append(time_fraction[i])
    time = np.array(time)
    model_coef = np.poly1d(np.polyfit(time, c1 * time, 1))
    ax4.scatter(time,c1, s=50)
    ax4.plot(np.append(time, expire_time), model_coef(np.append(time, expire_time)) / np.append(time, expire_time), 'r')

    #compute p
    p1 = []
    time = []
    for i in range(start_p, len(p)-2):
        p1.append(p[i])
        time.append(time_fraction[i])
    time = np.array(time)
    model_coef = np.poly1d(np.polyfit(time, p1 * time, 1))
    ax5.scatter(time,p1, s=50)
    ax5.plot(np.append(time, expire_time), model_coef(np.append(time, expire_time)) / np.append(time, expire_time), 'r')

    ax1.set_title('ATM volatility from Jump Wing  (v)')
    ax1.set_xlabel(r'maturity')
    ax1.set_ylabel(r'value')
    ax1.grid()

    ax2.set_title('minimum implied volatility from Jump Wing (v_tilda)')
    ax2.set_xlabel(r'maturity')
    ax2.set_ylabel(r'value')
    ax2.grid()

    ax3.set_title('ATM skew from Jump Wing (psi) for)')
    ax3.set_xlabel(r'maturity')
    ax3.set_ylabel(r'value')
    ax3.grid()

    ax4.set_title('Slope right wing from Jump Wing (c)')
    ax4.set_xlabel(r'maturity')
    ax4.set_ylabel(r'value')
    ax4.grid()

    ax5.set_title('Slope left wing from Jump Wing (p)')
    ax5.set_xlabel(r'maturity')
    ax5.set_ylabel(r'value')
    ax5.grid()

    plt.grid(True)
    plt.show()

    
    
def get_SVI_extrapolation(data_dict, expire_time, timetable, start_v, start_v_tilda, end_psi, start_c, start_p, number_droped_maturities_in_begin=0, number_droped_maturities_in_end=0, type_reggresion='linear', plot_graph = False):
            
    a_new, b_new, rho_new, m_new, sigma_new = get_param_for_extrapolate(data_dict, expire_time, start_v, start_v_tilda, end_psi, start_c, start_p, number_droped_maturities_in_begin, number_droped_maturities_in_end, type_reggresion,plot_graph)
    
#     print('a=', a_new)
#     print('b=', b_new)
#     print('rho=', rho_new)
#     print('m=', m_new)
#     print('sigma=', sigma_new)
    k = data_dict['implied_volatility_surface'][timetable]
    x = np.log(k['strikes'] / k['reference_forward'])
    w_volatility = np.linspace(0,0,len(x))
    for ind, i in enumerate(x):
        w_volatility[ind] = get_w_SVI_raw(a_new, b_new, rho_new, m_new, sigma_new, i)
    w_volatility = np.sqrt(w_volatility / k['expiry_date_in_act365_year_fraction'])
    return a_new, b_new, rho_new, m_new, sigma_new, w_volatility

def get_param_for_extrapolate(data_dict, expire_time, start_v, start_v_tilda, end_psi, start_c, start_p, number_droped_maturities_in_begin=0, number_droped_maturities_in_end=0, type_reggresion='linear', plot_graph=False):
    w, v, psi, p, c, v_tilda = get_jump_wing_param(data_dict, number_droped_maturities_in_begin, number_droped_maturities_in_end)
    time_fraction = []
    for i in range(number_droped_maturities_in_begin, len(data_dict['implied_volatility_surface']) - number_droped_maturities_in_end):
        time_fraction.append(data_dict['implied_volatility_surface'][i]['expiry_date_in_act365_year_fraction'])
    time_fraction = np.array(time_fraction)
    
    if type_reggresion == 'polynomial':
    
        #compute v
        v1 = []
        time = []
        for i in range(start_v, len(v)):
            v1.append(v[i])
            time.append(time_fraction[i])
        time = np.array(time)
        model_coef = np.poly1d(np.polyfit(np.log(time), v1, 2))
        v_new = model_coef(np.log(expire_time))

        #compute v_tilda
        v_tilda[len(v_tilda)-1] = v_tilda[len(v_tilda) -2]
        dif = v[len(v_tilda)-5] - v_tilda[len(v_tilda)-5]
        v_tilda[len(v_tilda)-4] = v[len(v_tilda)-4] - dif
        v_tilda[len(v_tilda)-3] = v[len(v_tilda)-3] - dif
        v_tilda[len(v_tilda)-2] = v[len(v_tilda)-2] - dif
        v_tilda[len(v_tilda)-1] = v[len(v_tilda)-1] - dif
        
        v1 = []
        time = []
        for i in range(start_v_tilda, len(v_tilda)):
            v1.append(v_tilda[i])
            time.append(time_fraction[i])
        time = np.array(time)
        model_coef = np.poly1d(np.polyfit(np.log(time), v1, 2))
        v_tilda_new = model_coef(np.log(expire_time))

        #compute psi
        psi1 = []
        time = []
        for i in range(len(psi)-end_psi, len(psi)):
            psi1.append(psi[i])
            time.append(time_fraction[i])
        time = np.array(time)
        model_coef = np.poly1d(np.polyfit(time, psi1, 1))
        psi_new = model_coef(expire_time)

        #compute c
        c1 = []
        time = []

        for i in range(start_c, len(c)):
            c1.append(c[i])
            time.append(time_fraction[i])
        time = np.array(time)

        model_coef = np.poly1d(np.polyfit(time, c1 * time, 1))
        c_new = model_coef(expire_time) / expire_time

        #compute p
        p1 = []
        time = []

        for i in range(start_p, len(p)-2):
            p1.append(p[i])
            time.append(time_fraction[i])
        time = np.array(time)

        model_coef = np.poly1d(np.polyfit(time, p1 * time, 1))
        p_new = model_coef(expire_time) / expire_time
        if plot_graph:
            graph_for_polynomial_regression(data_dict, v, v_tilda, psi, c,p, time_fraction, expire_time, start_v, start_v_tilda, end_psi, start_c, start_p,)
    
#     print(v_new, v_tilda_new, psi_new, c_new, p_new)
    w = v_new * expire_time
#     print(w)
    b = 0.5 * np.sqrt(w)*(c_new+p_new)
#     print('b =', b)
    rho = 1 - 2*p_new/(c_new+p_new)
    beta = rho - 2*psi_new*np.sqrt(w)/b
#     print('beta =', beta)
    
    if np.abs(beta) > 1:
        return "error"
    elif beta == 0:
        m = 0
        sigma = (v_new-v_tilda_new)*expire_time / (b*(1-np.sqrt(1-rho**2)))
    else:
        alpha = np.sign(beta) * np.sqrt(1/beta**2 - 1)
        
        m = (v_new-v_tilda_new)*expire_time / (b*(-rho+np.sign(alpha)*np.sqrt(1+alpha**2) -
                                    alpha*np.sqrt(1-rho**2)))
        sigma = alpha*m

    a = v_tilda_new*expire_time - b*sigma*np.sqrt(1-rho**2)   
    return a, b, rho, m, sigma

def get_w_SVI_raw(a_new, b_new, rho_new, m_new, sigma_new, x):
    return a_new + b_new *(rho_new *(x-m_new) + np.sqrt((x-m_new)**2 + sigma_new**2))

def get_grids_for_volatility_surface(start_v, start_v_tilda, end_psi, start_c, start_p, data_dict, lister_time, lister):    
    T_ACT_365 = lister_time 
    for k in data_dict['implied_volatility_surface']:
        T_ACT_365.append(k['expiry_date_in_act365_year_fraction'])
        T_ACT_365.sort()
   
    number = len(T_ACT_365)


    set_param = [0] * number
    x_grid = [0] * number
    for i in lister:
        a, b, rho, m, sigma, _ = get_SVI_extrapolation(data_dict, T_ACT_365[i], 0, start_v, start_v_tilda, end_psi, start_c, start_p,0, 0, type_reggresion = 'polynomial')
        set_param[i] = SVI(a, b, rho, m, sigma)
    # for k in Implied_Volatility_without_extrapolation['implied_volatility_surface']:    
    for ind, i in enumerate([0,1,2,3,4,5,6,7,8,9,10,11,12,14,16,19,21,22,26,31]):
        set_param[i] = data_dict['implied_volatility_surface'][ind]['set_param_raw']
        x_grid[i] = (data_dict['implied_volatility_surface'][ind]['strikes'] / data_dict['implied_volatility_surface'][ind]['reference_forward'])

    for i in lister:
        x_grid[i] = x_grid[i-1]

    Implied_survace_w = [0] * number

    for ind, n in enumerate(set_param):
    #     x = np.log(np.linspace(x_grid[ind][0], x_grid[ind][-1], 100))
    #     
        x = np.log(np.linspace(0.8 - ind / 150,1.2 + ind / 150,100))
        x_grid[ind] = np.exp(x)
        w_SVI_raw = np.linspace(0,0,100)
        for cnd, c in enumerate(x):
            w_SVI_raw[cnd] = get_w_SVI_raw(n.a,n.b,n.rho,n.m,n.sigma, c) # compute total varience for given parameters
        w_SVI_total_variance = w_SVI_raw
        w_SVI_raw = np.sqrt(w_SVI_total_variance / T_ACT_365[ind]) 
        Implied_survace_w[ind] = w_SVI_raw  


    T_AC = []
    for i in T_ACT_365:
        T_AC.append([i] * 100)
    return T_AC, x_grid, Implied_survace_w

def lines_2D_SVI_for_interpolate(ticker, data_dict, expiry_date, n):
    fig = plt.figure(figsize=(13, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    k = data_dict["implied_volatility_surface"][expiry_date]
    T_ACT_365 = k['expiry_date_in_act365_year_fraction']
    grid = np.array(k['strikes'])/k['reference_forward']
    ax1.scatter(grid, k['mid_implied_volatilities'], s = 5, color='y')
    ax1.plot(grid, k['mid_implied_volatilities'], label= 'mid IV from BM', color='y', linewidth=0.7)
    
    ax1.scatter(grid, k['bid_implied_volatilities'], s = 5, color='g')
    ax1.plot(grid, k['bid_implied_volatilities'], label= 'bid IV from BM', color='g', linewidth=0.7)
    
    ax1.scatter(grid, k['ask_implied_volatilities'], s = 5, color='r')
    ax1.plot(grid, k['ask_implied_volatilities'], label= 'ask IV from BM', color='r', linewidth=0.7)
    
    w_volatility = np.linspace(0,0,len(grid))
    for cnd, c in enumerate(np.log(grid)):
        w_volatility[cnd] = get_w_SVI_raw(n.a,n.b,n.rho,n.m,n.sigma, c) # compute total varience for given parameters
    w_SVI_total_variance = w_volatility
    w_volatility = np.sqrt(w_SVI_total_variance / T_ACT_365) 
    
    ax1.scatter(grid, w_volatility, s = 5, color='b')
    ax1.plot(grid, w_volatility, label= 'mid IV from SVI extrapolate', color='b', linewidth=1.5)
    
    relative_error = np.linspace(0,0,len(w_volatility))
    for i in range(0, len(relative_error)):
        if (w_volatility[i] <= k['ask_implied_volatilities'][i]) & (w_volatility[i] >= k['bid_implied_volatilities'][i]):
            relative_error[i] = 0
        elif (w_volatility[i] > k['ask_implied_volatilities'][i]):
            relative_error[i] = (w_volatility[i] - k['ask_implied_volatilities'][i])
        else:
            relative_error[i] = (k['bid_implied_volatilities'][i] - w_volatility[i])
        
    ax2.plot(grid, relative_error, label= 'error', color='grey', linewidth=3, linestyle='--')

    ax1.set_title(ticker + ' for expiration on '+ k['expiry_date'])
    ax1.set_xlabel(r'strike (fwd moneyness)')
    ax1.set_ylabel(r'Implied Volatility')
    ax1.legend()
    ax1.grid()
    ax2.set_title(ticker + ' for expiration on '+ k['expiry_date'] +' error graph')
    ax2.set_xlabel(r'strike (fwd moneyness)')
    ax2.set_ylabel(r'Implied Volatility')
    ax2.legend()
    ax2.grid()
    plt.show()


def lines_2D_SVI_example(ticker, data_dict, expiry_date, w_volatility):
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

    ax1.scatter(grid, w_volatility, s = 5, color='b')
    ax1.plot(grid, w_volatility, label= 'mid IV from SVI extrapolate', color='b', linewidth=1.5)
    
    relative_error = np.linspace(0,0,len(w_volatility))
    for i in range(0, len(relative_error)):
        if (w_volatility[i] <= k['ask_implied_volatilities'][i]) & (w_volatility[i] >= k['bid_implied_volatilities'][i]):
            relative_error[i] = 0
        elif (w_volatility[i] > k['ask_implied_volatilities'][i]):
            relative_error[i] = (w_volatility[i] - k['ask_implied_volatilities'][i])
        else:
            relative_error[i] = (k['bid_implied_volatilities'][i] - w_volatility[i])
        
    ax2.plot(grid, relative_error, label= 'error', color='grey', linewidth=3, linestyle='--')

    ax1.set_title(ticker + ' for expiration on '+ k['expiry_date'])
    ax1.set_xlabel(r'strike (fwd moneyness)')
    ax1.set_ylabel(r'Implied Volatility')
    ax1.legend()
    ax1.grid()
    ax2.set_title(ticker + ' for expiration on '+ k['expiry_date'] +' error graph')
    ax2.set_xlabel(r'strike (fwd moneyness)')
    ax2.set_ylabel(r'Implied Volatility')
    ax2.legend()
    ax2.grid()
    plt.show()