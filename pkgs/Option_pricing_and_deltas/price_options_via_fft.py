import numpy as np
from .carr_madan_function_aux import carr_madan_function

import logging

import numpy as np

def price_options_fft(S_0 : np.float64, T : np.float64, kou_params : dict, N=8192*4, d_v=0.01, alpha=0.75):
    """
    S_0 : Spot price, (stock value at initial time)

    T : Time to expiry

    kou_params : dictionary containing standard parameters defining kou process

    dv : spacing of frequency grid in fft

    alpha : damping factor used to make the carr-madan function square integrable, and hence amenable to fourier transforms. 

    Returns :
        strikes : np.ndarray 
            1D array of strikes for which (call) option prices have been calculated
        prices : np.ndarray
            1d array of (call) option prices

    This function calculates prices for call options via the fourier transform method of Carr&Madan(1999), for the kou process. 

    The theory behind this procedure is explained in the "Project_Report.pdf" file. 

    """
    # Defining Frequency grid (v) 
    v_grid = np.arange(N) * d_v
    
    # Defining the Log-moneyness grid (x)
    d_x = (2 * np.pi) / (N * d_v)
    x_grid = - (N * d_x) / 2 + np.arange(N) * d_x
    
    # Evaluating the Carr-Madan function
    psi_v = carr_madan_function(v_grid, alpha, T, kou_params)
    
    # FFT with Simpson's Rule weights
    weights = (3 + (-1)**(np.arange(N) + 1)) / 3.0
    weights[0] = 1.0 / 3.0
    
    # Shift input by x_min for taking FFT
    x_min = x_grid[0]
    fft_input = np.exp(-1j * v_grid * x_min) * psi_v * weights * d_v
    
    # Execute FFT
    fft_output = np.fft.fft(fft_input)
    
    # Price calculation
    call_prices = S_0 * (np.exp(-alpha * x_grid) / np.pi) * np.real(fft_output)
    strikes = S_0 * np.exp(x_grid)
    
    # Mask to keep relevant strikes
    mask = (x_grid > -0.7) & (x_grid < 0.7)
    
    return strikes[mask], call_prices[mask]

