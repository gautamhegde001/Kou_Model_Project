import numpy as np
from .carr_madan_function_aux import carr_madan_function
from scipy.interpolate import RegularGridInterpolator

import logging

def delta_surface_fft(T: np.float64, kou_params: dict, N=4096, d_v=0.1, alpha=0.75):
    """
    T : Time to expiry

    kou_params : Dictionary containing the list of parameters defining a kou process

    N : size of grid for FFT

    d_v : Grid spacing in frequency space

    alpha : Damping factor that makes the function fourier transformable

    Returns :
        
        x_grid : grid of log-moneyness across which deltas have been computer

        deltas : list of deltas calculated gives the time to expiry T and kou parameters as defined in kou_params

    Calculates Delta across a grid of log-moneyness (x) for a given T using the FFT method. 
    """
    # 1. Grids
    v_grid = np.arange(N) * d_v
    d_x = (2 * np.pi) / (N * d_v)
    x_grid = - (N * d_x) / 2 + np.arange(N) * d_x
    
    # 2. Get the integrand (Returns E[exp(iuX_T)])
    psi_v = carr_madan_function(v_grid, alpha, T, kou_params)
    
    # 3. Modify for Delta
    delta_integrand = psi_v * (1 + alpha + 1j * v_grid)
    
    # 4. Apply weights and phase shift
    weights = np.ones(N)
    weights[0] = 0.5
    weights[-1] = 0.5
    
    x_min = x_grid[0]
    fft_input = np.exp(-1j * v_grid * x_min) * delta_integrand * weights * d_v
    
    # 5. Execute FFT
    fft_output = np.fft.fft(fft_input)
    
    # 6. Post-process
    deltas = (np.exp(-alpha * x_grid) / np.pi) * np.real(fft_output)
    
    # 7. Mask (Keep reasonable log-moneyness range, e.g., -0.7 to 0.7)
    mask = (x_grid > -0.7) & (x_grid < 0.7)
    
    # Return the log-moneyness grid and the corresponding Deltas
    return x_grid[mask], deltas[mask]

def delta_interpolator_maker(T_values : np.ndarray,kou_params : dict) :

    """
    T_values : 1D array consisting of T_values which we find deltas for. 
    kou_params : dictionary of parameters defining a standard kou process

    Returns : 
    delta_interpolator 
    
    For a grid of T_values, computes deltas for a range of different strike prices by using the inverse fourier transform method
    This gives a surface of delta values which can be used to interpolate between and create a delta function for any given value 
    of log-moneyness k and time to expiry T
    (Carr and Madan 1999)
    """

    delta_surface = []
    x_values_container = None

    for T in T_values :
        x_values,prices = delta_surface_fft(T,kou_params)
        delta_surface.append(prices)

        if x_values_container is None : # Need to construct this array only once, hence this condition
            x_values_container = x_values

    delta_matrix = np.array(delta_surface)
    delta_interpolator = RegularGridInterpolator((T_values,x_values_container),delta_matrix,bounds_error=False,fill_value=None)

    return delta_interpolator   