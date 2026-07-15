import numpy as np
from pathlib import Path
import logging
import time

from pkgs.monte_carlo_simulators import monte_carlo
from pkgs import _kou_mc

def format_time(seconds):
    """Convert seconds (float) → h, m, s.ssssss"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60      # keep fractional part
    return f"{h} hours {m} minutes {s:.6f} seconds"

def main(S_0 : np.float64,T : np.float64, strikes : np.ndarray, n_sims : int,kou_params : dict, backend : str)  :
    """
    S_0 : spot price
    T : Time to expiry
    strikes : list of strike prices you wish to compute option prices for

    kou_params : Dictionary containing list of parameters defining kou process

    Given a specific S_0 and T, computes price options for a range of different strike prices by running monte carlo simulations and computing the mean

    """

    strikes = np.atleast_1d(strikes)

    if backend == "c++" : # Performs the monte carlo in C++ using pybind
        print("Using C++ to perform monte Carlo")
        #---Unpacking the kou_params dictionary----------

        r = kou_params['r']
        sigma = kou_params['sigma']
        lam = kou_params['lam']
        p = kou_params['p']
        eta1 = kou_params['eta1']
        eta2 = kou_params['eta2']
        
        
        
        prices = _kou_mc.calculate_price_monte_carlo(
        n_samples=n_sims, S_0=S_0, T=T, strikes=strikes, r=r,
        sigma=sigma, lam=lam, eta1=eta1, eta2=eta2, p=p, seed=12345)

    else : # Performs the monte carlo in python in a vectorized fashion

        stock_simulated = monte_carlo(n_sims,1,S_0,T,kou_params)

        stock_final = stock_simulated[:,-1]

        difference = stock_final[:,None] - strikes[None,:]

        difference = np.maximum(difference,0) #Take only the maximum of (S(t)-K,0)+

        r = kou_params['r']
        prices = np.exp(-r*T)*np.mean(difference,axis = 0) # Have to back-date expected pay-off to value at initial time

    savedata = (strikes,prices)
    
    #-----------------Saving strikes vs prices as a N x 2 array -----------------------------
    
    # We will be saving the data to the datafiles folder

    output_dir = Path(__file__).parent.parent / 'datafiles'
    output_filename = f'strikes_vs_prices_kou_monte_carlo_T={T}.npy'
    output_path = output_dir/output_filename

    np.save(output_path,savedata)
    
if __name__ == "__main__" : # This ensures that the main function is run only if this script is being run

    S_0 = 100.0

    T = 2.0

    strikes = np.linspace(50,200,50)
    n_sims = 10**6
    backend = "c++" # Set to "c++" for using C++; set to anything else if you wish to use python (vectorized) for MC instead

    # ---------- Choosing kou parameters ------------------
    r = 0.05
    sigma = 0.15
    lam = 1.5
    p = 0.3
    eta1 = 25.0
    eta2 = 10.0

    kou_params = {

        "r" : r,
        "sigma" : sigma,
        "lam" : lam,
        "p" : p,
        "eta1" : eta1,
        "eta2" : eta2

    }

    print("Calc price options via monte carlo")

    start = time.perf_counter()

    main(S_0, T, strikes,n_sims,kou_params,backend)

    end = time.perf_counter()

    print("Done calculating price options via monte carlo in ",format_time(end-start))

