import numpy as np
from pathlib import Path
import logging

from pkgs.monte_carlo_simulators import monte_carlo

def main(S_0 : np.float64,T : np.float64, strikes : np.ndarray, n_sims : int,kou_params : dict)  :
    """
    S_0 : spot price
    T : Time to expiry
    strikes : list of strike prices you wish to compute option prices for

    kou_params : Dictionary containing list of parameters defining kou process

    Given a specific S_0 and T, computes price options for a range of different strike prices by running monte carlo simulations and computing the mean

    """
    strikes = np.atleast_1d(strikes)

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

    T = 0.5

    strikes = np.linspace(50,200,50)
    n_sims = 10**6
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

    main(S_0, T, strikes,n_sims,kou_params)

    print("Done calculating price options via monte carlo")

