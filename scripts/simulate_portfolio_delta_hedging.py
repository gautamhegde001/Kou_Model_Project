import numpy as np
from pathlib import Path
import logging

from pkgs.monte_carlo_simulators import monte_carlo_hedging


def main(S_0 : np.float64, K: np.float64, T : np.float64, n_sims : int, n_steps : int, kou_params : dict)  :
    """
    S_0 : spot price
    K : strike price
    T : Time to expiry


    kou_params : Dictionary containing list of parameters defining kou process

    Given a specific S_0 and T, computes price options for a range of different strike prices by running monte carlo simulations and computing the mean

    """

    simulated_PnL = monte_carlo_hedging(n_sims,n_steps,S_0,K,T,kou_params)


    
    #-----------------Saving strikes vs prices as a N x 2 array -----------------------------
    
    # We will be saving the data to the datafiles folder

    output_dir = Path(__file__).parent.parent / 'datafiles'
    lam = kou_params['lam']
    output_filename = f'simulated_PnL_delta_hedging_lam={lam}.npy'
    output_path = output_dir/output_filename

    np.save(output_path,simulated_PnL)

if __name__ == "__main__" : # This ensures that the main function is run only if this script is being run

    S_0 = 100.0
    K = 80
    T = 1.0 # Represents one working year

    n_sims = 10**5

    n_steps = 252*2 #Hedge once every working day



    # ---------- Choosing kou parameters ------------------
    r = 0.05
    sigma = 0.15
    lam = 6.0
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

    print(f"Calculating PnL of delta hedged portfolio with 1 stock. Assuming kou model evolution with lamda {lam}")

    main(S_0,K,T,n_sims,n_steps,kou_params)

    print(f"Delta Hedged PnL calculated for {n_sims} simulations")
