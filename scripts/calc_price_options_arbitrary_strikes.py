import numpy as np
from pathlib import Path
import pandas as pd
import logging

from pkgs.Option_pricing_and_deltas.pricer_arbitrary_strikes import KouPricer

def main(S_0_array : np.ndarray, K_array : np.ndarray , T_array : np.ndarray, kou_params : dict)  :
    """
    Given a list of spot prices S_0, strike prices K and expiry times T  (all three as seperate arrays), computes price options for a 
    range of different strike prices by using the inverse fourier transform method (Carr and Madan 1999) and interpolation 

    """
   
    pricer = KouPricer(kou_params)

    price_options = pricer.generate_prices(S_0_array,K_array,T_array)

    df = pd.DataFrame({
        "S_0" : S_0_array,
        "K" : K_array,
        "T" : T_array,
        "C" : price_options

    })


    #-----------------Saving strikes vs prices as a N x 2 array -----------------------------

    # We will be saving the data to the datafiles folder

    output_dir = Path(__file__).parent.parent / 'datafiles'
    output_filename = "kou_prices_fft_vectorized.pkl"
    output_path = output_dir/output_filename

    df.to_pickle(output_path)

if __name__ == "__main__" : # This ensures that the main function is run only if this script is being run

    S_0_array = [100.0,100.0,100.0,100.0,100]
    K_array = [90,85,83,105,110]
    T_array = [1.0,1.0,1.0,1.0,1.0]



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

    print("Calc price options via fft")

    main(S_0_array, K_array, T_array, kou_params)

    print("Done calculating price options via fft")



