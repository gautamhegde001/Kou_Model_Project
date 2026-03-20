import numpy as np
from pathlib import Path
import logging

from pkgs.Option_pricing_and_deltas.price_options_via_fft import price_options_fft

def main(S_0 : np.float64,T : np.float64, kou_params : dict)  :
    """
    Given a specific S_0 and T, computes price options for a range of different strike prices by using the inverse fourier transform method
    (Carr and Madan 1999)
    """

    strikes,prices = price_options_fft(S_0,T,kou_params)

    savedata = (strikes,prices)


    #-----------------Saving strikes vs prices as a N x 2 array -----------------------------

    # We will be saving the data to the datafiles folder

    output_dir = Path(__file__).parent.parent / 'datafiles'
    output_filename = f'strikes_vs_prices_kou_fft_T={T}.npy'
    output_path = output_dir/output_filename

    np.save(output_path,savedata)

if __name__ == "__main__" : # This ensures that the main function is run only if this script is being run

    S_0 = 100.0

    T = 0.5

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

    main(S_0, T, kou_params)

    print("Done calculating price options via fft")



