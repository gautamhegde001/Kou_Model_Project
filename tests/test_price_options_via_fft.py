import numpy as np
from pkgs.Option_pricing_and_deltas.price_options_via_fft import price_options_fft
import logging

def test_price_options_fft() :

    S_0 = 100.0

    T = 1.0

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

    strikes,prices = price_options_fft(S_0,T,kou_params)
    logger = logging.getLogger(__name__)
    logger.debug("Strikes are %s",strikes)
    logger.debug("Prices are %s",prices)


    assert len(strikes)==len(prices)
