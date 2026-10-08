import numpy as np
from pathlib import Path
import logging
import time

import matplotlib.pyplot as plt

from pkgs.calibration.calibrator import calibrator
from pkgs.calibration.validation import plot_iv_vs_log_moneyness

def format_time(seconds):
    """Convert seconds (float) → h, m, s.ssssss"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60      # keep fractional part
    return f"{h} hours {m} minutes {s:.6f} seconds"

def initial_guess() -> list :
    """Initial guess [sigma, lam, p, eta1, eta2] for the optimizer"""

    sigma = 0.15
    lam = 1.5
    p = 0.3
    eta1 = 25.0
    eta2 = 10.0

    return [sigma,lam,p,eta1,eta2]

def calibrate(ticker_symbol : str, r : np.float64, fix_lam_zero : bool = False) :

    print("Ticker symbol is ",ticker_symbol)
    print("Using risk-free interest rate r as ",r)
    if fix_lam_zero :
        print("lam fixed to 0 : calibrating against Black-Scholes")
    kou_calibrator = calibrator(ticker_symbol,r)

    parameters = kou_calibrator.calibrate(initial_guess(), fix_lam_zero = fix_lam_zero)

    if fix_lam_zero :
        print("Black-Scholes parameters are as follows :")
        print("sigma (volatility) is ",parameters[0])
        return

    print("Kou parameters are as follows :")
    print("sigma (volatility) is ",parameters[0])
    print("lamda (average frequency of extreme events) is  ",parameters[1])
    print(" p ( probability of extreme event being upward jump is )",parameters[2])
    print("eta_1 (parameter characterizing upward jump distribution) is ",parameters[3])
    print("eta_2 (parameter characterizing downward jump distribution) is ",parameters[4])

def validate(ticker_symbol : str, r : np.float64, test_every : int = 5) :
    """
    Holds out every test_every-th expiry, calibrates both Kou and Black-Scholes (lam = 0) on the remaining expiries, and plots the 
    predicted implied volatility against the market implied volatility vs log-moneyness for the held out expiries (one subplot per model,
    in a single figure). The figure is saved to the Figures folder.
    """
    print("Ticker symbol is ",ticker_symbol)
    print("Using risk-free interest rate r as ",r)
    kou_calibrator = calibrator(ticker_symbol,r)
    kou_calibrator.split_by_expiry(test_every = test_every)

    models = {
        "Kou" : kou_calibrator.calibrate(initial_guess(), fix_lam_zero = False),
        "Black-Scholes" : kou_calibrator.calibrate(initial_guess(), fix_lam_zero = True)
    }

    output_dir = Path(__file__).parent.parent / 'Figures'

    results = {model_name : kou_calibrator.evaluate(parameters) for model_name, parameters in models.items()}

    #------Compare the models on the same options : those where the market and every model have an implied volatility------#
    common = np.logical_and.reduce([df['market_iv'].notna() & df['model_iv'].notna() for df in results.values()])
    print(f"RMS implied volatility error on held out expiries, over the {common.sum()} of {len(common)} options where every model has an implied volatility :")

    for model_name, parameters in models.items() :
        model_results = results[model_name]
        rms_iv_error = np.sqrt(np.mean((model_results['model_iv'][common] - model_results['market_iv'][common])**2))
        print(f"   {model_name} : {rms_iv_error:.4f}   (parameters {np.round(parameters, 4)})")

    output_path = output_dir / f"IV_validation_{ticker_symbol}.png"
    plot_iv_vs_log_moneyness(results, f"{ticker_symbol} : Prediction vs Market", save_path = output_path)
    print(f"Saved plot to {output_path}")

    plt.show()


if __name__ == "__main__" :
    start = time.perf_counter()
    ticker_symbol = "SPY"
    r = 0.05
    fix_lam_zero = False # Set to True to fix lam = 0 (no jumps), i.e. calibrate against Black-Scholes
    validate_on_held_out_expiries = True # Set to True to calibrate both Kou and Black-Scholes on 80% of expiries and plot predictions for the rest

    if validate_on_held_out_expiries :
        validate(ticker_symbol,r)
    else :
        calibrate(ticker_symbol,r,fix_lam_zero)
    end = time.perf_counter()
    
    print("Time taken to calibrate is ",format_time(end-start))


