import numpy as np
from pathlib import Path
import logging
import time

from pkgs.calibration.calibrator import calibrator
def format_time(seconds):
    """Convert seconds (float) → h, m, s.ssssss"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60      # keep fractional part
    return f"{h} hours {m} minutes {s:.6f} seconds"

def main(ticker_symbol : str, r : np.float64 ) :

    print("Ticker symbol is ",ticker_symbol)
    print("Using risk-free interest rate r as ",r)
    kou_calibrator = calibrator(ticker_symbol,r)

    #Choosing initial guess for parameters

    sigma = 0.15
    lam = 1.5
    p = 0.3
    eta1 = 25.0
    eta2 = 10.0
    
    initial_guess = [sigma,lam,p,eta1,eta2]

    parameters = kou_calibrator.calibrate(initial_guess)

    print("Kou parameters are as follows :")
    print("sigma (volatility) is ",parameters[0])
    print("lamda (average frequency of extreme events) is  ",parameters[1])
    print(" p ( probability of extreme event being upward jump is )",parameters[2])
    print("eta_1 (parameter characterizing upward jump distribution) is ",parameters[3])
    print("eta_2 (parameter characterizing downward jump distribution) is ",parameters[4])


if __name__ == "__main__" :
    start = time.perf_counter()
    ticker_symbol = "AAPL"
    r = 0.05
    main(ticker_symbol,r)
    end = time.perf_counter()
    
    print("Time taken to calibrate is ",format_time(end-start))


