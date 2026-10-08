import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

"""This file contains functions used to validate a calibration on the held out (test) expiries : Black-Scholes pricing,
implied volatility inversion, and the plot of implied volatility vs log-moneyness.
"""

def bs_call_price(S0_array : np.ndarray, K_array : np.ndarray, T_array : np.ndarray, r : float, sigma_array : np.ndarray) -> np.ndarray :
    """
    S0_array : Array of spot prices

    K_array : Array of strike prices

    T_array : Array of maturities

    r : risk-free interest rate

    sigma_array : Array of volatilities

    Returns :
        prices : 1D array containing Black-Scholes prices of call options
    """
    sqrt_T = np.sqrt(T_array)
    d1 = (np.log(S0_array / K_array) + (r + 0.5 * sigma_array**2) * T_array) / (sigma_array * sqrt_T)
    d2 = d1 - sigma_array * sqrt_T

    return S0_array * norm.cdf(d1) - K_array * np.exp(-r * T_array) * norm.cdf(d2)

def implied_volatility(prices : np.ndarray, S0_array : np.ndarray, K_array : np.ndarray, T_array : np.ndarray, r : float,
                       sigma_low = 1e-4, sigma_high = 5.0, n_iter = 100, min_time_value = 0.005) -> np.ndarray :
    """
    prices : Array of call option prices to invert

    S0_array, K_array, T_array : Arrays of spot prices, strike prices and maturities

    r : risk-free interest rate

    min_time_value : Prices within min_time_value of the no-arbitrage lower bound max(S0 - K e^{-rT}, 0) get NaN. Option prices are quoted in
                     $0.01 ticks, so less than half a tick of time value carries no information about volatility (e.g. a model price of
                     1e-15 for a far out of the money option gives an implied volatility determined by numerical noise).

    Returns :
        iv : 1D array containing the Black-Scholes implied volatilities. NaN where the price is outside the no-arbitrage
             bounds max(S0 - K e^{-rT}, 0) < C < S0 (e.g. deep in the money calls quoted below intrinsic value), since no volatility
             reproduces such a price, or where the time value is below min_time_value.

    Inverts the Black-Scholes formula by bisection, vectorized over all options at once. Bisection is used instead of Newton's method
    because it cannot diverge for options with small vega (deep in/out of the money).
    """
    prices, S0_array, K_array, T_array = np.broadcast_arrays(*map(np.atleast_1d, (prices, S0_array, K_array, T_array)))

    low = np.full(prices.shape, sigma_low)
    high = np.full(prices.shape, sigma_high)

    # Call prices increase with sigma, so the root is bracketed by [low, high]. 100 halvings of [1e-4, 5] is well below machine precision
    for _ in range(n_iter) :
        mid = 0.5 * (low + high)
        too_low = bs_call_price(S0_array, K_array, T_array, r, mid) < prices
        low = np.where(too_low, mid, low)
        high = np.where(too_low, high, mid)

    iv = 0.5 * (low + high)

    #------ Mask prices that no volatility in [sigma_low, sigma_high] can reproduce------#
    lower_bound = bs_call_price(S0_array, K_array, T_array, r, np.full(prices.shape, sigma_low))
    upper_bound = bs_call_price(S0_array, K_array, T_array, r, np.full(prices.shape, sigma_high))
    iv[(prices <= lower_bound + min_time_value) | (prices >= upper_bound)] = np.nan

    return iv

def plot_iv_vs_log_moneyness(results_by_model : dict, title : str, save_path = None) :
    """
    results_by_model : Dictionary {model name : Dataframe returned by calibrator.evaluate()}, each dataframe with columns T, log_moneyness,
                       market_iv, model_iv

    title : Title of the figure

    save_path : If given, the figure is saved to this path

    Returns :
        fig : matplotlib figure

    Plots implied volatility vs log-moneyness for the held out expiries, with one subplot per model stacked vertically. Market implied volatilities
    (from the mid prices) are plotted as points and the model's predicted implied volatilities as a dashed line of the same color.
    """
    fig, axes = plt.subplots(len(results_by_model), 1, figsize=(12, 6*len(results_by_model)), sharex=True, sharey=True, squeeze=False)

    for ax, (model_name, results) in zip(axes[:, 0], results_by_model.items()) :

        expiries = np.sort(results['T'].unique())
        expiries = np.array([expiries[0]])
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(expiries)))

        for T, color in zip(expiries, colors) :
            expiry_df = results[results['T'] == T].sort_values('log_moneyness')
            label = f"{T*365:.0f} days"

            market = expiry_df.dropna(subset=['market_iv'])
            model = expiry_df.dropna(subset=['model_iv'])

            ax.scatter(market['log_moneyness'], market['market_iv'], color=color, s=12, label=f"{label} (market mid)")
            ax.plot(model['log_moneyness'], model['model_iv'], color=color, linestyle='--', linewidth=1.5, label=f"{label} ({model_name} prediction)")

        ax.set_ylabel("Implied volatility",fontsize=22)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=14)
        ax.tick_params(labelsize=14)

    axes[-1][0].set_xlabel("Log-moneyness  k = ln(K / S0)",fontsize=22)
    fig.suptitle(title,fontsize=24)
    fig.tight_layout()

    if save_path is not None :
        fig.savefig(save_path, dpi=150)

    return fig
