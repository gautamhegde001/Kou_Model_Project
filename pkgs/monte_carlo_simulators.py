import numpy as np
from pkgs.Option_pricing_and_deltas.delta_calculator import delta_interpolator_maker

def monte_carlo(n_sims : int, n_steps: int, S_0 : np.float64, T : np.float64, kou_params : dict) :
    """
    Parameters :

        n_sims : Number of monte carlo simulation
    
        n_steps : Number of time steps in each monte carlo run
    
        S_0 : Spot price (Initial stock value)

        T : Time to expiry

        kou_params : A dictionary containing all the parameters defining the kou process

    Returns : 
        
        stock_simulated  : np.ndarray
            A 2D array. Each row represents one simulation, with each element of the row being the stock value at a given time-step. 
        
    Runs monte-carlo simulations with n_simulations, and n_timesteps, assuming initial spot prince S_0, and time to expiry T. T is in units of years. 
    """

    r = kou_params['r'] # risk-free interest rate
    sigma = kou_params['sigma'] # volatility of brownian motioni

    lam = kou_params['lam'] # rate of 'extreme' events per unit time, goes into poisson distribution
    eta1 = kou_params['eta1'] # exponential parameter determining strength of upward jump
    eta2 = kou_params['eta2'] # exponential parameter determining strength of downward jump 

    p = kou_params['p'] # probability of jump being upwards


    dt = T/n_steps #size of each time-step

    # Check for stability condition
    if eta1 <= 1:
        raise ValueError("eta1 must be greater than 1 for the expectation to exist.")

    #------------- Simulations -----------------------------------------------------#
    #Evaluating Brownian Noise terms 
    noise = np.random.normal(0, 1, size=(n_sims, n_steps))

    # Constructing Jump Compensator
    kappa = p * (eta1/(eta1 - 1)) + (1-p) * (eta2/(eta2 + 1)) - 1

    # Simulating Jumps (Vectorized for speed)
    N_t = np.random.poisson(lam * dt, size=(n_sims, n_steps))
    jump_log_returns = np.zeros((n_sims, n_steps))

    # We only iterate over cells where jumps actually occurred to save time. Else the code might take longer than the heat death of the universe.
    # Since the number of timesteps where jumps occured is a small fraction, we don't vectorize this as that would unneccesarily complicate the code. 
    # For very large lamda, this might have to be vectorized 
    rows, cols = np.where(N_t > 0)
    for r_idx, c_idx in zip(rows, cols):
        num_jumps = N_t[r_idx, c_idx]
        # Decide direction for each jump in this cell
        u = np.random.uniform(0, 1, size=num_jumps)
        pos_jumps = np.random.exponential(1/eta1, size=np.sum(u <= p))
        neg_jumps = -np.random.exponential(1/eta2, size=np.sum(u > p))
        jump_log_returns[r_idx, c_idx] = np.sum(pos_jumps) + np.sum(neg_jumps)

    # Combine all of the above to get Log Returns
    # drift = (r - 0.5 * sigma**2 - lambda * kappa)
    drift = (r - 0.5 * sigma**2 - lam * kappa) * dt
    diffusion = sigma * np.sqrt(dt) * noise
    log_returns = drift + diffusion + jump_log_returns

    # Path Construction
    cum_log_returns = np.cumsum(log_returns, axis=1)
    # Add the starting point (0 log return at t=0)
    padded_cum_log_returns = np.hstack((np.zeros((n_sims, 1)), cum_log_returns))
    stocks_simulated = S_0 * np.exp(padded_cum_log_returns)

    return stocks_simulated

import numpy as np

import numpy as np

def monte_carlo_hedging(n_sims : int, n_steps: int, S_0 : np.float64, K : np.float64, T : np.float64, kou_params : dict) :
    """
    Parameters :

        n_sims : Number of monte carlo simulation
    
        n_steps : Number of time steps in each monte carlo run
    
        S_0 : Spot price (Initial stock value)

        K : Strike Price

        T : Time to expiry

        kou_params : A dictionary containing all the parameters defining the kou process

    Returns : 
        
        stock_simulated  : np.ndarray
            A 1D array, with each element being the final value of the portfolio at the end of simulation at time T.
        
    Runs monte-carlo simulations with n_simulations, and n_timesteps. This code simulated the time evolution of a portfolio which contains
    exactly one stock option, with spot price S_0 and strike price K, with time to expiry given by T. Delta hedging is performed at every
    time step, such that the portfolio shorts stock equal to the delta of the stock option at any given time step. Output is the portfolio value
    of each simulation at each time step.  
    """

    r = kou_params['r'] # risk-free interest rate
    sigma = kou_params['sigma'] # volatility of brownian motion

    lam = kou_params['lam'] # rate of 'extreme' events per unit time, goes into poisson distribution
    eta1 = kou_params['eta1'] # exponential parameter determining strength of upward jump
    eta2 = kou_params['eta2'] # exponential parameter determining strength of downward jump 

    p = kou_params['p'] # probability of jump being upwards


    dt = T/n_steps #size of each time-step

    # Check for stability condition
    if eta1 <= 1:
        raise ValueError("eta1 must be greater than 1 for the expectation to exist.")

    #------------- Simulations -----------------------------------------------------#
    # Brownian Noise 
    noise = np.random.normal(0, 1, size=(n_sims, n_steps))

    # Jump Compensator
    kappa = p * (eta1/(eta1 - 1)) + (1-p) * (eta2/(eta2 + 1)) - 1

    # Simulating Jumps (Vectorized for speed)
    N_t = np.random.poisson(lam * dt, size=(n_sims, n_steps))
    jump_log_returns = np.zeros((n_sims, n_steps))

    # We only iterate over cells where jumps actually occurred
    rows, cols = np.where(N_t > 0)
    for r_idx, c_idx in zip(rows, cols):
        num_jumps = N_t[r_idx, c_idx]
        # Decide direction for each jump in this cell
        u = np.random.uniform(0, 1, size=num_jumps)
        pos_jumps = np.random.exponential(1/eta1, size=np.sum(u <= p))
        neg_jumps = -np.random.exponential(1/eta2, size=np.sum(u > p))
        jump_log_returns[r_idx, c_idx] = np.sum(pos_jumps) + np.sum(neg_jumps)

    # Combine to get Log Returns
    drift = (r - 0.5 * sigma**2 - lam * kappa) * dt
    diffusion = sigma * np.sqrt(dt) * noise
    log_returns = drift + diffusion + jump_log_returns

    # Path Construction
    cum_log_returns = np.cumsum(log_returns, axis=1)
    # Add the starting point (0 log return at t=0)
    padded_cum_log_returns = np.hstack((np.zeros((n_sims, 1)), cum_log_returns))
    stocks_simulated = S_0 * np.exp(padded_cum_log_returns)


    #------------- Delta Hedging Logic------------------#
    delta_interpolator = delta_interpolator_maker(np.linspace(1e-6,T,100),kou_params)
    #We construct a surface of Delta as a function of expiry time T and log-moneyness x=log(K/S_0) using the FFT method, and interpolate
    #from this surface to get the delta for any value of K,S_0,T as desired.
    def get_delta(S_array : np.ndarray , tau : np.float64):
        """
        S_array : array of spot prices for which you wish to calculate delta

        tau : Time to expiry used in calculating delta

        Returns :
        
        deltas : np.ndarray
            1D array containing deltas for the spot prices S_0, while assuming K as defined in the original argument passed to monte carlo

        """
        

        log_moneyness_array = np.log(S_array/K)

        query_points = np.c_[np.full_like(log_moneyness_array, tau), log_moneyness_array]

        deltas = delta_interpolator(query_points)
        return deltas

    # We will track the cumulative profit/loss from holding the Delta stock position
    trading_pnl = np.zeros(n_sims)

    # Note: We iterate n_steps times. i goes from 0 to n_steps-1
    for i in range(n_steps):
        S_current = stocks_simulated[:, i]
        S_next = stocks_simulated[:, i+1]
        tau = T - (i * dt)
        
        # Calculate Delta at the beginning of the step
        current_delta = get_delta(S_current, tau)
        
        # Calculate the profit/loss of holding 'Delta' shares over this time step.
        # The cost to hold the stock is the risk-free rate. 
        # So we subtract S_current * e^(r*dt) from the new stock price.
        step_pnl = current_delta * (S_next - S_current * np.exp(r * dt))
        
        # Accumulate PnL, compounding the previous balance by the risk-free rate
        trading_pnl = trading_pnl * np.exp(r * dt) + step_pnl
        
    # --- AT EXPIRY (Time T) ---
    S_final = stocks_simulated[:, -1]
    
    #  The Call Option Payoff at expiry
    option_payoff = np.maximum(S_final - K, 0)
    
    #  The Portfolio Value at T (Payoff minus the trading gains)
    portfolio_value_T = option_payoff - trading_pnl

    return portfolio_value_T