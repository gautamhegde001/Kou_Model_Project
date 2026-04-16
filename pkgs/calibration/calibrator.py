import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from ..Option_pricing_and_deltas.pricer_arbitrary_strikes import KouPricer
from datetime import datetime


class calibrator():
    def __init__(self, ticker_symbol: str, risk_free_interest: float):
        self.ticker_symbol = ticker_symbol
        self.ticker = yf.Ticker(ticker_symbol) # Initialize yfinance ticker
        self.risk_free_interest = risk_free_interest
        
        # Consistent naming for arrays
        self.S0_array = None
        self.T_array = None
        self.K_array = None
        self.market_prices = None

    def fetch_and_clean_data(self):
        """
        Pulls option chains from Yahoo Finance, cleans out illiquid strikes,
        calculates time to maturity (T), and prepares arrays for calibration.
        """
        # 1. Get current spot price
        self.S0 = self.ticker.history(period="1d")['Close'].iloc[-1]
        
        all_options_data = []
        today = datetime.now()

        # 2. Get available expirations
        expirations = self.ticker.options

        # ONLY FOR TESTING! Use this to check that the code runs by using only three expiry dates to run faster
        #expirations = expirations[0:3]

        for exp_date in expirations:
            # 3. Calculate Time to Maturity (T)
            dt_expiry = datetime.strptime(exp_date, '%Y-%m-%d')
            t_days = (dt_expiry - today).days
            
            if t_days <= 0:
                continue
            
            T = t_days / 365.0
            
            # 4. Fetch the option chain
            chain = self.ticker.option_chain(exp_date)
            calls = chain.calls
            
            # 5. Clean the data
            calls = calls[(calls['bid'] > 0) & (calls['ask'] > 0)]
            calls = calls[(calls['volume'] > 0) | (calls['openInterest'] > 0)]
            
            # Calculate Mid-Price
            calls['mid_price'] = (calls['bid'] + calls['ask']) / 2.0
            
            calls['T'] = T
            calls['S0'] = self.S0
            
            all_options_data.append(calls[['S0', 'strike', 'T', 'mid_price']])

        # 6. Consolidate into final numpy arrays
        full_df = pd.concat(all_options_data)
        
        # Filter by log-moneyness range [-0.7, 0.7]
        log_moneyness = np.log(full_df['strike'] / full_df['S0'])
        mask = (log_moneyness > -0.7) & (log_moneyness < 0.7)
        final_df = full_df[mask]

        self.S0_array = final_df['S0'].values
        self.K_array = final_df['strike'].values
        self.T_array = final_df['T'].values
        self.market_prices = final_df['mid_price'].values

        print(f"Successfully loaded {len(self.market_prices)} liquid call options.")
    
    def objective_function(self, params_list: list) -> float:
        """
        params_list : List containing the kou parameters defining a process. 

        Returns :
            mean_square_difference : float
                Calculates the "distance" between the prices taken from real life data (y-finance) and prices calculated using 
                optimizers current guess. The metric for this distance is the mean of the square of differences between
                individual prices. 
        """
    
        kou_params = {
            'r' : self.risk_free_interest,
            'sigma' : params_list[0],
            'lam': params_list[1],
            'p': params_list[2],
            'eta1': params_list[3],
            'eta2': params_list[4]
        }

        try:
            current_pricer = KouPricer(kou_params)

            calculated_prices = current_pricer.generate_prices(self.S0_array, self.K_array, self.T_array)
        except Exception as e:
            print(f"CRASH in KouPricer: {e}")
            return 1e6 # Assign a large penalty if the FFT fails to converge due to bad values of parameter choice.
        

        mse = np.mean((calculated_prices - self.market_prices)**2)
        return mse
    
    def calibrate(self, initial_guess: list):
        if self.market_prices is None:
            self.fetch_and_clean_data()
        
        bounds = (
            (0.04, 2.0),   # sigma
            (1e-4, 10.0),  # lambda
            (0.0, 1.0),    # p
            (1.01, 50.0),  # eta1
            (1e-4, 50.0)   # eta2
        )

        print("Starting optimization... this may take a few minutes.")

        result = minimize(
            fun=self.objective_function, 
            x0=initial_guess,
            method='L-BFGS-B',
            bounds=bounds,
            options={'disp': True, 'ftol': 1e-6}
        )

        print("Optimized Parameters:", result.x)
        return result.x