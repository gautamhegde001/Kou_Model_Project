import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from ..Option_pricing_and_deltas.pricer_arbitrary_strikes import KouPricer
from .validation import implied_volatility, bs_call_price
from datetime import datetime
from sqlalchemy import create_engine
from scipy.stats import norm

#-------Define auxillary function used to calculate vegas during calibration process------------#
def calculate_bs_vega(S0_array : np.ndarray , K_array : np.ndarray , T_array : np.ndarray , r : float , iv_array : np.ndarray) -> np.ndarray:
    """
    S0_array : Array of spot prices

    K_array : Array of strike prices

    T_array : Array of maturities

    r : risk-free interest rate

    iv_array : The market implied volatility for each option (e.g., from yfinance)

    Returns :
        vega : 1D array containing vegas
    
    Calculates the Black-Scholes Vega for an array of options.
    
    """
    # Protect against 0 DTE (days to expiration) or 0 IV causing division by zero
    T = np.maximum(T_array, 1e-5)
    iv = np.maximum(iv_array, 1e-5)
    
    # Calculate d1
    d1 = (np.log(S0_array / K_array) + (r + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
    
    # Calculate the PDF of d1
    pdf_d1 = norm.pdf(d1)
    
    # Calculate raw Vega
    vega = S0_array * np.sqrt(T) * pdf_d1
    
    return vega

class calibrator():
    def __init__(self, ticker_symbol: str, risk_free_interest: float):
        self.ticker_symbol = ticker_symbol
        self.ticker = yf.Ticker(ticker_symbol) # Initialize yfinance ticker
        self.risk_free_interest = risk_free_interest

        #creating SQL Engine for cleaning data
        self.engine = create_engine('sqlite:///:memory:')
        
        # Consistent naming for arrays
        self.S0_array = None
        self.T_array = None
        self.K_array = None
        self.market_prices = None

        # Held out (test) data, set by split_by_expiry(). None means all the data is used for calibration
        self.test_data = None

    def fetch_and_clean_data_SQL(self) :
        """
        Pull option chains from Yfinance, loads data into an in-memory SQL database, 
        cleans it using SQL queries (faster than cleaning it using pandas), and prepared calibration arrays
        """

        #-----Fetch current spot price------#
        self.S0 = self.ticker.history(period = "1d")['Close'].iloc[-1]

        #-----Fetch all available options and combine them into one dataframe------#
        expirations = self.ticker.options
        all_raw_calls = []

        print("Fetching data from yfinance....")

        for exp_date in expirations :
            chain = self.ticker.option_chain(exp_date)
            calls = chain.calls

            calls["expiration_date"] = exp_date
            all_raw_calls.append(calls)
        
        if not all_raw_calls:
            raise ValueError(f"No options data found for {self.ticker_symbol}.")
        
        #------Create pandas dataframe containing raw data about all options, BEFORE cleaning------#

        raw_options_df = pd.concat(all_raw_calls, ignore_index = True)

        #------Create SQL database  named "options_data" containing raw data about all options, from the above raw_options_df------#
        #------We save it to the RAM since it is temporary. If such a database exists in the ram, it is replaced-----------#

        raw_options_df.to_sql('options_data', self.engine, index=False, if_exists='replace')

        #----------Clean the data using SQL, and save to dataframe named clean_calls--------#
        #--------Cleaned data will only include calls with non-zero liquidity (volume or openinterest>0) and bid & ask >0 -----#
        query = """
            SELECT 
                expiration_date, 
                strike, 
                bid, 
                ask,
                impliedVolatility
            FROM options_data 
            WHERE bid > 0 
              AND ask > 0
              AND (volume > 0 OR openInterest > 0)
        """
        clean_calls = pd.read_sql(query, self.engine) # Contains the exp_date, strike, bid and ask for options

        #------Final Kou Model Preparations (Math & T calculation)----------#
        today = pd.to_datetime(datetime.now().date())
        clean_calls['expiration_date'] = pd.to_datetime(clean_calls['expiration_date'])
        
        #--------Calculate Time to Maturity (T)-----------#
        clean_calls['t_days'] = (clean_calls['expiration_date'] - today).dt.days
        clean_calls = clean_calls[clean_calls['t_days'] > 0].copy()
        clean_calls['T'] = clean_calls['t_days'] / 365.0
        
        #---------Calculate Mid-Price as estimate of actual price of call option-----------#
        clean_calls['mid_price'] = (clean_calls['bid'] + clean_calls['ask']) / 2.0
        clean_calls['S0'] = self.S0
        
        #----------Filter by log-moneyness, only include strikes in the range [-0.7, 0.7]---------#
        log_moneyness = np.log(clean_calls['strike'] / clean_calls['S0'])
        mask = (log_moneyness > -0.7) & (log_moneyness < 0.7)
        final_df = clean_calls[mask]

        #---------Consolidate into final numpy arrays-----------------#
        self.S0_array = final_df['S0'].values
        self.K_array = final_df['strike'].values
        self.T_array = final_df['T'].values
        self.market_prices = final_df['mid_price'].values
        self.market_ivs = final_df['impliedVolatility'].values
        self.market_vegas = calculate_bs_vega(
            self.S0_array, 
            self.K_array, 
            self.T_array, 
            self.risk_free_interest, 
            self.market_ivs
        )
        
        self.market_spreads = final_df['ask'].values - final_df['bid'].values

        print(f"Successfully loaded {len(self.market_prices)} liquid call options from in-memory DB.")


    def fetch_and_clean_data_legacy(self):
        """
        THIS IS LEGACY CODE! DOES NOT USE SQL FOR DATA CLEANING.

        Pulls option chains from Yahoo Finance, cleans out illiquid strikes,
        calculates time to maturity (T), and prepares arrays for calibration.
        """
        # 1. Get current spot price
        self.S0 = self.ticker.history(period="1d")['Close'].iloc[-1]
        
        all_options_data = []
        today = datetime.now()

        # 2. Get available expirations
        expirations = self.ticker.options


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
            
            all_options_data.append(calls[['S0', 'strike', 'T', 'mid_price', 'impliedVolatility']])

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
        self.market_ivs = final_df['impliedVolatility'].values
        
        self.market_vegas = calculate_bs_vega(
            self.S0_array, 
            self.K_array, 
            self.T_array, 
            self.risk_free_interest, 
            self.market_ivs
        )

        

        print(f"Successfully loaded {len(self.market_prices)} liquid call options.")

    def split_by_expiry(self, test_every: int = 5, first_test: int = 2):
        """
        test_every : Every test_every-th expiry (sorted by maturity) is held out

        first_test : Index of the first held out expiry. Must be >= 1 so the shortest expiry stays in the training set

        Splits the data by whole expiries : all the options of the held out expiries form the test set, and the rest are used for calibration.
        The shortest and longest expiries are always kept for calibration, so the model is tested on maturities lying between the ones it was
        calibrated on (interpolation in T, not extrapolation). After this call, calibrate() only uses the training expiries, and evaluate()
        prices the test expiries.
        """
        if self.market_prices is None:
            self.fetch_and_clean_data_SQL()

        if self.test_data is not None:
            raise RuntimeError("Data has already been split. Create a new calibrator to split again.")

        if first_test < 1:
            raise ValueError("first_test must be >= 1 so that the shortest expiry is used for calibration.")

        #------Every test_every-th expiry, starting at first_test, never the last one------#
        unique_T = np.unique(self.T_array)
        test_T = unique_T[first_test:-1:test_every]
        test_mask = np.isin(self.T_array, test_T)

        #------Move the held out options from the calibration arrays to self.test_data------#
        array_names = ['S0_array', 'K_array', 'T_array', 'market_prices', 'market_ivs', 'market_vegas', 'market_spreads']
        self.test_data = {}
        for name in array_names:
            full_array = getattr(self, name)
            self.test_data[name] = full_array[test_mask]
            setattr(self, name, full_array[~test_mask])

        print(f"Calibrating on {len(unique_T) - len(test_T)} expiries ({len(self.market_prices)} options). "
              f"Holding out {len(test_T)} expiries ({test_mask.sum()} options) : {', '.join(f'{T*365:.0f}' for T in test_T)} days.")

    def evaluate(self, parameters: list) -> pd.DataFrame:
        """
        parameters : Calibrated parameters [sigma, lam, p, eta1, eta2], e.g. returned by calibrate()

        Returns :
            results : pd.DataFrame
                One row per held out option, with columns T, K, log_moneyness, market_price, model_price, market_iv, model_iv.
                Implied volatilities are obtained by inverting Black-Scholes, using the same r for the market mid prices and the model prices.
                NaN where a price is outside the no-arbitrage bounds (e.g. deep in the money calls quoted below intrinsic value).

        Uses the calibrated parameters to predict the prices of the options of the held out expiries (see split_by_expiry())
        """
        if self.test_data is None:
            raise RuntimeError("No held out data. Call split_by_expiry() before calibrate() and evaluate().")

        kou_params = {
            'r' : self.risk_free_interest,
            'sigma' : parameters[0],
            'lam': parameters[1],
            'p': parameters[2],
            'eta1': parameters[3],
            'eta2': parameters[4]
        }

        S0 = self.test_data['S0_array']
        K = self.test_data['K_array']
        T = self.test_data['T_array']
        market_prices = self.test_data['market_prices']

        #------Black-Scholes (lam = 0) : the implied volatility is sigma at every strike by definition. Inverting the price instead would give NaN
        #------wherever the price has no time value left (far from the money at short maturities), cutting the predicted curve short------#
        if kou_params['lam'] == 0:
            model_prices = bs_call_price(S0, K, T, self.risk_free_interest, np.full(len(K), kou_params['sigma']))
            model_ivs = np.full(len(K), kou_params['sigma'])
        else:
            model_prices = KouPricer(kou_params).generate_prices(S0, K, T)
            model_ivs = implied_volatility(model_prices, S0, K, T, self.risk_free_interest)

        results = pd.DataFrame({
            'T' : T,
            'K' : K,
            'log_moneyness' : np.log(K / S0),
            'market_price' : market_prices,
            'model_price' : model_prices,
            'market_iv' : implied_volatility(market_prices, S0, K, T, self.risk_free_interest),
            'model_iv' : model_ivs
        })

        return results

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
        
        #---- MSE is calculated as calc_price - market_price, weighed by market spread----------#
        #---- Higher Market spread represents more uncertainty, and more uncertain points are weighed less. Minimum spread is assigned to ensure division by zero doesn't take place-----#
        spread_floor = 0.05
        mse = np.mean( ((calculated_prices - self.market_prices) / np.maximum(self.market_spreads, spread_floor))**2 )

        #---- NaN/inf prices don't raise an exception, so penalize them here too----------#
        if not np.isfinite(mse):
            return 1e6

        return mse
    
    def calibrate(self, initial_guess: list, fix_lam_zero: bool = False):
        """
        initial_guess : List [sigma, lam, p, eta1, eta2] used as the starting point of the optimizer

        fix_lam_zero : If True, lam is fixed to 0 (no jumps) and only sigma is calibrated, i.e. the model is calibrated as Black-Scholes.
                       p, eta1, eta2 have no effect on prices when lam = 0, so they are held at their values in initial_guess.

        Returns :
            parameters : np.ndarray
                Calibrated [sigma, lam, p, eta1, eta2]
        """
        if self.market_prices is None:
            self.fetch_and_clean_data_SQL()
        
        bounds = (
            (0.04, 2.0),   # sigma
            (1e-4, 10.0),  # lambda
            (0.0, 1.0),    # p
            (2.0, 50.0),   # eta1 : Carr-Madan with alpha=0.75 needs E[S_T^(1+alpha)] finite, i.e. eta1 > 1.75
            (1e-4, 50.0)   # eta2
        )

        #---- Optimize over parameters rescaled to [0,1]. The raw parameters differ in scale by ~50x (p vs eta), and the objective has a narrow curved
        #---- valley along the jump parameters. Without rescaling, the finite-difference gradients make L-BFGS-B stall far from the minimum.
        lower = np.array([b[0] for b in bounds])
        upper = np.array([b[1] for b in bounds])

        #---- Parameters that are not free are held at their value in fixed_params. Black-Scholes : only sigma is free, lam = 0-------#
        fixed_params = np.array(initial_guess, dtype=float)
        free = np.ones(len(bounds), dtype=bool)
        if fix_lam_zero:
            fixed_params[1] = 0.0
            free[:] = False
            free[0] = True
            print("lam fixed to 0 : calibrating Black-Scholes (sigma only).")

        def to_params(u):
            params = fixed_params.copy()
            params[free] = lower[free] + u*(upper[free] - lower[free])
            return params

        scaled_objective = lambda u : self.objective_function(list(to_params(u)))
        scaled_guess = (fixed_params[free] - lower[free])/(upper[free] - lower[free])

        print("Starting optimization... this may take a few minutes.")

        result = minimize(
            fun=scaled_objective,
            x0=scaled_guess,
            method='L-BFGS-B',
            bounds=[(0.0, 1.0)]*int(free.sum()),
            options={'disp': True, 'ftol': 1e-12, 'gtol': 1e-10} # objective is flat along the jump parameters, a loose ftol stops early
        )

        parameters = to_params(result.x)
        print("Optimized Parameters:", parameters)
        return parameters