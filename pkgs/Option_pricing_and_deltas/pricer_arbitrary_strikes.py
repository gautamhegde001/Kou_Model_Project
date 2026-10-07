import numpy as np
from .carr_madan_function_aux import carr_madan_function, carr_madan_function_vectorized
from scipy.interpolate import CubicSpline



class KouPricer:
    """
    The class generates an object for a unique set of kou parameters to generate price options for an array of (S_0,K,T) simultaneously in a vectorized fashion. 

    _price_options_ratio_fft() creates a grid of C/S_0 for different values of x=ln(K/S_0) and T

    generate_prices() is the final function you call to produce a list of price options for a list of (S_0,K,T). It runs the FFT at exactly the
    distinct expiries requested (no interpolation in T), and interpolates in x with a cubic spline.

    """

    def __init__(self, kou_params: dict):
        """
        Initializes the Kou Pricer for the given parameters.
        """
        self.kou_params = kou_params

    def _price_options_ratio_fft(self, T_array: np.ndarray, N=4096, d_v=0.125, alpha=0.75):
        """
        T_array : 1D Array of size M containing the expiration times T

        kou_params : dictionary containing standard parameters defining kou process

        dv : spacing of frequency grid in fft

        alpha : damping factor used to make the carr-madan function square integrable, and hence amenable to fourier transforms.

        Note : The log-moneyness spacing is d_x = 2*pi/(N*d_v), so a small d_v gives a coarse strike grid. N=4096, d_v=0.125 gives
        d_x ~ 0.012 (prices between grid points are interpolated in generate_prices) while integrating up to v = N*d_v = 512.
        d_v=0.25 leaves a constant pricing bias of ~2.7e-5 * S_0 from the Simpson quadrature, d_v=0.125 removes it.

        Returns :
            x_grid : np.ndarray
                1D array of size N containing x_values (x is the log_moneyness, x = ln(K/S_0)) for which (call) 
                option prices have been calculated

            price_ratios : np.ndarray
                2D array of size N x M containing (call) option price ratios (C/S_0). Rows correspond to x-values, columns correspond
                to T values 

        This function calculates the ratio C(x,t)/S_0, where C is the call option price, and x the log-moneyness (x = ln (K/S_0)), and t is the time
        to expiry. The ratios are calculated via the fourier transform method of Carr&Madan(1999), for the kou process. 
        """

        # Defining Frequency grid (v) 
        v_grid = np.arange(N) * d_v
        
        # Defining the Log-moneyness grid (x)
        d_x = (2 * np.pi) / (N * d_v)
        x_grid = - (N * d_x) / 2 + np.arange(N) * d_x
        
        # Mask to keep relevant strikes. Slightly wider than the [-0.7, 0.7] used elsewhere so the spline is not extrapolated at the edges
        mask = (x_grid > -0.8) & (x_grid < 0.8)
        truncated_x_grid = x_grid[mask]
        
        # Evaluating the Carr-Madan function for all values of v and T
        # v-grid is 1D array of size N. T_array is 1d array of size M. carr_mada_function_vectorized is 2D array of size M x N.
        psi_v = carr_madan_function_vectorized(v_grid, alpha, T_array, self.kou_params)
            
        # FFT with Simpson's Rule weights
        weights = (3 + (-1)**(np.arange(N) + 1)) / 3.0
        weights[0] = 1.0 / 3.0
            
        # Shift input by x_min for taking FFT
        x_min = x_grid[0]
        fft_input = np.exp(-1j * v_grid * x_min) * psi_v * weights * d_v
            
        # Execute FFT
        fft_output = np.fft.fft(fft_input, axis=-1)
            
        # Price calculation
        # price ratios is a 2D array of size N x M. Rows correspond to fixed x-values, columns correspond to fixed T-values.
        price_ratios = (np.exp(-alpha * x_grid[:,None]) / np.pi) * np.real(fft_output).T
        
        # Slice all columns (:), but only keep the rows that match the mask
        truncated_price_ratios = price_ratios[mask, :]

        return truncated_x_grid, truncated_price_ratios

    def generate_prices(self, S_0_array: np.ndarray, K_array: np.ndarray, T_array: np.ndarray) -> np.ndarray:
        """
        S_0_array : 1d array of size N containing spot prices S_0

        K_array : 1D array of size N containing strike prices K

        T_array : 1D array of size N containing expiry times T 

        Calculates the prices of N call options, each one with its own S_0, K, and T. (The three are entered seperately through the three arrays S_0_array, K_array
        T_array). The FFT is run once for each distinct T (market data only has ~20 expiries), and prices between strike grid points are 
        obtained with a cubic spline in x.

        Returns :

            prices : np.float64 

                Price for the given S_0, K, T and given kou parameters
        """
        #--------Ensure that input lists are numpy arrays----------------#
        S_0_array = np.atleast_1d(S_0_array)
        K_array = np.atleast_1d(K_array)
        T_array = np.atleast_1d(T_array)
        
        S_0_array, K_array, T_array = np.broadcast_arrays(S_0_array, K_array, T_array)

        x_array = np.log(K_array/S_0_array) # 1D array of N x-values, x here is log moneyness

        #---------FFT at each distinct expiry. T_index maps every option to its column in price_ratio_grid---------------
        unique_T, T_index = np.unique(T_array, return_inverse=True)
        log_moneyness_grid, price_ratio_grid = self._price_options_ratio_fft(unique_T)

        #------------Cubic spline in x for every expiry at once, then pick each option's own expiry----------------
        spline = CubicSpline(log_moneyness_grid, price_ratio_grid, axis=0)
        price_option_ratios = spline(x_array)[np.arange(len(x_array)), T_index]

        #------------Multiplying by S_0 to get price option values ----------------
        price_option_values = price_option_ratios * S_0_array

        return price_option_values
