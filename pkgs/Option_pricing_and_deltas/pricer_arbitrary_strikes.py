import numpy as np
from .carr_madan_function_aux import carr_madan_function, carr_madan_function_vectorized
from scipy.interpolate import RegularGridInterpolator



class KouPricer:
    """
    The class generates an object for a unique set of kou parameters to generate price options for an array of (S_0,K,T) simultaneously in a vectorized fashion. 

    price_options_ratio_fft() creates a grid of C/S_0 for different values of x=ln(K/S_0) and T

    price_options_interpolator_maker() uses that grid to interpolate and create a function that can calculate price options for any (x,T) for a fixed set of kou parameters

    price_options_interpolated_vectorized() is the final function you call to produce a list of price options for a list of (S_0,K,T)

    """

    def __init__(self, kou_params: dict):
        """
        Initializes the Kou Pricer by building the interpolation surface
        for the given parameters.
        """
        self.kou_params = kou_params
        # We generate the interpolator once during initialization
        self.interpolator = self._price_options_interpolator_maker()

    def _price_options_ratio_fft(self, T_array: np.ndarray, N=8192, d_v=0.01, alpha=0.75):
        """
        T_array : 1D Array of size M containing the expiration times T

        kou_params : dictionary containing standard parameters defining kou process

        dv : spacing of frequency grid in fft

        alpha : damping factor used to make the carr-madan function square integrable, and hence amenable to fourier transforms. 

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
        
        # Mask to keep relevant strikes
        mask = (x_grid > -0.7) & (x_grid < 0.7)
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

    def _price_options_interpolator_maker(self) -> RegularGridInterpolator:
        """
        kou_params : Dictionary containing various kou parameters

        Returns :

            price_option_interpolator : RegularGridInterpolator

                2D function that gives price of (call) option ratio (C/S_0) for desired value of x=ln(K/S_0) (log-moneyness) and T (time).

                price_option_interpolator(x,T) yields C(x,T)/S_0 for set of kou parameters.
        """

        T_array = np.linspace(0.1, 5, 100)
        log_moneyness_grid, price_ratio_grid = self._price_options_ratio_fft(T_array) # generating grid of strikes and option-prices

        price_ratio_surface = np.array(price_ratio_grid)

        price_option_interpolator = RegularGridInterpolator((log_moneyness_grid, T_array), price_ratio_surface, bounds_error=False, fill_value=None)

        return price_option_interpolator

    def generate_prices(self, S_0_array: np.ndarray, K_array: np.ndarray, T_array: np.ndarray) -> np.ndarray:
        """
        S_0_array : 1d array of size N containing spot prices S_0

        K_array : 1D array of size N containing strike prices K

        T_array : 1D array of size N containing expiry times T 

        price_option_interpolator : Interpolator that generates price options for a given set of Kou parameters

        Calculates the prices of N call options, each one with its own S_0, K, and T. (The three are entered seperately through the three arrays S_0_array, K_array
        T_array). Prices are calculated using the interpolator (price_option_interpolator)

        Returns :

            prices : np.float64 

                Price for the given S_0, K, T and given kou parameters
        """
        #--------Ensure that input lists are numpy arrays----------------#
        S_0_array = np.atleast_1d(S_0_array)
        K_array = np.atleast_1d(K_array)
        T_array = np.atleast_1d(T_array)
        
        #---------Making N x 2 array that will be the input of price_option_interpolator---------------
        x_array = np.log(K_array/S_0_array) # 1D array of N x-values

        x_T_matrix = np.column_stack((x_array, T_array))

        #------------Calculating price option ratios ----------------
        price_option_ratios = self.interpolator(x_T_matrix)

        #------------Multiplying by S_0 to get price option values ----------------
        price_option_values = price_option_ratios * S_0_array

        return price_option_values
