# Kou Model Project

This repository contains Python tools for pricing options under the **Kou Jump Diffusion Model** and simulating portfolio delta hedging. It provides implementations using multiple quantitative finance methodologies.

## Valuation Methods & Features Implemented
1. **Model Calibration**: Includes a module to calibrate the Kou model parameters to real or simulated market data. Real market data is taken using yfinance, with data-cleaning done using SQL. 
2. **Inverse Fast Fourier Transform (FFT)**: Prices options efficiently based on the characteristic function of the Kou model using the Carr and Madan (1999) approach. The theory behind this is explained in the Project_Report.pdf file. 
3. **Vectorized Pricing for Arbitrary Strikes**: Option prices for several options can be calculated simultaneously in a vectorized way allowing arbitrary strikes, rather than being restricted to a predetermined standard FFT grid.
4. **Monte Carlo Simulations**: Computes the option prices by generating simulated stock paths with Brownian motion and Poisson-driven exponential jumps.
5. **Delta Hedging Simulation**: Simulates the performance of a delta-hedged portfolio through time.

## Project Structure

- **`pkgs/`**: Contains core modules and functions for quantitative calculations.
  - `monte_carlo_simulators.py`: Core Monte Carlo simulation engine for the jump-diffusion paths.
  - **`Option_pricing_and_deltas/`**: Subpackage focused on pricing algorithms and Greek calculations.
    - `carr_madan_function_aux.py`: Auxiliary functions for the Carr-Madan pricing equations.
    - `delta_calculator.py`: Logic for computing option deltas and required hedging ratios.
    - `price_options_via_fft.py`: Implements the Fast Fourier Transform routine for option pricing.
  
- **`scripts/`**: Executable scripts acting as the primary entry points.
  - `calc_price_options_arbitrary_strikes.py`: Runs vectorized calculations for option prices at arbitrary, non-grid strikes.
  - `calc_price_options_via_fft.py`: Runs FFT to compute option prices vs. strikes and saves output.
  - `calc_price_options_via_monte_carlo.py`: Runs Monte Carlo simulators to evaluate options prices.
  - `simulate_portfolio_delta_hedging.py`: Runs a simulated delta hedging scenario.
  - `calibrate_model.py`: Script to calibrate the Kou Model parameters.

- **`datafiles/`**: Output directory where calculated numerical datasets (`.npy` files) are saved.
- **`Data_Analysis/`** : Contains jupyter notebooks allowing for easy analysis and visualization of data.

- **`Figures/`**: Figures from Data_Analysis are saved here. 


## Usage

Ensure you have your virtual environment activated, then run the scripts as modules from the root path:

```bash
# Calibrate the Kou Model to market data
python -m scripts.calibrate_model

# Calculate prices simultaneously for arbitrary strikes in a vectorized way
python -m scripts.calc_price_options_arbitrary_strikes

# Calculate prices using the FFT approach (Carr-Madan)
python -m scripts.calc_price_options_via_fft

# Calculate prices using Monte Carlo simulations
python -m scripts.calc_price_options_via_monte_carlo

# Run delta hedging simulation
python -m scripts.simulate_portfolio_delta_hedging
```

## Dependencies
This project is built utilizing the scientific computing libraries, and the yfinance library to extract market data. The list of libraries is :
- numpy
- scipy
- matplotlib
- pandas
- yfinance
- sqlalchemy

