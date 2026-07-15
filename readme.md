# Kou Model Project

This repository contains tools for pricing options under the **Kou Jump Diffusion Model** and simulating portfolio delta hedging. It provides implementations using multiple quantitative finance methodologies, and includes a performance-critical Monte Carlo pricing engine written in **C++** and exposed to Python via **pybind11**.

## Valuation Methods & Features Implemented
1. **Model Calibration**: Includes a module to calibrate the Kou model parameters to real or simulated market data. Real market data is taken using yfinance, with data-cleaning done using SQL.
2. **Inverse Fast Fourier Transform (FFT)**: Prices options efficiently based on the characteristic function of the Kou model using the Carr and Madan (1999) approach. The theory behind this is explained in the Project_Report.pdf file.
3. **Vectorized Pricing for Arbitrary Strikes**: Option prices for several options can be calculated simultaneously in a vectorized way allowing arbitrary strikes, rather than being restricted to a predetermined standard FFT grid.
4. **Monte Carlo Simulations**: Computes option prices by generating simulated stock paths with Brownian motion and Poisson-driven double-exponential jumps. Available with two interchangeable backends: a vectorized NumPy implementation and a C++ engine (via pybind11) that parallelizes the path simulation with OpenMP, giving roughly a 100x speedup. The C++ pricer is cross-validated against the semi-analytic Carr-Madan FFT pricer for correctness.
5. **Delta Hedging Simulation**: Simulates the performance of a delta-hedged portfolio through time.

## Project Structure

- **`pkgs/`**: Contains core modules and functions for quantitative calculations.
  - `monte_carlo_simulators.py`: Monte Carlo simulation engine for the jump-diffusion paths. Exposes a `backend` switch selecting between the Python and C++ pricing implementations.
  - `_kou_mc`: Compiled C++ extension module (built from `src/cpp/`, see below) providing the parallelized Monte Carlo pricer.
  - **`Option_pricing_and_deltas/`**: Subpackage focused on pricing algorithms and Greek calculations.
    - `carr_madan_function_aux.py`: Auxiliary functions for the Carr-Madan pricing equations.
    - `delta_calculator.py`: Logic for computing option deltas and required hedging ratios.
    - `price_options_via_fft.py`: Implements the Fast Fourier Transform routine for option pricing.

- **`src/cpp/`**: C++ source for the compiled extension.
  - **`kou_price_monte_carlo/`**: The Kou Monte Carlo pricing engine.
    - `kou_price_mc.cpp` / `kou_price_mc.hpp`: The pricing core — terminal-sampling Monte Carlo for the Kou model, pure C++ with no Python dependencies.
    - `bindings.cpp`: pybind11 bindings exposing the pricer to Python as `_kou_mc`.

- **`scripts/`**: Executable scripts acting as the primary entry points.
  - `calc_price_options_arbitrary_strikes.py`: Runs vectorized calculations for option prices at arbitrary, non-grid strikes.
  - `calc_price_options_via_fft.py`: Runs FFT to compute option prices vs. strikes and saves output.
  - `calc_price_options_via_monte_carlo.py`: Runs Monte Carlo simulators to evaluate option prices (Python or C++ backend).
  - `simulate_portfolio_delta_hedging.py`: Runs a simulated delta hedging scenario.
  - `calibrate_model.py`: Script to calibrate the Kou Model parameters.

- **`datafiles/`**: Output directory where calculated numerical datasets (`.npy` files) are saved.
- **`Data_Analysis/`**: Contains Jupyter notebooks allowing for easy analysis and visualization of data.
- **`Figures/`**: Figures from Data_Analysis are saved here.

## Building the C++ Extension

The Monte Carlo C++ backend is compiled as part of installing the project. The build uses **scikit-build-core** and **CMake**, and requires a C++ compiler (MSVC on Windows, or GCC/Clang on Linux/macOS). pybind11, CMake, and Ninja are provisioned automatically during the build, so no manual installation of the build tools is required.

From the repository root, with your virtual environment activated:

```bash
pip install -e .
```

This compiles the extension and makes it importable as `pkgs._kou_mc`. Re-run this command after any changes to the C++ source. On Windows, run it from a shell with the MSVC toolchain available (e.g. the *x64 Native Tools Command Prompt for VS*, or a terminal launched from it) so the compiler is on the path.

## Usage

After building (`pip install -e .`), run the scripts as modules from the root path:

```bash
# Calibrate the Kou Model to market data
python -m scripts.calibrate_model

# Calculate prices simultaneously for arbitrary strikes in a vectorized way
python -m scripts.calc_price_options_arbitrary_strikes

# Calculate prices using the FFT approach (Carr-Madan)
python -m scripts.calc_price_options_via_fft

# Calculate prices using Monte Carlo simulations (Python or C++ backend)
python -m scripts.calc_price_options_via_monte_carlo

# Run delta hedging simulation
python -m scripts.simulate_portfolio_delta_hedging
```

## Dependencies

This project is built using standard scientific computing libraries, the yfinance library to extract market data, and pybind11 / scikit-build-core / CMake for the C++ extension (the latter three are handled automatically at build time). The core runtime libraries are:
- numpy
- scipy
- matplotlib
- pandas
- yfinance
- sqlalchemy

Building the C++ extension additionally requires a C++ compiler with OpenMP support (bundled with MSVC on Windows; standard on GCC/Clang).
