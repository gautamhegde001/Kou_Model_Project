// bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "kou_price_mc.hpp"

namespace py = pybind11;

py::array_t<double> calculate_price_monte_carlo_py(
        unsigned long n_samples,
        double S_0, double T,
        py::array_t<double, py::array::c_style | py::array::forcecast> strikes,
        double r, double sigma, double lam,
        double eta1, double eta2, double p,
        unsigned long seed) {

    // Read the strikes array: get length and a raw pointer into its buffer
    auto strikes_view = strikes.unchecked<1>();
    int n_strikes = static_cast<int>(strikes_view.shape(0));
    const double* strikes_ptr = strikes.data();

    // Allocate the NumPy output array the C++ core will write into
    auto prices = py::array_t<double>(n_strikes);
    double* prices_ptr = prices.mutable_data();

    {
        py::gil_scoped_release release;
        calculate_price_monte_carlo(
            n_samples, S_0, T, n_strikes, strikes_ptr,
            r, sigma, lam, eta1, eta2, p, seed,
            prices_ptr);
    }

    return prices;
}

PYBIND11_MODULE(_kou_mc, m) {
    m.doc() = "Kou model Monte Carlo European option pricer";
    m.def("calculate_price_monte_carlo", &calculate_price_monte_carlo_py,
          py::arg("n_samples"),
          py::arg("S_0"),
          py::arg("T"),
          py::arg("strikes"),
          py::arg("r"),
          py::arg("sigma"),
          py::arg("lam"),        // 'lambda' is a reserved word in Python
          py::arg("eta1"),
          py::arg("eta2"),
          py::arg("p"),
          py::arg("seed") = 12345,
          "Price European calls across an array of strikes via Kou MC.");
}