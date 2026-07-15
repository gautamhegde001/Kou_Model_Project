#include <random>
#include <vector>
#include <algorithm>
#include <cmath>
#include <omp.h>
#include "kou_price_mc.hpp"


void calculate_price_monte_carlo(unsigned long n_samples, double S_0, double T,
                                 int n_strikes, const double* strikes,
                                 double r, double sigma, double lam,
                                 double eta1, double eta2, double p,
                                 unsigned long seed, double* prices) {

    // Constants across all paths — computed once, shared read-only
    double kappa = p * (eta1 / (eta1 - 1.0)) + (1.0 - p) * (eta2 / (eta2 + 1.0)) - 1.0;
    double x_T_from_drift = (r - 0.5 * sigma * sigma - lam * kappa) * T;
    double diffusion_scale = sigma * std::sqrt(T);

    // Global accumulators, written once per thread at the end (not per path)
    std::vector<double> sum(n_strikes, 0.0);

    #pragma omp parallel
    {
        // --- Per-thread state: each thread gets its own RNG and its own accumulator ---
        int tid = omp_get_thread_num();

        // Seed each thread distinctly so their random streams don't overlap
        std::mt19937_64 rng(seed + static_cast<unsigned long>(tid));

        std::normal_distribution<double> norm(0.0, 1.0);
        std::exponential_distribution<double> up_jump_expo(eta1);
        std::exponential_distribution<double> down_jump_expo(eta2);
        std::poisson_distribution<long> poisson(lam * T);
        std::uniform_real_distribution<double> unif(0.0, 1.0);

        std::vector<double> local_sum(n_strikes, 0.0);   // private to this thread

        // --- Split the sample loop across threads ---
        #pragma omp for
        for (long long i = 0; i < static_cast<long long>(n_samples); ++i) {
            double x_T_from_jumps = 0.0;
            long n_jumps = poisson(rng);
            for (long j = 0; j < n_jumps; ++j) {
                double jump = (unif(rng) <= p) ? up_jump_expo(rng) : -down_jump_expo(rng);
                x_T_from_jumps += jump;
            }

            double x_T_from_diffusion = norm(rng) * diffusion_scale;
            double x_T_total = x_T_from_drift + x_T_from_diffusion + x_T_from_jumps;
            double S_T = S_0 * std::exp(x_T_total);

            for (int k = 0; k < n_strikes; ++k) {
                local_sum[k] += std::max(S_T - strikes[k], 0.0);
            }
        }

        // --- Combine each thread's local sums into the global sum, one thread at a time ---
        #pragma omp critical
        {
            for (int k = 0; k < n_strikes; ++k) {
                sum[k] += local_sum[k];
            }
        }
    }

    double discount = std::exp(-r * T);
    for (int k = 0; k < n_strikes; ++k) {
        prices[k] = discount * sum[k] / static_cast<double>(n_samples);
    }
}