#pragma once

void calculate_price_monte_carlo(unsigned long n_samples,double S_0, double T, int n_strikes, const double* strikes, double r, double sigma, double lam, double eta1, double eta2, double p, unsigned long seed,double* prices); 