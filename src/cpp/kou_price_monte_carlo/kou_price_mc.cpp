#include <random>
#include <vector>
#include <algorithm>
#include <cmath>
#include "kou_price_mc.hpp"


void calculate_price_monte_carlo(unsigned long n_samples,double S_0, double T, int n_strikes, const double* strikes, double r, double sigma, double lam, double eta1, double eta2, double p, unsigned long seed,double* prices){
    

    std::mt19937_64 rng(seed);
    std::normal_distribution<double> norm(0,1.0);
    std::exponential_distribution<double> up_jump_expo(eta1);
    std::exponential_distribution<double> down_jump_expo(eta2);
    std::poisson_distribution<long> poisson(lam*T);
    std::uniform_real_distribution<double> unif(0.0, 1.0);


    //------Calculating martingale compensator and drift term-----------------
    double kappa = p * (eta1/(eta1 - 1)) + (1-p) * (eta2/(eta2 + 1)) - 1;
    double x_T_from_drift = (r - 0.5 * sigma*sigma - lam * kappa) * T;

    std::vector<double> sum(n_strikes,0.0); // Variable sized array which will store sums of option prices calculated for all samples

    for(unsigned long i=0; i<n_samples; ++i){
        //--------Calculating net log-returns (x_T) from jumps-------
        double x_T_from_jumps = 0.0;
        long n_jumps = poisson(rng);

        for(long j=0;j<n_jumps;j++){

            double up_or_down = unif(rng);

            double jump = 0;
            if (up_or_down <= p) {
                jump = up_jump_expo(rng);
            }

            else{
                 jump = -down_jump_expo(rng);

            }

            x_T_from_jumps+=jump;
        }


        //--------Calculating net log-returns (x_T) from brownian motion------
        double x_T_from_diffusion = norm(rng) * sigma * std::sqrt(T);





        double x_T_total = x_T_from_drift + x_T_from_diffusion + x_T_from_jumps ;

        double S_T = S_0 * std::exp(x_T_total); 


        //-----Calculating option prices for all strikes-----

        

        for(int k=0;k<n_strikes;k++){
            
            double option_price_sample = std::max(S_T - strikes[k],0.0);

            sum[k] += option_price_sample;
            

        }
    }


    for(int k=0;k<n_strikes;k++){
        prices[k] = std::exp(-r*T)*sum[k]/static_cast<double>(n_samples); // Calculates final prices by back-dating results of monte carlo to signing date
    }

    
   
    
};

