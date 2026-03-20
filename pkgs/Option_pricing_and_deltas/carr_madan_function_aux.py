import numpy as np

def levy_exponent(z: np.complex128, r: np.float64, sigma : np.float64, lam : np.float64, eta1 : np.float64, eta2 : np.float64, p: np.float64) :
    """
    z : argument of function
    r: risk free interest rate
    sigma : volatility of brownian motion
    lam : poisson parameter

    eta1 : exponential parameter corresponding to upward jump
    eta2 : exponential parameter corresponding to downward jump
    p : probability of upward jump

    Returns :

    psi_z : np.float
        The levy exponent

    This function returns the levy exponent psi(z) given all the parameters defining the kou process.

    For a levy process X_T, the levy exponent is defined as :

    psi(z) = 1/T * log (E[e^{z X_T}])


    
    """

    kappa = p * eta1/(eta1-1) + (1-p)*eta2/(eta2+1) - 1 


    drift_term = (r - 0.5*sigma**2 - lam*kappa)*z #comes from the deterministic part of the process
    brownian_term = 0.5*sigma**2 * z**2 #comes from the brownian motion part of the process

    jump_term = lam*(p * eta1/(eta1-z) + (1-p)*eta2/(eta2+z)-1) # comes from the jumps in the process

    psi_z = drift_term + brownian_term + jump_term

    return psi_z

def characteristic_function(u : np.complex128 ,T : np.float64 ,kou_parameters : dict) :

    """
    u : argument of characteristic function

    S_0 : initial stock price 

    T : Expiry time/ Time of evaluation

    kou_parameters : A dictionary containing all the kou parameters needed to compute the levy exponent. Defines the kou process.

    Returns :

    phi_u : np.complex128
        The characteristic function of the kou process evaluated at a specific u

    The characteristic function of the kou process is defined as 

    phi(u)  = E[e^{iu log(S(t)) }]

    And it can be broken down as :

    phi(u) = e^{iu log(S_0)} * e^{T psi(iu)}


    """
    
    
    #-------------Levy exponent------------------
    r = kou_parameters['r']
    sigma = kou_parameters['sigma']
    lam = kou_parameters['lam']
    eta1 = kou_parameters['eta1']
    eta2 = kou_parameters['eta2']
    p = kou_parameters['p']

    z = 1j*u
    psi_z = levy_exponent(z,r,sigma,lam,eta1,eta2,p)

    levy_exp_contrib = np.exp(T * psi_z)

    phi_u = levy_exp_contrib

    return phi_u

def carr_madan_function(v : np.float64, alpha : np.float64, T : np.float64, kou_params : dict) :
    """
    v : Argument of carr-madan function

    alpha : Damping factor used to make function square integrable, and hence amenable to fourier transforms

    S_0 : initial stock value

    T : time to expiry/evaluation

    kou_params : A dictionary containing all the parameters characterizing the kou process. 

    Returns :
        CM_func : np.float64
            The carr_madan function evaluated at a specific v 

    The carr-madan function is a useful function that can be used to calculate the option prices numerically in a feasible manner.
    It is defined as the fourier transform of the call price multiplied by a damping exponential factor e^{alpha k}
    """
    z = v - (alpha+1)*1j
    phi = characteristic_function(z,T,kou_params)

    r = kou_params['r']

    Numerator = np.exp(-r*T)*phi

    Denominator = alpha**2 + alpha - v**2 + 1j*(2*alpha + 1)*v

    return Numerator/Denominator