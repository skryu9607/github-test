# Using surrogate model and acquisition functions and Expected improvement
'''
We will implement the acquisition function and its optimization in plain NumPy and SciPy
and use scikit-learn for the Gaussian Process implementation.
f : black box and iteratively approximate it with a Gaussian process during Bayesian optimization. 
Samples drawn from the objective function are noisy and the noise level is given by the noise variable. 
'''
import numpy as np
import matplotlib.pyplot as plt
bounds = np.array([[-1.0,2.0]])
noise = 0.2

def f(X,noise = noise ):
    '''
    Numpy.random.rand : 값 1개가 추출
    Numpy.random.randn : array만들 수 있다.
    Numpy.random.random : 원하는 차원의 형태를 튜플 자료형으로 넣어주어야한다는 차이점. 
    -> np.random.rand(5,3) and np.random.random((5,3)) 
    '''   
    return -np.sin(3*X)-X**2 + 0.7 * X + noise * np.random.randn(*X.shape)
x_init = np.array([[-0.9],[1.1]])
y_init = f(x_init)

x = np.arange(bounds[:,0], bounds[:,1],0.01).reshape(-1,1)
y = f(x,0)

plt.plot(x, y, 'y--', lw=2, label='Noise-free objective')
plt.plot(x, f(x), 'bx', lw=1, alpha=0.1, label='Noisy samples')
plt.plot(x_init, y_init, 'kx', mew=3, label='Initial samples')
plt.legend()
# Goal is to find the global optimum in a small number of steps.

from scipy.stats import norm
def expected_improvements(x,x_sample,y_sample, gpr,xi = 0.01):
    '''
        x : POINTS at which EI shall be computed
        x_sample : sample locations
        y_sample : samle values
        gpr : A gaussianProcessRegressor fitted to samples
        xi : exploitation-exploration trade-off parameter

        Returns: expected improvements at points x

    '''
    mu,sigma = gpr.predict(x,return_std = True)
    mu_sample = gpr.predict(x_sample)

    sigma = sigma.reshape(-1,1)

    mu_sample_opt = np.max(mu_sample)

    with np.errstate(divide = 'warn'):
        imp = mu - mu_sample_opt - xi
        z = imp/sigma
        ei = imp*norm.cdf(z) + sigma * norm.pdf(z)
        ei[sigma == 0.0] = 0.0
    return ei
# We also need a function that proposes the next sampling point by computing the location
# of the acquisition function maximum. 