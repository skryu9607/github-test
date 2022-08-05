from matplotlib import pyplot as plt
import numpy as np
import math
from matplotlib import pyplot as plt
from warnings import catch_warnings
from warnings import simplefilter
from sklearn.gaussian_process import GaussianProcessRegressor

'''
Bayesian optimization is a powerful strategy for finding the extrema of objective functions that are expensive to evaluate. 
It is particulary useful when these evaluations are costly, when one does not have access to derivations, or when the problem at hand
is non-convex. 
Quantify the beliefs about an unknown objective function given samples from the domain and their evaluation via the objective function.
The posterior probability is a surrogate objective function. The posterior captures the updated beliefs about the unknown objective function. 
One may also interpret this step of Bayesian optimization as estimating the objective function with a surrogate function. 

'''
'''
Surrogate function 
Bayesian approximation of the objective function that can be sampled efficiently.
1. Acquistion function
2. Evaluation funciton
3. Surrogate function


'''



def objective_function(x, noise= 0.1):
    noise = np.random.normal(loc=0,scale = noise)
    return(x**2*math.sin(5*math.pi*x)**6.0) + noise


# Grid - based sample of the domain [0,1]
x = np.arange(0,1, 0.01)

# We can then evaluate these sampls using the target function without any noise to see what the real objective funciton looks like:
y = [objective_function(i,0) for i in x]

ynoise = [objective_function(i,0.1) for i in x]



ix = np.argmax(x)
print('Optima : x =%.3f , y = %.3f' % (x[ix], y[ix]))
plt.scatter(x,ynoise)
plt.plot(x,y)
plt.show()

# Treat the problem as a regression predictive modeling problem with the data representing the input and the score representing the output to the model. 
model = GaussianProcessRegressor()
yhat = model.predict(X, return_std = True)
# Surrogate or approximation for the objective function
# This funciton any time to estimate the cost of one or more samples.
def surrogate(model,x):
    # catch any warning generated when making a prediction
    with catch_warnings():
        simplefilter("ignore")
        return model.predict(x, return_std = True)


