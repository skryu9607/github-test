from matplotlib import pyplot as plt
import numpy as np
import math
from warnings import catch_warnings
from warnings import simplefilter
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import norm
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


# Treat the problem as a regression predictive modeling problem with the data representing the input and the score representing the output to the model. 
# Surrogate or approximation for the objective function
# This funciton any time to estimate the cost of one or more samples.
def surrogate(model,X):
    # catch any warning generated when making a prediction
    with catch_warnings():
        simplefilter("ignore")
        return model.predict(X, return_std = True)

def plot(X,y,model):
    plt.scatter(X,y)
    xsamples= np.asarray(np.arange(0, 1, 0.01))
    xsamples = xsamples.reshape(len(xsamples),1)
    ysamples,_ = surrogate(model,xsamples)
    plt.plot(xsamples,ysamples)
    plt.show()
# Acquistion function : to interpret and score the response from the surrogate function.
# BFGS Algorithm
def opt_acquisition(X,y,model):
    xsamples = np.random.random(100)
    xsamples = xsamples.reshape(len(xsamples),1)
    # calculate the acquistion function for each sample
    scores = acquisition(X,xsamples,model)
    ix = np.argmax(scores)
    return xsamples[ix,0]
 # Probability of imporovement acquisition function   
def acquisition(X,xsamples,model):
    # Calculate the best surrogate score found so far
    yhat,_ = surrogate(model,X)
    best = np.max(yhat)
    # Calculate the mean and stdev via surrogate function
    mu,std = surrogate(model,xsamples)
    # Calculate the probability of improvement
    probs = norm.cdf((mu-best)/(std+1e-9))
    # cdf is the normal cumulative distribution function
    return probs

# sample the domain
X = np.random.random(100)
y = np.asarray([objective_function(x) for x in X])

X = X.reshape(len(X),1)
y = y.reshape(len(y),1)
# Define the model
model = GaussianProcessRegressor()
model.fit(X,y)
# Plot before handle
plot(X,y,model)
# Perform the optimization process
for i in range(100):
    # Select the next point to sample
    x = opt_acquisition(X,y,model)
    actual = objective_function(x)
    est,_ = surrogate(model,[[x]])
    print('>x=%.3f, f()=%3f, actual=%.3f' % (x, est.item(), actual))
    X = np.vstack((X,[[x]]))
    y = np.vstack((y,[[actual]]))
    # Update the model
    model.fit(X,y)
plot(X,y,model)
ix = np.argmax(y)
print('Best result: x = %.3f, y = %.3f' %(X[ix],y[ix]))






