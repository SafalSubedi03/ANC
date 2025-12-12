import numpy as np 
import matplotlib.pyplot as plt 
from Cross_Correlation import Cross_Corelation
from AutoCorrelation import auto_Correlation

def optimalCoeff(autocorrelation,crosscorrelation):
    if(len(autocorrelation) != len(crosscorrelation)):
        raise ValueError("Incorrect length of input relation matrices")
    return crosscorrelation/autocorrelation

w0_n = optimalCoeff(auto_Correlation,Cross_Corelation)
print(w0_n[:10])
        