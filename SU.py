# Sveuciliste u Zagrebu
# Fakultet elektrotehnike i racunarstva
#
# Strojno ucenje 
# http://www.fer.hr/predmet/su
#
# (c) 2015 Jan Snajder

import pandas as pd
import scipy as sp
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode
        self.encoders ={}

    def fit(self,X,y=None):
        if self.columns is not None:
            for colname in self.columns:
                self.encoders[colname] = LabelEncoder().fit(X[colname])
        else:
            for colname,col in X.iteritems():
                self.encoders[colname] = LabelEncoder().fit(col)
        return self

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        for colname in self.encoders.keys():
            output[colname] = self.encoders[colname].transform(output[colname])
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)

    def encoders(self):
        return self.encoders

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

class PolyRegression:

    def __init__(self, order):
        self.order = order
        self.h = LinearRegression()
        
    def fit(self, X, y): 
        Xt = PolynomialFeatures(self.order).fit_transform(X)
        self.h.fit(Xt, y)
        return self
    
    def predict(self, X):
        Xt = PolynomialFeatures(self.order).fit_transform(X)
        return self.h.predict(Xt)
    
    def __call__(self, x):
        return self.predict(x)[0]

def plot_problem(X, y, h=None, surfaces=True) :
    '''
    Plots a two-dimensional labeled dataset (X,y) and, if function h(x) is given, 
    the decision boundaries (surfaces=False) or decision surfaces (surfaces=True)
    '''
    assert X.shape[1] == 2, "Dataset is not two-dimensional"
    if h!=None : 
        # Create a mesh to plot in
        r = 0.02  # mesh resolution
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = sp.meshgrid(sp.arange(x_min, x_max, r),
                             sp.arange(y_min, y_max, r))
        XX = sp.c_[xx.ravel(), yy.ravel()]
        try:
            #Z_test = h(XX)
            #if sp.shape(Z_test) == () :
            #    # h returns a scalar when applied to a matrix; map explicitly
            #    Z = sp.array(map(h,XX))
            #else :
            #    Z = Z_test
            Z = sp.array(map(h,XX))
        except ValueError:
            # can't apply to a matrix; map explicitly
            Z = sp.array(map(h,XX))
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        if surfaces :
            plt.contourf(xx, yy, Z, cmap=plt.cm.Pastel1)
        else :
            plt.contour(xx, yy, Z)
    # Plot the dataset
    plt.scatter(X[:,0],X[:,1],c=y, cmap=plt.cm.Paired,marker='o',s=50);
