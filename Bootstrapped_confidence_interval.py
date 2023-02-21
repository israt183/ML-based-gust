#!/usr/bin/env python3
# -*- coding: utf-8 -*-
© 2021-2023 Israt Jahan, University of Connecticut. All rigts reserved.

import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime
import array as ar
#from statistics import mean
from scipy import stats
import matplotlib as mpl
import joblib
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
#import pickle as pkl
import random
import scikits.bootstrap as boot
#from scipy.stats import bootstrap

obs=pd.read_csv('All_obs_test_RF.csv')
pred=pd.read_csv('All_pred_test_RF.csv')
obs=np.array(obs)
pred=np.array(pred)
obs=obs.astype(np.float32)
pred=pred.astype(np.float32)
obs=obs.ravel()
pred=pred.ravel()

#r_value
# =============================================================================
# cis_r = boot.ci( (obs,pred), statfunction=stats.linregress,alpha=0.05,n_samples=10000,method='pi',
#               multi='paired',return_dist=True )
# =============================================================================

# =============================================================================
# #MAE
# cis_mae = boot.ci( (obs,pred), statfunction=mean_absolute_error,alpha=0.05,n_samples=10000,method='pi',
#               multi='paired',return_dist=True )
# =============================================================================

#RMSE
def rmse(obs,pred):
    RMSE = mean_squared_error(obs,pred)**0.5
    return RMSE

cis_rmse = boot.ci( (obs,pred), statfunction=rmse,alpha=0.05,n_samples=10000,method='pi',
              multi='paired',return_dist=True)

#BIAS
def bias(obs,pred):
    BIAS = np.mean(pred-obs)
    return BIAS

cis_bias = boot.ci( (obs,pred), statfunction=bias,alpha=0.05,n_samples=10000,method='pi',
              multi='paired',return_dist=True)

#CRMSE
def crmse(obs,pred):
    BIAS = np.mean(pred-obs)
    RMSE = mean_squared_error(obs,pred)**0.5
    CRMSE = (RMSE**2-BIAS**2)**0.5
    return CRMSE

cis_crmse = boot.ci( (obs,pred), statfunction=crmse,alpha=0.05,n_samples=10000,method='pi',
              multi='paired',return_dist=True) 

