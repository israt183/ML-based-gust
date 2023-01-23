#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 20:36:18 2022

@author: itu
"""
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime
import array as ar
from statistics import mean
from scipy import stats
import matplotlib as mpl
import joblib
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import pickle as pkl

cd=os.getcwd()

#Path_merged='/Volumes/Disk 2/Study/UCONN/Research/ML_WG_project/Merged_obs_WRF/'
Path_merged=cd+'/Combined_obsWRF/'
#importing the csv files
RW1=pd.read_csv(Path_merged+"Sorted_merged_obs_WRF_10.csv",sep=',')
RW2=pd.read_csv(Path_merged+"Sorted_merged_obs_WRF_14.csv",sep=',')
RW3=pd.read_csv(Path_merged+"Sorted_merged_obs_WRF_24.csv",sep=',')


# Remove column 'index'
RW1=RW1.drop(['index'],axis = 1)
RW2=RW2.drop(['index'],axis = 1)
RW3=RW3.drop(['index'],axis = 1)

RW_comb=pd.concat([RW1,RW2,RW3],axis=0,ignore_index=True)

## Converting units of the following features: PSFC(Pa),POT_2m(K),PBLH(m)
## change PSFC to kPa from Pa, PBLH to km and POT_2m to deg C
## be carfule that this cell should not run multiple times. Otherwise, values of the above features will keep changing
def to_kPa(x):
    return x/1000
RW_comb['PSFC(Pa)']= RW_comb['PSFC(Pa)'].apply(to_kPa)
def to_degC(x): 
    return x-273.15
RW_comb['POT_2m(K)']= RW_comb['POT_2m(K)'].apply(to_degC)
def to_km(x):
    return x/1000
RW_comb['PBLH(m)']= RW_comb['PBLH(m)'].apply(to_km)

## renaming the columns for which units have been converted
RW_comb = RW_comb.rename({'PSFC(Pa)': 'PSFC(kPa)', 'POT_2m(K)': 'POT_2m(C)','PBLH(m)':'PBLH(km)'}, axis=1)
RW_comb.head()

RW_comb['Valid_Time_x'] = pd.to_datetime(RW_comb['Valid_Time_x'], format='%Y%m%d%H')

a=RW_comb.at[0,'Valid_Time_x']
# empty list row_index will be used to store how many rows of the dataframe belong to each storm
row_index=[]
for i in range(len(RW_comb)):
    b=RW_comb.at[i,'Valid_Time_x']
    diff=abs(a-b)
    diff_in_hours = diff.total_seconds() / 3600
    if diff_in_hours<= 48:
        continue
    else:
        a=RW_comb.iloc[i,23]
        row_index.append(i)

    
row_index=[0]+row_index+[len(RW_comb)]
# 1st row_index(0) is the row index of the beginning of 1st event in RW_comb, 2nd row_index(1) is the row index of the beginning of 2nd event in RW_comb,
#row index(47) is the row index of the begnning of last event in RW_comb and row index(48) is the index of the end of the last event+1  

#Number of events in the dataset
Total_events=48
#Number of events for test dataset
Test_storms=1
r=Total_events/Test_storms

arr=np.arange(0, Total_events, 1)
all_score=pd.DataFrame()

#j=2
for j in range(int(r)):
    #the no. of test events
    Test_events=np.arange(j*Test_storms,(j+1)*Test_storms)
    #the number of train events
    Train_events= [x for x in arr if x not in Test_events]
    # data for all the test events
    test_data=RW_comb.iloc[row_index[Test_events[0]]:row_index[j+1],:]
    RW_copy=RW_comb.copy(deep=True)
    #data for all the train events
    train_data=RW_copy.drop(RW_copy.index[list(range(row_index[Test_events[0]],row_index[j+1]))],axis=0)
    # Creating dataframe for the input features
    X_train=pd.DataFrame(train_data.loc[:,'PSFC(kPa)':'Pot_temp_grad(PBLH_sfc)'])
    X_train=X_train.drop(['SfcWG_UPP(m/s)','WG_ECMWF(m/s)'],axis = 1)
    X_test=pd.DataFrame(test_data.loc[:,'PSFC(kPa)':'Pot_temp_grad(PBLH_sfc)'])
    X_test=X_test.drop(['SfcWG_UPP(m/s)','WG_ECMWF(m/s)'],axis = 1)
    UPP_test=pd.DataFrame(test_data['SfcWG_UPP(m/s)'])
    # Creating dataframe for targets
    Y_train=pd.DataFrame(train_data['WG_o'])
    Y_test=pd.DataFrame(test_data['WG_o'])
    # Creating dataframe for the input features by dropping highly correlated attributes
    X_train= X_train.drop(['PSFC(kPa)','POT_2m(C)','T2(K)','Pot_temp_grad(1km_sfc)','Pot_temp_grad(2km_sfc)','Pot_temp_grad(PBLH_sfc)'],axis=1)
    X_test=X_test.drop(['PSFC(kPa)','POT_2m(C)','T2(K)','Pot_temp_grad(1km_sfc)','Pot_temp_grad(2km_sfc)','Pot_temp_grad(PBLH_sfc)'],axis=1)
    X_train_copy=X_train.copy(deep=True)
    #fisrt, converting the angles to radians from degree
    X_train_copy['WindDC(cos)']= np.deg2rad(X_train_copy['WindDC(degree)']
                                               .sub(1).div(360).mul(360))
    X_train_copy['WindDC(cos)']=np.cos(X_train_copy['WindDC(cos)'])
    X_train_copy['WindDC(sin)']= np.deg2rad(X_train_copy['WindDC(degree)']
                                               .sub(1).div(360).mul(360))
    X_train_copy['WindDC(sin)']=np.sin(X_train_copy['WindDC(sin)'])
    X_train_copy=X_train_copy.drop(['WindDC(degree)'],axis=1)
    X_test_copy=X_test.copy(deep=True)
    #fisrt, converting the angles to radians from degree
    X_test_copy['WindDC(cos)']= np.deg2rad(X_test_copy['WindDC(degree)']
                                               .sub(1).div(360).mul(360))
    X_test_copy['WindDC(cos)']=np.cos(X_test_copy['WindDC(cos)'])
    X_test_copy['WindDC(sin)']= np.deg2rad(X_test_copy['WindDC(degree)']
                                               .sub(1).div(360).mul(360))
    X_test_copy['WindDC(sin)']=np.sin(X_test_copy['WindDC(sin)'])
    X_test_copy=X_test_copy.drop(['WindDC(degree)'],axis=1)
    
    #convert “Y_train” from a Pandas “Series” object into a NumPy array for the model 
    #to accept the target training data while fitting the model
    Y_train = np.array(Y_train)
    Y_train=Y_train.ravel()
    #FIXME hyperparameters

    Tuned_HPs={'n_estimators': 246,
           'max_depth': 16,
           'min_samples_split': 10,
           'max_features': 'sqrt',
           'bootstrap': True,
           'max_samples': 0.095}
    

    #FIXME
    from sklearn.ensemble import RandomForestRegressor
    #FIXME n_jobs
    model = RandomForestRegressor(random_state=10,n_jobs=-1,**Tuned_HPs)
    model.fit(X_train_copy, Y_train)
    
    #Feature importance_permutation method
    #This method will randomly shuffle each feature and compute the change in the model’s performance. 
    #The features which impact the performance the most are the most important one.
    #The permutation based importance is computationally expensive. 
    #The permutation based method can have problem with highly-correlated features, it can report them as unimportant.
    from sklearn.inspection import permutation_importance
    perm_importance = permutation_importance(model, X_train_copy, Y_train,scoring='neg_mean_squared_error',random_state=10)
    imp=perm_importance.importances_mean
    score=pd.DataFrame(X_train_copy.columns,imp)
    score=score.reset_index()
    score.columns=['Increase in MSE', 'Feature']
    score=score[score.columns[::-1]]
    all_score=all_score.append(score)  
    #all_score.append(score)
    print('iteration:'+str(j)+'_done')

all_score_final=all_score.reset_index()
#FIXME filename
all_score_final.to_csv('Scores_FI_RF_tuned.csv', index = False)

#%% 
import seaborn as sns
all_score_final=all_score_final.drop(['index'],axis = 1)
Tplot = sns.boxplot(y='Increase in MSE', x='Feature',data=all_score_final, 
                width=0.4,palette='Set1',flierprops = dict(markerfacecolor = '0.50', markersize = 2))

#plt.legend(loc='upper left',frameon=False)


Tplot.axes.set_title("Permutation importance plot",
                    fontsize=16)
 
Tplot.set_xlabel("Features", 
                fontsize=14)
 
Tplot.set_ylabel("Increase in MSE",
                fontsize=14)
 
Tplot.tick_params(labelsize=10)
plt.xticks(rotation=90, ha='right')
# output file name
#FIXME filename
plot_file_name="FI_RF_Tuned.jpg"
 
# save as jpeg
Tplot.figure.savefig(plot_file_name,bbox_inches='tight',
                    format='jpeg',
                    dpi=300)
