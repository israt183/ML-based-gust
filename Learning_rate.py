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
from statistics import mean
from scipy import stats
import matplotlib as mpl
import joblib
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import pickle as pkl
import random

cd=os.getcwd()

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
        a=RW_comb.at[i,'Valid_Time_x']
        row_index.append(i)

        
row_index=[0]+row_index+[len(RW_comb)]
# 1st row_index(0) is the index of the beginning of 1st event, 2nd row_index(1) is the index of the beginning of 2nd event,
#row_index(47) is the index of the begnning of last event and row_index(48) is the index of the end of the last event+1 

#Number of events in the training stroms
Total_storms=48
Training_storms=45
Test_storms=3
Sample_size=5
r=Training_storms/Sample_size
Trials=16

BIAS_piled=pd.DataFrame()
RMSE_piled=pd.DataFrame()
CRMSE_piled=pd.DataFrame()
MAE_piled=pd.DataFrame()
#fisrt for loop for toal number of trials
for i in range(int(Trials)):
    #all storms ids
    arr=np.arange(0, Total_storms, 1)
    #test storms ids
    Test_events=np.arange(i*Test_storms,(i+1)*Test_storms)
    #preparing test storms
    test_storm_1=RW_comb.iloc[row_index[Test_events[0]]:row_index[Test_events[1]],:]
    test_storm_2=RW_comb.iloc[row_index[Test_events[1]]:row_index[Test_events[2]],:]
    test_storm_3=RW_comb.iloc[row_index[Test_events[2]]:row_index[Test_events[2]+1],:]
    
    X_test_1=pd.DataFrame(test_storm_1.loc[:,'PSFC(kPa)':'Pot_temp_grad(PBLH_sfc)'])
    X_test_1=X_test_1.drop(['SfcWG_UPP(m/s)','WG_ECMWF(m/s)','PSFC(kPa)','POT_2m(C)','T2(K)','Pot_temp_grad(1km_sfc)','Pot_temp_grad(2km_sfc)','Pot_temp_grad(PBLH_sfc)'],axis = 1)
    
    X_test_2=pd.DataFrame(test_storm_2.loc[:,'PSFC(kPa)':'Pot_temp_grad(PBLH_sfc)'])
    X_test_2=X_test_2.drop(['SfcWG_UPP(m/s)','WG_ECMWF(m/s)','PSFC(kPa)','POT_2m(C)','T2(K)','Pot_temp_grad(1km_sfc)','Pot_temp_grad(2km_sfc)','Pot_temp_grad(PBLH_sfc)'],axis = 1)
    
    X_test_3=pd.DataFrame(test_storm_3.loc[:,'PSFC(kPa)':'Pot_temp_grad(PBLH_sfc)'])
    X_test_3=X_test_3.drop(['SfcWG_UPP(m/s)','WG_ECMWF(m/s)','PSFC(kPa)','POT_2m(C)','T2(K)','Pot_temp_grad(1km_sfc)','Pot_temp_grad(2km_sfc)','Pot_temp_grad(PBLH_sfc)'],axis = 1)
    
    #fisrt, converting the angles to radians from degree
    X_test_1['WindDC(cos)']= np.deg2rad(X_test_1['WindDC(degree)']
                                                .sub(1).div(360).mul(360))
    X_test_1['WindDC(cos)']=np.cos(X_test_1['WindDC(cos)'])
    X_test_1['WindDC(sin)']= np.deg2rad(X_test_1['WindDC(degree)']
                                                .sub(1).div(360).mul(360))
    X_test_1['WindDC(sin)']=np.sin(X_test_1['WindDC(sin)'])
    X_test_1=X_test_1.drop(['WindDC(degree)'],axis=1)
    
    #fisrt, converting the angles to radians from degree
    X_test_2['WindDC(cos)']= np.deg2rad(X_test_2['WindDC(degree)']
                                                .sub(1).div(360).mul(360))
    X_test_2['WindDC(cos)']=np.cos(X_test_2['WindDC(cos)'])
    X_test_2['WindDC(sin)']= np.deg2rad(X_test_2['WindDC(degree)']
                                                .sub(1).div(360).mul(360))
    X_test_2['WindDC(sin)']=np.sin(X_test_2['WindDC(sin)'])
    X_test_2=X_test_2.drop(['WindDC(degree)'],axis=1)
    
    #fisrt, converting the angles to radians from degree
    X_test_3['WindDC(cos)']= np.deg2rad(X_test_3['WindDC(degree)']
                                                .sub(1).div(360).mul(360))
    X_test_3['WindDC(cos)']=np.cos(X_test_3['WindDC(cos)'])
    X_test_3['WindDC(sin)']= np.deg2rad(X_test_3['WindDC(degree)']
                                                .sub(1).div(360).mul(360))
    X_test_3['WindDC(sin)']=np.sin(X_test_3['WindDC(sin)'])
    X_test_3=X_test_3.drop(['WindDC(degree)'],axis=1)
    
    Y_test_1=pd.DataFrame(test_storm_1['WG_o'])
    Y_test_2=pd.DataFrame(test_storm_2['WG_o'])
    Y_test_3=pd.DataFrame(test_storm_3['WG_o'])
    
    Y_test_1 = np.asarray(Y_test_1)
    Y_test_2 = np.asarray(Y_test_2)
    Y_test_3 = np.asarray(Y_test_3)
    
    Y_test_1=Y_test_1.ravel()
    Y_test_2=Y_test_2.ravel()
    Y_test_3=Y_test_3.ravel()
    
    #train storms ids
    Train_events= [x for x in arr if x not in Test_events]
    Train_events_random = Train_events.copy()
    random.shuffle(Train_events_random)
    Train_events_random=pd.DataFrame(Train_events_random)
    # save the train and test storm ids dor each trial
    Train_events_random.columns=['Storm_ID']
    Train_events_random.to_csv(cd+'/Training_storms_trial_'+str(i)+'.csv')
    Test_events=pd.DataFrame(Test_events)
    Test_events.columns=['Storm_ID']
    Test_events.to_csv(cd+'/Test_storms_trial_'+str(i)+'.csv')
    Test_events= Test_events['Storm_ID'].values.tolist()
    #each trial will continue untill sample size for train 
    #storms reach 45
    for j in range(int(r)):
        Sample_size=Train_events_random[0:(j+1)*5]
        Sample_size = Sample_size['Storm_ID'].values.tolist()
        train_data=pd.DataFrame()
        for k in range(len(Sample_size)):
            df_sample=RW_comb.iloc[row_index[Sample_size[k]]:row_index[Sample_size[k]+1],:]
            train_data=pd.concat([df_sample,train_data])
        train_data=train_data.reset_index(drop=True)    
        X_train=pd.DataFrame(train_data.loc[:,'PSFC(kPa)':'Pot_temp_grad(PBLH_sfc)'])
        X_train=X_train.drop(['SfcWG_UPP(m/s)','WG_ECMWF(m/s)','PSFC(kPa)','POT_2m(C)','T2(K)','Pot_temp_grad(1km_sfc)','Pot_temp_grad(2km_sfc)','Pot_temp_grad(PBLH_sfc)'],axis = 1)
        X_train_copy=X_train.copy(deep=True)
        #fisrt, converting the angles to radians from degree
        X_train_copy['WindDC(cos)']= np.deg2rad(X_train_copy['WindDC(degree)']
                                                   .sub(1).div(360).mul(360))
        X_train_copy['WindDC(cos)']=np.cos(X_train_copy['WindDC(cos)'])
        X_train_copy['WindDC(sin)']= np.deg2rad(X_train_copy['WindDC(degree)']
                                                   .sub(1).div(360).mul(360))
        X_train_copy['WindDC(sin)']=np.sin(X_train_copy['WindDC(sin)'])
        X_train_copy=X_train_copy.drop(['WindDC(degree)'],axis=1)
        # Creating dataframe for targets
        Y_train=pd.DataFrame(train_data['WG_o'])
        #convert “Y_train” from a Pandas “Series” object into a NumPy array for the model 
        #to accept the target training data while fitting the model
        Y_train = np.array(Y_train)
        Y_train=Y_train.ravel()
    
  
        Tuned_HPs={'n_estimators': 246,
               'max_depth': 16,
               'min_samples_split': 10,
               'max_features': 'sqrt',
               'bootstrap': True,
               'max_samples': 0.095}
    
     
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(random_state=10,n_jobs=-1,**Tuned_HPs)
        model.fit(X_train_copy, Y_train)
        Y_pred_1 = model.predict(X_test_1)
        Y_pred_2 = model.predict(X_test_2)
        Y_pred_3 = model.predict(X_test_3)
        
        from sklearn.metrics import mean_squared_error
        #for test storm 1
        MSE_1=mean_squared_error(Y_test_1,Y_pred_1)
        MSE_1=round(MSE_1,3)
        # Bias= Prediction-Observation
        # This is mean bias
        BIAS_1 = np.mean(Y_pred_1-Y_test_1)
        BIAS_1=round(BIAS_1,3)
        BIAS_1=pd.Series(BIAS_1)
        
        RMSE_1 = mean_squared_error(Y_test_1,Y_pred_1)**0.5
        RMSE_1=round(RMSE_1,3)
        RMSE_1=pd.Series(RMSE_1)
        
        CRMSE_1 = (RMSE_1**2-BIAS_1**2)**0.5
        CRMSE_1=round(CRMSE_1,3)
        CRMSE_1=pd.Series(CRMSE_1)
        
        MAE_1=mean_absolute_error(Y_test_1,Y_pred_1)
        MAE_1=round(MAE_1,3) 
        MAE_1=pd.Series(MAE_1)
        
        #for test storm 2
        MSE_2=mean_squared_error(Y_test_2,Y_pred_2)
        MSE_2=round(MSE_2,3)
        # Bias= Prediction-Observation
        # This is mean bias
        BIAS_2 = np.mean(Y_pred_2-Y_test_2)
        BIAS_2=round(BIAS_2,3)
        BIAS_2=pd.Series(BIAS_2)
        
        RMSE_2 = mean_squared_error(Y_test_2,Y_pred_2)**0.5
        RMSE_2=round(RMSE_2,3)
        RMSE_2=pd.Series(RMSE_2)
        
        CRMSE_2 = (RMSE_2**2-BIAS_2**2)**0.5
        CRMSE_2=round(CRMSE_2,3)
        CRMSE_2=pd.Series(CRMSE_2)
        
        MAE_2=mean_absolute_error(Y_test_2,Y_pred_2)
        MAE_2=round(MAE_2,3)
        MAE_2=pd.Series(MAE_2)
        
        #for test storm 3
        MSE_3=mean_squared_error(Y_test_3,Y_pred_3)
        MSE_3=round(MSE_3,3)
        # Bias= Prediction-Observation
        # This is mean bias
        BIAS_3 = np.mean(Y_pred_3-Y_test_3)
        BIAS_3=round(BIAS_3,3)
        BIAS_3=pd.Series(BIAS_3)
        
        RMSE_3 = mean_squared_error(Y_test_3,Y_pred_3)**0.5
        RMSE_3=round(RMSE_3,3)
        RMSE_3=pd.Series(RMSE_3)
        
        CRMSE_3 = (RMSE_3**2-BIAS_3**2)**0.5
        CRMSE_3=round(CRMSE_3,3)
        CRMSE_3=pd.Series(CRMSE_3)
        
        MAE_3=mean_absolute_error(Y_test_3,Y_pred_3)
        MAE_3=round(MAE_3,3)
        MAE_3=pd.Series(MAE_3)
        
        TS=['TS_'+str(Test_events[0]),'TS_'+str(Test_events[1]),'TS_'+str(Test_events[2])]
        Storms=[5,10,15,20,25,30,35,40,45]
        #Error metrics of individual test storms
        BIAS=pd.concat([BIAS_1,BIAS_2,BIAS_3],axis=0,ignore_index=True)
        BIAS=BIAS.to_frame()
        BIAS.rename( columns={0 :'Value'}, inplace=True)
        BIAS['Test_storms']=TS
        BIAS['Sample_size']=Storms[j]
        BIAS_piled=pd.concat([BIAS,BIAS_piled])
        
        
        RMSE=pd.concat([RMSE_1,RMSE_2,RMSE_3],axis=0,ignore_index=True)
        RMSE=RMSE.to_frame()
        RMSE.rename( columns={0 :'Value'}, inplace=True)
        RMSE['Test_storms']=TS
        RMSE['Sample_size']=Storms[j]
        RMSE_piled=pd.concat([RMSE,RMSE_piled])
        
        CRMSE=pd.concat([CRMSE_1,CRMSE_2,CRMSE_3],axis=0,ignore_index=True)
        CRMSE=CRMSE.to_frame()
        CRMSE.rename( columns={0 :'Value'}, inplace=True)
        CRMSE['Test_storms']=TS
        CRMSE['Sample_size']=Storms[j]
        CRMSE_piled=pd.concat([CRMSE,CRMSE_piled])
        
        MAE=pd.concat([MAE_1,MAE_2,MAE_3],axis=0,ignore_index=True)
        MAE=MAE.to_frame()
        MAE.rename( columns={0 :'Value'}, inplace=True)
        MAE['Test_storms']=TS
        MAE['Sample_size']=Storms[j]
        MAE_piled=pd.concat([MAE,MAE_piled])
    
BIAS_piled.to_csv(cd+'/BIAS_all_trials'+'.csv', index = False)
RMSE_piled.to_csv(cd+'/RMSE_all_trials'+'.csv', index = False)
CRMSE_piled.to_csv(cd+'/CRMSE_all_trials'+'.csv', index = False)
MAE_piled.to_csv(cd+'/MAE_all_trials'+'.csv', index = False)
    
