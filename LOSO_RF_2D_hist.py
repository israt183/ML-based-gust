#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# © 2021-2023 Israt Jahan, University of Connecticut. All rigts reserved.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime
import array as ar
from statistics import mean
from scipy import stats
import matplotlib as mpl
import joblib
from sklearn.metrics import mean_absolute_error

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
        a=RW_comb.iloc[i,23]
        row_index.append(i)
   
row_index=[0]+row_index+[len(RW_comb)]

#Number of events in the dataset
Total_events=48
#Number of events for test dataset
Test_storms=1
r=Total_events/Test_storms

arr=np.arange(0, Total_events, 1)

#following empty lists will be used to store error values of RF model on test data for each iteration
all_MSE_RF=[]
all_BIAS_RF=[]
all_RMSE_RF=[]
all_CRMSE_RF=[]
all_MAE_RF=[]
#following empty lists will be used to store error values of UPP test data for each iteration
all_MSE_UPP=[]
all_BIAS_UPP=[]
all_RMSE_UPP=[]
all_CRMSE_UPP=[]
all_MAE_UPP=[]
#following empty lists will be used to store error values of RF model on train data for each iteration
all_MSE_train=[]
all_BIAS_train=[]
all_RMSE_train=[]
all_CRMSE_train=[]
all_MAE_train=[]

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
    X_train=X_train.drop(['WG_ECMWF(m/s)','WindDC(degree)'],axis = 1)
    X_test=pd.DataFrame(test_data.loc[:,'PSFC(kPa)':'Pot_temp_grad(PBLH_sfc)'])
    X_test=X_test.drop(['WG_ECMWF(m/s)','WindDC(degree)'],axis = 1)
    UPP_test=pd.DataFrame(test_data['SfcWG_UPP(m/s)'])
    # Creating dataframe for targets
    Y_train=pd.DataFrame(train_data['WG_o'])
    Y_test=pd.DataFrame(test_data['WG_o'])
    # Creating dataframe for the input features by dropping highly correlated attributes
    X_train= X_train.drop(['PSFC(kPa)','POT_2m(C)','T2(K)','Pot_temp_grad(1km_sfc)','Pot_temp_grad(2km_sfc)','Pot_temp_grad(PBLH_sfc)'],axis=1)
    X_test=X_test.drop(['PSFC(kPa)','POT_2m(C)','T2(K)','Pot_temp_grad(1km_sfc)','Pot_temp_grad(2km_sfc)','Pot_temp_grad(PBLH_sfc)'],axis=1)
    
    #convert “Y_train” from a Pandas “Series” object into a NumPy array for the model 
    #to accept the target training data while fitting the model
    Y_train = np.array(Y_train)
    Y_train=Y_train.ravel()
    
    
    from sklearn.ensemble import RandomForestRegressor
    RF = RandomForestRegressor(random_state=10,n_jobs=-1)
    RF.fit(X_train, Y_train)
    Y_pred = RF.predict(X_test)
    Y_pred=Y_pred.reshape(len(Y_test),1)
    Y_test = np.asarray(Y_test)
    Y_pred=Y_pred.ravel()
    Y_test=Y_test.ravel()
    from sklearn.metrics import mean_squared_error
    MSE=mean_squared_error(Y_test,Y_pred)
    MSE=round(MSE,3)
    print("MSE on test data:",MSE)
    # Bias= Prediction-Observation
    # This is mean bias
    BIAS = np.mean(Y_pred-Y_test)
    BIAS=round(BIAS,3)
    print("Bias on test data:",BIAS)
    RMSE = mean_squared_error(Y_test,Y_pred)**0.5
    RMSE=round(RMSE,3)
    print("RMSE on test data:",RMSE)
    CRMSE = (RMSE**2-BIAS**2)**0.5
    CRMSE=round(CRMSE,3)
    print("CRMSE on test data:",CRMSE)
    MAE=mean_absolute_error(Y_test,Y_pred)
    MAE=round(MAE,3)
    print("MAE on test data:",MAE)
    
    #need to check how the model does on train data
    Y_pred_train=RF.predict(X_train)
    Y_pred_train=Y_pred_train.reshape(len(Y_train),1)
    
    #Error metrics on train dataset which should be closer to zero
    MSE_train=mean_squared_error(Y_train,Y_pred_train)
    MSE_train=round(MSE_train,3)
    print("MSE on train data:",MSE_train)
    # Bias= Prediction-Observation
    # This is mean bias
    BIAS_train = np.mean(Y_pred_train-Y_train)
    BIAS_train=round(BIAS_train,3)
    print("Bias on train data:",BIAS_train)
    RMSE_train = mean_squared_error(Y_train,Y_pred_train)**0.5
    RMSE_train=round(RMSE_train,3)
    print("RMSE on train data:",RMSE_train)
    CRMSE_train = (RMSE_train**2-BIAS_train**2)**0.5
    CRMSE_train=round(CRMSE_train,3)
    print("CRMSE on train data:",CRMSE_train)
    MAE_train=mean_absolute_error(Y_train,Y_pred_train)
    MAE_train=round(MAE_train,3)
    print("MAE on train data:",MAE_train) 
    
    fontszt   =12
    titlesize =12
    fontsz    =16
    line_x = np.arange(-1000,10000,10)
    c_min  = 1
    c_max  = 1000
    LWIDTH=2
    trp    = 0.6
    RTT    = 25.
    def Heat_bin_plots(MINXY,MAXXY,INCR,Y_pred,Y_test,c_min,c_max,
                       xlabel_log,ylabel_log,title_log,yticks_log):
        fig, ax = plt.subplots(1)
        bins    = (np.arange(MINXY,MAXXY+INCR,step=INCR),np.arange(MINXY,MAXXY+INCR,step=INCR))
        img = plt.hist2d(Y_test, Y_pred,norm=mpl.colors.LogNorm(), bins=bins, cmin = 1,cmap=plt.cm.jet)
        cbar=plt.colorbar(label="Density", orientation="vertical")
        plt.clim(c_min,c_max)
        cbar.set_ticks([1,10, 100, 1000])
        cbar.set_ticklabels(["1","10", "100", "1000"])
        plt.plot(line_x, line_x,color='black',linewidth=LWIDTH)
        slope, intercept, r_value, p_value, std_err = stats.linregress(Y_test,Y_pred)
        line_y = slope*line_x + intercept
        plt.plot(line_x, line_y,color='gray',linestyle='--',linewidth=LWIDTH)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05,0.95,"Cor. C. = "+str(round(r_value,2))+'\n' "Bias = "+str(BIAS)+'(m/s)'+'\n' "MAE = "+str(MAE)+'(m/s)'+'\n' "RMSE = "+str(RMSE)+'(m/s)'+'\n' "CRMSE = "+str(CRMSE)+'(m/s)'+'\n' "N Obs. = "+str(len(Y_test)),
                va='top', transform=ax.transAxes, fontsize = 10, color='black',bbox=props)
        if title_log==1:
            ax.set_title("Predicted wind gust(RF_baseline) vs. observed wind gust",fontsize = titlesize)
        if xlabel_log==1:
            ax.set_xlabel("Observed wind gust(m/s)", fontsize = titlesize)
        if ylabel_log==1:
            ax.set_ylabel("Predicted wind gust(m/s)",fontsize = titlesize )
        if yticks_log==0:
            ax.set_yticklabels([])
        plt.xticks(rotation=RTT)
        ax.tick_params(axis='both',direction='in')
        plt.grid(b=None, which='major', axis='both',linestyle=':')
        ax.set_xlim([MINXY,MAXXY])
        ax.set_ylim([MINXY,MAXXY]) 
    Heat_bin_plots(5,35,0.5,Y_pred,Y_test,c_min,c_max,1,1,1,1)

    #Feature importance_permutation method
    from sklearn.inspection import permutation_importance
    perm_importance = permutation_importance(RF, X_train, Y_train,scoring='neg_mean_squared_error')
    sorted_idx = perm_importance.importances_mean.argsort()
    plot1 = plt.figure(1)
    plt.barh(X_train.columns[sorted_idx], perm_importance.importances_mean[sorted_idx])
    plt.xlabel("Increase in MSE")
    plt.title("Permutation importance")
    
    all_MSE_RF.append(MSE)
    all_BIAS_RF.append(BIAS)
    all_RMSE_RF.append(RMSE)
    all_CRMSE_RF.append(CRMSE)
    all_MAE_RF.append(MAE)
    
    all_MSE_train.append(MSE_train)
    all_BIAS_train.append(BIAS_train)
    all_RMSE_train.append(RMSE_train)
    all_CRMSE_train.append(CRMSE_train)
    all_MAE_train.append(MAE_train)
    
    #Evaluation of WG_UPP correponding to the test data
    UPP_test=np.asarray(UPP_test)
    UPP_test=UPP_test.ravel()
    # This is mean bias of UPP predicted wind gust
    BIAS_UPP = np.mean(UPP_test-Y_test)
    BIAS_UPP=round(BIAS_UPP,3)
    print("Bias on WG_UPP:",BIAS_UPP)
    #This is MSE of UPP predicted wind gust
    MSE_UPP=mean_squared_error(Y_test,UPP_test)
    MSE_UPP=round(MSE_UPP,3)
    print("MSE on WG_UPP:",MSE_UPP)
    RMSE_UPP = mean_squared_error(Y_test,UPP_test)**0.5
    RMSE_UPP=round(RMSE_UPP,3)
    print("RMSE on WG_UPP:",RMSE_UPP)
    CRMSE_UPP = (RMSE_UPP**2-BIAS_UPP**2)**0.5
    CRMSE_UPP=round(CRMSE_UPP,3)
    print("CRMSE on WG_UPP:",CRMSE_UPP)
    MAE_UPP=mean_absolute_error(Y_test,UPP_test)
    MAE_UPP=round(MAE_UPP,3)
    print("MAE on WG_UPP:",MAE_UPP) 
    
    def Heat_bin_plots(MINXY,MAXXY,INCR,UPP_test,Y_test,c_min,c_max,
                       xlabel_log,ylabel_log,title_log,yticks_log):
        fig, ax = plt.subplots(1)
        bins    = (np.arange(MINXY,MAXXY+INCR,step=INCR),np.arange(MINXY,MAXXY+INCR,step=INCR))
        img = plt.hist2d(Y_test, UPP_test,norm=mpl.colors.LogNorm(), bins=bins, cmin = 1,cmap=plt.cm.jet)
        cbar=plt.colorbar(label="Density", orientation="vertical")
        plt.clim(c_min,c_max)
        cbar.set_ticks([1,10, 100, 1000])
        cbar.set_ticklabels(["1","10", "100", "1000"])
        plt.plot(line_x, line_x,color='black',linewidth=LWIDTH)
        slope_UPP, intercept_UPP, r_UPP, p_value, std_err = stats.linregress(Y_test,UPP_test)
        line_y_UPP = slope_UPP*line_x + intercept_UPP
        plt.plot(line_x, line_y_UPP,color='gray',linestyle='--',linewidth=LWIDTH)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05,0.95,"Cor. C. = "+str(round(r_UPP,2))+'\n' "Bias = "+str(BIAS_UPP)+'(m/s)'+'\n' "MAE = "+str(MAE_UPP)+'(m/s)'+'\n' "RMSE = "+str(RMSE_UPP)+'(m/s)'+'\n' "CRMSE = "+str(CRMSE_UPP)+'(m/s)'+'\n' "N Obs. = "+str(len(Y_test)),
                va='top', transform=ax.transAxes, fontsize = 10, color='black',bbox=props)
        if title_log==1:
            ax.set_title("UPP Predicted wind gust vs. observed wind gust",fontsize = titlesize)
        if xlabel_log==1:
            ax.set_xlabel("Observed wind gust(m/s)", fontsize = titlesize)
        if ylabel_log==1:
            ax.set_ylabel("UPP Predicted wind gust vs. observed wind gust",fontsize = titlesize )
        if yticks_log==0:
            ax.set_yticklabels([])
        plt.xticks(rotation=RTT)
        ax.tick_params(axis='both',direction='in')
        plt.grid(b=None, which='major', axis='both',linestyle=':')
        ax.set_xlim([MINXY,MAXXY])
        ax.set_ylim([MINXY,MAXXY])
    Heat_bin_plots(5,35,0.5,UPP_test,Y_test,c_min,c_max,1,1,1,1)
    
    all_MSE_UPP.append(MSE_UPP)
    all_BIAS_UPP.append(BIAS_UPP)
    all_RMSE_UPP.append(RMSE_UPP)
    all_CRMSE_UPP.append(CRMSE_UPP)
    all_MAE_UPP.append(MAE_UPP)
    
    
    #converting to dataframes so that I can save them as csv files
    Y_train=pd.DataFrame(Y_train,columns=['WG_o'])
    Y_test=pd.DataFrame(Y_test,columns=['WG_o'])
    Y_pred=pd.DataFrame(Y_pred,columns=['WG_pred_on_testset'])
    UPP_test=pd.DataFrame(UPP_test,columns=['UPP_for_testset'])
    #Saving the test, train, UPP and prediction dataframes as csv
    X_train.to_csv('X_train_'+str(j)+'.csv')
    Y_train.to_csv('Y_train_'+str(j)+'.csv')
    X_test.to_csv('X_test_'+str(j)+'.csv')
    Y_test.to_csv('Y_test_'+str(j)+'.csv')
    Y_pred.to_csv('Y_pred_'+str(j)+'.csv')
    UPP_test.to_csv('UPP_test_'+str(j)+'.csv')
    
    # save the model to disk
    #filename = r'/Volumes/Disk 2/Study/UCONN/Research/ML_WG_project/ML_code_output/Tuned_model_allX.sav'
    joblib.dump(RF,cd+'/LOSO_48storms_UPP_as_feature/LOSO_'+str(j)+'.sav')


# These are for test data
Avg_MSE_RF=mean(all_MSE_RF)
Avg_MSE_RF=round(Avg_MSE_RF,3)
Avg_MSE_RF = pd.Series(Avg_MSE_RF)
print("Avg_MSE over all the iterations:",Avg_MSE_RF)
Avg_BIAS_RF=mean(all_BIAS_RF)
Avg_BIAS_RF=round(Avg_BIAS_RF,3)
Avg_BIAS_RF= pd.Series(Avg_BIAS_RF)
print("Avg_BIAS over all the iterations:",Avg_BIAS_RF)
Avg_RMSE_RF=mean(all_RMSE_RF)
Avg_RMSE_RF=round(Avg_RMSE_RF,3)
Avg_RMSE_RF= pd.Series(Avg_RMSE_RF)
print("Avg_RMSE over all the iterations:",Avg_RMSE_RF)
Avg_CRMSE_RF=mean(all_CRMSE_RF) 
Avg_CRMSE_RF=round(Avg_CRMSE_RF,3)
Avg_CRMSE_RF= pd.Series(Avg_CRMSE_RF) 
print("Avg_CRMSE over all the iterations:",Avg_CRMSE_RF)
Avg_MAE_RF=mean(all_MAE_RF)
Avg_MAE_RF=round(Avg_MAE_RF,3)
Avg_MAE_RF=pd.Series(Avg_MAE_RF)
print("Avg_MAE over all the iterations:",Avg_MAE_RF)
Error_RF=pd.concat([Avg_MSE_RF,Avg_BIAS_RF,Avg_RMSE_RF,Avg_CRMSE_RF,Avg_MAE_RF],axis=0,ignore_index=True)
Error_RF=Error_RF.to_frame()
Names=["Avg_MSE", "Avg_BIAS","Avg_RMSE","Avg_CRMSE","Avg_MAE"]
Error_RF['Error_metric'] = Names

# These are for train data
Avg_MSE_train=mean(all_MSE_train)
Avg_MSE_train=round(Avg_MSE_train,3)
Avg_MSE_train = pd.Series(Avg_MSE_train)
print("Avg_MSE over all the iterations for train data:",Avg_MSE_train)
Avg_BIAS_train=mean(all_BIAS_train)
Avg_BIAS_train=round(Avg_BIAS_train,3)
Avg_BIAS_train = pd.Series(Avg_BIAS_train)
print("Avg_BIAS over all the iterations for train data:",Avg_BIAS_train)
Avg_RMSE_train=mean(all_RMSE_train)
Avg_RMSE_train=round(Avg_RMSE_train,3)
Avg_RMSE_train = pd.Series(Avg_RMSE_train)
print("Avg_RMSE over all the iterations for train data:",Avg_RMSE_train)
Avg_CRMSE_train=mean(all_CRMSE_train) 
Avg_CRMSE_train=round(Avg_CRMSE_train,3) 
Avg_CRMSE_train = pd.Series(Avg_CRMSE_train)
print("Avg_CRMSE over all the iterations for train data:",Avg_CRMSE_train)
Avg_MAE_train=mean(all_MAE_train)
Avg_MAE_train=round(Avg_MAE_train,3)
Avg_MAE_train=pd.Series(Avg_MAE_train)
print("Avg_MAE over all the iterations:",Avg_MAE_train)
Error_train=pd.concat([Avg_MSE_train,Avg_BIAS_train,Avg_RMSE_train,Avg_CRMSE_train,Avg_MAE_train],axis=0,ignore_index=True)
Error_train=Error_train.to_frame()
Error_train['Error_metric'] = Names


#these are for UPP test data
Avg_MSE_UPP=mean(all_MSE_UPP)
Avg_MSE_UPP=round(Avg_MSE_UPP,3)
Avg_MSE_UPP = pd.Series(Avg_MSE_UPP)
print("Avg_MSE over all the iterations for UPP test data:",Avg_MSE_UPP)
Avg_BIAS_UPP=mean(all_BIAS_UPP)
Avg_BIAS_UPP=round(Avg_BIAS_UPP,3)
Avg_BIAS_UPP = pd.Series(Avg_BIAS_UPP)
print("Avg_BIAS over all the iterations for UPP test data:",Avg_BIAS_UPP)
Avg_RMSE_UPP=mean(all_RMSE_UPP)
Avg_RMSE_UPP=round(Avg_RMSE_UPP,3)
Avg_RMSE_UPP = pd.Series(Avg_RMSE_UPP)
print("Avg_RMSE over all the iterations for UPP test data:",Avg_RMSE_UPP)
Avg_CRMSE_UPP=mean(all_CRMSE_UPP) 
Avg_CRMSE_UPP=round(Avg_CRMSE_UPP,3)
Avg_CRMSE_UPP = pd.Series(Avg_CRMSE_UPP) 
print("Avg_CRMSE over all the iterations for UPP test data:",Avg_CRMSE_UPP)
Avg_MAE_UPP=mean(all_MAE_UPP)
Avg_MAE_UPP=round(Avg_MAE_UPP,3)
Avg_MAE_UPP=pd.Series(Avg_MAE_UPP)
print("Avg_MAE over all the iterations:",Avg_MAE_UPP) 
Error_UPP=pd.concat([Avg_MSE_UPP,Avg_BIAS_UPP,Avg_RMSE_UPP,Avg_CRMSE_UPP,Avg_MAE_UPP],axis=0,ignore_index=True)
Error_UPP=Error_UPP.to_frame()
Error_UPP['Error_metric'] = Names

Error_RF.to_csv('Average_Error_RF_test.csv', index = False)
Error_train.to_csv('Average_Error_RF_train.csv', index = False)
Error_UPP.to_csv('Average_Error_UPP_test.csv', index = False)

