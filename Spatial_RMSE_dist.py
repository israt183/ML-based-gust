#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 13:47:18 2022

@author: itu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime
import array as ar
from statistics import mean
from scipy import stats
import shapely
import cartopy.crs as crs
import cartopy.feature as cfeature
from sklearn.metrics import mean_squared_error
from matplotlib.cm import get_cmap

#from cartopy.feature import NaturalEarthFeature
#import matplotlib.ticker as mticker
#from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

cd=os.getcwd()
Path_merged=cd+'/Combined_obsWRF/'
#Path_station='/Volumes/Disk 2/Study/UCONN/Research/ISD_data/'
#importing the csv files
RW1=pd.read_csv(Path_merged+"Sorted_merged_obs_WRF_10.csv",sep=',')
RW2=pd.read_csv(Path_merged+"Sorted_merged_obs_WRF_14.csv",sep=',')
RW3=pd.read_csv(Path_merged+"Sorted_merged_obs_WRF_24.csv",sep=',')


# Remove column 'index'
RW1=RW1.drop(['index'],axis = 1)
RW2=RW2.drop(['index'],axis = 1)
RW3=RW3.drop(['index'],axis = 1)


RW_comb=pd.concat([RW1,RW2,RW3],axis=0,ignore_index=True)
New_RW=RW_comb.loc[:,["Station_x","WG_o","SfcWG_UPP(m/s)"]]

#to calculate how many WG observaion each station has
count_WG = RW_comb.groupby(['Station_x', 'WG_o']).size().reset_index(name="Times")
count_WG=count_WG.rename(columns={"Station_x" : "Station"})
Unique_stations=count_WG['Station'].unique()
Freq_staions_WG =np.zeros((len(Unique_stations), 2),dtype=np.int64)

for x in range(len(Unique_stations)): 
    Freq_staions_WG[x,0] = Unique_stations[x]
    Freq_staions_WG[x,1]=count_WG.loc[count_WG['Station'] == Freq_staions_WG[x,0] , 'Times'].sum()
 
#converting the arry to dataframe
df_freq_WG = pd.DataFrame(Freq_staions_WG, columns = ['Station','WG_freq'])
#df_freq_WG.to_excel('WG_Frequency_48events.xlsx')

New_RW=New_RW.rename(columns={"Station_x" : "Station"})
#create an empty dataframe to store average RMSE of individual stations
Holder=np.zeros([len(Unique_stations),2])

for y in range(len(Unique_stations)):
    UPP_gust= New_RW.loc[New_RW['Station'] == Unique_stations[y],'SfcWG_UPP(m/s)']
    Obs_gust= New_RW.loc[New_RW['Station'] == Unique_stations[y],'WG_o']
    RMSE=mean_squared_error(Obs_gust,UPP_gust)**0.5
    RMSE=round(RMSE,3)
    Holder[y,0]=Unique_stations[y]
    Holder[y,1]=RMSE
    
df = pd.DataFrame(Holder, columns = ['Station','Avg_RMSE'])
df= df.astype({'Station': 'int64','Avg_RMSE': 'float64' })

Station_info=pd.read_csv("ISD_Coords_mountWashingtonexcluded.csv",sep=',')
Station_info=Station_info.rename(columns={"File_Name" : "Station"})
Data = df.merge(Station_info,on='Station',how = 'inner')


figure = plt.figure(figsize=(8,6))
ax = figure.add_subplot(1,1,1, projection=crs.Mercator())
# adds a stock image as a background
#ax.stock_img()
# adds national borders
ax.add_feature(cfeature.BORDERS)
# add coastlines
ax.add_feature(cfeature.COASTLINE)
#sequence for ax.set_extent: min lon,max lon,min lat,max lat
ax.set_extent([-79,-68,38 ,47,],crs=crs.PlateCarree())
#ax.set_extent([-85,-60,35 ,50,],crs=crs.PlateCarree())
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.LAND, color='white',edgecolor='black')
ax.add_feature(cfeature.STATES,linewidth=0.5, edgecolor='black')
ax.add_feature(cfeature.RIVERS)
ax.gridlines()
#plt.show()
plt.scatter(
    Data["Lon"],
    Data["Lat"],
    c=Data["Avg_RMSE"],
    s=50,
    cmap="hsv",
    vmin=0,
    vmax=8,
    transform=crs.PlateCarree(),
)
plt.colorbar().set_label("Avgerage wind gust RMSE (m/s) by WRF ")
plt.savefig('Avg_RMSE_over_48storms_hsv.png',dpi=300,bbox_inches='tight')


