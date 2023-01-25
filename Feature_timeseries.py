#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import datetime 
from os import listdir
from datetime import timedelta
import time
import seaborn as sns
import os

cd=os.getcwd()
df1=pd.read_csv(cd+"/test_storm_1.csv",sep=',')

Station_list= np.array([72504614707,72508714752])
Station_name=np.array(['KGON','KHFD'])
for i in range(len(Station_list)):
    Single_station= df1[df1['Station_x']==Station_list[i]]
    WG_timeseries_obs=Single_station.loc[:,['Valid_Time_y','WG_o']]
    Feature=Single_station.loc[:,['Valid_Time_y','PBLH(km)']]
    WG_timeseries_obs['Valid_Time_y'] = pd.to_datetime(WG_timeseries_obs['Valid_Time_y'],
                                            format='%Y%m%d%H')
    Feature['Valid_Time_y'] = pd.to_datetime(Feature['Valid_Time_y'],
                                            format='%Y%m%d%H')
    x2=WG_timeseries_obs.loc[:,['Valid_Time_y']]
    y2=WG_timeseries_obs.loc[:,['WG_o']]
    x3=Feature.loc[:,['Valid_Time_y']]
    y3=Feature.loc[:,['PBLH(km)']]
    fig, ax = plt.subplots(1)
    fig.autofmt_xdate()
    ax.scatter(x2,y2,s=4,marker='^', label='WG_obs')
    ax.scatter(x3,y3,s=4, marker='s',label='PBLH(km)')
    ax.set_facecolor('white')
    ax.patch.set_edgecolor('k')  
    ax.patch.set_linewidth('2')
    ax.grid(b=True,which='both', axis='both',linestyle=':', linewidth=1, color='k', alpha=0.6,zorder=4)
    leg = ax.legend()
    # Adjust the x-axis
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6)) # Month intervals
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d-%H')) # date formatting
    ax.set_ylim([0, 40])
    # automatically set font and rotation for date tick labels
    plt.gcf().autofmt_xdate()
    plt.axhline(y = 15, color = 'r', linestyle = '-')
    ax.set_xlabel('Time (UTC)')
    ax.set_ylabel('Wind(m/s)')
    ax.set_title('Storm data for station_'+Station_name[i])
    plt.savefig('TS_1_obs_PBLH_'+Station_name[i]+'.png',dpi=300,
                bbox_inches='tight')
