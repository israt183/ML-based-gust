#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 19:53:57 2022

@author: itu
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import datetime 
from os import listdir
#from matplotlib import pyplot as plt
#from itertools import compress
from datetime import timedelta
#from sklearn.metrics import r2_score,mean_squared_error
import time
import seaborn as sns
import os

cd=os.getcwd()
#FIXME dataframe for test storm
df1=pd.read_csv(cd+"/test_storm_1.csv",sep=',')
# =============================================================================
#df2=pd.read_csv(cd+"/test_storm_2.csv",sep=',')
#df3=pd.read_csv(cd+"/test_storm_3.csv",sep=',')
# =============================================================================
# =============================================================================
# df1['Storm'] = 'TS_1'
# df2['Storm'] = 'TS_2'
# df3['Storm'] = 'TS_3'
# =============================================================================
#df=pd.concat([df1,df2,df3],axis=0) 
Station_list= np.array([72504614707,72508714752])
Station_name=np.array(['KGON','KHFD'])
for i in range(len(Station_list)):
    Single_station= df1[df1['Station_x']==Station_list[i]]
    #Single_station_obs= WG_obs[WG_obs['Station']==Station_list[i]]
    #Time=Single_station.loc[:,'Valid_Time_x']
    #WG_timeseries_UPP=Single_station.loc[:,['Valid_Time_y','SfcWG_UPP(m/s)']]
    WG_timeseries_obs=Single_station.loc[:,['Valid_Time_y','WG_o']]
    #FIXME which feature to use for timeseries
    Feature=Single_station.loc[:,['Valid_Time_y','PBLH(km)']]
# =============================================================================
#     WG_timeseries_UPP['Valid_Time_y'] = pd.to_datetime(WG_timeseries_UPP['Valid_Time_y'],
#                                             format='%Y%m%d%H')
# =============================================================================
    WG_timeseries_obs['Valid_Time_y'] = pd.to_datetime(WG_timeseries_obs['Valid_Time_y'],
                                            format='%Y%m%d%H')
    Feature['Valid_Time_y'] = pd.to_datetime(Feature['Valid_Time_y'],
                                            format='%Y%m%d%H')
# =============================================================================
#     x1=WG_timeseries_UPP.loc[:,['Valid_Time_y']]
#     y1=WG_timeseries_UPP.loc[:,['SfcWG_UPP(m/s)']]
# =============================================================================
    x2=WG_timeseries_obs.loc[:,['Valid_Time_y']]
    y2=WG_timeseries_obs.loc[:,['WG_o']]
    x3=Feature.loc[:,['Valid_Time_y']]
    #FIXME which feature to use for timeseries
    y3=Feature.loc[:,['PBLH(km)']]
    fig, ax = plt.subplots(1)
    #fig.tight_layout(h_pad=12,w_pad=5)
    #fig.tight_layout()
    fig.autofmt_xdate()
    #ax.scatter(x1, y1,s=4, marker='o',label='WG_UPP')
    ax.scatter(x2,y2,s=4,marker='^', label='WG_obs')
    #FIXME change the label according to feature
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
    #xlabels = ax[0,0].get_xticklabels()
    #ax[0,0].set_xticklabels(ax.get_xticklabels, rotation=45, ha='right')
    #ax[0,0].tick_params(axis='x', labelrotation=45)
    #fig.autofmt_xdate()
    # automaticall set font and rotation for date tick labels
    #plt.gcf().autofmt_xdate()
    plt.axhline(y = 15, color = 'r', linestyle = '-')
    ax.set_xlabel('Time (UTC)')
    ax.set_ylabel('Wind(m/s)')
    #plt.title('Station_'+str(Stations_tseries[i]))
    #ax.set_title('Storm data for station_'+str(Station_list[i]))
    ax.set_title('Storm data for station_'+Station_name[i])
# =============================================================================
#     plt.savefig('WG_UPP_timeseries.png',dpi=300,
#                 bbox_inches='tight')
# =============================================================================
    
    #plt.show()
# =============================================================================
#     plt.savefig('TS_1_obs_WS950mb'+str(Station_list[i])+'.png',dpi=300,
#                 bbox_inches='tight')
# =============================================================================
    #FIXME test storm name and feature
    plt.savefig('TS_1_obs_PBLH_'+Station_name[i]+'.png',dpi=300,
                bbox_inches='tight')