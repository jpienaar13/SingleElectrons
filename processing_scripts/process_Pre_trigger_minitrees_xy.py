import sys
import hax
import numpy as np
import pandas as pd
from pax import units, configuration, datastructure
from collections import defaultdict
import math
from multihist import Histdd, Hist1d
import pickle
from tqdm import tqdm

#Run Number to post-process
run_number=dataset=sys.argv[1]

#Main
pax_config = configuration.load_configuration('XENON1T')
R_tpc = pax_config['DEFAULT']['tpc_radius']

#Find and load processed minitree
hax.init(experiment='XENON1T', 
        pax_version_policy = '6.8.0',
        main_data_paths = ['/project2/lgrandi/xenon1t/processed', '/project/lgrandi/xenon1t/processed'],
        minitree_paths = ['/scratch/midway2/jpienaar/minitrees/',
                         '/project2/lgrandi/xenon1t/minitrees/pax_v6.8.0',
                         '/project/lgrandi/xenon1t/minitrees/pax_v6.8.0',
                         ],
        ) 

#Load File Location
cache_file_name = '/scratch/midway2/jpienaar/cache_files/'+run_number+ '_Pre_trigger.hdf5'
print (run_number, cache_file_name)
df = hax.minitrees.load(cache_file = cache_file_name)

#Binning
#large_window_length=10**9
#t_bins=np.linspace(0, large_window_length, 501)
#t_bin_width=t_bins[1]-t_bins[0]
r_bins=np.linspace(0, (2*R_tpc)**2, 101)

small_window_length=2*10**8
t_reduced_bins=np.linspace(0, small_window_length, 101)
t_bin_red_width=t_reduced_bins[1]-t_reduced_bins[0]

_spatial_bins=50
x_bins=np.linspace(-50, 50, _spatial_bins+1)
y_bins=np.linspace(-50, 50, _spatial_bins+1)
_xbins=np.linspace(-50, 50, _spatial_bins+1)
_ybins=np.linspace(-50, 50, _spatial_bins+1)
xv, yv = np.meshgrid(_xbins, _ybins)
array_hist = np.zeros((_spatial_bins, _spatial_bins))
xc = np.zeros_like(array_hist)
yc = np.zeros_like(array_hist)
dict_hist={}
                      
#Empty Dict for Results

for idx, x_edge in tqdm(enumerate(xv[:-1])):
    for idy, y_edge in enumerate(yv[:-1]):
        df_xy=df.loc[(df.x_s2_tpf>xv[idx][idy])&(df.x_s2_tpf<xv[idx][idy+1])
                     &(df.y_s2_tpf>yv[idx][idy])&(df.y_s2_tpf<yv[idx+1][idy])]
        unique_s2s=pd.unique(df_xy[['s2_time', 'x_s2_tpf', 'y_s2_tpf']].values)
        num_events=len(unique_s2s)
        
        #Define Blank Hists
        dt_r2_histogram=Histdd(bins=[t_reduced_bins,
                                 r_bins,
                                 ], 
                            axis_names=['delta_T',
                                        'r_dist',
                                        ]) 

        xy_histogram = Histdd(bins=[t_reduced_bins,
                               x_bins,
                               y_bins,
                               ], 
                            axis_names=['delta_T',
                                     'x_p_pos',
                                     'y_p_pos',
                                     ]) 
        
        xc[idx, idy]=xv[idx][idy]+(xv[idx][idy+1]-xv[idx][idy])/2
        yc[idx, idy]=yv[idx][idy]+(yv[idx+1][idy]-yv[idx][idy])/2
        
        for s2 in unique_s2s[:num_events]:
            live_time=0

            #Find all peaks within 1s
            window_length=small_window_length
            temp_df = df_xy.loc[(df_xy.global_time>(s2[0]))&(df_xy.global_time<(s2[0]+window_length))]

            #Count up livetime considered within 1ms
            subsequent_events=pd.unique(temp_df[['s1_time', 'event_start']].values)
            for event in subsequent_events:
                live_time += (event[0]-event[1])

            #Some Extra Branches
            r_s2=s2[1]**2+s2[1]**2

            temp_df['x_dist'] = temp_df['x_p_tpf']-s2[1]
            temp_df['y_dist'] = temp_df['y_p_tpf']-s2[2]
            temp_df['r_dist'] = np.sqrt(temp_df['x_dist']**2+temp_df['y_dist']**2)

            temp_df['alpha'] = np.arccos((temp_df['r_dist']**2+r_s2**2-R_tpc**2)/(2*r_s2*temp_df['r_dist']))
            temp_df['area_norm'] = temp_df['alpha'].fillna(np.pi) 
            temp_df['norm'] = temp_df['area_norm']/np.pi

            ##Fill TR_Dist Histogram
            histogram=Histdd(temp_df['global_time'].values-s2[0],
                             (temp_df['r_dist'].values)**2,
                             weights=(window_length)/(t_bin_red_width*(window_length-live_time))/temp_df['norm'].values,
                             axis_names=['delta_T',
                                         'r_dist',
                                         ], 
                             bins=[t_reduced_bins,
                                   r_bins,
                                   ])
            dt_r2_histogram+=histogram
            
            #Pos Hist
            histogram=Histdd(temp_df['global_time'].values-s2[0],
                             temp_df['x_p_tpf'].values, 
                             temp_df['y_p_tpf'].values,
                             weights=[(window_length)/(t_bin_red_width*(window_length-live_time))]*len(temp_df),
                             axis_names=['delta_T',
                                         'x_p_pos',
                                         'y_p_pos',
                                         ], 
                             bins=[t_reduced_bins,
                                   x_bins,
                                   y_bins,
                                   ])
            xy_histogram+=histogram
    
        #Renormalize to number of primary S2s considered    
        dt_r2_histogram=dt_r2_histogram/num_events
        xy_histogram=xy_histogram/num_events
        #print(xc[idx, idy], yc[idx, idy])
        dict_hist['%d_%d' %(xc[idx, idy], yc[idx, idy])] = {'dt_r2':dt_r2_histogram, 'xy': xy_histogram, 
                                                            'x': xc[idx, idy], 'y': yc[idx, idy], 'events': len(unique_s2s)}

with open('/scratch/midway2/jpienaar/cache_files/%s_xy.pkl' %run_number, 'wb') as handle:
#with open('/scratch/midway2/jpienaar/cache_files/%s.pkl' %run_number, 'wb') as handle:
    pickle.dump(dict_hist, handle, protocol=pickle.HIGHEST_PROTOCOL)


