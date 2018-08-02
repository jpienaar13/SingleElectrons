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

#Load AFT Cut Values
with open('/home/jpienaar/SingleElectrons/aft_fit_values.pkl', 'rb') as handle:
    dict_aft_fits = pickle.load(handle)

#Define AFT Cut    
def aft_peak_cut(df):
    aft_means=np.array(dict_aft_fits['Background']['means'])
    aft_sigmas=np.array(dict_aft_fits['Background']['sigmas'])
    bins=dict_aft_fits['Background']['bins']
    df['CutPeakAFT']= True ^ (df.area_fraction_top>np.interp(np.log10(df.area), bins, aft_means-3*aft_sigmas)) ^ (df.area_fraction_top<np.interp(np.log10(df.area), bins, aft_means+3*aft_sigmas)) 
    return df

#Determine Cut PAss/Fail for whole DF
df = aft_peak_cut(df)

#Binning
window_length=10**8
t_bins     = np.linspace(0, window_length, 201)
t_bin_width= t_bins[1:]-t_bins[:-1]
r_bins     = np.linspace(0, (R_tpc)**2, 101)
dist_bins  = np.linspace(-R_tpc, R_tpc, 101)
s2_bins    = np.linspace(2, 6, 51)
s2_p_bins  = np.linspace(0, 4, 51)

#Define Blank Hists
livet_histogram=Histdd(bins=[t_bins,
                             s2_bins,
                             ], 
                        axis_names=['delta_T',
                                    's2_area',
                                    ]) 

livet_weights_histogram=Histdd(bins=[t_bins,
                             s2_bins,
                             ], 
                        axis_names=['delta_T',
                                    's2_area',
                                    ]) 

events_histogram=Histdd(bins=[t_bins,
                              s2_bins,
                             ], 
                        axis_names=['delta_T',
                                    's2_area'
                                    ]) 

weights_histogram=Histdd(bins=[t_bins,
                              s2_bins,
                             ], 
                        axis_names=['delta_T',
                                    's2_area'
                                    ]) 


peaks_histogram=Histdd(bins=[t_bins,
                             s2_p_bins,
                             ], 
                        axis_names=['delta_T',
                                    'peak_area'
                                    ]) 


dt_r2_histogram=Histdd(bins=[t_bins,
                             r_bins,
                             ], 
                        axis_names=['delta_T',
                                    'r_dist',
                                    ]) 

weights_dt_r2_histogram=Histdd(bins=[t_bins,
                             r_bins,
                             ], 
                        axis_names=['delta_T',
                                    'r_dist',
                                    ]) 


dt_xy_histogram=Histdd(bins=[t_bins,
                           dist_bins,
                           dist_bins,
                           ],
                    axis_names=['delta_T',
                                 'x_dist',
                                 'y_dist',
                                 ])

#Find all unique primary S2 events
all_events=pd.unique(df['event_number'].values)
unique_s2s=[]
#num_s2s=np.min([500000, len(all_events)])
num_s2s=len(all_events)
for s2 in tqdm(all_events[:num_s2s]):
    s2_df=df[df.event_number==s2].iloc[0]
    unique_s2s.append([s2_df.s2_time, s2_df.x_s2_tpf, s2_df.y_s2_tpf, s2_df.s2])

#For each unique s2 investigate subsquent S2s
num_s2s=len(unique_s2s)
for s2 in tqdm(unique_s2s):
    #window_length=window_length
    temp_df = df.loc[(df.global_time>(s2[0]))&(df.global_time<(s2[0]+window_length))]
    
    #Find unique events within 1ms
    subsequent_events=pd.unique(temp_df['event_number'].values)
    event_info=[]
    for event_id in subsequent_events:
        event_df=temp_df[temp_df.event_number==event_id].iloc[0]
        event_info.append([np.nanmin([event_df.s1_time, event_df.s2_time]), event_df.event_start])
    
    #Binning Info of events
    bin_allocation_start = np.digitize([x[1]-s2[0] for x in event_info], bins=t_bins)
    bin_allocation_end = np.digitize([x[0]-s2[0] for x in event_info], bins=t_bins)
    bin_difference=bin_allocation_end-bin_allocation_start
    
    #Determine Live_time conisdered in each time bin
    start_time=[x[1]-s2[0] for x in event_info]
    end_time=[np.min([x[0]-s2[0], window_length]) for x in event_info]
    
    live_time_array=[]
    #Bin number in digitize of by 1
    for idx, bin_allocation in enumerate(bin_difference):
        if bin_allocation == 0:
            live_time_array.append([t_bins[bin_allocation_start[idx]-1], 
                                    (end_time[idx]-start_time[idx])/t_bin_width[bin_allocation_start[idx]-1]])
        elif bin_allocation == 1:
            live_time_array.append([t_bins[bin_allocation_start[idx]-1], 
                                   (t_bins[bin_allocation_start[idx]]-start_time[idx])/t_bin_width[bin_allocation_start[idx]-1]])
            if bin_allocation_start[idx]<len(t_bins)-1:
                live_time_array.append([t_bins[bin_allocation_start[idx]], 
                                    (end_time[idx]-t_bins[bin_allocation_start[idx]])/t_bin_width[bin_allocation_start[idx]]])
        elif bin_allocation >1:
            live_time_array.append([t_bins[bin_allocation_start[idx]-1], 
                                   (t_bins[bin_allocation_start[idx]]-start_time[idx])/t_bin_width[bin_allocation_start[idx]-1]])
            for index in range(bin_allocation-1):
                if bin_allocation_start[idx]+index<len(t_bins)-1:
                    live_time_array.append([t_bins[bin_allocation_start[idx]+index],1])
            if bin_allocation_start[idx]+(bin_allocation-1)<len(t_bins)-1:
                live_time_array.append([t_bins[bin_allocation_start[idx]+(bin_allocation-1)], 
                                        (end_time[idx]-t_bins[bin_allocation_start[idx]+(bin_allocation-1)])/t_bin_width[bin_allocation_start[idx]+(bin_allocation-1)]])
    
    #Apply AFT Cut
    temp_df=hax.cuts.selection(temp_df, temp_df['CutPeakAFT'], "CutPeakAFT")
    
    #Some Extra Branches
    r_s2=s2[1]**2+s2[1]**2
    
    temp_df['x_dist'] = temp_df['x_p_tpf']-s2[1]
    temp_df['y_dist'] = temp_df['y_p_tpf']-s2[2]
    temp_df['r_dist'] = np.sqrt(temp_df['x_dist']**2+temp_df['y_dist']**2)
    
    temp_df['alpha'] = np.arccos((temp_df['r_dist']**2+r_s2**2-R_tpc**2)/(2*r_s2*temp_df['r_dist']))
    temp_df['area_norm'] = temp_df['alpha'].fillna(np.pi) 
    temp_df['norm'] = temp_df['area_norm']/np.pi
    
    #Binning for bin width rate correction
    peak_bins=np.digitize(temp_df['global_time'].values-s2[0], t_bins)
  
    
    #Fill Live_T Histogram
    histogram=Histdd([x[0]+1 for x in live_time_array],
                     [np.log10(s2[3])]*len(live_time_array),
                     weights=[x[1] for x in live_time_array],
                     axis_names=['delta_T',
                                 's2_area',
                                 ], 
                     bins=[t_bins,
                           s2_bins,
                           ])
    livet_histogram+=histogram

    #Fill Live_T Histogram
    histogram=Histdd([x[0]+1 for x in live_time_array],
                     [np.log10(s2[3])]*len(live_time_array),
                     weights=[x[1]**2 for x in live_time_array],
                     axis_names=['delta_T',
                                 's2_area',
                                 ], 
                     bins=[t_bins,
                           s2_bins,
                           ])
    livet_weights_histogram+=histogram

    
    #Fill Events Histogram according to Peak Size
    histogram=Histdd(temp_df['global_time'].values-s2[0],
                     np.log10(temp_df['area'].values),
                     weights=1/t_bin_width[peak_bins-1],
                     axis_names=['delta_T',
                                 'peak_area',
                                 ], 
                     bins=[t_bins,
                           s2_p_bins,
                           ])
    peaks_histogram+=histogram
    
    #Fill Events Histogram according to S2 Size
    histogram=Histdd(temp_df['global_time'].values-s2[0],
                     [np.log10(s2[3])]*len(temp_df),
                     weights=1/t_bin_width[peak_bins-1],
                     axis_names=['delta_T',
                                 's2_area',
                                 ], 
                     bins=[t_bins,
                           s2_bins,
                           ])
    events_histogram+=histogram
    
    #Fill weights histogram for Events Histogram according to S2 Size
    histogram=Histdd(temp_df['global_time'].values-s2[0],
                     [np.log10(s2[3])]*len(temp_df),
                     weights=(1/t_bin_width[peak_bins-1])**2,
                     axis_names=['delta_T',
                                 's2_area',
                                 ], 
                     bins=[t_bins,
                           s2_bins,
                           ])
    weights_histogram+=histogram
            
    ##Fill TR_Dist Histogram
    histogram=Histdd(temp_df['global_time'].values-s2[0],
                     (temp_df['r_dist'].values)**2,
                     weights=1/temp_df['norm'].values,
                     axis_names=['delta_T',
                                 'r_dist',
                                 ], 
                     bins=[t_bins,
                           r_bins,
                           ])
    dt_r2_histogram+=histogram
    
    ##Fill Weight TR_Dist Histogram
    histogram=Histdd(temp_df['global_time'].values-s2[0],
                     (temp_df['r_dist'].values)**2,
                     weights=(1/temp_df['norm'].values)**2,
                     axis_names=['delta_T',
                                 'r_dist',
                                 ], 
                     bins=[t_bins,
                           r_bins,
                           ])
    weights_dt_r2_histogram+=histogram
    
    ##Fill TXY_Dist Histogram
    histogram=Histdd(temp_df['global_time'].values-s2[0],
                     temp_df['x_dist'].values,
                     temp_df['y_dist'].values,
                     axis_names=['delta_T',
                                 'x_dist',
                                 'y_dist',
                                 ], 
                     bins=[t_bins,
                           dist_bins,
                           dist_bins,
                           ])
    dt_xy_histogram+=histogram
        
#Store in Dict for Later
dict_hist={'version' : 1.0,
           'deltat':events_histogram, 'deltat_weights':weights_histogram,
           'peaks': peaks_histogram, 
           'livet_hist': livet_histogram, 'livet_weights': livet_weights_histogram, 
           'dt_r2':dt_r2_histogram, 'dt_r2_weights':weights_dt_r2_histogram,
           'dt_xy':dt_xy_histogram, 'events': num_s2s}

with open('/scratch/midway2/jpienaar/cache_files/%s_dt.pkl' %run_number, 'wb') as handle:
#with open('/scratch/midway2/jpienaar/cache_files/%s.pkl' %run_number, 'wb') as handle:
    pickle.dump(dict_hist, handle, protocol=pickle.HIGHEST_PROTOCOL)


