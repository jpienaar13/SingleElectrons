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
from Pre_trigger_peaks import Primary_Times

#Run Number to post-process
run_number=dataset=sys.argv[1]

#Init Hax and Studd
hax.init(experiment='XENON1T', 
        pax_version_policy = '6.8.0',
        main_data_paths = ['/project2/lgrandi/xenon1t/processed', '/project/lgrandi/xenon1t/processed'],
        minitree_paths = ['/scratch/midway2/jpienaar/minitrees/',
                         '/project2/lgrandi/xenon1t/minitrees/pax_v6.8.0',
                         '/project/lgrandi/xenon1t/minitrees/pax_v6.8.0',
                         ],
        ) 
    
#Set window to look for previous S2s
window_length=10**9

#Main
pax_config = configuration.load_configuration('XENON1T')
R_tpc = pax_config['DEFAULT']['tpc_radius']

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

#Determine Cut Pass/Fail for whole DF
df = aft_peak_cut(df)

#Load Minitree with Primary S2 information
primaries = hax.minitrees.load(run_number,
                         treemakers=[Primary_Times, 'Basics'],
                         preselection=None,
                         force_reload=True,
                                   )


pre_existing_fields = []
for field in df.columns:
    pre_existing_fields.append(field)
primary_fields=['x_dist', 'y_dist', 'delay', 's2']


#For each unique s2 investigate subsquent S2s
num_s2s=len(primaries)
#loop_count=0
new_df=[]
for key, event in tqdm(df.iterrows()):    
    primaries['delay'] = event.global_time - primaries.s2_time
    temp_primaries = primaries.loc[(primaries.delay<window_length)&(primaries.delay>0)]
    #print(len(temp_primaries))
    temp_primaries = temp_primaries.sort_values(by=['delay'])
    temp_primaries['x_dist'] = temp_primaries['x_s2_tpf']-event.x_p_tpf
    temp_primaries['y_dist'] = temp_primaries['y_s2_tpf']-event.y_p_tpf

    
    row_entry={}
    for field in pre_existing_fields:
        row_entry[field]=event[field]
        
    primary_index=1
    for key, primary in temp_primaries.iterrows():
        for field in primary_fields:
            row_entry['%i_%s' %(primary_index, field)]=primary[field]
        primary_index+=1
    
    new_df.append(row_entry)
    #loop_count+=1
    #if loop_count>1000:
    #    break

new_df = pd.DataFrame(new_df)

new_df.to_hdf('/scratch/midway2/jpienaar/cache_files/%s_Delay.hdf5' %(run_number), key='df') 