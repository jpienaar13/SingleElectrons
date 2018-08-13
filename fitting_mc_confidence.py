import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import colors
import matplotlib.gridspec as gridspec
from multihist import Histdd, Hist1d
import pandas as pd
import pickle
from tqdm import tqdm
import scipy
from scipy import stats
import sys

#Input Fit Values:
known_c = float(sys.argv[1])
known_tau = float(sys.argv[2])
known_source = sys.argv[3]
file_iteration =  int(sys.argv[4])

#Load S2 size histograms (produced from data):
with open('/home/jpienaar/SingleElectrons/s2_spectrum_per_source_sr1.pkl', 'rb') as handle:
    sampling_dict = pickle.load(handle)

#Define Single Exponential
def rate_function(scaling_constant, time_scale, dt_bins):
    y = scaling_constant*np.exp(-dt_bins/time_scale)
    return (y)

#Define "Waveform" Function
def se_histogram(event_rate, # Number of events assumed to occur within 1s
                 background_rate, # Flat background rate of Single Electrons 
                 time_scale, # Time Scale of exponential decay of trapped electrons
                 scaling_constant, #Fraction of electrons relative to S2 size 
                 source = 'Background', # Primary S2 spectrum for sampling from
                 time_bins=100, # Number of bins in time
                 window_length=1000 #Length after given S2 to look at
                ):
    
    #Determine number of events in timw window
    number_of_events = np.random.poisson(event_rate) #Event rate in window....
    
    #Assign random times to S2s in window following S2, leaving first S2 at t=0 
    if number_of_events>1:
        time_of_s2s=np.random.random(number_of_events-1)
        time_of_s2s=np.concatenate(([0], time_of_s2s))
    else:
        number_of_events = 1
        time_of_s2s=np.array([0])
    
    bin_width=window_length/time_bins
    
    #Sample from Appropriate S2 Size Spectrum (produced previously)
    s2_spectrum_hist = sampling_dict[source]['hist']    
    s2_size = np.random.choice(s2_spectrum_hist.bin_centers, 
                               number_of_events, 
                               p =(s2_spectrum_hist[:])/np.sum(s2_spectrum_hist[:]))
    s2_size=10**s2_size
    
    #Initialize Arrays of SE rate
    se_rate=[background_rate]*time_bins
    dt_bins=np.linspace(0, window_length, time_bins+1)
    
    #Loop over all primary S2s in waveform
    for is2, s2 in enumerate(time_of_s2s):
        s2 = s2 * window_length # Span window length, assumed to be in ms
        bin_offset = np.digitize(s2, dt_bins) 
        counts = rate_function(scaling_constant = scaling_constant*s2_size[is2], 
                               time_scale=time_scale, 
                               dt_bins=dt_bins)
        event_len =  time_bins-bin_offset
        zero_array = np.array([0]*int(bin_offset))
        counts = np.concatenate((np.array(zero_array), counts[:event_len]))
        se_rate += counts
    
    dt_bin_centers = dt_bins[1:]-(dt_bins[1:]-dt_bins[:-1])/2
    
    return dt_bin_centers, se_rate
 
##    
##Produce "True" Waveform:
##
waveforms = 10000
time_bins = 200
window_length=100
dict_mc_known={}
   
#Setup Bins
x_bins = [0]*time_bins
y_bins = [0]*time_bins
    
#Set source rate based on data and window length
source_rate = sampling_dict[known_source]['event_rate']
event_rate = source_rate*window_length/1000 #Event rate in s, window length in ms
    
for sim in tqdm(range(waveforms)):
    dt_bin_centers, se_rate = se_histogram(event_rate = event_rate, 
                                           background_rate=0, 
                                           time_scale = known_tau,
                                           scaling_constant = known_c, 
                                           source = known_source,
                                           time_bins=time_bins,
                                           window_length = window_length)
    y_bins += se_rate
x_bins = dt_bin_centers
true_waveform=y_bins/waveforms

##
##Fit "True" Waveform to Toy MC and find best fit_value.
##
#Store fit values for parameter being scanned
scaling_constant_values=np.linspace(1*10**-5, 1*10**-4, 51)
time_scale_values=np.linspace(10, 50, 41)

#Settings for ToyMC for other parameters
source=known_source
source_rate = sampling_dict[source]['event_rate']
background_rate=0

#Match to MC Itself,  
rate_obs = true_waveform

#Reduce number of waveform for time reasons
waveforms =1000
mc_iterations=100

best_fit_chi =[]
best_fit_p   =[]
best_fit_c   =[]
best_fit_tau =[]
for it in tqdm(range(mc_iterations)):
    fit_values=[]
    p_values=[]
    for  scaling_constant in scaling_constant_values:
        temp_array=[]
        temp_p_array=[]
        for  time_scale in time_scale_values:
            #Initialize Bins
            x_bins = [0]*time_bins
            y_bins = [0]*time_bins

            #Run 1000 TImes
            event_rate = source_rate*window_length/1000
            for sim in range(waveforms):
                dt_bin_centers, se_rate = se_histogram(event_rate = event_rate, 
                                                        background_rate=background_rate, 
                                                        time_scale = time_scale,
                                                        scaling_constant = scaling_constant, 
                                                        source = source,
                                                        time_bins=time_bins,
                                                        window_length = 100)
                y_bins += se_rate/waveforms
            rate_exp = y_bins

            fit_value, fit_probability = scipy.stats.chisquare(rate_obs[9:198], rate_exp[9:198])
            temp_array = np.append(temp_array, fit_value)
            temp_p_array = np.append(temp_p_array, fit_probability)
        fit_values.append(temp_array)
        p_values.append(temp_p_array)
    fit_values=np.array(fit_values)
    p_values=np.array(p_values)     

    i_min = np.unravel_index(fit_values.argmin(), fit_values.shape)
    best_fit_chi.append(fit_values[i_min[0]][i_min[1]])
    best_fit_p.append(p_values[i_min[0]][i_min[1]])
    best_fit_c.append(scaling_constant_values[i_min[0]])
    best_fit_tau.append(time_scale_values[i_min[1]])


results_dict = {'chi': best_fit_chi, 'p': best_fit_p, 'c': best_fit_c, 'tau': best_fit_tau, 'known_c': known_c, 'known_tau':known_tau}
with open('/home/jpienaar/SingleElectrons/processing_scripts/fitting_results_%s_%i.pkl' %(known_source, file_iteration), 'wb') as handle:
    pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
