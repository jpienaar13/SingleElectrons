import sys
import hax
import hax
import numpy as np
from pax import units
from collections import defaultdict
import math

def get_largest_indices(peaks, exclude_indices=tuple()):

    """Return a dic with the indices in peaks of the largest peak of each type (s1, s2, etc)
    excluding the indices in exclude_peak_indices from consideration
    """

    largest_area_of_type = defaultdict(float)
    largest_indices = dict()

    for i, p in enumerate(peaks):
        if i in exclude_indices:
            continue

        if p.detector == 'tpc':
            peak_type = p.type

        else:

            if p.type == 'lone_hit':
                peak_type = 'lone_hit_%s' % p.detector    # Will not be saved
            else:
                peak_type = p.detector
        if p.area > largest_area_of_type[peak_type]:
            largest_area_of_type[peak_type] = p.area
            largest_indices[peak_type] = i
            
    return largest_indices   

def yield_peak_info(event):
    #NB peaks and event.peaks are not the same!!!
    #Skip waveforms with no recorded interaction
    if len(event.interactions) == 0:
        if len(event.s2s):
            i_s2 = event.s2s[0]
            i_s1 = np.nan
            time_s2 = event.peaks[i_s2].area_midpoint
            time_s1 = np.nan
        else:
            return []   #skips over events that do not have an "interaction" (pair of s1 & s2)
    else:
        i_s2=event.interactions[0].s2
        i_s1=event.interactions[0].s1
        
        #Time of primary S2/S1
        time_s1=event.peaks[i_s1].area_midpoint
        time_s2=event.peaks[i_s2].area_midpoint
            
    #s2 position
    rp_s2 = event.peaks[i_s2].reconstructed_positions
    for rp in rp_s2:
        if rp.algorithm == 'PosRecTopPatternFit':
            x_s2_tpf = rp.x
            y_s2_tpf = rp.y
        elif rp.algorithm == 'PosRecNeuralNet':
            x_s2_nn = rp.x
            y_s2_nn = rp.y
            
    # Get all non-lone-hit peaks in the TPC
    peaks_tmp = []
    for i, peak in enumerate(event.peaks):
        if i == i_s2 or i == i_s1  or (peak.n_hits<15 and peak.type!='s1'):
            continue
        else:
            peaks_tmp.append(peak) 
    
    peaks = [p for p in peaks_tmp if p.detector == 'tpc' and p.type!='lone_hit' and p.type!='unknown']
    
    #Ordering by left bound time [10ns]
    peaks = sorted(peaks, key=lambda p: p.left) 
    
    #If peaks list is empty, nothing to do
    if not len(peaks):
        return []
    
    for i, peak in enumerate(peaks):      
        #Make sure event.peak is to the right of s1 by looking at time (ns)
        time_peak = peak.area_midpoint
        if not np.isnan(time_s1):
            if not time_peak < time_s1: 
                continue
            else:
                yield peak, x_s2_tpf, y_s2_tpf, x_s2_nn, y_s2_nn, time_s1, time_s2, time_peak
        if np.isnan(time_s1):
            if not time_peak < time_s2: 
                continue
            else:
                yield peak, x_s2_tpf, y_s2_tpf, x_s2_nn, y_s2_nn, time_s1, time_s2, time_peak


class Pre_Trigger(hax.minitrees.MultipleRowExtractor):
    __version__ = '4.0.0'
    uses_arrays=True
    extra_branches = ['peaks.left', 'peaks.n_hits', 'peaks.area', 'peaks.type',
                      'peaks.n_contributing_channels', 'peaks.n_contributing_channels_top',
                      'peaks.reconstructed_positions*', 'peaks.area_midpoint']
    #extra_branches = ['*']

 
    def extract_data(self, event):       
        results = []      
   
        for peak, x_s2_tpf, y_s2_tpf, x_s2_nn, y_s2_nn, time_s1, time_s2, time_peak in yield_peak_info(event):
            result = dict({x: getattr(peak, x) for x in ['area', 'area_fraction_top', 'n_hits']})
            result['x_s2_tpf'] = x_s2_tpf
            result['y_s2_tpf'] = y_s2_tpf
            result['x_s2_nn'] = x_s2_nn
            result['y_s2_nn'] = y_s2_nn
            result['n_hits'] = peak.n_hits
            #result['type']= peak.type
            result['area'] = peak.area
            result['global_time'] = time_peak+event.start_time
            result['event_stop'] = event.stop_time
            result['event_start'] = event.start_time
            result['s1_time']=time_s1+event.start_time
            result['s2_time']=time_s2+event.start_time  
            result['p_range_50p_area'] = peak.range_area_decile[5]
            result['n_contributing_channels'] = peak.n_contributing_channels
            result['n_contributing_channels_top'] = peak.n_contributing_channels_top
            #result['sum_waveform'] = np.array(list(peak.sum_waveform))
            for rp in peak.reconstructed_positions:
                if rp.algorithm == 'PosRecTopPatternFit':
                    result['x_p_tpf'] = rp.x
                    result['y_p_tpf'] = rp.y
                    result['xy_gof_tpf'] = rp.goodness_of_fit
                elif rp.algorithm == 'PosRecNeuralNet':
                    result['x_p_nn'] = rp.x
                    result['y_p_nn'] = rp.y
                    result['xy_gof_nn'] = rp.goodness_of_fit
            results.append(result)


        return results
