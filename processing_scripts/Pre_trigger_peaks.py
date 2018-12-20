import sys
import hax
import numpy as np
from pax import units
from collections import defaultdict
import math

def yield_peak_info(event, event_info):  
    #Get all non-lone-hit peaks in the TPC
    peaks_tmp = []
    for i, peak in enumerate(event.peaks):
        if i == event_info.i_s2 or i == event_info.i_s1  or (peak.n_hits<10 and peak.type!='s1'):
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
        #Make sure event.peak is to the right of S1/S2 by looking at time (ns)
        time_peak = peak.area_midpoint
        if not np.isnan(event_info.time_s1):
            if not time_peak < event_info.time_s1: 
                continue
            else:
                yield peak
        if np.isnan(event_info.time_s1):
            if not time_peak < event_info.time_s2: 
                continue
            else:
                yield peak

class Event_Info:
    def __init__ (self):
        self.x_s2_tpf = np.nan
        self.y_s2_tpf = np.nan
        self.x_s2_nn = np.nan
        self.y_s2_nn = np.nan
        self.time_s2 = np.nan
        self.time_s1 = np.nan
        self.i_s2 = np.nan
        self.i_s1 = np.nan
        self.first_trigger = np.nan

                
class Pre_Trigger(hax.minitrees.MultipleRowExtractor):
    __version__ = '6.0.0'
    uses_arrays=True
    #extra_branches = ['peaks.left', 'peaks.n_hits', 'peaks.area', 'peaks.type',
    #                  'peaks.n_contributing_channels', 'peaks.n_contributing_channels_top',
    #                  'peaks.reconstructed_positions*', 'peaks.area_midpoint']
    extra_branches = ['*']
 
    def extract_data(self, event):       
        results = []
        
        #Initialize Data Structure for storing event level information
        event_info = Event_Info()
        
        #If we have an interaction get both S1 and S2 information
        if len(event.interactions):
            #Get indices of S1/S2 in primary interaction
            event_info.i_s2=event.interactions[0].s2
            event_info.i_s1=event.interactions[0].s1

            #Get time of S1/S2 from start of event
            event_info.time_s1=event.peaks[event_info.i_s1].area_midpoint
            event_info.time_s2=event.peaks[event_info.i_s2].area_midpoint
        #If no interaction use largest S2 greater than 150 PE as "primary S2"
        elif len(event.s2s):
            if event.s2s[0]>150:
                event_info.i_s2 = event.s2s[0]
                
                event_info.time_s2 = event.peaks[event_info.i_s2].area_midpoint

        #Get S2 positions (TPF/NN) of primary S2
        if np.isnan(event_info.i_s2) == False:
            rp_s2 = event.peaks[event_info.i_s2].reconstructed_positions
            for rp in rp_s2:
                if rp.algorithm == 'PosRecTopPatternFit':
                    event_info.x_s2_tpf = rp.x
                    event_info.y_s2_tpf = rp.y
                elif rp.algorithm == 'PosRecNeuralNet':
                    event_info.x_s2_nn = rp.x
                    event_info.y_s2_nn = rp.y
        
        #Identify first peak large enough to be considered a trigger.
        peaks_trigger = [p.area_midpoint for p in event.peaks if p.type=='s1' or p.area>150]
        event_info.first_trigger = min(peaks_trigger, default=1.0e6)   
        
        
        peak_branches = ['area', 'area_fraction_top', 'n_hits', 
                         'n_contributing_channels', 'n_contributing_channels_top']
        for peak in yield_peak_info(event, event_info):
            result = dict({x: getattr(peak, x) for x in peak_branches})
            result['x_s2_tpf'] = event_info.x_s2_tpf
            result['y_s2_tpf'] = event_info.y_s2_tpf
            result['x_s2_nn'] = event_info.x_s2_nn
            result['y_s2_nn'] = event_info.y_s2_nn
            result['global_time'] = peak.area_midpoint + event.start_time
            result['event_stop'] = event.stop_time
            result['event_start'] = event.start_time
            result['s1_time'] = event_info.time_s1+event.start_time
            result['s2_time'] = event_info.time_s2+event.start_time  
            result['p_range_50p_area'] = peak.range_area_decile[5]
            result['p_range_90p_area'] = peak.range_area_decile[9]
            result['rise_time'] = peak.area_decile_from_midpoint[1]
            result['time_before_trigger'] = event_info.first_trigger - peak.area_midpoint
            result['window_length'] = event_info.first_trigger
            for rp in peak.reconstructed_positions:
                if rp.algorithm == 'PosRecTopPatternFit':
                    result['x_p_tpf'] = rp.x
                    result['y_p_tpf'] = rp.y
                    result['xy_gof_tpf'] = rp.goodness_of_fit
                elif rp.algorithm == 'PosRecNeuralNet':
                    result['x_p_nn'] = rp.x
                    result['y_p_nn'] = rp.y
                    result['xy_gof_nn'] = rp.goodness_of_fit        
            if peak.type == 's1': 
                result['type'] = 1
            if peak.type == 's2': 
                result['type'] = 2
            if peak.type == 'lone_hit': 
                result['type'] = 3
            if peak.type == 'unknown': 
                result['type'] = 4   
            results.append(result)
        
        #If no peak objects, still need to store S1/S2 info:
        if not results:
            result = dict({x: np.nan for x in peak_branches})
            result['x_s2_tpf'] = event_info.x_s2_tpf
            result['y_s2_tpf'] = event_info.y_s2_tpf
            result['x_s2_nn'] = event_info.x_s2_nn
            result['y_s2_nn'] = event_info.y_s2_nn
            result['global_time'] = np.nan
            result['event_stop'] = event.stop_time
            result['event_start'] = event.start_time
            result['s1_time'] = event_info.time_s1+event.start_time
            result['s2_time'] = event_info.time_s2+event.start_time  
            result['p_range_50p_area'] = np.nan
            result['p_range_90p_area'] = np.nan
            result['rise_time'] = np.nan
            result['time_before_trigger'] = np.nan
            result['window_length'] = event_info.first_trigger
            result['x_p_tpf'] = np.nan
            result['y_p_tpf'] = np.nan
            result['xy_gof_tpf'] = np.nan
            result['x_p_nn'] = np.nan
            result['y_p_nn'] = np.nan
            result['xy_gof_nn'] = np.nan        
            result['type'] = np.nan   
            results.append(result)
        
        return results
       
class Primary_Times(hax.minitrees.TreeMaker):
    """
    Provides:
     - event_stop: 
     - event_start: 
     - s1_time: 
     - s2_time: 
    """
    __version__ = '1.0.0'
    extra_branches = ['peaks.*']

    def extract_data(self, event):
        result={}
        
        if len(event.interactions) == 0:
            if len(event.s2s):
                i_s2 = event.s2s[0]
                i_s1 = np.nan
                time_s2 = event.peaks[i_s2].area_midpoint
                time_s1 = np.nan
            else:
                return result  
        else:
            i_s2=event.interactions[0].s2
            i_s1=event.interactions[0].s1

            #Time of primary S2/S1
            time_s1=event.peaks[i_s1].area_midpoint
            time_s2=event.peaks[i_s2].area_midpoint
            
        #S2 position
        x_s2_tpf = np.nan
        y_s2_tpf = np.nan
        x_s2_nn = np.nan
        y_s2_nn = np.nan
        rp_s2 = event.peaks[i_s2].reconstructed_positions
        for rp in rp_s2:
            if rp.algorithm == 'PosRecTopPatternFit':
                x_s2_tpf = rp.x
                y_s2_tpf = rp.y
            elif rp.algorithm == 'PosRecNeuralNet':
                x_s2_nn = rp.x
                y_s2_nn = rp.y
               
        result['event_stop'] = event.stop_time
        result['event_start'] = event.start_time
        result['s1_time'] = time_s1 + event.start_time
        result['s2_time'] = time_s2 + event.start_time 
        result['x_s2_tpf'] = x_s2_tpf
        result['y_s2_tpf'] = y_s2_tpf
        result['x_s2_nn'] = x_s2_nn
        result['y_s2_nn'] = y_s2_nn
        return result
