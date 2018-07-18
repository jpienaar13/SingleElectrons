#################################################################
# Batchbuilder for building minitrees
# Remember to choose the source and minitree name correctly
#
#################################################################
queue_name='xenon1t'  ##Turn this into command line input at some stage

x = """#!/bin/bash
#SBATCH --job-name={run}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8000
#SBATCH --output=minitree_%J.log
#SBATCH --error=minitree_%J.log
#SBATCH --account=pi-lgrandi
#SBATCH --partition={queue}
export PATH=/project/lgrandi/anaconda3/bin:$PATH
export PROCESSING_DIR=/scratch/midway2/jpienaar/minitrees/minitree_{run}
        
mkdir -p ${{PROCESSING_DIR}}
cd ${{PROCESSING_DIR}}
rm -f pax_event_class*
source activate pax_head
python /home/jpienaar/SingleElectrons/processing_scripts/process_Pre_trigger_minitrees.py {run}
rm -r ${{PROCESSING_DIR}}
"""
import os
import sys
import subprocess
from subprocess import PIPE
import time
import glob
import numpy as np

# Use submit procedure from CAX
os.system('source activate pax_head')
from cax.qsub import submit_job

####################################################################
#Functions to be used                                              #
####################################################################
def get_file_list(path, pattern, remove_string='', remove_path=True):
    '''
    Get a list of files matching pattern in path. Optional to remove 
    a part of the path (i.e. the extention).
    Optional to remove the path.
    '''
    file_list = glob.glob(path + pattern)
    # Remove path
    for i, f in enumerate(file_list):
        if remove_path:
            f = f.replace(path, '')
        if remove_string != '':
            f = f.replace(remove_string, '')
        file_list[i] = f
    file_list = np.sort(file_list)
    return file_list

def check_queue():
    command='squeue -u jpienaar --partition=%s| wc -l' %queue_name
    var=subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    (var, err)=var.communicate()
    var=int(str(var, 'utf-8'))
    return var

####################################################################
#Find datasets for which minitrees already exist                   #
####################################################################
#Identify available minitrees
print('Finding Minitree HDF5 Files')
simpath= '/scratch/midway2/jpienaar/cache_files/'
datasets_hdf5 = get_file_list(simpath, '*Pre_trigger.hdf5', '_Pre_trigger.hdf5')
print('Found %d minitree files' % len(datasets_hdf5))

#Identify available minitrees
print('Finding Process PKL Files')
simpath= '/scratch/midway2/jpienaar/cache_files/'
datasets_pickle = get_file_list(simpath, '*dt.pkl', '_dt.pkl')
print('Found %d processed files' % len(datasets_pickle))

processing_list=np.setdiff1d(datasets_hdf5, datasets_pickle, assume_unique=True)

#For every run, make and submit the script
for dataset in processing_list[:]:
    while check_queue()>30:
        print("Jobs in queue: ", check_queue())
        time.sleep(60)
    y = x.format(run=dataset, queue=queue_name)
    submit_job(y)

