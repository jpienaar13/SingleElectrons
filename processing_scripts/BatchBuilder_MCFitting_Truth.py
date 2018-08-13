#################################################################
# Batchbuilder for building minitrees
# Remember to choose the source and minitree name correctly
#python /home/jpienaar/batchjobs/process_Pre_trigger_minitrees_xy.py {run} NB!!! Update this script!!!!!
#python /home/jpienaar/SingleElectrons/make_Pre_trigger_minitrees.py {run}
#################################################################
queue_name='dali'

x = """#!/bin/bash
#SBATCH --job-name={source}_{it}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8000
#SBATCH --output=minitree_%J.log
#SBATCH --error=minitree_%J.log
#SBATCH --account=pi-lgrandi
#SBATCH --partition={queue}
export PATH=/project/lgrandi/anaconda3/bin:$PATH
export PROCESSING_DIR=/scratch/midway2/jpienaar/minitrees/mcfit_{source}_{it}
        
mkdir -p ${{PROCESSING_DIR}}
cd ${{PROCESSING_DIR}}
rm -f pax_event_class*
source activate pax_head
python /home/jpienaar/SingleElectrons/fitting_mc_confidence.py {c} {tau} {source} {it}
rm -r ${{PROCESSING_DIR}}
"""
import os
import sys
import subprocess
from subprocess import PIPE
import time
import numpy as np

# Use submit procedure from CAX
os.system('source activate pax_head')
from cax.qsub import submit_job

####################################################################
#Define Values for which to fit#
####################################################################
scaling_constant_values=np.linspace(1*10**-5, 1*10**-4, 51)
time_scale_values=np.linspace(10, 50, 41)

def check_queue():
    command='squeue -u jpienaar --partition=%s| wc -l' %queue_name
    var=subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    (var, err)=var.communicate()
    var=int(str(var, 'utf-8'))
    return var


known_source='Background'
#For every run, make and submit the script
it=0
for  scaling_constant in scaling_constant_values:
    for  time_scale in time_scale_values:
        while check_queue()>50:
            print("Jobs in queue: ", check_queue())
            time.sleep(60)
        print('Submit job with c=%f and tau=%f' %(scaling_constant, time_scale))
        y = x.format(queue=queue_name, c=scaling_constant, tau=time_scale, source = known_source, it=it)
        submit_job(y)
        it+=1

# Check your jobs with: 'squeue -u <username>'
# Check number of submitted jobs with 'squeue -u <username> | wc -l' (is off by +2 btw)
