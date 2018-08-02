#################################################################
# Batchbuilder for building minitrees
# Remember to choose the source and minitree name correctly
#python /home/jpienaar/batchjobs/process_Pre_trigger_minitrees_xy.py {run} NB!!! Update this script!!!!!
#python /home/jpienaar/SingleElectrons/make_Pre_trigger_minitrees.py {run}
#################################################################
queue_name='dali'

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
python /home/jpienaar/SingleElectrons/processing_scripts/make_Pre_trigger_minitrees.py {run}
python /home/jpienaar/SingleElectrons/processing_scripts/process_Pre_trigger_minitrees.py {run}
rm -r ${{PROCESSING_DIR}}
"""
import os
import sys
import subprocess
from subprocess import PIPE
import time

# Use submit procedure from CAX
os.system('source activate pax_head')
from cax.qsub import submit_job

from pax import units, configuration, datastructure
pax_config = configuration.load_configuration('XENON1T')
n_channels = pax_config['DEFAULT']['n_channels']
pmts = pax_config['DEFAULT']['pmts']
tpc_height = pax_config['DEFAULT']['tpc_length']
tpc_radius = pax_config['DEFAULT']['tpc_radius']
gains = pax_config['DEFAULT']['gains']
import hax
from hax.data_extractor import DataExtractor
print("Hax version :", hax.__version__)

####################################################################
#Define which runs we want to process (max per submit is 400 jobs!)#
####################################################################
#Initialize Hax
hax.init(experiment='XENON1T', 
        pax_version_policy = '6.8.0',
        main_data_paths = ['/project2/lgrandi/xenon1t/processed', '/project/lgrandi/xenon1t/processed'],
        minitree_paths = ['/scratch/midway2/jpienaar/minitrees/',
                         '/project2/lgrandi/xenon1t/minitrees/pax_v6.8.0',
                         '/project/lgrandi/xenon1t/minitrees/pax_v6.8.0',
                         ],
        ) 
datasets = hax.runs.datasets 
datasets = hax.runs.tags_selection(include=['*'],
                                  exclude=['bad','messy', 'test',
                                           'nofield','lowfield',
                                           'commissioning',
                                           'pmttrip','trip','_pmttrip',
                                           'source_opening',
                                           ],
                                  )
datasets= hax.cuts.selection(datasets, datasets['source__type']=='none', 'Source in place')
datasets= hax.cuts.selection(datasets, datasets['location'] != '', 'Processed data available')
run_numbers = datasets['number'].values
dataset_names = datasets['name']
print('Total of %d datasets' % len(run_numbers))

def check_queue():
    command='squeue -u jpienaar --partition=%s| wc -l' %queue_name
    var=subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    (var, err)=var.communicate()
    var=int(str(var, 'utf-8'))
    return var



#For every run, make and submit the script
for dataset in dataset_names[8000::]:# in run_numbers[:1]:
    while check_queue()>30:
        print("Jobs in queue: ", check_queue())
        time.sleep(60)
    y = x.format(run=dataset, queue=queue_name)
    submit_job(y)

# Check your jobs with: 'squeue -u <username>'
# Check number of submitted jobs with 'squeue -u <username> | wc -l' (is off by +2 btw)
