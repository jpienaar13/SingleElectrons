import sys
import hax
import hax
import numpy as np
from pax import units
from collections import defaultdict
import math
#from Pre_trigger_peaks import Pre_Trigger
from PI_after_s2 import S2_ionization


#Main
hax.init(experiment='XENON1T', 
        pax_version_policy = '6.8.0',
        main_data_paths = ['/project2/lgrandi/xenon1t/processed', '/project/lgrandi/xenon1t/processed'],
        minitree_paths = ['/scratch/midway2/jpienaar/minitrees/',
                         '/project2/lgrandi/xenon1t/minitrees/pax_v6.8.0',
                         '/project/lgrandi/xenon1t/minitrees/pax_v6.8.0',
                         ],
        ) 
run_number=dataset=sys.argv[1]
cache_file_name = '/scratch/midway2/jpienaar/cache_files/'+run_number+ '_PI_after_S2.hdf5'
print (run_number)
df_PI = hax.minitrees.load(run_number,
                         treemakers=[S2_ionization, 'Basics'],
                         preselection=None,
                         force_reload=True,
                         )
#df_PI.to_pickle(cache_file_name)
hax.minitrees.save_cache_file(df_PI,cache_file_name)
#hax.minitrees.save_cache_file(df_PI,cache_file_name)