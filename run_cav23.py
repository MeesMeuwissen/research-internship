# %run "~/documents/sensitivity-prmdps/prmdp-sensitivity-git/run_cav23.py"

from core.main_pmc import run_pmc
from core.main_prmc import run_prmc
from core.parse_inputs import parse_inputs

import os
import math
import numpy as np
from pathlib import Path
from tabulate import tabulate
from datetime import datetime

# Parse arguments
args = parse_inputs()

# Load PRISM model with STORM
args.root_dir = os.path.dirname(os.path.abspath(__file__))

# args.model = 'models/pmdp/CSMA/csma2_4_param.nm'
# args.formula = 'R{"time"}max=? [ F "all_delivered" ]'
# args.default_valuation = 0.1
# args.goal_label = 'observe0Greater1'

### pMC execution

current_time = datetime.now().strftime("%H:%M:%S")
print('Program started at {}'.format(current_time))
print('\n',tabulate(vars(args).items(), headers=["Argument", "Value"]),'\n')

model_path = Path(args.root_dir, args.model)
param_path = Path(args.root_dir, args.parameters) if args.parameters else False

pmc, T, inst, solution, deriv = run_pmc(args, model_path, param_path, verbose = args.verbose)

current_time = datetime.now().strftime("%H:%M:%S")
print('\npMC code ended at {}\n'.format(current_time))
print('=============================================')

### prMC execution

# Check if we should also run the prMC part
if not args.no_prMC:
    
    np.random.seed(0)
    
    if inst['sample_size'] is None:
        print('- Create arbitrary sample sizes')
        inst['sample_size'] = {par.name: 10000 + np.random.rand()*10 for par in pmc.parameters}
    
    # scale rewards for prMC:
    if args.scale_reward:
        pmc.reward = pmc.reward / np.max(pmc.reward)
    
    print('Start prMC code')
    
    # Normalize reward vector
    # NORM = 10**(-math.floor(math.log(pmc.model.nr_states, 10)))
    # print('- Normalize reward vector by factor {}'.format(NORM))
    # pmc.reward *= NORM
    
    prmc, T, inst, solution, deriv = run_prmc(pmc, args, inst, verbose = args.verbose)
    
    current_time = datetime.now().strftime("%H:%M:%S")
    print('\nprMC code ended at {}\n'.format(current_time))
    print('=============================================')
