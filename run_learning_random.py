from core.classes import PMC
from core.pmc_functions import pmc_load_instantiation, pmc_instantiate, \
    assert_probabilities
from core.verify_pmc import pmc_verify, pmc_get_reward
from core.io.export import timer
from core.io.parser import parse_main
from pathlib import Path
from datetime import datetime
import sys
import time 

from core.learning.classes import learner

import os
import numpy as np
import pandas as pd


'''
This file is used to compare sampling according to the exploration MDP with randoml movement.

'''


# Parse arguments
args = parse_main(learning = True)
args.no_gradient_validation = True

args.root_dir = os.path.dirname(os.path.abspath(__file__))

model_path = Path(args.root_dir, args.model)
param_path = Path(args.root_dir, args.parameters) if args.parameters else False
true_param_path = Path(args.root_dir, args.true_param_file) if args.true_param_file else False
statespace_path = Path(args.root_dir, args.statespace) if args.statespace else False

T = timer()

# %%

pmc = PMC(model_path = model_path, args = args)

args.robust_probabilities = np.full(pmc.model.nr_states, True)
args.robust_dependencies = 'parameter' # Can be 'none' or 'parameter'

### pMC execution    
if true_param_path:
    # Load parameter valuation
    inst_true = pmc_load_instantiation(pmc, true_param_path, args.default_valuation)
    
else:
    # Create parameter valuation
    inst_true = {'valuation': {}}
    
    # Create parameter valuations on the spot
    for v in pmc.parameters:
        inst_true['valuation'][v.name] = args.default_valuation
        


# Compute True solution

# Define instantiated pMC based on parameter valuation
instantiated_model, inst_true['point'] = pmc_instantiate(pmc, inst_true['valuation'], T)
assert_probabilities(instantiated_model)

pmc.reward = pmc_get_reward(pmc, instantiated_model, args)

print('\n',instantiated_model,'\n')

# Verify true pMC
solution_true, J, Ju = pmc_verify(instantiated_model, pmc, inst_true['point'], T)

print('Optimal solution under the true parameter values: {:.3f}'.format(solution_true))

# %%

DFs = {}
DFs_stats = {}

modes = ['derivative','expVisits_sampling','expVisits','samples','random', 'exploration_random_policy']
modes = ['exploration20', 'exploration50', 'exploration_random_policy'] #Compare the methods

mes = 20
max_exploration_steps_list = [mes]
max_samples = 100


DF_time = pd.DataFrame(columns= ['exploration20', 'exploration50', 'exploration_random_policy'])
for mod in modes:
        
    #Set the --exploration_steps argument correctly:
    args.exploration_steps = mes
    if mod == 'exploration20':
        mode = mod
        mod = 'exploration'
        args.exploration_steps = 20
    elif mod == 'exploration50':
        mode = mod
        mod = 'exploration'
        args.exploration_steps = 50
    else:
        mode = mod 
        args.exploration_steps = 50
    args.learning_steps = int(max_samples / args.exploration_steps)
    print("LEARNING ITERATIONS:", args.learning_steps)

    #Set the correct string for the plots

    DFs[mode] = pd.DataFrame()

    times = []
    for q, seed in enumerate(np.arange(args.learning_iterations)):
        t1 = time.time() #Start timing when the learning starts
        print('>>> Start iteration {} <<<'.format(seed + 1))
        
        if param_path:
            inst = pmc_load_instantiation(pmc, param_path, args.default_valuation)
            
        else:
            inst = {'valuation': {}, 'sample_size': {}}
            
            # Set seed
            np.random.seed(seed)
            
            # Create parameter valuations on the spot
            for v in pmc.parameters:
                
                # Sample MLE value
                p = inst_true['valuation'][v.name]
                N = args.default_sample_size
                delta = 1e-4
                MLE = np.random.binomial(N, p) / N
                
                # Store
                inst['valuation'][v.name] = max(min(MLE , 1-delta), delta)
                inst['sample_size'][v.name] = args.default_sample_size
        
        # Define instantiated pMC based on parameter 
        instantiated_model, inst['point'] = pmc_instantiate(pmc, inst['valuation'], T)
        assert_probabilities(instantiated_model)

        pmc.reward = pmc_get_reward(pmc, instantiated_model, args)
        
        # Define learner object 
        print(mod)
        L = learner(pmc, inst, args.learning_samples_per_step, seed, args, mod)
        samples_collected = 0
        
        for i in range(args.learning_steps):
            print('----------------\nMethod {}, Iteration {}, Step {}\n----------------'.format(mod, q, i))
            
            # Compute robust solution for current step
            L.solve_step()
            
            # Determine for which parameter to obtain additional samples
            PAR = L.sample_method(L)
            
            # Get additional samples
            L.sample(PAR, inst_true['valuation'], mod) 
            samples_collected += args.learning_samples_per_step * args.exploration_steps

            # Update learnined object
            L.update(PAR, mod) 
        #after collecting all samples, recalculate the solution one last time
        L.solve_step()
        DFs[mode] = pd.concat([DFs[mode], pd.Series(L.solution_list)], axis=1)
        t2 = time.time() #End timing after learning terminates 
        times.append(t2-t1)
    

    current_time = datetime.now().strftime("%H:%M:%S")
    print('\nprMC code ended at {}\n'.format(current_time))
    print('=============================================')
    
    dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    # If desired, export the detailed results for each of the different
    # sampling methods
    # DFs[mode].to_csv('output/learning_{}_{}.csv'.format(dt,mode), sep=';')    
        
    DFs_stats[mod] = pd.DataFrame({
        '{}_mean'.format(mode): DFs[mode].mean(axis=1),
        '{}_min'.format(mode): DFs[mode].min(axis=1),
        '{}_max'.format(mode): DFs[mode].max(axis=1)
        })
    DF_time[mode] = times
# %%
    
print('\nExport data of learning experiment and create plot...')

import matplotlib.pyplot as plt

dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

DF_stats = pd.concat(list(DFs_stats.values()), axis=1)

df_merged = pd.concat([df.mean(axis=1) for df in DFs.values()], axis=1)
df_merged_max = pd.concat([df.max(axis=1) for df in DFs.values()], axis=1)
df_merged_min = pd.concat([df.min(axis=1) for df in DFs.values()], axis=1)

df_merged.columns = list(DFs.keys())
df_merged_max.columns = list(DFs.keys())
df_merged_min.columns = list(DFs.keys())

#For fairness, compare each run based on samples collected instead of iterations done.
# learning_steps = int(max_samples / value)

plt.rcParams.update({'font.size': 22})


value = 20

print(df_merged)
learning_steps = int(max_samples / value) + 1
#this is amount of times a value was added to L.solution_list, so amount of times the sol was calculated. Every calculation took place after collecting value * args.learning_samples_per_step samples. 1 is added to account for the extra solution calculation after all samples were collected.

#plt.plot(args.learning_samples_per_step * np.array(range(0, value*learning_steps, value)), [x for x in DF_stats['exploration20_mean'].to_list() if str(x) != 'nan'], ':', label = 'Exploration 20')
plt.plot(args.learning_samples_per_step * np.array(range(0, value*learning_steps, value)), df_merged['exploration20'], ':', label = 'Exploration 20')

#Above line plots the amount of samples collected (per learning iteration, value * learning*steps * samples_per_step samples are collected) against this: Every learning iteration the model outputs multiple approximations of the true value, at different points in the iteration. These values are collected, and the means of them are calculated and plotted. 

plt.fill_between(args.learning_samples_per_step * np.array(range(0, value*learning_steps, value)), list(filter(pd.notna, df_merged_min["exploration20"].values)), list(filter(pd.notna,  df_merged_max["exploration20"])), alpha = 0.2)

value = 50
learning_steps = int(max_samples / value) + 1
plt.plot(args.learning_samples_per_step * np.array(range(0, value*learning_steps, value)), list(filter(pd.notna,df_merged['exploration50'])), ':', label = 'Exploration 50')  

plt.fill_between(args.learning_samples_per_step * np.array(range(0, value*learning_steps, value)), list(filter(pd.notna, df_merged_min["exploration50"].values)), list(filter(pd.notna,  df_merged_max["exploration50"])), alpha = 0.2)
#Above plots shaded areas between min/max over learning iterations


#plot the random policy
plt.plot(args.learning_samples_per_step * np.array(range(0, value*learning_steps, value)), list(filter(pd.notna,df_merged['exploration_random_policy'])), ':', label = 'Random Policy 50') 

plt.fill_between(args.learning_samples_per_step * np.array(range(0, value*learning_steps, value)), list(filter(pd.notna, df_merged_min['exploration_random_policy'].values)), list(filter(pd.notna,  df_merged_max['exploration_random_policy'])), alpha = 0.2)

#df_merged.plot()

ax = plt.gca()
ax.set_ylim([0, None])

plt.axhline(y=solution_true, color='gray', linestyle='--')



#Set visuals
plt.legend()
plt.xlabel("Total samples collected")
plt.ylabel("Robust solution")

d = {   'color': 'black',
        'family': 'Times New Roman'
    }
plt.text(solution_true + 10, solution_true + 10 , "True solution", fontdict = d)
#plt.title("Solution versus samples collected") #

DF_stats.to_csv('output/learning_{}_{}.csv'.format(args.instance, dt), sep=';') 

#plt.savefig('output/{}_samples_mes_{}_vs_random_{}.png'.format(args.learning_samples_per_step * max_samples, mes, dt), bbox_inches='tight')
plt.savefig('output/{}_samples_mes_{}_vs_random_{}.pdf'.format(args.learning_samples_per_step * max_samples, mes, dt), bbox_inches='tight')

print('Data exported and plot saved.')