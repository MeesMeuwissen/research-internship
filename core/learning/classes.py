from core.prmc_functions import pmc2prmc, pmc_instantiate
from core.sensitivity import gradient, solve_cvx_gurobi
from core.verify_prmc import verify_prmc

from core.learning.exp_visits import parameter_importance_exp_visits
from core.learning.validate import validate

from gurobipy import GRB

import numpy as np
import random
import time
import stormpy
import stormpy.simulator

# %%
class learner:
    
    def __init__(self, pmc, inst, samples_per_step, seed, args, mode):
        
        self.UPDATE = True
        
        self.pmc = pmc
        self.inst = inst
        
        self.SAMPLES_PER_STEP = samples_per_step
        self.args = args
        
        self._set_sampler(mode)
        
        np.random.seed(seed) 
        random.seed(seed)
        
        self.solution_list = []
    
        # Define prMC
        self.prmc = pmc2prmc(self.pmc.model, self.pmc.parameters, self.pmc.scheduler_prob, self.inst['point'], self.inst['sample_size'], self.args, verbose = self.args.verbose)
        
        print("INITIAL STATES:", self.prmc.initial_states)
        assert 1==2

        #Zelf toegevoegd:
        self.max_exploration_steps = 20 #Variable to set the max length of exploratory mdp, could be variable in the class instead
        self.statespace_fp = '/Users/Mees_1/prmc-sensitivity/models/slipgrid_learning/model_mees_more_terrains_statespace.txt' # Dit nog aanpassen per run. Misschien als extra argument toevoegen aan de class.
        self.state_dict, self.terrain_dict, self.transition_matrix = make_exploration_utils(self.statespace_fp, self.max_exploration_steps) 
        self.current_state = self.state_dict[(0,0)] # The drone starts in the initial state. This will be updated throughout sample collection

        self.CVX = verify_prmc(self.prmc, self.pmc.reward, self.args.beta_penalty, self.args.robust_bound, verbose = self.args.verbose)
        self.CVX.cvx.tune()
        try:
            self.CVX.cvx.getTuneResult(0)
        except:
            print('Exception: could not set tuning results')
        
        self.CVX.cvx.Params.NumericFocus = 3
        self.CVX.cvx.Params.ScaleFlag = 1
        
        if self.opposite:
            if self.args.robust_bound == 'lower':
                bound = 'upper'
            else:
                bound = 'lower'
            
            self.CVX_opp = verify_prmc(self.prmc, self.pmc.reward, self.args.beta_penalty, bound, verbose = self.args.verbose)
            self.CVX_opp.cvx.tune()
            try:
                self.CVX_opp.cvx.getTuneResult(0)
            except:
                print('Ecception: could not set tuning results')
        
        
    def _set_sampler(self, mode):
        
        if mode == 'random':
            self.sample_method = sample_uniform
            self.opposite = False
        elif mode == 'samples':
            self.sample_method = sample_lowest_count
            self.opposite = False
        elif mode == 'expVisits':
            self.sample_method = sample_importance
            self.opposite = True
        elif mode == 'expVisits_sampling':
            self.sample_method = sample_importance_proportional
            self.opposite = True
        elif mode == 'derivative':
            self.sample_method = sample_derivative
            self.opposite = False
        elif mode == 'exploration':
            self.sample_method = sample_exploration
            self.opposite = False
        else:
            print('ERROR: unknown mode')
            assert False
        
        
    def solve_step(self):
        
        start_time = time.time()
        self.CVX.solve(store_initial = True, verbose=False)
        print('Solver time:', time.time() - start_time)
        
        self.solution_current = self.CVX.x_tilde[self.prmc.sI['s']] @ self.prmc.sI['p']
        self.solution_list += [np.round(self.solution_current, 9)]
        
        print('Range of solutions: [{}, {}]'.format(np.min(self.CVX.x_tilde), np.max(self.CVX.x_tilde)))
        print('Solution in initial state: {}\n'.format(self.solution_current))
        
        SLACK = self.CVX.get_active_constraints(self.prmc, verbose=False)
        
        if self.opposite:
            self.CVX_opp.solve(store_initial = True, verbose=self.args.verbose)
            SLACK = self.CVX_opp.get_active_constraints(self.prmc, verbose=False)
        
        
        
    def sample(self, params, true_valuation, mode):
        if mode == "exploration":
            # params will have a different shape in this case, hence the different code.
            for q,var in enumerate(params):
            
                if type(true_valuation) == dict:
                    true_prob = true_valuation[var[0].name]
                else:
                    true_prob = self.args.default_valuation
                
                samples = np.random.binomial(var[1], true_prob)
                
                old_sample_mean = self.inst['valuation'][var[0].name]
                new_sample_mean = (self.inst['valuation'][var[0].name] * self.inst['sample_size'][var[0].name] + samples) / (self.inst['sample_size'][var[0].name] + var[1])
                
                self.inst['valuation'][var[0].name] = new_sample_mean
                self.inst['sample_size'][var[0].name] += var[1]
                
                print('\n>> Drawn {} more samples for parameter {} ({} positives)'.format(var[1], var[0], samples))
                print('>> MLE is now: {:.3f} (difference: {:.3f})'.format(new_sample_mean, new_sample_mean - old_sample_mean))
                print('>> Total number of samples is now: {}\n'.format(self.inst['sample_size'][var[0].name]))
        else: 
            for q,var in enumerate(params):
            
                if type(true_valuation) == dict:
                    true_prob = true_valuation[var.name]
                else:
                    true_prob = self.args.default_valuation
                
                samples = np.random.binomial(self.SAMPLES_PER_STEP, true_prob)
                
                old_sample_mean = self.inst['valuation'][var.name]
                new_sample_mean = (self.inst['valuation'][var.name] * self.inst['sample_size'][var.name] + samples) / (self.inst['sample_size'][var.name] + self.SAMPLES_PER_STEP)
                
                self.inst['valuation'][var.name] = new_sample_mean
                self.inst['sample_size'][var.name] += self.SAMPLES_PER_STEP
                
                print('\n>> Drawn {} more samples for parameter {} ({} positives)'.format(self.SAMPLES_PER_STEP, var, samples))
                print('>> MLE is now: {:.3f} (difference: {:.3f})'.format(new_sample_mean, new_sample_mean - old_sample_mean))
                print('>> Total number of samples is now: {}\n'.format(self.inst['sample_size'][var.name]))
        
        
        
    def update(self, params, mode):
        
        ##### UPDATE PARAMETER POINT
        _, self.inst['point'] = pmc_instantiate(self.pmc, self.inst['valuation'])
        
        if mode == "exploration": 
            if self.UPDATE:
            
                for var in params:
                    # Update sample size
                    self.prmc.parameters[var[0]].value = self.inst['sample_size'][var[0].name]
                    
                    # Update mean
                    self.prmc.update_distribution(var[0], self.inst)
                    
                    self.CVX.update_parameter(self.prmc, var[0])
                    
                    if self.opposite:
                        self.CVX_opp.update_parameter(self.prmc, var[0])
                    
                    # Update ordering over robust constraints
                    self.prmc.set_robust_constraints()
                
            else:

                self.prmc = pmc2prmc(self.pmc.model, self.pmc.parameters, self.pmc.scheduler_prob, self.inst['point'], self.inst['sample_size'], self.args, verbose = self.args.verbose)
                self.CVX = verify_prmc(self.prmc, self.pmc.reward, self.args.beta_penalty, self.args.robust_bound, verbose = self.args.verbose)
            
        else: 
            if self.UPDATE:
            
                for var in params:
                    # Update sample size
                    self.prmc.parameters[var].value = self.inst['sample_size'][var.name]
                    
                    # Update mean
                    self.prmc.update_distribution(var, self.inst)
                    
                    self.CVX.update_parameter(self.prmc, var)
                    
                    if self.opposite:
                        self.CVX_opp.update_parameter(self.prmc, var)
                    
                    # Update ordering over robust constraints
                    self.prmc.set_robust_constraints()
                
            else:

                self.prmc = pmc2prmc(self.pmc.model, self.pmc.parameters, self.pmc.scheduler_prob, self.inst['point'], self.inst['sample_size'], self.args, verbose = self.args.verbose)
                self.CVX = verify_prmc(self.prmc, self.pmc.reward, self.args.beta_penalty, self.args.robust_bound, verbose = self.args.verbose)
            
        
        
    def validate_derivatives(self, obj):
        
        print('\nValidation by perturbing parameters by +{}'.format(self.args.validate_delta))
        
        empirical_der = validate(self.solution_current, self.pmc.parameters, self.args, self.pmc, self.inst)
        relative_diff = (empirical_der/obj[0])-1
        
        for q,x in enumerate(self.pmc.parameters):
            print('- Parameter {}, val: {:.3f}, LP: {:.3f}, diff: {:.3f}'.format(
                    x,  empirical_der[q], obj[0], relative_diff[q]))
            
        min_deriv_val = self.pmc.parameters[ np.argmin(empirical_der) ]
        assert min_deriv_val in self.PAR
        assert np.isclose( np.min(empirical_der), obj[0] )
        
    
def sample_uniform(L):
    """
    Sample uniformly (randomly)
    """
    
    print('Sample uniformly...')
    
    idx = np.array([np.random.randint(0, len(L.prmc.parameters_pmc))])            
    PAR = L.prmc.parameters_pmc[idx]
    
    return PAR

def sample_lowest_count(L):
    """
    Always sample from parameter with lowest sample size so far.
    """
    
    print('Sample biggest interval...')
    
    # Get parameter with minimum number of samples so far
    par_samples = {}
    for key in L.prmc.parameters_pmc:
        par_samples[key] = L.inst['sample_size'][key.name]
        
    PAR = [min(par_samples, key=par_samples.get)]

    return PAR

def sample_importance(L):
    """
    Sample greedily based on importance factor
    """

    print('Sample based on importance factor (expVisits * intervalWidth)...')

    importance, dtmc = parameter_importance_exp_visits(L.pmc, L.prmc, L.inst, L.CVX_opp)
    PAR = [max(importance, key=importance.get)]
    
    return PAR

def sample_importance_proportional(L):
    """
    Sample proportional to importance factors
    """
    
    print('Weighted sampling based on importance factor (expVisits * intervalWidth)...')
    
    importance, dtmc = parameter_importance_exp_visits(L.pmc, L.prmc, L.inst, L.CVX_opp)
    
    keys    = list(importance.keys())
    weights = np.array(list(importance.values()))
    weights_norm = weights / sum(weights)
    
    PAR = [np.random.choice(keys, p=weights_norm)]
    
    return PAR

def sample_derivative(L):
    """
    Sample greedily based on the highest/lowest derivative
    """
    
    print('Sample based on biggest absolute derivative...')
    
    # Create object for computing gradients
    G = gradient(L.prmc, L.args.robust_bound)
    
    # Update gradient object with current solution
    G.update_LHS(L.prmc, L.CVX)
    G.update_RHS(L.prmc, L.CVX)

    # Check if matrix has correct size
    assert G.J.shape[0] == G.J.shape[1]
    
    if L.args.robust_bound == 'lower':
        direction = GRB.MAXIMIZE
    else:
        direction = GRB.MINIMIZE
        
    _, idx, obj = solve_cvx_gurobi(G.J, G.Ju, L.prmc.sI, L.args.num_deriv,
                                direction=direction, verbose=L.args.verbose)

    PAR = [L.prmc.paramIndex[v] for v in idx]
    
    if not L.args.no_gradient_validation:  
        L.validate_derivatives(obj)
    
    return PAR
    
def sample_exploration(L):
    # Dit is waar de Exploration MDP gebruikt moet worden om de juiste variabeles te returnen 

    #Stappen: Bij de __init__ een functie callen die de exploration MDP bouwt. Hier de transition rewards berekenen door GRB te doen. De reward voor een transition hangt af van de afgeleide van de parameter van de ondergrond waar je terecht komt. 

    # DAN: Maak je reward model met die net gemaakte transition matrix. Laat dat MDP-probleem oplossen door stormpy. Die geeft een scheduler. 

    # Dan: Run door je mdp met die specifieke scheduler, bijhoudend welke parameters gesampled worden. Die lijst van parameters wordt gereturned.

    # Example van PAR: [<Variable v73 [id = 46]>]
    print("Sample based on solution to the exploration mdp")

    # Create object for computing gradients
    G = gradient(L.prmc, L.args.robust_bound)
    
    # Update gradient object with current solution
    G.update_LHS(L.prmc, L.CVX)
    G.update_RHS(L.prmc, L.CVX)

    # Check if matrix has correct size
    assert G.J.shape[0] == G.J.shape[1]
    
    if L.args.robust_bound == 'lower':
        direction = GRB.MAXIMIZE
    else:
        direction = GRB.MINIMIZE

    _, idx, obj = solve_cvx_gurobi(G.J, G.Ju, L.prmc.sI, len(L.prmc.parameters),
                                direction=direction, verbose=L.args.verbose) # The model has len(L.prmc.parameters) parameters. This finds the derivative of all of them. 

    derivatives = {}
    
    for i in range(len(idx)):   
        derivatives[idx[i]] = obj[i]

    
    if not L.args.no_gradient_validation:  
        L.validate_derivatives(obj)

    transition_reward_matrix = make_reward_matrix(derivatives, L.state_dict, L.terrain_dict, L.max_exploration_steps)
    PAR = [L.prmc.paramIndex[1]]

    #define the reward model
    reward_models = {}
    reward_models['transition rewards'] = stormpy.SparseRewardModel(optional_transition_reward_matrix=transition_reward_matrix)

    # State labeling is required, every state has a label
    state_labeling = stormpy.storage.StateLabeling(len(L.state_dict) * (L.max_exploration_steps + 1))
    
    # Add a label for every terrain in the dictionary
    labels = {str(x) for x in L.terrain_dict.values()}
    for label in labels:
        state_labeling.add_label(label)
    
    #Add initial state
    state_labeling.add_label('init')
    state_labeling.add_label_to_state('init', L.state_dict[(0,0)]) #Initial state = (0,0)
    #Loop over every (x,y) per layer, add the correct terrain label to the correct state.
    for layer in range(L.max_exploration_steps + 1): 
        for (x,y) in L.state_dict.keys(): 
            state_labeling.add_label_to_state(str(L.terrain_dict[(x,y)]), L.state_dict[(x,y)] + (layer) * len(L.state_dict))


    #Collect components and build the mdp
    components = stormpy.SparseModelComponents(transition_matrix=L.transition_matrix, state_labeling=state_labeling,
                                               reward_models=reward_models, rate_transitions=False)
    mdp = stormpy.storage.SparseMdp(components)

    #Find the optimal scheduler for the mdp:
    prop = "Rmax=? [C]" # maximize cumulative reward
    properties = stormpy.parse_properties(prop)

    result = stormpy.model_checking(mdp, properties[0], only_initial_states = True, extract_scheduler=True)

    assert result.has_scheduler
    scheduler = result.scheduler

    #Assuming we do have a scheduler, we simulate it for L.max_exploration_steps, keeping track of the terrains we have visited.

    current_state = L.current_state
    print(current_state)
    parameters_to_sample = []
    simulator = stormpy.simulator.create_simulator(mdp, seed = 42)

    for _ in range(L.max_exploration_steps):
        choice = scheduler.get_choice(current_state)
        action = choice.get_deterministic_choice()
        
        new_state, _, labels = simulator.step(action)
        # Extract the parameter from the label and add it to the list
        parameters_to_sample.append(L.prmc.paramIndex[int(list(labels)[0])])  #This only works if all the states only have a single label! But we build the mdp ourselves so should be manageable.
        current_state = new_state

    # Ensure the next step continues where we left off
    L.current_state = current_state

    # Count occurrences of parameters along the optimal path
    parameters_to_sample = [(par, parameters_to_sample.count(par) * L.SAMPLES_PER_STEP) for par in set(parameters_to_sample)]
    return parameters_to_sample


def make_reward_matrix(derivatives, state_dict, terrain_dict, max_exploration_steps):
    # This matrix will be very similar to the transition matrix used to build the mdp. The difference is that in this case, the values in the matrix will not be transition probabilities but transition rewards instead.
    state_dict_keys = state_dict.keys()
    row_counter = 0
    builder = stormpy.SparseMatrixBuilder(rows=0, columns=0, entries=0, force_dimensions=False,
                                          has_custom_row_grouping=True, row_groups=0)
    

    for layer in range(max_exploration_steps):
        for (x,y), value in state_dict.items():
            
            #idle in place
            builder.new_row_group(row_counter)
            builder.add_next_value(row_counter, value + (layer + 1) * len(state_dict), -1 * derivatives[terrain_dict[(x,y)]]) 
            row_counter += 1

            # step to the left:
            if (x-1,y) in state_dict_keys:
                builder.add_next_value(row_counter, state_dict[(x-1, y)] + (layer + 1) * len(state_dict), -1 * derivatives[terrain_dict[(x-1,y)]])
                row_counter += 1 
            
            # step to the right:
            if (x+1,y) in state_dict_keys:
                builder.add_next_value(row_counter, state_dict[(x+1, y)] + (layer + 1) * len(state_dict), -1 * derivatives[terrain_dict[(x+1,y)]]) 
                row_counter += 1 # increment row_counter since we have added a row. 

            # step up:
            if (x,y-1) in state_dict_keys:
                builder.add_next_value(row_counter, state_dict[(x, y-1)] + (layer + 1) * len(state_dict), -1 * derivatives[terrain_dict[(x,y-1)]])
                row_counter += 1
            
            # step down:
            if (x,y+1) in state_dict_keys:
                builder.add_next_value(row_counter, state_dict[(x, y+1)] + (layer + 1) * len(state_dict), -1 * derivatives[terrain_dict[(x,y+1)]])
                row_counter += 1

            # Make sure all final state transitions yield no reward: 
        if layer == max_exploration_steps - 1:
                
            #Make all final state transitions yield no reward: 
            for value in state_dict.values(): 
                builder.new_row_group(row_counter)
                builder.add_next_value(row_counter, value + (layer+1) * len(state_dict), 0)
                row_counter += 1

            

    #After processing all key-value pairs we build the matrix
    reward_matrix = builder.build()
    return reward_matrix

def make_exploration_utils(statespace_file_path, max_exploration_steps):
    # The mdp can be constructed from only the statespace file. 
    # This function creates the state dictionary, terrain dictionary as well as  the transition matrix for the mdp.

    #Hier: Per state in de MC willen we een state in de mdp. De transities van die state worden bepaald door de omliggende states in de MC. Tijdens het definieren van de MC wordt er ook een file gemaakt die bijhoudt welke x,y bij welke state_id horen. Het kan zijn dat zelfde x,y bij meerdere state_ids hoort, namelijk 1 state met pakket, 1 zonder pakket. Voor elke combinatie x,y moet er een state in de mdp zijn, met als omliggende states (x-1, y), (x+1, y), ... etc. Het kan handig zijn om de maximale x,y te hebben, dan kan je makkelijker de randen van de grid bepalen. Ook moet ergens de ondergrond van een state gedefinieerd zijn. 

    # Format voor dat bestand: 
    # x y state_id ondergrond. 
    # De eerste regel bevat de dimensies van de grid, spatie ertussen: X Y

    f = open(r'{}'.format(statespace_file_path), 'r')
    firstline = f.readline() #Lees de eerste regel met info over dimensies
    X = int(firstline.split(" ")[-2])
    Y = int(firstline.split(" ")[-1]) # De dimensies van de grid. 

    print("The dimensions of the grid: {} by {}".format(X,Y))

    lines = f.readlines() #Misschien niet handig als het heel veel lines zijn. 

    state_dict = {}
    terrain_dict = {}

    #The state_dict and terrain_dict are separate since we don't have access to the terrain of a (x,y) pair at the point we assign it a state id. 

    state_counter = 0 #Counter tracks what states we have defined in the mdp. Used when assigning unseen (x,y) a fresh state id.
    for line in lines:
        # Per line you get x,y, state_id, terrain. If this x,y has not yet been processed, add it to the dictts. 
        
        x, y, state_id, terrain = line.split(" ")
        x, y, terrain = int(x), int(y), int(terrain)       
        
        if not (x,y) in state_dict.keys(): 
            # This pair requires a state id since it doesn't have one yet:
            state_dict[(x,y)] = state_counter 
            state_counter += 1

            terrain_dict[(x,y)] = terrain 

    # Now: loop through the dicts to create the transition matrix. 
    row_counter = 0 #Counter tracks how many rows we have added to the matrix. This is used when deciding where to add new actions from new states. 
    state_dict_keys = state_dict.keys()
    builder = stormpy.SparseMatrixBuilder(rows=0, columns=0, entries=0, force_dimensions=False,
                                          has_custom_row_grouping=True, row_groups=0)
    
    # Add a layer of our statespace for the max amount of steps we want per sampling run:
    # The offset is the number of states. The idea is that the final layer does not yield any reward, allowing for scheduler extraction of optimal policy

    for layer in range(max_exploration_steps):

        for (x,y), value in state_dict.items():

            builder.new_row_group(row_counter)
            # idle in place
            builder.add_next_value(row_counter, value + (layer + 1) * len(state_dict), 1)
            row_counter += 1

            # step to the left:
            if (x-1,y) in state_dict_keys:
                builder.add_next_value(row_counter, state_dict[(x-1, y)] + (layer + 1) * len(state_dict), 1) # from this state, to the state (x-1,y), prob 1
                row_counter += 1 # increment row_counter since we have added a row. 
            
            # step to the right:
            if (x+1,y) in state_dict_keys:
                builder.add_next_value(row_counter, state_dict[(x+1, y)] + (layer + 1) * len(state_dict), 1) 
                row_counter += 1 # increment row_counter since we have added a row. 

            # step up:
            if (x,y-1) in state_dict_keys:
                builder.add_next_value(row_counter, state_dict[(x, y-1)] + (layer + 1) * len(state_dict), 1)
                row_counter += 1
            
            # step down:
            if (x,y+1) in state_dict_keys:
                builder.add_next_value(row_counter, state_dict[(x, y+1)] + (layer + 1) * len(state_dict), 1)
                row_counter += 1

        if layer == max_exploration_steps - 1:
            #Make all final states absorbing:
            for value in state_dict.values():
                builder.new_row_group(row_counter)
                builder.add_next_value(row_counter, value + (layer+1) * len(state_dict), 1)
                row_counter += 1


            
    #After processing all lines:
    transition_matrix = builder.build()

    #for info:
    print("States in mdp: {}".format(len(state_dict) * (max_exploration_steps + 1)))
    print("Nr of choices in mdp: {}".format(row_counter))

    return state_dict, terrain_dict, transition_matrix


# %%