from core.experiments.generate_slipgrids import generate_pmc_learning_drn
from core.learning.classes import make_exploration_utils
import numpy as np
import random 


ROOT_DIR = "/Users/Mees_1/prmc-sensitivity/models/slipgrid_learning"
N = 20


terrain = np.reshape([random.randint(0,19) for _ in range(400)], [20,20])#Voor testen is alles hetzelfde terrein
model_name = "20x20_20_terrains_random"
loc_package = (random.randint(1,19), random.randint(1,19))
loc_warehouse = (random.randint(1,19),random.randint(1,19))
reward = 0.01

statespace_file_path = "/Users/Mees_1/prmc-sensitivity/models/slipgrid_learning/model_mees_more_terrains_statespace.txt"

generate_pmc_learning_drn(ROOT_DIR, N, terrain, model_name, loc_package, loc_warehouse, reward)

#state_dict, terrain_dict, trans_matrix = make_exploration_mdp(statespace_file_path, 20)

print("Terrain generated")

#print(state_dict)

