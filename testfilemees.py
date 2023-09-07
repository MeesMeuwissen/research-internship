from core.experiments.generate_slipgrids import generate_pmc_random_drn
from core.learning.classes import make_exploration_mdp
import numpy as np


ROOT_DIR = "/Users/Mees_1/prmc-sensitivity/models/slipgrid_learning"
N = 20
terrain = np.array([[1]* 20] *20) #Voor testen is alles hetzelfde terrein
terrain2 = np.array([[i] * 20 for i in range(20)]) # Terrein per rij verschillend. 
model_name = "model_mees_more_terrains"
loc_package = (12,14)
loc_warehouse = (19,19)
reward = 0.01

statespace_file_path = "/Users/Mees_1/prmc-sensitivity/models/slipgrid_learning/model_mees_more_terrains_statespace.txt"

#generate_pmc_random_drn(ROOT_DIR, N, terrain2, model_name, loc_package, loc_warehouse, reward)

state_dict, terrain_dict, trans_matrix = make_exploration_mdp(statespace_file_path, 20)

print(state_dict)

