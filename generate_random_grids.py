from core.experiments.generate_slipgrids import generate_pmc_learning_drn
from core.learning.classes import make_exploration_utils
import numpy as np
import random 

# Can be used to create grids of random terrain 

ROOT_DIR = "/Users/Mees_1/prmc-sensitivity/models/slipgrid_learning"
N = 20 # Size of grid
num_terrains = 20


terrain = np.reshape([random.randint(0,num_terrains - 1) for _ in range(N**2)], [N,N])#
model_name = "{}x{}_{}_terrains_random_double".format(N,N,num_terrains)
loc_package = (N-1,0)
loc_warehouse = (0,N-1)
reward = 0.01

statespace_file_path = "/Users/Mees_1/prmc-sensitivity/models/slipgrid_learning/model_mees_more_terrains_statespace.txt"

generate_pmc_learning_drn(ROOT_DIR, N, terrain, model_name, loc_package, loc_warehouse, reward)
print("Terrain generated")


