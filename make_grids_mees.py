from core.experiments.generate_slipgrids import generate_pmc_learning_drn
import numpy as np

ROOT_DIR = "/Users/Mees_1/prmc-sensitivity/models/slipgrid_learning"
N = 20
terrain = np.array([[1]* 20] *20) #Voor testen is alles hetzelfde terrein
model_name = "model_mees_one_terrain"
loc_package = (12,14)
loc_warehouse = (20,20)
reward = 0.01

generate_pmc_learning_drn(ROOT_DIR, N, terrain, model_name, loc_package, loc_warehouse, reward)