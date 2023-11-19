Efficiently Improving Estimations Of Markov Chains Using Exploring Agents
=============================

This repository contains python code used in the creation of a report for Research Internship. It is largely based on a [repository by LAVA_LAB](https://github.com/LAVA-LAB/prmc-sensitivity "repository by LAVA_LAB"), user instructions are under the link provided.

In this repository, a sampling method based on an exploration MDP is added. It is the default in the run_learning_RI file.

Additional arguments in this repo: 

 -- statespace 

In order to run experimentes using the exploration method, models require an additional statespace file. This is a txt file describing the dimensions and terrain of the model. The first line must contains the dimensions of the grid separated by a space: X Y. For every cell in the grid, there should be a line of the form "x y state_id terrain", specifying the relationship between the coordinates of the cell and the terrain of the cell. 
 
 
 
-- exploration_steps

This argument set the size of the finite horizon used. Default is 50. Higher values lead to faster run-time, but potentially less optimal sampling.

Example command to run the program and test it against random behaviour:

<code>
python run_learning_random.py --instance gridworld --model models/slipgrid_learning/20x20_20_terrains_random_double.drn --parameters models/slipgrid_learning/pmc_size=20_params=100_mle.json --formula 'Rmin=? [F "goal"]' --output_folder 'output/learning/' --num_deriv 1 --robust_bound 'upper' --uncertainty_model 'Hoeffding' --true_param_file models/slipgrid_learning/pmc_size=20_params=100.json --learning_iterations 3 --learning_samples_per_step 15 --statespace models/slipgrid_learning/20x20_20_terrains_random_double_statespace.txt</code>