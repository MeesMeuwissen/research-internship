Efficiently Improving Estimations Of Markov Chains Using Exploring Agents
=============================

This repository contains python code used in the creation of a report for Research Internship. It is largely based on a [repository by LAVA_LAB](https://github.com/LAVA-LAB/prmc-sensitivity "repository by LAVA_LAB"), user instructions are under the link provided.

In this repository, a sampling method based on an exploration MDP is added. It is the default in the run_learning_RI file.

Additional arguments in this repo: 
 -- statespace 

In order to run experimentes using the exploration method, models require an additional statespace file. This is a txt file describing the dimensions and terrain of the model. The first line must contains the dimensions of the grid separated by a space: X Y. For every cell in the grid, there should be a line of the form "x y state_id terrain", specifying the relationship between the coordinates of the cell and the terrain of the cell. 
 
 
 
-- exploration_steps

This argument set the size of the finite horizon used. Default is 50. Higher values lead to faster run-time, but potentially less optimal sampling.