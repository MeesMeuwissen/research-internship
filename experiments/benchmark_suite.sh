#!/bin/bash
cd ..;
echo -e "START BENCHMARK SUITE...";
# pDTMC benchmarks
timeout 3600s python3 run_cav23.py --model 'models/pdtmc/brp16_2.pm' --formula 'P=? [ F s=5 ]' --validate_delta 1e-3 --output_folder 'output/benchmark_suite_cav/' --explicit_baseline --goal_label '(s = 5)';
timeout 3600s python3 run_cav23.py --model 'models/pdtmc/brp64_3.pm' --formula 'P=? [ F s=5 ]' --validate_delta 1e-3 --output_folder 'output/benchmark_suite_cav/' --explicit_baseline --goal_label '(s = 5)';
timeout 3600s python3 run_cav23.py --model 'models/pdtmc/brp512_5.pm' --formula 'P=? [ F s=5 ]' --validate_delta 1e-3 --output_folder 'output/benchmark_suite_cav/' --explicit_baseline --goal_label '(s = 5)';
timeout 3600s python3 run_cav23.py --model 'models/pdtmc/crowds3_5.pm' --formula 'P=? [F "observe0Greater1" ]' --validate_delta 1e-3 --output_folder 'output/benchmark_suite_cav/' --explicit_baseline --goal_label 'observe0Greater1';
timeout 3600s python3 run_cav23.py --model 'models/pdtmc/crowds6_5.pm' --formula 'P=? [F "observe0Greater1" ]' --validate_delta 1e-3 --output_folder 'output/benchmark_suite_cav/' --explicit_baseline --goal_label 'observe0Greater1';
timeout 3600s python3 run_cav23.py --model 'models/pdtmc/crowds10_5.pm' --formula 'P=? [F "observe0Greater1" ]' --validate_delta 1e-3 --output_folder 'output/benchmark_suite_cav/' --explicit_baseline --goal_label 'observe0Greater1';
timeout 3600s python3 run_cav23.py --model 'models/pdtmc/crowds20_10.pm' --formula 'P=? [F "observe0Greater1" ]' --validate_delta 1e-3 --output_folder 'output/benchmark_suite_cav/' --explicit_baseline --goal_label 'observe0Greater1';
timeout 3600s python3 run_cav23.py --model 'models/pdtmc/nand2_4.pm' --formula 'P=? [F "target" ]' --parameters 'models/pdtmc/nand.json' --validate_delta 1e-3 --output_folder 'output/benchmark_suite_cav/' --explicit_baseline --goal_label 'target';
timeout 3600s python3 run_cav23.py --model 'models/pdtmc/nand5_10.pm' --formula 'P=? [F "target" ]' --parameters 'models/pdtmc/nand.json' --validate_delta 1e-3 --output_folder 'output/benchmark_suite_cav/' --explicit_baseline --goal_label 'target';
timeout 3600s python3 run_cav23.py --model 'models/pdtmc/nand10_15.pm' --formula 'P=? [F "target" ]' --parameters 'models/pdtmc/nand.json' --validate_delta 1e-3 --output_folder 'output/benchmark_suite_cav/' --explicit_baseline --goal_label 'target';
#
# pMDP benchmarks
timeout 3600s python3 run_cav23.py --model 'models/pmdp/coin/coin4.pm' --formula 'Pmin=? [ F "finished" & "all_coins_equal_1" ]' --default_valuation 0.5 --validate_delta 1e-3 --output_folder 'output/benchmark_suite_cav/' --explicit_baseline --goal_label 'finished';
timeout 3600s python3 run_cav23.py --model 'models/pmdp/CSMA/csma2_4_param.nm' --formula 'R{"time"}max=? [ F "all_delivered" ]' --default_valuation 0.1 --validate_delta 1e-3 --output_folder 'output/benchmark_suite_cav/' --explicit_baseline;
timeout 3600s python3 run_cav23.py --model 'models/pmdp/virus/virus.pm' --formula 'R{"attacks"}max=? [F s11=2 ]' --default_valuation 0.1 --validate_delta 1e-3 --output_folder 'output/benchmark_suite_cav/' --explicit_baseline;
timeout 3600s python3 run_cav23.py --model 'models/pmdp/wlan/wlan0_param.nm' --formula 'R{"time"}max=? [ F s1=12 | s2=12 ]' --default_valuation 0.1 --validate_delta 1e-3 --output_folder 'output/benchmark_suite_cav/' --explicit_baseline;
#
# STTT Drone
timeout 3600s python3 run_cav23.py --model 'models/sttt-drone/drone_model.nm' --formula 'Pmax=? [F attarget ]' --validate_delta 1e-3 --output_folder 'output/benchmark_suite_cav/' --default_valuation 0.07692307692 --explicit_baseline --goal_label '(((x > (15 - 2)) & (y > (15 - 2))) & (z > (15 - 2)))';
#
# POMDP benchmarks
timeout 3600s python3 run_cav23.py --model 'models/pomdp/maze/maze_simple_extended_m5.drn' --formula 'Rmin=? [F "goal"]' --validate_delta 1e-3 --output_folder 'output/benchmark_suite_cav/' --explicit_baseline; 
timeout 3600s python3 run_cav23.py --model 'models/pomdp/network/network2K-20_T-8_extended-simple_full.drn' --formula 'R{"dropped_packets"}=? [F "goal"]' --validate_delta 1e-3 --output_folder 'output/benchmark_suite_cav/' --explicit_baseline --goal_label 'goal'; 
timeout 3600s python3 run_cav23.py --model 'models/pomdp/drone/pomdp_drone_4-2-mem1-simple.drn' --formula 'P=? ["notbad" U "goal"]' --validate_delta 1e-3 --output_folder 'output/benchmark_suite_cav/' --explicit_baseline --goal_label 'goal';
timeout 3600s python3 run_cav23.py --model 'models/pomdp/drone/pomdp_drone_4-2-mem5-simple.drn' --formula 'P=? ["notbad" U "goal"]' --validate_delta 1e-3 --output_folder 'output/benchmark_suite_cav/' --explicit_baseline --goal_label 'goal';
timeout 3600s python3 run_cav23.py --model 'models/satellite/pomdp_satellite_36_sat_5_act_0_65_dist_5_mem_06_sparse_full.drn' --formula 'P=? [F "goal"]' --default_valuation 0.1 --validate_delta 1e-3 --output_folder 'output/benchmark_suite_cav/' --explicit_baseline --verbose --no_prMC;
timeout 3600s python3 run_cav23.py --model 'models/satellite/pomdp_prob_36_sat_065_dist_1_obs_diff_orb_len_40.drn' --formula 'P=? [F "goal"]' --default_valuation 0.2 --validate_delta 1e-3 --output_folder 'output/benchmark_suite_cav/' --explicit_baseline --no_prMC;
%
python3 parse_output.py --folder 'output/benchmark_suite_cav/' --table_name 'tables/benchmark_suite_cav'