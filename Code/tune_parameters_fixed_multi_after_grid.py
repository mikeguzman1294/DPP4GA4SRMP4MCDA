#####################################################################################################################################################
##################################################################### RESOURCES #####################################################################
#####################################################################################################################################################

"""
    
"""

#####################################################################################################################################################
#################################################################### GLOBAL STUFF ###################################################################
#####################################################################################################################################################

# Imports
import itertools
import subprocess
import sys
import os
import ast
import numpy
import random

# Hyperparameters optimization
NB_TESTS = 1
VERBOSE = True
#OUTPUT = "./tune_parameters_fixed_multi_after_grid.csv"
OUTPUT = "tune_parameters_fixed_multi_after_grid.csv"

# Hyperparameters to optimize
HYPERPARAMETERS = {"--nb_criteria" : {"init" : 11}, # Fixed
                   "--nb_profiles" : {"init" : 3}, # Fixed
                   "--nb_alternatives" : {"init" : 50}, # Fixed
                   "--eval_on_test_set_1" : {"init" : True}, # Fixed
                   "--nb_alternatives_test_set_2" : {"init" : 300}, # Fixed
                   "--nb_comparisons" : {"init" : 100}, # Fixed
                   "--debug_mode" : {"init" : False}, # Fixed
                   "--random_seed" : {"init" : None}, # Will be set at hand
                   "--output_directory" : {"init" : "."}, # Fixed
                   "--mutation_random_profile_perturbation__perturbation_scale" : {"init" : 0.1}, # Fixed
                   "--mutation_random_profile_perturbation__individual_criterion_proba" : {"init" : 1.0}, # Fixed
                   "--mutation_random_weights_perturbation__perturbation_scale" : {"init" : 0.1}, # Fixed
                   "--mutation_random_weights_perturbation__individual_criterion_proba" : {"init" : 1.0}, # Fixed
                   "--mutation_shrink_profiles__shrink_factor" : {"init" : 0.7}, # Fixed
                   "--mutation_shrink_profiles__individual_criterion_proba" : {"init" : 1.0}, # Fixed
                   "--mutation_expand_profiles__expand_factor" : {"init" : 0.7}, # Fixed
                   "--mutation_expand_profiles__individual_criterion_proba" : {"init" : 1.0}, # Fixed
                   "--prepare_new_population__elitism_ratio" : {"init" : 0.4}, # Fixed
                   "--prepare_new_population__random_ratio" : {"init" : 0.1}, # Fixed
                   "--select_solutions__nb_solutions" : {"init" : 2}, # Fixed
                   "--select_solutions__strategy" : {"init" : "roulette"}, # Fixed
                   "--make_crossover__crossover_swap_weights_probability" : {"init" : 0.0}, # Fixed
                   "--make_crossover__crossover_swap_orders_probability" : {"init" : 0.0}, # Fixed
                   "--make_crossover__crossover_swap_profiles_probability" : {"init" : 0.0}, # Fixed
                   "--make_crossover__crossover_mix_criteria_probability" : {"init" : 0.5}, # Fixed
                   "--make_crossover__crossover_mix_criteria_and_weights_probability" : {"init" : 0.5}, # Fixed
                   "--make_mutation__mutation_random_profile_perturbation_probability" : {"init" : 0.0}, # Fixed
                   "--make_mutation__mutation_random_weights_perturbation_probability" : {"init" : 0.2}, # Fixed
                   "--make_mutation__mutation_shrink_profiles_probability" : {"init" : 0.2}, # Fixed
                   "--make_mutation__mutation_expand_profiles_probability" : {"init" : 0.0}, # Fixed
                   "--make_mutation__mutation_partially_reverse_order_probability" : {"init" : 0.2}, # Fixed
                   "--keep_or_drop_children__survival_probability" : {"init" : 0.0}, # Fixed
                   "--estimate_decision_maker__return_k_best" : {"init" : 1}, # Fixed
                   "--estimate_decision_maker__population_size" : {"init" : 100}, # Fixed ONLY CHANGED ORIGINAL 300
                   "--estimate_decision_maker__stop_after_non_evolving" : {"init" : 50}, # Fixed
                   "--estimate_decision_maker__check_identical_ratio" : {"init" : 0.1}, # Fixed
                   "--estimate_decision_maker__nb_profiles" : {"init" : 3}} # Fixed

#####################################################################################################################################################
##################################################################### FUNCTIONS #####################################################################
#####################################################################################################################################################

def run_experiment (arguments, output) :

    """
        Runs the given experiment for all seeds.
    """
    
    # Run command for each seed
    perfs = {"train" : [], "test_1" : [], "test_2" : [], "time" : []}
    if VERBOSE : print(arguments, flush=True)
    for seed in range(NB_TESTS) :
        if VERBOSE : print("%d/%d" % (seed+1, NB_TESTS), end="", flush=True)
        arguments["--random_seed"] = seed
        #command = "python3 learn_SRMP.py " + " ".join(arg + " " + str(arguments[arg]) for arg in arguments)
        command = "python learn_SRMP.py " + " ".join(arg + " " + str(arguments[arg]) for arg in arguments)
        result = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).stdout.read().decode("utf-8")
        result = ast.literal_eval(result)
        for entry in result :
            perfs[entry].append(float(result[entry]))
        print(arguments, ";", perfs["train"][-1], ";", perfs["test_1"][-1], ";", perfs["test_2"][-1], ";", perfs["time"][-1], file=output)
        if VERBOSE : print(" -> %f / %f / %f in %fs" % (perfs["train"][-1], perfs["test_1"][-1], perfs["test_2"][-1], perfs["time"][-1]), flush=True)
    
    # Return averages
    average_result = {entry : numpy.mean(perfs[entry]) for entry in perfs}
    if VERBOSE : print("Average perf: %f / %f / %f" % (average_result["train"], average_result["test_1"], average_result["test_2"]))
    if VERBOSE : print("Average time: %fs" % (average_result["time"]))
    return average_result
    
#####################################################################################################################################################
####################################################################### SCRIPT ######################################################################
#####################################################################################################################################################

# Work with the latest ipynb
'''if len(sys.argv) == 1 :
    command = ["ipynb-py-convert", "learn_SRMP.ipynb", "learn_SRMP.py"]
    subprocess.call(command)'''

# Output file
output = open(OUTPUT, "w", buffering=1)

# We make realizations with random probabilities for all operators, normalization is handled by the main program
arguments = {arg : HYPERPARAMETERS[arg]["init"] for arg in HYPERPARAMETERS if "init" in HYPERPARAMETERS[arg]}
results = run_experiment(arguments, output)

#####################################################################################################################################################
#####################################################################################################################################################