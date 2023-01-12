#####################################################################################################################################################
##################################################################### RESOURCES #####################################################################
#####################################################################################################################################################

"""
    Performs a grid search of (a subset of) the hyperparameters of "learn_SRMP".
    Run with no argument, and it will spawn processes for all pairs crossover/mutation.
    Run with arguments crossover and mutation, and it will run for that specific pair.
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
NB_TESTS = 10
PROCESS_IDS = range(10)
NB_REALIZATIONS_PER_PROCESS = 1000
VERBOSE = True
OUTPUT = "./tune_parameters_random_multi_process.csv"

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
                   "--prepare_new_population__elitism_ratio" : {"init" : 0.1}, # Fixed
                   "--prepare_new_population__random_ratio" : {"init" : 0.1}, # Fixed
                   "--select_solutions__nb_solutions" : {"init" : 2}, # Fixed
                   "--select_solutions__strategy" : {"init" : "roulette"}, # Fixed
                   "--make_crossover__crossover_swap_weights_probability" : {"init" : None}, # Will be set at hand
                   "--make_crossover__crossover_swap_orders_probability" : {"init" : None}, # Will be set at hand
                   "--make_crossover__crossover_swap_profiles_probability" : {"init" : None}, # Will be set at hand
                   "--make_crossover__crossover_mix_criteria_probability" : {"init" : None}, # Will be set at hand
                   "--make_crossover__crossover_mix_criteria_and_weights_probability" : {"init" : None}, # Will be set at hand
                   "--make_mutation__mutation_random_profile_perturbation_probability" : {"init" : None}, # Will be set at hand
                   "--make_mutation__mutation_random_weights_perturbation_probability" : {"init" : None}, # Will be set at hand
                   "--make_mutation__mutation_shrink_profiles_probability" : {"init" : None}, # Will be set at hand
                   "--make_mutation__mutation_expand_profiles_probability" : {"init" : None}, # Will be set at hand
                   "--make_mutation__mutation_partially_reverse_order_probability" : {"init" : None}, # Will be set at hand
                   "--keep_or_drop_children__survival_probability" : {"init" : 0.0}, # Fixed
                   "--estimate_decision_maker__return_k_best" : {"init" : 1}, # Fixed
                   "--estimate_decision_maker__population_size" : {"init" : 250}, # Fixed
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
        command = "python3 learn_SRMP.py " + " ".join(arg + " " + str(arguments[arg]) for arg in arguments)
        result = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).stdout.read().decode("utf-8")
        result = ast.literal_eval(result)
        for entry in result :
            perfs[entry].append(float(result[entry]))
        print(arguments, ";", perfs["train"][-1], ";", perfs["test_1"][-1], ";", perfs["test_2"][-1], ";", perfs["time"][-1], file=output)
        if VERBOSE : print(" -> %f / %f / %f in %fs" % (perfs["train"][-1], perfs["test_1"][-1], perfs["test_2"][-1], perfs["time"][-1]), flush=True)
    
    # Return averages
    average_result = {entry : numpy.mean(perfs[dataset]) for entry in perfs}
    if VERBOSE : print("Average perf: %f / %f / %f" % (average_result["train"], average_result["test_1"], average_result["test_2"]))
    if VERBOSE : print("Average time: %fs" % (average_result["time"]))
    return average_result
    
#####################################################################################################################################################
####################################################################### SCRIPT ######################################################################
#####################################################################################################################################################

# Work with the latest ipynb
if len(sys.argv) == 1 :
    command = ["ipynb-py-convert", "learn_SRMP.ipynb", "learn_SRMP.py"]
    subprocess.call(command)

# If this file has been run with no arguments, we spawn processes
if len(sys.argv) == 1 :
    for i in PROCESS_IDS :
        subprocess.Popen("python3 %s %s" % (sys.argv[0], i), shell=True, stdin=None, stdout=None, stderr=None, close_fds=True)
    sys.exit()
if VERBOSE : print("Spawned process %s" % (sys.argv[1]))
numpy.random.seed(int(sys.argv[1]))

# Output file
OUTPUT = OUTPUT.replace("process", sys.argv[1])
output = open(OUTPUT, "w", buffering=1)

# We make realizations with random probabilities for all operators, normalization is handled by the main program
crossovers = [arg for arg in HYPERPARAMETERS if arg[:16] == "--make_crossover"]
mutations = [arg for arg in HYPERPARAMETERS if arg[:15] == "--make_mutation"]
arguments = {arg : HYPERPARAMETERS[arg]["init"] for arg in HYPERPARAMETERS if "init" in HYPERPARAMETERS[arg]}
for _ in range(int(NB_REALIZATIONS_PER_PROCESS)) :
    for arg in crossovers + mutations :
        arguments[arg] = numpy.random.rand()
    average_result = run_experiment(arguments, output)

#####################################################################################################################################################
#####################################################################################################################################################
