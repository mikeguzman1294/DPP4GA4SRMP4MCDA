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
VERBOSE = True
OUTPUT = "./tune_parameters_grid_crossover_mutation.csv"

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
                   "--mutation_random_profile_perturbation__perturbation_scale" : {"min" : 0.1, "max" : 0.5, "step" : 0.1}, # Will be set at hand
                   "--mutation_random_profile_perturbation__individual_criterion_proba" : {"init" : 1.0}, # Fixed
                   "--mutation_random_weights_perturbation__perturbation_scale" : {"min" : 0.1, "max" : 0.5, "step" : 0.1}, # Will be set at hand
                   "--mutation_random_weights_perturbation__individual_criterion_proba" : {"init" : 1.0}, # Fixed
                   "--mutation_shrink_profiles__shrink_factor" : {"min" : 0.2, "max" : 1.0, "step" : 0.2}, # Will be set at hand
                   "--mutation_shrink_profiles__individual_criterion_proba" : {"init" : 1.0}, # Fixed
                   "--mutation_expand_profiles__expand_factor" : {"min" : 0.2, "max" : 1.0, "step" : 0.2}, # Will be set at hand
                   "--mutation_expand_profiles__individual_criterion_proba" : {"init" : 1.0}, # Fixed
                   "--prepare_new_population__elitism_ratio" : {"min" : 0.1, "max" : 0.5, "step" : 0.1},
                   "--prepare_new_population__random_ratio" : {"min" : 0.1, "max" : 0.5, "step" : 0.1},
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
                   "--estimate_decision_maker__population_size" : {"min" : 100, "max" : 300, "step" : 50},
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

# If this file has been run with no arguments, we spawn processes for particular pairs of crossover/mutation
crossovers = [arg for arg in HYPERPARAMETERS if arg[:16] == "--make_crossover"]
mutations = [arg for arg in HYPERPARAMETERS if arg[:15] == "--make_mutation"]
if len(sys.argv) == 1 :
    for crossover in crossovers :
        for mutation in mutations :
            subprocess.Popen("python3 %s %s %s" % (sys.argv[0], crossover, mutation), shell=True, stdin=None, stdout=None, stderr=None, close_fds=True)
    sys.exit()
if VERBOSE : print("Spawned process %d for crossover %s and mutation %s" % (os.getpid(), sys.argv[1], sys.argv[2]))

# Set probabilities of crossovers and mutations to match those to use in the process
for crossover in crossovers :
    HYPERPARAMETERS[crossover]["init"] = float(sys.argv[1] == crossover)
for mutation in mutations :
    HYPERPARAMETERS[mutation]["init"] = float(sys.argv[2] == mutation)
    for arg in HYPERPARAMETERS :
        if mutation[17:-12] in arg and mutation != sys.argv[2] and "min" in HYPERPARAMETERS[arg] :
            print("." * 10, arg, "is not used")
            HYPERPARAMETERS[arg] = {"init" : HYPERPARAMETERS[arg]["min"]}
            
# Output file
OUTPUT = OUTPUT.replace("crossover", sys.argv[1])
OUTPUT = OUTPUT.replace("mutation", sys.argv[2])
output = open(OUTPUT, "w", buffering=1)

# Grid search of all the parameters that can vary
varying_parameters = [arg for arg in HYPERPARAMETERS if "init" not in HYPERPARAMETERS[arg]]
varying_values = [list(numpy.arange(HYPERPARAMETERS[arg]["min"], HYPERPARAMETERS[arg]["max"] + HYPERPARAMETERS[arg]["step"], HYPERPARAMETERS[arg]["step"])) for arg in varying_parameters]
arguments = {arg : HYPERPARAMETERS[arg]["init"] for arg in HYPERPARAMETERS if "init" in HYPERPARAMETERS[arg]}
for config in itertools.product(*varying_values) :
    for i in range(len(config)) :
        arguments[varying_parameters[i]] = config[i]
    results = run_experiment(arguments, output)

#####################################################################################################################################################
#####################################################################################################################################################
