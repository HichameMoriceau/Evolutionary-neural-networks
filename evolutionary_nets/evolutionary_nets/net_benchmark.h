#ifndef NET_BENCHMARK_H
#define NET_BENCHMARK_H

#include <iostream>
#include <fstream>
#include <string>
#include <cstdio>
#include <chrono>
#include <thread>
#include <omp.h>
#include "data_set.h"
#include "neuralnet.h"
#include "trainer.h"
#include "trainer_de.h"
#include "trainer_pso.h"
#include "trainer_ais.h"

using namespace std;
using namespace chrono;

enum OPTIMIZATION_ALG{ DE=0, PSO=1, AIS=2};


struct exp_files{
  vector<string> dataset_filenames;
  unsigned int max_nb_err_func_calls;
  unsigned int nb_reps;
  unsigned int pop_size;
};

class Net_benchmark
{

private:
    unsigned int    nb_replicates;

    Data_set        data_set;

    NeuralNet       net;

    net_topology    max_topo;

    Trainer_DE      evo_trainer;

    ofstream        experiment_file;

public:
                // ctor
                Net_benchmark();
                // dtor
                ~Net_benchmark();

   void         run_benchmark(exp_files ef);

   double       find_termination_criteria_epsilon(unsigned int many_generations);

   void         train_topology(NeuralNet &net);

   void         set_topology(net_topology t);

   void         compute_perfs_test_validation(double &model_score_training_set,
                                              double &model_prediction_acc_training_set,
                                              double &model_score_validation_set,
                                              double &model_prediction_acc_validation_set);

   // print as matrix
   void         print_results_octave_format(ofstream &result_file, mat recorded_performances, string octave_variable_name);

private:
   // helper classes

   unsigned int count_nb_positive_examples(vec A);

   mat          evaluate_backprop_general_performances();

   // print as cell array
   void         print_results_octave_format(ofstream &result_file, vector<mat> recorded_performances, string octave_variable_name);

   double       corrected_sample_std_dev(mat best_scores);

   /**
    * @brief train_net_and_save_performances
    *           Train neural net up to largest topology,
    *           also writes the <experiment_file> and print the results
    * @param pop_size_GA        Size of the population for the Genetic Algorithm
    * @param nb_generations_GA  Maximum number of generations for the Genetic Algorithm
    * @param epsilon            Value of variance of the score of all individuals when Genetic Algorithm has converged
    */
   void         train_net_and_save_performances(unsigned int pop_size_GA, unsigned int nb_generations_GA, unsigned int selected_opt_alg, double epsilon, unsigned int selected_mutation_scheme);

   void         training_task(unsigned int i, string data_set_filename, unsigned int selected_opt_algorithm, double epsilon, unsigned int selected_mutation_scheme);

   mat          compute_learning_curves_perfs(vector<mat> &result_matrices_training_perfs, unsigned int selected_opt_alg, double epsilon, unsigned int selected_mutation_scheme);

   void         compute_scores_task(unsigned int i, net_topology max_topo, vector<mat> &replicated_results, mat &pop_sizes, double epsilon, unsigned int selected_mutation_scheme);

   mat          compute_learning_curves_population_size(vector<mat> &result_matrices_perfs_pop_size, mat &pop_sizes, double epsilon, unsigned int selected_mutation_scheme);

   mat          compute_learning_curves_dataset_size(vector<mat> &result_matrices_perfs_data_set_sizes, unsigned int selected_mutation_scheme);

   mat          average_matrices(vector<mat> results);

   mat          to_matrix(double a);

   mat          compute_replicate_error(vector<mat> result_matrices_training_perfs);

   mat          compute_pop_size_replicate_error(vector<mat> result_matrices_perfs_pop_size);

   const string get_current_date_time();
};

#endif // NET_BENCHMARK_H
