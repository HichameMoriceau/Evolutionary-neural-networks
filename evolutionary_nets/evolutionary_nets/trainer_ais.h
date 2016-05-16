#ifndef TRAINER_CLONAL_SELECTION_H
#define TRAINER_CLONAL_SELECTION_H

#include "trainer.h"

class Trainer_AIS : public Trainer
{
public:
    vector<NeuralNet>   population;

    // if variance < epsilon then stop GA
    double              epsilon = 1.0f;

public:
                        Trainer_AIS();

    void                train(Data_set data_set, NeuralNet &net);

    void                train(Data_set data_set, NeuralNet &net, mat &results_score_evolution);

    void                train_weights(data_subset training_set, data_subset validation_set, NeuralNet &net, unsigned int nb_epochs, mat &results_score_evolution);

    NeuralNet           train_topology_plus_weights(Data_set data_set, net_topology max_topo, mat &results_score_evolution, unsigned int selected_mutation_scheme);

    NeuralNet           cross_validation_training(Data_set data_set, net_topology min_topo, net_topology max_topo, mat &results_score_evolution, double &avrg_score, double &avrg_acc, unsigned int selected_mutation_scheme);

    NeuralNet           evolve_through_generations(Data_set data_set, net_topology min_topo, net_topology max_topo, unsigned int nb_epochs, mat &results_cost_and_score_evolution, unsigned int index_cross_validation_section, unsigned int selected_mutation_scheme);

    void                elective_accuracy(vector<NeuralNet> pop, Data_set data_set, double &ensemble_accuracy, double &ensemble_score);

    unsigned int        get_nb_identical_elements(mat A, mat B);

    void                initialize_random_population(unsigned int pop_size, net_topology max_topo);

    // helper methods
private:
    double              compute_score_variance(vector<NeuralNet> population, data_subset data_set);

    double              compute_score_variance(vector<vec> population, data_subset data_set);

    double              compute_score_stddev(vector<NeuralNet> population, data_subset data_set);

    double              compute_score_stddev(vector<vec> population, data_subset data_set);

    double              compute_score_mean(vector<NeuralNet> population, data_subset data_set);

    double              compute_score_mean(vector<vec> population, data_subset data_set);

    double              compute_score_median(vector<NeuralNet> population, data_subset data_set);

    double              compute_score_median(vector<vec> population, data_subset data_set);

    vector<NeuralNet>   generate_population(unsigned int pop_size, net_topology t, data_subset training_set);

    vector<vec>         generate_random_genome_population(unsigned int quantity, NeuralNet template_net);

    vector<vec>         generate_random_topology_genome_population(unsigned int quantity, NeuralNet largest_net);

    vector<vec>         generate_random_topology_genome_population(unsigned int quantity, net_topology min_topo, net_topology max_topo);

    void                evaluate_population(vector<NeuralNet> &pop, data_subset d);

    void                differential_evolution(vector<NeuralNet> &population, data_subset data_set);

    void                clonal_selection_topology_evolution(vector<vec> &population, data_subset training_set, net_topology min_topo, net_topology max_topology, unsigned int selected_mutation_scheme);

    void                mutative_crossover(unsigned int R, double CR, double F, unsigned int genome_length, net_topology min_topo, net_topology max_topo, vec original_model, vec &candidate_model, vec indiv_a, vec indiv_b, vec indiv_c);

    vector<vec>         select(unsigned int quantity, vector<vec> pop, data_subset training_set, net_topology max_topo);

    vector<vec>         generate_clones(unsigned int nb_clones, vec indiv);

    unsigned int        compute_nb_clones(double beta, int pop_size, int index);

    vec                 hypermutate(vec original, double mutation_rate);

    vector<vec>         add_all(vector<vector<vec>> all_populations, unsigned int* nb_clones_array);

    double              clip(double n, double min, double max);

    double              mutation_scheme_DE_rand_1(double F, double x_rand_1, double x_rand_2, double x_rand_3);

    vector<NeuralNet>   convert_population_to_nets(vector<vec> genome_population);

    vector<vec>         convert_population_to_genomes(vector<NeuralNet> net_pop, net_topology largest_topology);

public:
    NeuralNet           get_best_model(vector<NeuralNet> pop);

    NeuralNet           get_best_model(vector<vec> genome_pop);

    vec                 get_genome(NeuralNet n, net_topology largest_topology);

    NeuralNet           generate_net(vec genome);

    unsigned int        get_genome_length(net_topology t);

    unsigned int        get_population_size();

    mat                 get_population_scores(data_subset d);

    vector<NeuralNet>   get_population(){   return population;}

    void                set_population( vector<NeuralNet> pop) { population = pop; }

    void                insert_individual( NeuralNet indiv){    population[population.size()/2] = indiv;}

    double              get_epsilon() const;

    void                set_epsilon(double e);

    double              f_rand(double fMin, double fMax);

};

#endif // TRAINER_CLONAL_SELECTION_H
