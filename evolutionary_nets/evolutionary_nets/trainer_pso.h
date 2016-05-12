#ifndef TRAINER_PSO_H
#define TRAINER_PSO_H

#include "trainer.h"

class Trainer_PSO : public Trainer
{

private:
    vector<NeuralNet>   population;

    // if variance < epsilon then stop GA
    double              epsilon = 1.0f;
public:
    Trainer_PSO();

    void                train(Data_set data_set, NeuralNet &net);

    void                train(Data_set data_set, NeuralNet &net, mat &results_score_evolution);

    NeuralNet           train_topology_plus_weights(Data_set data_set, net_topology max_topo, mat &results_score_evolution);

    NeuralNet           cross_validation_training(Data_set data_set, net_topology min_topo, net_topology max_topo, mat &results_score_evolution, double &avrg_score, double &avrg_acc);

    NeuralNet           evolve_through_PSO(Data_set data_set, net_topology min_topo, net_topology max_topo, unsigned int nb_epochs, mat &results_cost_and_score_evolution, unsigned int index_cross_validation_section);

    double              f_rand(double fMin, double fMax);

    void                elective_accuracy(vector<NeuralNet> pop, Data_set data_set, double &ensemble_accuracy, double &ensemble_score);

    unsigned int        get_nb_identical_elements(mat A, mat B);

    void                initialize_random_population(unsigned int pop_size, net_topology max_topo);

    double              compute_score_variance(vector<NeuralNet> pop, data_subset data_set);

    double              compute_score_variance(vector<vec> pop, data_subset data_set);

    double              compute_score_stddev(vector<NeuralNet> pop, data_subset data_set);

    double              compute_score_stddev(vector<vec> pop, data_subset data_set);

    double              compute_score_mean(vector<NeuralNet> pop, data_subset data_set);

    double              compute_score_mean(vector<vec> pop, data_subset data_set);

    double              compute_score_median(vector<NeuralNet> pop, data_subset data_set);

    double              compute_score_median(vector<vec> pop, data_subset data_set);

    vector<NeuralNet>   generate_population(unsigned int pop_size, net_topology t, data_subset training_set);

    vector<vec>         generate_random_genome_population(unsigned int quantity, NeuralNet largest_net);

    vector<vec>         generate_random_topology_genome_population(unsigned int quantity, NeuralNet largest_net);

    vector<vec>         generate_random_topology_genome_population(unsigned int quantity, net_topology min_topo, net_topology max_topo);

    void                evaluate_population(vector<NeuralNet> &pop, data_subset d);

    double              clip(double x, double min, double max);

    void                PSO_topology_evolution(vector<vec> &pop, vector<vec> &velocities, data_subset training_set, net_topology max_topo, vector<NeuralNet> &pBest, NeuralNet gBest, double pop_score_variance);

    vector<NeuralNet>   convert_population_to_nets(vector<vec> genome_pop);

    vector<vec>         convert_population_to_genomes(vector<NeuralNet> net_pop, net_topology largest_topology);

    NeuralNet           get_best_model(vector<NeuralNet> pop);

    NeuralNet           get_best_model(vector<vec> genome_pop);

    vec                 get_genome(NeuralNet net, net_topology max_topo);

    NeuralNet           generate_net(vec genome);

    unsigned int        get_genome_length(net_topology t);

    unsigned int        get_population_size();

    mat                 get_population_scores(data_subset d);

    vector<NeuralNet>   get_population(){   return population;}

    void                set_population( vector<NeuralNet> pop) { population = pop; }

    void                insert_individual( NeuralNet indiv){    population[population.size()/2] = indiv;}

    double              get_epsilon() const;

    void                set_epsilon(double e);
};

#endif // TRAINER_PSO_H
