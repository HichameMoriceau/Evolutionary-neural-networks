#ifndef TRAINER_PSO_H
#define TRAINER_PSO_H

#include "trainer.h"

class Trainer_PSO : public Trainer
{
public:
    //---
                        Trainer_PSO();
    void                train(Data_set data_set, NeuralNet &net);
    void                train(Data_set data_set, NeuralNet &net, mat &results_score_evolution);
    NeuralNet           evolve_through_iterations(Data_set data_set, net_topology min_topo, net_topology max_topo, unsigned int nb_gens, mat &results_score_evolution, unsigned int index_cross_validation_section, unsigned int selected_mutation_scheme, unsigned int current_gen);

    void                PSO_topology_evolution(vector<vec> &pop, vector<vec> &velocities, data_subset training_set, net_topology max_topo, vector<NeuralNet> &pBest, NeuralNet gBest, double pop_score_variance);
    double              clip(double x, double min, double max);
    //---
/*
    // stats routines
    double              compute_score_variance(vector<NeuralNet> population, data_subset data_set);
    double              compute_score_stddev  (vector<NeuralNet> population, data_subset data_set);
    double              compute_score_mean    (vector<NeuralNet> population, data_subset data_set);
    double              compute_score_median  (vector<NeuralNet> population, data_subset data_set);
    double              compute_score_variance(vector<vec> population, data_subset data_set);
    double              compute_score_stddev  (vector<vec> population, data_subset data_set);
    double              compute_score_median  (vector<vec> population, data_subset data_set);
    double              compute_score_mean    (vector<vec> population, data_subset data_set);

    // cross-validation routines
    NeuralNet           train_topology_plus_weights(Data_set data_set, net_topology max_topo, mat &results_score_evolution);
    NeuralNet           cross_validation_training(Data_set data_set, net_topology min_topo, net_topology max_topo, mat &results_score_evolution, double &avrg_score, double &avrg_acc);
    NeuralNet           evolve_through_iterations(Data_set data_set, net_topology min_topo, net_topology max_topo, unsigned int nb_epochs, mat &results_cost_and_score_evolution, unsigned int index_cross_validation_section);

    // neural nets ensembles interpretation routines
    void                elective_accuracy(vector<NeuralNet> pop, Data_set data_set, double &ensemble_accuracy, double &ensemble_score);
    unsigned int        get_nb_identical_elements(mat A, mat B);

    // general population-based subroutines
    void                initialize_random_population(unsigned int pop_size, net_topology max_topo);
    vector<NeuralNet>   generate_population(unsigned int pop_size, net_topology t, data_subset training_set);
    vector<vec>         generate_random_genome_population(unsigned int quantity, NeuralNet largest_net);
    /
     * @brief Evolutionary_trainer::generate_genome_population
     * @param quantity pop size
     * @param largest_net biggest possible network architecture
     * @return A pop of random neural nets (represented as
     *         vector : topology desc. followed by params) where
     *         each neural net has a topology of smaller or equal
     *         size to largest_net.
     /
    vector<vec>         generate_random_topology_genome_population(unsigned int quantity, NeuralNet largest_net);
    /
     * @brief Evolutionary_trainer::generate_random_topology_genome_population
     * @param quantity pop size
     * @param min_topo smallest possible network architecture
     * @param max_topo biggest possible network architecture
     * @return A pop of random neural nets (represented as
     *         vector : topology desc. followed by params) where
     *         each neural net belong to the same species (between min_topo and max_topo).
     /
    vector<vec>         generate_random_topology_genome_population(unsigned int quantity, net_topology min_topo, net_topology max_topo);
    void                evaluate_population(vector<NeuralNet> &pop, data_subset d);

    // conversion routines & accessors
    vector<NeuralNet>   convert_population_to_nets(vector<vec> genome_pop);
    vector<vec>         convert_population_to_genomes(vector<NeuralNet> net_pop, net_topology largest_topology);

    NeuralNet           get_best_model(vector<NeuralNet> pop);
    NeuralNet           get_best_model(vector<vec> genome_pop);

    vec                 get_genome(NeuralNet net, net_topology max_topo);
    NeuralNet           generate_net(vec genome);

    vector<NeuralNet>   get_population(){   return population;}
    void                set_population( vector<NeuralNet> pop) { population = pop; }
    unsigned int        get_population_size();
    mat                 get_population_scores(data_subset d);

    double              get_epsislon() const{return epsilon;}
    void                set_epsilon(double e){epsilon=e;}

    //other routines
    void                insert_individual( NeuralNet indiv){    population[population.size()/2] = indiv;}
    unsigned int        get_genome_length(net_topology t);
    double              f_rand(double fMin, double fMax);
    */
};

#endif // TRAINER_PSO_H
