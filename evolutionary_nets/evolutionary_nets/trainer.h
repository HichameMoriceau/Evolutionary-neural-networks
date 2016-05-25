#ifndef TRAINER_H
#define TRAINER_H

#include "neuralnet.h"
#include "data_set.h"
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>

using namespace std;

class Trainer
{
protected:
public:
    vector<NeuralNet>   population;
    // if variance < epsilon then stop GA
    double              epsilon = 1.0f;

    unsigned int        nb_epochs;

public:

    virtual void    train(Data_set data_set, NeuralNet &net) = 0;
    virtual void    train(Data_set data_set, NeuralNet &net, mat &results_cost_and_score_evolution) = 0;
    //virtual void    single_epoch(vector<vec> &population, data_subset training_set, net_topology min_topo, net_topology max_topology, unsigned int selected_mutation_scheme) =0;

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
    NeuralNet           train_topology_plus_weights(Data_set data_set, net_topology max_topo, mat &results_score_evolution, unsigned int selected_mutation_scheme);
    NeuralNet           cross_validation_training(Data_set data_set, net_topology min_topo, net_topology max_topo, mat &results_score_evolution, double &avrg_score, double &avrg_acc, unsigned int selected_mutation_scheme);
    virtual NeuralNet   evolve_through_iterations(Data_set data_set, net_topology min_topo, net_topology max_topo, unsigned int nb_epochs, mat &results_cost_and_score_evolution, unsigned int index_cross_validation_section, unsigned int selected_mutation_scheme)=0;

    // neural nets ensembles interpretation routines
    void                elective_accuracy(vector<NeuralNet> pop, Data_set data_set, double &ensemble_accuracy, double &ensemble_score);
    unsigned int        return_highest(map<unsigned int, unsigned int> votes);
    unsigned int        count_nb_identicals(unsigned int predicted_class, unsigned int expected_class, mat predictions, mat expectations);
    mat                 to_multiclass_format(mat predictions);

    // general population-based subroutines
    void                initialize_random_population(unsigned int pop_size, net_topology max_topo);
    vector<NeuralNet>   generate_population(unsigned int pop_size, net_topology t, data_subset training_set);
    vector<vec>         generate_random_genome_population(unsigned int quantity, NeuralNet template_net);
    /**
     * @brief Evolutionary_trainer::generate_genome_population
     * @param quantity pop size
     * @param largest_net biggest possible network architecture
     * @return A pop of random neural nets (represented as
     *         vector : topology desc. followed by params) where
     *         each neural net has a topology of smaller or equal
     *         size to largest_net.
     */
    vector<vec>         generate_random_topology_genome_population(unsigned int quantity, NeuralNet largest_net);
    vector<vec>         generate_random_topology_genome_population(unsigned int quantity, net_topology min_topo, net_topology max_topo);
    void                evaluate_population(vector<NeuralNet> &pop, data_subset d);

    // conversion routines & accessors
    vector<vec>         convert_population_to_genomes(vector<NeuralNet> net_pop, net_topology largest_topology);
    // returns a vector of neural networks corresponding to the provided genomes
    vector<NeuralNet>   convert_population_to_nets(vector<vec> genome_population);

    NeuralNet           get_best_model(vector<NeuralNet> pop);
    NeuralNet           get_best_model(vector<vec> genome_pop);

    vec                 get_genome(NeuralNet n, net_topology largest_topology);
    NeuralNet           generate_net(vec genome);

    unsigned int        get_genome_length(net_topology t);

    vector<NeuralNet>   get_population(){   return population;}
    void                set_population( vector<NeuralNet> pop) { population = pop; }
    unsigned int        get_population_size(){return population.size();}
    mat                 get_population_scores(data_subset d);

    double              get_epsislon() const{return epsilon;}
    void                set_epsilon(double e){epsilon=e;}
    unsigned int        get_nb_epochs(){return nb_epochs;}
    void                set_nb_epochs(unsigned int e){ nb_epochs  = e;}

    // other routines
    void                insert_individual( NeuralNet indiv){    population[population.size()/2] = indiv;}
    double              f_rand(double fMin, double fMax);
    double              clip(double n, double min, double max);
    unsigned int        generate_random_integer_between_range(unsigned int min, unsigned int max);
};

#endif // TRAINER_H
