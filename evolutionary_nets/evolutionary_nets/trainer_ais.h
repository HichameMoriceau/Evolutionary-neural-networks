#ifndef TRAINER_CLONAL_SELECTION_H
#define TRAINER_CLONAL_SELECTION_H

#include "trainer.h"

class Trainer_AIS : public Trainer
{
public:
                        Trainer_AIS();
    void                train(Data_set data_set, NeuralNet &net);
    void                train(Data_set data_set, NeuralNet &net, mat &results_score_evolution);
    NeuralNet           evolve_through_iterations(Data_set data_set, net_topology min_topo, net_topology max_topo, unsigned int nb_gens, mat &results_score_evolution, unsigned int index_cross_validation_section, unsigned int selected_mutation_scheme, unsigned int current_gen);

    void                clonal_selection_topology_evolution(vector<genome> &population, data_subset training_set, net_topology min_topo, net_topology max_topology, unsigned int selected_mutation_scheme);
    vector<genome>      select(unsigned int quantity, vector<genome> pop, data_subset training_set, net_topology max_topo);
    // cloning is inversely proportional to affinity (fitness)
    vector<genome>      generate_clones(unsigned int nb_clones, genome indiv);
    unsigned int        compute_nb_clones(double beta, int pop_size, int index);
    vector<genome>      add_all(vector<vector<genome>> all_populations, unsigned int* nb_clones_array);
    void                mutative_crossover(unsigned int R, double CR, double F, unsigned int genome_length, net_topology min_topo, net_topology max_topo, vec original_model, vec &candidate_model, vec indiv_a, vec indiv_b, vec indiv_c);
    double              mutation_scheme_DE_rand_1(double F, double x_rand_1, double x_rand_2, double x_rand_3);

};

#endif // TRAINER_CLONAL_SELECTION_H
