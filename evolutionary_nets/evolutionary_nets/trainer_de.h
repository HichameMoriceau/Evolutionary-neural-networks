#ifndef EVOLUTIONARY_TRAINER_H
#define EVOLUTIONARY_TRAINER_H

#include "trainer.h"

class Trainer_DE : public Trainer
{
public:
    //---
                        Trainer_DE();
    void                train(Data_set data_set, NeuralNet &net);
    void                train(Data_set data_set, NeuralNet &net, mat &results_score_evolution);
    NeuralNet           evolve_through_iterations(Data_set data_set, net_topology min_topo, net_topology max_topo, unsigned int nb_gens, mat &results_score_evolution, unsigned int index_cross_val_section, unsigned int selected_mutation_scheme, unsigned int current_gen);

    void                differential_evolution_topology_evolution(Data_set data_set, net_topology min_topo, net_topology max_topology, unsigned int selected_mutation_scheme, mat& results_score_evolution, unsigned int gen);
    void                differential_evolution(vector<NeuralNet> &population, data_subset data_set);
    void                train_weights(Data_set data_set, NeuralNet &net, unsigned int nb_epochs, mat &results_score_evolution);
    void                mutative_crossover(double CR, double F, net_topology min_topo, net_topology max_topo, vec original_model, vec &candidate_model, vec indiv_a, vec indiv_b, vec indiv_c);
    double              mutation_scheme_DE_rand_1(double F, double x_rand_1, double x_rand_2, double x_rand_3);
    //---

};

#endif // EVOLUTIONARY_TRAINER_H
