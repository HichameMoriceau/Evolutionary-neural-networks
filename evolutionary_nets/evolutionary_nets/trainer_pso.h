#ifndef TRAINER_PSO_H
#define TRAINER_PSO_H

#include "trainer.h"

class Trainer_PSO : public Trainer
{
public:
                        Trainer_PSO();
    void                train(Data_set data_set, NeuralNet &net);
    void                train(Data_set data_set, NeuralNet &net, mat &results_score_evolution);
    NeuralNet           evolve_through_iterations(Data_set data_set, net_topology min_topo, net_topology max_topo, unsigned int nb_gens, mat &results_score_evolution, unsigned int index_cross_validation_section, unsigned int selected_mutation_scheme, unsigned int current_gen);

    void                PSO_topology_evolution(vector<NeuralNet> &pop, vector<vec> &velocities, Data_set data_set, net_topology max_topo, vector<NeuralNet> &pBest, NeuralNet gBest, double pop_score_variance);
    double              clip(double x, double min, double max);
};

#endif // TRAINER_PSO_H
