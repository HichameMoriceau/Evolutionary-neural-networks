#ifndef TRAINER_H
#define TRAINER_H

#include "neuralnet.h"
#include "data_set.h"
#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;

class Trainer
{
protected:
    unsigned int    nb_epochs;

public:

    virtual void    train(Data_set data_set, NeuralNet &net) = 0;

    virtual void    train(Data_set data_set, NeuralNet &net, mat &results_cost_and_score_evolution) = 0;

    // helper methods
protected:

    unsigned int    generate_random_integer_between_range(unsigned int min, unsigned int max);

public:
    // accessors
    unsigned int    get_nb_epochs(){return nb_epochs;}
    void            set_nb_epochs(unsigned int e){ nb_epochs  = e;}

};

#endif // TRAINER_H
