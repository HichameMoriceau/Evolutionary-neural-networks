#ifndef BACKPROPAGATION_TRAINER_H
#define BACKPROPAGATION_TRAINER_H

#include "trainer.h"
#include <fann.h>
#include <iomanip>

class Backpropagation_trainer : public Trainer
{

    // learning rate
    double          alpha;

    // regularization coefficient
    double          lambda;

public:

                    Backpropagation_trainer();

    void            train(Data_set data_set, NeuralNet &net);

    void            train(Data_set data_set, NeuralNet &net, mat &results_cost_and_score_evolution);

private:

    void            train_backprop(Data_set training_set, NeuralNet &net, mat &results_cost_and_score_evolution);

    //
    // backpropagation specific helper
    //

    bool            is_close(mat& A, mat& B, double tolerance);

    double          compute_cost(data_subset data_set, NeuralNet net);

    double          compute_mean_squared_error_cost(mat Y, mat H, vector<mat> Thetas);

    double          compute_cross_entropy_cost(mat Y, mat H, vector<mat> Thetas);

    double          sigmoid(double z);

    mat             sigmoid_matrix(mat Z);

public:
    // accessors
    double          get_alpha()          { return alpha;}
    void            set_alpha(double a ) { alpha  = a;}
    double          get_lambda()         { return lambda;}
    void            set_lambda(double d) { lambda = d;}
};

#endif // BACKPROPAGATION_TRAINER_H
