#ifndef NEURALNET_H
#define NEURALNET_H

#include <iomanip>
#include <iostream>
#include <armadillo>
#include <vector>
#include <cmath>
#include "data_set.h"
#include <string>

using namespace std;
using namespace arma;


/**
 * \brief Describes the architecture of a NeuralNet instance.
 */
struct net_topology{
  unsigned int nb_input_units;
  unsigned int nb_units_per_hidden_layer;
  unsigned int nb_output_units;
  unsigned int nb_hidden_layers;

  unsigned int get_total_nb_weights(){
      // return variable
      unsigned int nb_weights = 0;
      unsigned int nb_input_weights  = (this->nb_input_units+1) * this->nb_units_per_hidden_layer;
      unsigned int nb_hidden_weights = (this->nb_units_per_hidden_layer+1) * (this->nb_units_per_hidden_layer) * (this->nb_hidden_layers-1);
      unsigned int nb_output_weights = (this->nb_units_per_hidden_layer+1) * this->nb_output_units;
      nb_weights = nb_input_weights + nb_output_weights;
      if(this->nb_hidden_layers > 1)
          nb_weights += nb_hidden_weights;
      return nb_weights;
  }

  string to_string(){
      stringstream ss;
      ss << nb_input_units << "_" << nb_units_per_hidden_layer << "_" << nb_output_units << "_" << nb_hidden_layers;
      return ss.str();
  }
};

class NeuralNet
{
private:
    vec             params;
    net_topology    topology;
    double          accuracy;
    double          score;
    double          matthews_coeficient;

public:
                    // ctor
                    NeuralNet();
                    // ctor
                    NeuralNet(net_topology t);

    /**
     * \brief forward_propagate
     * \param X     Input data as matrix, whether it contains a single row or several. (Must fit the number of input neurons)
     * \return      The predictions made by the net on the input data X
     */
    mat             forward_propagate(mat X);

    /**
     * \brief forward_propagate
     * \param X     Input data as matrix, whether it contains a single row or several. (Must fit the number of input neurons)
     * \param Zs    vector of matrices for the summed weights*inputs(not yet been through sigmoid) to be returned by reference
     * \param As    vector of matrices for the activations (outputs) of the neurons to be returned by reference
     * \return      The predictions made by the net on the input data X
     *              Also returns (by reference) the updated state of the vectors Z and A
     */
    mat             forward_propagate(mat X, vector<mat> &Zs, vector<mat> &As);


    /**
     * \brief reshape_weights
     * \return  returns vector of Theta (weights) matrices of the neural network
                e.g. if net has 1 input layer, 1 hidden layer, 1 output layer
                        reshape_weights() will return a vector of the two weight matrices
                        respectively Theta[0] and Theta[1]

     */
    vector<mat>     reshape_weights();

    void            save_net(ofstream &model_file);

    unsigned int    get_total_nb_weights();

    vec             generate_random_model();

    void            print_topology();

    // getters and setters

    vec             get_params();

    void            set_params(vec p);

    net_topology    get_topology();

    void            set_topology(net_topology t);

    /**
     * \brief get_accuracy
     * \param d data portion used for accuracy calculation
     * \return  returns percentage representing how often the model
     *          correctly predicts <Y> on the data-set <X>
     */
    double          get_accuracy(data_subset d);

    /**
     * \brief get_f1_score
     * \param d
     * \return (score function) returns an indication of the quality of the model [0, 1]
     *         The F1 Score function calculates the *precision* and *recall* of the model.
     *         This function is used as fitness function by the Differential Evolution algorithm.
     */
    double          get_f1_score(data_subset d);

    /**
     * \brief get_matthews_correlation_coefficient
     * \param d
     * \return (score function) returns an indication of the quality of the model [-1, +1]
     */
    double          get_matthews_correlation_coefficient(data_subset d);

    void            print_topology(net_topology t);

    /**
     * \brief operator < (comparator-function for sorting by highest score)
     * \param n
     * \return true if the provided net is less fit than this net
     */
    bool            operator<(const NeuralNet &n) const {   return n.score < this->score; }

    // returns the Mean Squared Error of a net on <data_set>
    double          get_MSE(data_subset d);

    // helper methods
private:
    double          sigmoid(double z);

    /**
     * \brief sigmoid_matrix
     * \param Z
     * \return returns element-wise computation of the sigmoid function on provided matrix.
     */
    mat             sigmoid_matrix(mat Z);

    /**
     * \brief get_nb_identical_elements
     * \param A
     * \param B
     * \return returns the number of identical elements of two matrices A and B(3 doubleing points close)
               returns 0 if the matrices aren't of same dimensions
     */
    unsigned int    get_nb_identical_elements(mat A, mat B);

};
#endif // NEURALNET_H
