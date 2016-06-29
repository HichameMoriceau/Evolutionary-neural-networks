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
      unsigned int nb_input_weights =(this->nb_input_units+1)*this->nb_units_per_hidden_layer;
      unsigned int nb_hidden_weights=(this->nb_units_per_hidden_layer+1)*(this->nb_units_per_hidden_layer)*(this->nb_hidden_layers-1);
      unsigned int nb_output_weights=(this->nb_units_per_hidden_layer+1)*this->nb_output_units;
      nb_weights = nb_input_weights + nb_output_weights;
      if(this->nb_hidden_layers > 1)
          nb_weights += nb_hidden_weights;
      return nb_weights;
  }

  unsigned int get_genome_length(){
      return get_total_nb_weights()+4;
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
    // contains topology + weights (params is genotype)
    vec             params;
    net_topology    topology;

    double          train_acc;
    double          train_score;
    double          train_mse;

    double          val_score;
    double          val_acc;
    double          val_mse;

    double          test_score;
    double          test_acc;
    double          test_mse;

public:
                    // ctors
                    NeuralNet();
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

    void            randomize_weights();

    void            print_topology();

    // getters and setters
    //vec             get_genome(net_topology max_topo);

    vec             get_params();
    void            set_params(vec p);

    net_topology    get_topology();
    void            set_topology(net_topology t);

    mat             generate_conf_mat(unsigned int nb_classes, mat preds, mat labels);
    double          compute_score(mat confusion_matrix, unsigned int nb_classes, unsigned int nb_local_classes);

    /**
     * \brief get_train_acc
     * \param d data portion used for accuracy calculation
     * \return  returns percentage representing how often the model
     *          correctly predicts <Y> on the data-set <X>
     */
    double          get_train_acc(data_subset d);
    /**
     * \brief get_train_score
     * \param d
     * \return (score function) returns an indication of the quality of the model [0, 1]
     *         The F1 Score function calculates the *precision* and *recall* of the model.
     *         This function is used as fitness function by the Differential Evolution algorithm.
     */
    double          get_train_score(data_subset d);

    double          get_train_acc(){return train_acc;}
    double          get_train_score(){return train_score;}

    double          get_val_acc(){return val_acc;}
    void            set_val_acc(double v_a){ val_acc=v_a;}

    double          get_test_acc(){return test_acc;}
    void            set_test_acc(double v_a){ test_acc=v_a;}

    double          get_val_score(){ return val_score; }
    void            set_val_score(double v_s){ val_score=v_s;}

    double          get_test_score(){ return test_score; }
    void            set_test_score(double v_s){ test_score=v_s;}

    void            set_train_score(double s){ train_score=s;}
    void            set_acc(double a){ train_acc=a;}

    void            get_fitness_metrics(Data_set D);

    unsigned int    count_nb_classes(mat labels);

    unsigned int    count_nb_identicals(unsigned int predicted_class, unsigned int expected_class, mat predictions, mat expectations);

    mat             to_multiclass_format(mat predictions);

    void            print_topology(net_topology t);

    // returns the Mean Squared Error of a net on <data_set>
    double          get_mse(data_subset d);
    double          get_train_mse(){return train_mse;}
    void            set_train_mse(double e){ train_mse=e;}
    double          get_val_mse(){return val_mse;}
    void            set_val_mse(double e){ val_mse=e;}
    double          get_test_mse(){return test_mse;}
    void            set_test_mse(double e){ test_mse=e;}

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

public:
    /**
     * \brief operator < (comparator-function for sorting by highest score)
     * \param n
     * \return true if the provided net is less fit than this net
     */
    bool            operator<(const NeuralNet &n) const {   return n.train_score < this->train_score; }

};
#endif // NEURALNET_H
