#include "backpropagation_trainer.h"

Backpropagation_trainer::Backpropagation_trainer(){
    nb_epochs = 1000;

    // default Gradient Descent settings
    alpha  = 0.001;
    lambda = 1;
}

void Backpropagation_trainer::train(Data_set data_set, NeuralNet &net){
    mat results_cost_and_score_evolution;
    train_backprop(data_set, net, results_cost_and_score_evolution);
}

void Backpropagation_trainer::train(Data_set data_set, NeuralNet &net, mat &results_cost_and_score_evolution){
    train_backprop(data_set, net, results_cost_and_score_evolution);
}

// see fann C++ library: fann_set_callback()
int FANN_API fann_callback(struct fann *ann, struct fann_train_data *train,
                           unsigned int max_epochs, unsigned int epochs_between_reports,
                           float desired_error, unsigned int epochs){

   string line = to_string(epochs) + ", " + to_string(fann_get_MSE(ann));
   cout << line << endl;
   ofstream output;
   output.open("data/backprop_perfs.csv", ios::app | ios::out);
   output << line << endl;
   output.close();
   return 0;
}

void Backpropagation_trainer::train_backprop(Data_set data_set, NeuralNet &net, mat &results_score_evolution) {
    // set backprop settings
    const float desired_error = (const float) 0.0;
    const unsigned int epochs_between_reports = 1;

    net_topology t = net.get_topology();
    unsigned int layers[t.nb_hidden_layers+2];
    for(unsigned int i=0; i<t.nb_hidden_layers+2;i++)
        layers[i] = t.nb_units_per_hidden_layer;
    // set nb inputs
    layers[0] = t.nb_input_units;
    // set nb outputs
    layers[t.nb_hidden_layers+1] = t.nb_output_units;
    struct fann* fann = fann_create_standard_array(t.nb_hidden_layers+2, layers);

    // set neural net settings
    fann_set_activation_function_hidden(fann, FANN_SIGMOID_SYMMETRIC);
    fann_set_activation_function_output(fann, FANN_SIGMOID_SYMMETRIC);
    fann_set_train_error_function(fann, FANN_ERRORFUNC_LINEAR);
    fann_set_learning_rate(fann, alpha);
    cout << "done here2" << endl;


    // setting data set
    string data_filename = data_set.data_set_filename.substr(0,data_set.data_set_filename.find_last_of(".")) + ".data";
    // set callback function to output result on file
    fann_set_callback(fann, fann_callback);

    cout << "done here3" << endl;

    fann_train_on_file(fann, data_filename.c_str(), nb_epochs, epochs_between_reports, desired_error);

    cout << "done here4" << endl;


    fann_destroy(fann);
}

// compares equality of matrices <A> and <B> with <tolerance>
bool Backpropagation_trainer::is_close(mat &A, mat &B, double tolerance){
    // abs returns a mat type then max checks columns and returns a row_vec
    // max used again will return the biggest element in the row_vec
    bool close = false;
    if(max(max(abs(A-B))) < tolerance)
    {
        close = true;
    }
    return close;
}

double Backpropagation_trainer::compute_cost(data_subset data_set, NeuralNet net){
    double cost = -1.0f;

    net_topology t = net.get_topology();
    vector<mat> Thetas = net.reshape_weights();

    unsigned int m = data_set.X.n_rows;

    mat I = eye(t.nb_output_units, t.nb_output_units);
    mat Y = zeros(m, t.nb_output_units);
    if(t.nb_output_units > 1){
        for(unsigned int i = 0 ; i < m ; ++i){
            Y.row(i) = I.row( data_set.Y(i) );
        }
    }
    mat H = net.forward_propagate(data_set.X);

    // compute cost (with regularization)
    cost = compute_mean_squared_error_cost(data_set.Y, H, Thetas);
    return cost;
}

double Backpropagation_trainer::compute_mean_squared_error_cost(mat Y, mat H, vector<mat> Thetas) {
    // return value
    double cost = 0.0f;

    // mean squared error
    cost = 0.5 * as_scalar(sum( square(Y - H)));

    /*
  // compute value of regularization term
  double sum_of_thetas_squared = 0.0f;
  for(unsigned int i = 0 ; i < Thetas.size(); ++i)
    sum_of_thetas_squared += sum(sum(sum(square(Thetas[i].cols(1,Thetas[i].n_cols-1)))));
  double regularization_penalty = (lambda / (2*m)) * sum_of_thetas_squared;

  // add regularization penalty (prevents overfitting)
  cost += regularization_penalty;
  */

    return cost;
}

// returns cost for corresponding model using the cross-entropy calculation
// (aka. Bernoulli negative log-likelihood and Binary Cross-Entropy)
double Backpropagation_trainer::compute_cross_entropy_cost(mat Y, mat H, vector<mat> Thetas) {
    // return value
    double cost = 0.0f;
    // nb of examples
    unsigned int m = Y.n_rows;

    Y = Y.t();

    // compute cross-entropy cost
    cost = (-0.1 * m) * sum(sum( (Y * log(H)) + ((1 - Y) * log(1 - H)) ));

    /*
  // compute value of regularization term
  double sum_of_thetas_squared = 0.0f;
  for(unsigned int i = 0 ; i < Thetas.size(); ++i)
    sum_of_thetas_squared += sum(sum(sum(square(Thetas[i].cols(1,Thetas[i].n_cols-1)))));
  double regularization_penalty = ((0.5 * lambda) / m) * sum_of_thetas_squared;

  // add regularization penalty (prevents overfitting)
  cost += regularization_penalty;
  */

    return cost;
}

double Backpropagation_trainer::sigmoid(double z) {
    // compute function
    double result = 1/(1+exp(-z));

    // check +inf
    if(result>0 && result/result != result/result)
        result = 0.99;
    // check -inf
    else if(result<0 && result/result != result/result)
        result = -0.99;

    return result;
}

// returns element-wise computation of the sigmoid function on provided matrix.
mat Backpropagation_trainer::sigmoid_matrix(mat Z){
    for(unsigned int i = 0 ; i < Z.n_rows ; ++i)
        for(unsigned int j = 0 ; j < Z.n_cols ; ++j)
            Z(i,j) = sigmoid(-Z(i,j));
    return Z;
}
