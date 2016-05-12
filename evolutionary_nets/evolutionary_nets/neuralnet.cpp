#include "neuralnet.h"

NeuralNet::NeuralNet(){
    // default topology
    net_topology t;
    t.nb_input_units   = 1;
    t.nb_hidden_layers = 1;
    t.nb_output_units  = 1;
    t.nb_units_per_hidden_layer = 1;

    accuracy     = 0.0f;
    score        = 0.0f;
    matthews_coeficient = 0.0f;
    topology     = t;
    params       = generate_random_model();
}

NeuralNet::NeuralNet(net_topology t)
{
    accuracy     = 0.0f;
    score        = 0.0f;
    matthews_coeficient = 0.0f;
    topology     = t;
    params       = generate_random_model();
}

mat NeuralNet::forward_propagate(mat X) {

    // return variable (output value of hypothesis)
    mat H;
    unsigned int total_nb_layers = get_topology().nb_hidden_layers + 2;
    vector<mat> Thetas = reshape_weights();
    vector<mat> Zs(total_nb_layers);
    vector<mat> As(total_nb_layers);
    unsigned int m = X.n_rows;
    // add bias unit
    mat bias = ones(m, 1);
    // feed input
    mat prev_activation = X;
    // forward propagate each layer
    for(unsigned int i=0; i < total_nb_layers-1 ; ++i) {
        // append bias
        As[i] = join_horiz(bias, prev_activation);
        // compute Z
        Zs[i+1] = As[i] * (Thetas[i]).t();
        // compute A
        prev_activation = sigmoid_matrix( Zs[i+1]);
    }
    // memorize predictions
    H = As[total_nb_layers-1] = prev_activation;
    return H;
}

mat NeuralNet::forward_propagate(mat X, vector<mat> &Zs, vector<mat> &As) {

    // return variable (output value of hypothesis)
    mat H;
    unsigned int total_nb_layers = topology.nb_hidden_layers+2;
    unsigned int m = X.n_rows;
    // add bias unit
    mat bias = ones(m, 1);
    // feed input
    mat prev_activation = X;
    // retrieve weights
    vector<mat> Thetas = reshape_weights();
    // forward propagate each layer
    for(unsigned int i=0; i < total_nb_layers-1 ; ++i) {
        // append bias
        As[i] = join_horiz(bias, prev_activation);
        // compute Z
        Zs[i+1] = As[i] * (Thetas[i]).t();
        // compute A
        prev_activation = sigmoid_matrix( Zs[i+1]);
    }
    // memorize all predictions
    H = As[total_nb_layers-1] = prev_activation;

    return H;
}

vector<mat> NeuralNet::reshape_weights(){
    net_topology t = topology;
    // declaring variable for reshaping all Theta matrices
    unsigned int size_thetas_vector = t.nb_hidden_layers+1;
    // return variable
    vector<mat> Thetas(size_thetas_vector);

    unsigned int index_start  = -1;
    unsigned int index_end    = -1;
    unsigned int height = -1;
    unsigned int width  = -1;
    unsigned int previous_last_index = -1;

    // breaks down single vector<double> into corresponding weight matrices
    for(unsigned int i = 0 ; i < Thetas.size() ; ++i) {
        index_start  = 0;
        index_end    = 0;
        height = 0;
        width  = 0;

        // special case: first to second layer
        if(i == 0) {
            index_start  = 0;
            index_end    = (t.nb_input_units+1) * t.nb_units_per_hidden_layer; //(+1 for bias)
            height = t.nb_units_per_hidden_layer;
            width  = t.nb_input_units+1;
            // special case : pre-last to last layer
        } else if(i == Thetas.size()-1) {
            index_start  = previous_last_index;
            index_end    = params.n_elem-1;
            height = t.nb_output_units;
            width  = t.nb_units_per_hidden_layer + 1;
            // n to n+1 layer
        } else {
            index_start  = previous_last_index;
            index_end    = previous_last_index +  (t.nb_units_per_hidden_layer * (t.nb_units_per_hidden_layer + 1) );
            height = t.nb_units_per_hidden_layer;
            width  = t.nb_units_per_hidden_layer + 1;
        }
        /*
        // where Thetas[0] is first matrix of parameters of net
        Thetas[i] = reshape(params( span( index_start , index_end )),
                            height,
                            width);
        */
        //cout << endl;

        unsigned int sub_range = width-1;//((index_end-index_start) / height) ;
        /*
        cout << "dimensions  = " << height << "by" << width << endl;
        cout << "sub-range   = " << sub_range << endl;
        cout << "index start = " << index_start << endl;
        cout << "index end   = " << index_end << endl;
        cout << endl;
        */
        mat tmp;

        // each row *i* influence its corresponding neuron *i* in the next layer
        for(unsigned int i=0; i<height; ++i){
            unsigned int from = (index_start+i*sub_range)+i;
            unsigned int to   = index_start+(i+1)*sub_range+i;
            //cout << "from " << from << " to " << to << endl;
            tmp = join_vert(tmp, params(span(from, to)).t());
        }
        //cout << "tmp dimensions: " << size(tmp) << endl;
        Thetas[i] = tmp;

        // memorize end-index of this chunk
        previous_last_index = index_end;
    }
    return Thetas;
}

void NeuralNet::save_net(ofstream &model_file){
    net_topology t = topology;
    // print topology
    model_file << "------ NET TOPOLOGY ------" << endl
               << "nb input units        : " << t.nb_input_units << endl
               << "nb units/hidden layer : " << t.nb_units_per_hidden_layer << endl
               << "nb output units       : " << t.nb_output_units << endl
               << "nb hidden layers      : " << t.nb_hidden_layers << endl
               << "--------------------------" << endl;
    // print weights
    vector<mat> Thetas = reshape_weights();
    for(unsigned int i = 0 ; i < Thetas.size() ; ++i){
        model_file << "Theta"    << i << ":" << endl
                   << Thetas[i]  << endl;
    }
}

unsigned int NeuralNet::get_total_nb_weights(){
    // return variable
    unsigned int nb_weights = 0;
    unsigned int nb_input_weights  = (topology.nb_input_units+1) * topology.nb_units_per_hidden_layer;
    unsigned int nb_hidden_weights = (topology.nb_units_per_hidden_layer+1) * (topology.nb_units_per_hidden_layer) * (topology.nb_hidden_layers-1);
    unsigned int nb_output_weights = (topology.nb_units_per_hidden_layer+1) * topology.nb_output_units;
    nb_weights = nb_input_weights + nb_output_weights;
    if(topology.nb_hidden_layers > 1)
        nb_weights += nb_hidden_weights;
    return nb_weights;
}

vec NeuralNet::generate_random_model(){
    vec random_model = randu<vec>(get_total_nb_weights());
    return random_model;
}

void NeuralNet::print_topology(){
  net_topology t = topology;
  cout << "------ NET TOPOLOGY ------" << endl
       << "nb input units        : " << t.nb_input_units << endl
       << "nb units/hidden layer : " << t.nb_units_per_hidden_layer << endl
       << "nb output units       : " << t.nb_output_units << endl
       << "nb hidden layers      : " << t.nb_hidden_layers << endl
       << "--------------------------" << endl;
}

// getters and setters
vec NeuralNet::get_params(){                    return params;    }

void NeuralNet::set_params(vec v){              params = v;       }

net_topology NeuralNet::get_topology(){         return topology;  }

void NeuralNet::set_topology(net_topology t) {  topology = t;     }

double NeuralNet::get_accuracy( data_subset data_set) {
    //return variable
    double computed_accuracy = 0.0f;

    // make predictions over entire data-set
    mat H = forward_propagate(data_set.X);
    mat Predictions = round(H);

    // compute average accuracy using expected results
    computed_accuracy = (get_nb_identical_elements(Predictions.col(0), data_set.Y.col(0)) / double(data_set.X.n_rows)) * 100;
    // update neural network accuracy
    accuracy = computed_accuracy;
    // return computed accuracy
    return computed_accuracy;
}

double NeuralNet::get_f1_score(data_subset data_set) {
    // number of correctly classified positives results divided by the number of all positive results
    double precision = 0.0f;
    // number of correctly classified positives that were recalled (found)
    double recall = 0.0f;
    // computed score is based on precision and recall
    double computed_score = 0.0f;
    // perform predictions on provided data-set
    mat H = forward_propagate(data_set.X);
    mat Predictions = round(H);
    unsigned int true_positives  = sum(sum(Predictions==1 && data_set.Y==1));
    unsigned int false_positives = sum(sum(Predictions==1 && data_set.Y==0));
    unsigned int false_negatives = sum(sum(Predictions==0 && data_set.Y==1));
    
    if( !((true_positives + false_positives)==0 || (true_positives + false_negatives)==0)){
        precision =  ( (double) true_positives) / (true_positives + false_positives);
        recall    =  ( (double) true_positives) / (true_positives + false_negatives);
        // compute score
        computed_score = 2*((precision*recall) / (precision + recall));
        // make score of same scale as accuracy (better for plotting)
        computed_score = computed_score * 100;
    }
    // check for -NaN
    if(computed_score != computed_score)
        computed_score = 0;
    // update net score
    score = computed_score;
    // return net score
    return score;
}

double NeuralNet::get_matthews_correlation_coefficient(data_subset data_set){
    // return variable
    double MCC = -2;
    // perform predictions on provided data-set
    mat H = forward_propagate(data_set.X);
    mat Predictions = round(H);
    // compute true positives
    unsigned int tp = sum(sum(Predictions==1 && data_set.Y==1));
    // compute true negatives
    unsigned int tn = sum(sum(Predictions==0 && data_set.Y==0));
    // compute false positives
    unsigned int fp = sum(sum(Predictions==1 && data_set.Y==0));
    // compute false negatives
    unsigned int fn = sum(sum(Predictions==0 && data_set.Y==1));
    // compute coef (prevent NaN)
    if( ( (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) ) <= 0 )
        MCC = 0;
    else
        MCC = (tp*tn - fp*fn) / sqrt( (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn));
    // update net coef
    matthews_coeficient = MCC;
    // return net coef
    return matthews_coeficient;
}

double NeuralNet::get_MSE(data_subset d){
    double total_error = 0;
    for(unsigned int i=0; i<d.X.n_rows; i++){
        mat prediction = forward_propagate(d.X.row(i));
        total_error += pow(d.Y(i) - prediction(0), 2);
    }
    return total_error/d.X.n_rows;
}

double NeuralNet::sigmoid(double z) {
    //return (exp(z)-exp(-z))/(exp(z)+exp(-z)); // tanh
    return 1/(1+exp(-z));                       // sigmoid
}

mat NeuralNet::sigmoid_matrix(mat Z){
    for(unsigned int i = 0 ; i < Z.n_rows ; ++i)
        for(unsigned int j = 0 ; j < Z.n_cols ; ++j)
            Z(i,j) = sigmoid(-Z(i,j));
    return Z;
}

unsigned int NeuralNet::get_nb_identical_elements(mat A, mat B){
    if(A.n_rows != B.n_rows || A.n_cols != B.n_cols)
        return 0;
    unsigned int count = 0;
    for(unsigned int i = 0 ; i < A.n_rows ; ++i)
        for(unsigned int j = 0 ; j < A.n_cols ; ++j)
            if(A(i,j) == B(i,j))
                ++count;
    return count;
}

