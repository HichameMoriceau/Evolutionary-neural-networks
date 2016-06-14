#include "neuralnet.h"

NeuralNet::NeuralNet(){
    // default topology
    net_topology t;
    t.nb_input_units   = 1;
    t.nb_hidden_layers = 1;
    t.nb_output_units  = 2;
    t.nb_units_per_hidden_layer = 1;

    accuracy     = 0.0f;
    score        = 0.0f;
    mse = 0.0f;
    topology     = t;
    params       = generate_random_model();
}

NeuralNet::NeuralNet(net_topology t){
    accuracy     = 0.0f;
    score        = 0.0f;
    mse = 0.0f;
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
        prev_activation = sigmoid_matrix(Zs[i+1]);
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

    // breaks down single vector<double> into multiple corresponding weight matrices
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

        unsigned int sub_range = width-1;
        mat tmp;

        // each row *i* influence its corresponding neuron *i* in the next layer
        for(unsigned int i=0; i<height; ++i){
            unsigned int from = (index_start+i*sub_range)+i;
            unsigned int to   = index_start+(i+1)*sub_range+i;
            tmp = join_vert(tmp, params(span(from, to)).t());
        }
        Thetas[i] = tmp;
        // memorize end-index of this chunk
        previous_last_index = index_end;
    }
    return Thetas;
}

void NeuralNet::save_net(ofstream &model_file) {
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
    random_model[0]=topology.nb_input_units;
    random_model[1]=topology.nb_units_per_hidden_layer;
    random_model[2]=topology.nb_output_units;
    random_model[3]=topology.nb_hidden_layers;
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

mat NeuralNet::generate_conf_mat(unsigned int nb_classes, mat preds, mat labels){
    // generate confusion matrix
    mat confusion_matrix(nb_classes, nb_classes);
    for(unsigned int i=0; i<nb_classes; i++)
        for(unsigned int j=0; j<nb_classes; j++)
            confusion_matrix(i,j) = count_nb_identicals(i,j, to_multiclass_format(preds), labels);
    return confusion_matrix;
}

double NeuralNet::compute_score(mat confusion_matrix, unsigned int nb_classes, unsigned int nb_local_classes){
    double computed_score=0;
    vec scores(nb_classes);
    // computing f1 score for each label
    for(unsigned int i=0; i<nb_classes; i++){
        double TP = confusion_matrix(i,i);
        double TPplusFN = sum(confusion_matrix.col(i));
        double TPplusFP = sum(confusion_matrix.row(i));
        double tmp_precision=TP/TPplusFP;
        double tmp_recall=TP/TPplusFN;
        scores[i] = 2*((tmp_precision*tmp_recall)/(tmp_precision+tmp_recall));
        // check for -NaN
        if(scores[i] != scores[i])
            scores[i] = 0;
        computed_score += scores[i];
    }
    // general f1 score = average of all classes score
    return (computed_score/nb_local_classes)*100;
}

double NeuralNet::get_accuracy(data_subset data_set) {
    //return variable
    double computed_accuracy = 0;
    // make predictions over entire data-set
    mat H = forward_propagate(data_set.X);
    unsigned int nb_classes = topology.nb_output_units;
    mat confusion_matrix=generate_conf_mat(nb_classes, H, data_set.Y);
    double TP =0;
    for(unsigned int i=0; i<nb_classes; i++){
        TP += confusion_matrix(i,i);
    }
    computed_accuracy = (TP/H.n_rows)*100;
    // update neural network accuracy
    accuracy = computed_accuracy;
    // return computed accuracy
    return computed_accuracy;
}

double NeuralNet::get_f1_score(data_subset data_set) {
    // computed score is based on precision and recall
    double computed_score = 0;
    // perform predictions on provided data-set
    mat H = forward_propagate(data_set.X);
    unsigned int nb_classes = topology.nb_output_units;
    // generate confusion matrix
    mat confusion_matrix=generate_conf_mat(nb_classes, H, data_set.Y);
    unsigned int nb_local_classes=count_nb_classes(data_set.Y);
    vec scores(nb_classes);
    // computing f1 score for each label
    for(unsigned int i=0; i<nb_classes; i++){
        double TP = confusion_matrix(i,i);
        double TPplusFN = sum(confusion_matrix.col(i));
        double TPplusFP = sum(confusion_matrix.row(i));
        double tmp_precision=TP/TPplusFP;
        double tmp_recall=TP/TPplusFN;
        scores[i] = 2*((tmp_precision*tmp_recall)/(tmp_precision+tmp_recall));
        // check for -NaN
        if(scores[i] != scores[i])
            scores[i] = 0;
        computed_score += scores[i];
    }
    // general f1 score = average of all classes score
    computed_score = (computed_score/nb_local_classes)*100;
    // update net score
    score = computed_score;
    // return net score
    return score;
}

void NeuralNet::get_fitness_metrics(Data_set D){
    /*SCORE CALCULATION*/
    // perform predictions on provided data-set
    mat H_train = forward_propagate(D.training_set.X);
    mat H_val   = forward_propagate(D.validation_set.X);
    unsigned int nb_classes = topology.nb_output_units;
    // generate confusion matrix
    mat confusion_matrix_train=generate_conf_mat(nb_classes,H_train, D.training_set.Y);
    mat confusion_matrix_val=generate_conf_mat(nb_classes,H_val, D.validation_set.Y);
    unsigned int nb_local_classes_t=count_nb_classes(D.training_set.Y);
    unsigned int nb_local_classes_val=count_nb_classes(D.validation_set.Y);
    // update score values
    score=compute_score(confusion_matrix_train,nb_classes, nb_local_classes_t);
    validation_score=compute_score(confusion_matrix_val,nb_classes, nb_local_classes_val);

    /*ACC CALCULATION*/
    double TP =0;
    for(unsigned int i=0; i<nb_classes; i++)
        TP += confusion_matrix_train(i,i);
    accuracy=(TP/H_train.n_rows)*100;
    TP=0;
    for(unsigned int i=0; i<nb_classes; i++)
        TP += confusion_matrix_val(i,i);
    validation_acc=(TP/H_val.n_rows)*100;

    /*MSE CALCULATION*/
    mse=get_MSE(D.training_set);
}

unsigned int NeuralNet::count_nb_classes(mat labels){
  vector<unsigned int> array;
  bool is_known_class = false;
  // compute nb output units required
  for(unsigned int i=0; i<labels.n_rows; i++) {
    unsigned int current_pred_class = labels(i);
    is_known_class=false;
    // for each known prediction classes
    for(unsigned int j=0;j<array.size(); j++) {
      // if current output is different from prediction class
      if(current_pred_class==array[j]){
    is_known_class = true;
      }
    }
    if(array.empty() || (!is_known_class)){
      array.push_back(current_pred_class);
    }
  }
  return array.size();
}

unsigned int NeuralNet::count_nb_identicals(unsigned int predicted_class, unsigned int expected_class, mat predictions, mat expectations){
    unsigned int count=0;
    // for each example
    for(unsigned int i=0; i<predictions.n_rows; i++){
        if(predictions(i)==predicted_class && expectations(i)==expected_class)
            count++;
    }
    return count;
}

mat NeuralNet::to_multiclass_format(mat predictions){
    unsigned int nb_classes = predictions.n_cols;
    mat formatted_predictions(predictions.n_rows, 1);
    double highest_activation = 0;
    // for each example
    for(unsigned int i=0; i<predictions.n_rows; i++){
        unsigned int index = 0;
        highest_activation = 0;
        // the strongest activation is considered the prediction
        for(unsigned int j=0; j<nb_classes; j++){
            if(predictions(i,j) > highest_activation){
                highest_activation = predictions(i,j);
                index = j;
            }
        }
        formatted_predictions(i) = index;
    }
    return formatted_predictions;
}

double NeuralNet::get_MSE(data_subset d){
    double total_error = 0;
    unsigned int nb_classes=get_topology().nb_output_units;
    for(unsigned int i=0; i<d.X.n_rows; i++){
        mat prediction = forward_propagate(d.X.row(i));
        double max=-1;
        unsigned int index=-1;
        // search for most activated output
        for(unsigned int j=0; j<nb_classes; j++){
            unsigned int p=prediction(j);
            if(p>max){
                max=p;
                index=j;
            }
        }
        total_error += pow(as_scalar(to_multiclass_format(prediction))-d.Y(i), 2);
    }
    return (total_error/d.X.n_rows);
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
