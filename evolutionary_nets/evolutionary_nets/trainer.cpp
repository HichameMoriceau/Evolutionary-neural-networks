#include "trainer.h"


double Trainer::compute_score_variance(vector<NeuralNet> pop, data_subset data_set){
    double variance=0.0f;
    vec score_values(pop.size());
    for(unsigned int i=0; i< pop.size(); ++i) {
        score_values(i)=pop[i].get_f1_score(data_set);
        // round after two decimal places
        score_values(i)=(round(score_values(i)) * 100) / 100.0f;
    }
    variance=var(score_values);
    return variance;
}

double Trainer::compute_score_stddev(vector<NeuralNet> pop, data_subset data_set){
    double std_dev=0.0f;
    vec score_values(pop.size());
    for(unsigned int i=0; i< pop.size(); ++i) {
        score_values(i)=pop[i].get_f1_score(data_set);
        // round after two decimal places
        score_values(i)=(round(score_values(i)) * 100) / 100.0f;
    }
    std_dev=stddev(score_values);
    return std_dev;
}

double Trainer::compute_score_mean(vector<NeuralNet> pop, data_subset data_set){
    double mean_pop=0.0f;
    vec score_values(pop.size());
    for(unsigned int i=0; i< pop.size(); ++i) {
        score_values(i)=pop[i].get_f1_score(data_set);
        // round after two decimal places
        score_values(i)=(round(score_values(i)) * 100) / 100.0f;
    }
    mean_pop=mean(score_values);
    return mean_pop;
}

double Trainer::compute_score_median(vector<NeuralNet> pop, data_subset data_set){
    double median_pop=0.0f;
    vec score_values(pop.size());
    for(unsigned int i=0; i< pop.size(); ++i) {
        score_values(i)=pop[i].get_f1_score(data_set);
        // round after two decimal places
        score_values(i)=(round(score_values(i)) * 100) / 100.0f;
    }
    median_pop=median(score_values);
    return median_pop;
}

double Trainer::compute_score_variance(vector<vec> pop, data_subset data_set){
    double variance=0.0f;
    vec score_values(pop.size());
    for(unsigned int i=0; i< pop.size(); ++i) {
        score_values(i)=(generate_net(pop[i])).get_f1_score(data_set);
        // round after two decimal places
        score_values(i)=(round(score_values(i)) * 100) / 100.0f;
    }
    variance=var(score_values);
    return variance;
}

double Trainer::compute_score_stddev(vector<vec> pop, data_subset data_set){
    double std_dev=0.0f;
    vec score_values(pop.size());
    for(unsigned int i=0; i< pop.size(); ++i) {
        score_values(i)=(generate_net(pop[i])).get_f1_score(data_set);
        // round after two decimal places
        score_values(i)=(round(score_values(i)) * 100) / 100.0f;
    }
    std_dev=stddev(score_values);
    return std_dev;
}

double Trainer::compute_score_mean(vector<vec> pop, data_subset data_set){
    double mean_pop=0.0f;
    vec score_values(pop.size());
    for(unsigned int i=0; i< pop.size(); ++i) {
        score_values(i)=(generate_net(pop[i])).get_f1_score(data_set);
        // round after two decimal places
        score_values(i)=(round(score_values(i)) * 100) / 100.0f;
    }

    // if mean= NaN
    if(mean(score_values) != mean(score_values)){
        // display details error
        cout << "mean=" << mean(score_values) << endl;
        cout << endl;
        cout << "scores=" ;
        for(unsigned int i=0; i<pop.size(); ++i){
            cout << score_values(i) << " ";
        }
        exit(0);
    }
    mean_pop=mean(score_values);
    return mean_pop;
}

double Trainer::compute_score_median(vector<vec> pop, data_subset data_set){
    double median_pop=0.0f;
    vec score_values(pop.size());
    for(unsigned int i=0; i< pop.size(); ++i) {
        score_values(i)=(generate_net(pop[i])).get_f1_score(data_set);
        // round after two decimal places
        score_values(i)=(round(score_values(i)) * 100) / 100.0f;
    }
    median_pop=median(score_values);
    return median_pop;
}

NeuralNet Trainer::train_topology_plus_weights(Data_set data_set, net_topology max_topo, mat &results_score_evolution, unsigned int selected_mutation_scheme) {
    // return variable
    NeuralNet cross_validated_net;
    net_topology min_topo;
    min_topo.nb_input_units=max_topo.nb_input_units;
    min_topo.nb_units_per_hidden_layer=1;
    min_topo.nb_output_units=max_topo.nb_output_units;
    min_topo.nb_hidden_layers=1;

    double test_score=0;
    double test_acc  =0;
    cross_validated_net=cross_validation_training(data_set, min_topo, max_topo, results_score_evolution, test_score, test_acc, selected_mutation_scheme);
    // append Cross Validation error to result matrix
    mat test_score_m=ones(results_score_evolution.n_rows,1) * test_score;
    mat test_acc_m  =ones(results_score_evolution.n_rows,1) * test_acc;
    results_score_evolution=join_horiz(results_score_evolution, test_score_m);
    results_score_evolution=join_horiz(results_score_evolution, test_acc_m);
    cout<<"Performances on test set: "<<test_score<<", acc="<<test_acc<<endl;
    mat mutation_scheme=ones(results_score_evolution.n_rows,1) * selected_mutation_scheme;
    results_score_evolution=join_horiz(results_score_evolution, mutation_scheme);
    return cross_validated_net;
}

NeuralNet Trainer::cross_validation_training(Data_set data_set, net_topology min_topo, net_topology max_topo, mat &results_score_evolution, double &test_score, double &test_acc, unsigned int selected_mutation_scheme){
    unsigned int nb_folds=10;
    NeuralNet tmp_net(max_topo), cross_validated_net(max_topo);
    tmp_net.set_topology(max_topo);
    cross_validated_net.set_topology(max_topo);
    mat tmp_results_perfs,perfs_cross_validation;

    unsigned int pop_size=population.size();
    population=convert_population_to_nets(generate_random_topology_genome_population(pop_size,min_topo, max_topo));

    // for each generation
    for(unsigned int i=0; i<(nb_epochs)-(nb_epochs/nb_folds); i++) {
        unsigned int k=i%nb_folds;
        cout << "Using validation-set" << k << " of" << nb_folds-1 << endl;
        data_set.subdivide_data_cross_validation(k, nb_folds);
        // make sure topology is adequate to data-set
        max_topo.nb_input_units=data_set.training_set.X.n_cols;
        max_topo.nb_output_units=data_set.find_nb_prediction_classes(data_set.data);

        // empty temporary result matrix
        tmp_results_perfs.reset();
        tmp_net=evolve_through_iterations(data_set, min_topo, max_topo, 1, tmp_results_perfs, k, selected_mutation_scheme, i);

        if(tmp_net.get_f1_score(data_set.validation_set)>=cross_validated_net.get_f1_score(data_set.validation_set)){
            // update best model
            cross_validated_net.set_topology(tmp_net.get_topology());
            cross_validated_net.set_params(tmp_net.get_params());
        }
        // append results for this fold to results to be printed
        perfs_cross_validation=join_vert(perfs_cross_validation, tmp_results_perfs);

    }
    cout << "start last fold" << endl;
    // force last training cycle to do all epochs
    set_epsilon(-1);


    //tmp_net=evolve_through_iterations(data_set, min_topo, max_topo, nb_epochs, tmp_results_perfs, 0, selected_mutation_scheme);

    // force last training cycle to make use of entire training set
    data_set.training_set.X=join_vert(data_set.training_set.X,   data_set.training_set.X);
    data_set.training_set.Y=join_vert(data_set.training_set.Y, data_set.training_set.Y);
    // train net
    mat perfs_entire_training_set;
    for(unsigned int i=0;i<nb_epochs/nb_folds;i++)
        tmp_net=evolve_through_iterations(data_set, min_topo, max_topo, 1, perfs_entire_training_set, nb_folds, -1, nb_epochs-(nb_epochs/nb_folds)+i);

    if(tmp_net.get_f1_score(data_set.validation_set)>=cross_validated_net.get_f1_score(data_set.validation_set)){
        // update best model
        cross_validated_net.set_topology(tmp_net.get_topology());
        cross_validated_net.set_params(tmp_net.get_params());
    }

    // return test score as reference
    test_score=cross_validated_net.get_f1_score(data_set.test_set);
    // return test accuracy as reference
    test_acc  =cross_validated_net.get_accuracy(data_set.test_set);
    // return result matrix as reference
    results_score_evolution=join_vert(perfs_cross_validation, perfs_entire_training_set);
    // return trained net
    return cross_validated_net;
}

void Trainer::elective_accuracy(vector<NeuralNet> pop, Data_set data_set, double &ensemble_accuracy, double &ensemble_score){
    // sort pop by fitness
    evaluate_population(pop, data_set.validation_set);
    unsigned int nb_individuals=pop.size();
    unsigned int nb_classes=pop[0].get_topology().nb_output_units;
    unsigned int nb_examples=data_set.validation_set.Y.n_rows;
    mat elected_votes(data_set.validation_set.X.n_rows,1);
    mat pos_votes(data_set.validation_set.X.n_rows,1);
    mat neg_votes(data_set.validation_set.X.n_rows,1);
    // keeps track of number of votes for each class
    vector<map<unsigned int, unsigned int> >votes(nb_examples);

    // for each indiv
    for(unsigned int i=0;i<pop.size(); i++){
        // perform predictions on data-set
        mat H=pop[i].forward_propagate(data_set.validation_set.X);
        mat Predictions=to_multiclass_format(H);
        // memorize vote
        for(unsigned int p=0; p<Predictions.n_rows; p++){
            for(unsigned int i=0;i<nb_classes;i++){
                if(Predictions[p]==i)
                    votes[p][i]++;
            }
        }
    }

    // generate the ensemble's predictions
    for(unsigned int p=0; p<nb_examples; p++){
        // use majority as prediction
        elected_votes(p)=return_highest(votes[p]);
    }

    // generate confusion matrix
    mat confusion_matrix(nb_classes, nb_classes);
    for(unsigned int i=0; i<nb_classes; i++) {
        for(unsigned int j=0; j<nb_classes; j++){
            confusion_matrix(i,j)=count_nb_identicals(i,j, elected_votes, data_set.validation_set.Y);
        }
    }

    vec scores(nb_classes);
    // averaged f1 score based on precision and recall
    double computed_score=0;
    double computed_accuracy=0;
    // computing f1 score for each label
    for(unsigned int i=0; i<nb_classes; i++){
        double TP=confusion_matrix(i,i);
        double TPplusFN=sum(confusion_matrix.col(i));
        double TPplusFP=sum(confusion_matrix.row(i));
        double tmp_precision=TP/TPplusFP;
        double tmp_recall=TP/TPplusFN;
        scores[i]=2*((tmp_precision*tmp_recall)/(tmp_precision+tmp_recall));
        // check for -NaN
        if(scores[i] != scores[i])
            scores[i]=0;
        computed_score += scores[i];
    }
    // general f1 score=average of all classes score
    computed_score=(computed_score/nb_classes)*100;

    double TP=0;
    for(unsigned int i=0;i<nb_classes;i++)
        TP+=confusion_matrix(i,i);
    computed_accuracy=(TP/elected_votes.n_rows)*100;

    // return ensemble accuracy
    ensemble_accuracy=computed_accuracy;
    // return ensemble score
    ensemble_score=computed_score;
}

unsigned int Trainer::return_highest(map<unsigned int, unsigned int> votes){
    unsigned int vote=-1;
    double max=-1;
    for(unsigned int i=0;i<votes.size();i++){
        if(votes[i]>max){
            max=votes[i];
            vote=i;
        }
    }
    return vote;
}

unsigned int Trainer::count_nb_identicals(unsigned int predicted_class, unsigned int expected_class, mat predictions, mat expectations){
    unsigned int count=0;
    // for each example
    for(unsigned int i=0; i<predictions.n_rows; i++){
        if(predictions(i)==predicted_class && expectations(i)==expected_class)
            count++;
    }
    return count;
}

mat Trainer::to_multiclass_format(mat predictions){
    unsigned int nb_classes=predictions.n_cols;
    mat formatted_predictions(predictions.n_rows, 1);
    double highest_activation=0;
    // for each example
    for(unsigned int i=0; i<predictions.n_rows; i++){
        unsigned int index=0;
        highest_activation=0;
        // the strongest activation is considered the prediction
        for(unsigned int j=0; j<nb_classes; j++){
            if(predictions(i,j) > highest_activation){
                highest_activation=predictions(i,j);
                index=j;
            }
        }
        //cout << "formatted prediction=" << endl << formatted_predictions << endl;
        formatted_predictions(i)=index;
    }
    return formatted_predictions;
}

void Trainer::initialize_random_population(unsigned int pop_size, net_topology max_topo){
    if(pop_size < 10) pop_size=10;
    net_topology min_topo;
    min_topo.nb_input_units=max_topo.nb_input_units;
    min_topo.nb_units_per_hidden_layer=1;
    min_topo.nb_output_units=max_topo.nb_output_units;
    min_topo.nb_hidden_layers=1;

    population=convert_population_to_nets(generate_random_topology_genome_population(pop_size,min_topo, max_topo));
}

vector<NeuralNet> Trainer::generate_population(unsigned int pop_size, net_topology t, data_subset training_set) {
    vector<NeuralNet> pop(pop_size);
    for(unsigned int i=0 ; i < pop_size; ++i) {
        NeuralNet tmp_net(t);
        pop[i]=tmp_net;
    }
    // update all fitness values
    evaluate_population(pop, training_set);
    return pop;
}

vector<vec> Trainer::generate_random_genome_population(unsigned int quantity, NeuralNet largest_net) {
    vector<vec> pop(quantity);
    for(unsigned int i=0 ; i < quantity ; ++i) {
        // instantiate new random neural net with set topology
        vec tmp_vec(largest_net.get_total_nb_weights() + 4);
        // assign random params values [0, 1]
        tmp_vec.randu();
        // make sure genome represents net of same nature (identical nb layers)
        tmp_vec[0]=largest_net.get_topology().nb_input_units;
        tmp_vec[1]=generate_random_integer_between_range(1, largest_net.get_topology().nb_units_per_hidden_layer);
        tmp_vec[2]=largest_net.get_topology().nb_output_units;
        tmp_vec[3]=largest_net.get_topology().nb_hidden_layers;
        // add it to pop
        pop[i]=tmp_vec;
    }
    return pop;
}

vector<vec> Trainer::generate_random_topology_genome_population(unsigned int quantity, NeuralNet largest_net) {
    // return variable
    vector<vec> pop(quantity);
    for(unsigned int i=0 ; i < quantity ; ++i) {
        // instantiate new random neural net with set topology
        vec tmp_vec(largest_net.get_total_nb_weights() + 4);
        // assign random values [0, 1]
        tmp_vec.randu();
        // make sure topology is valid
        tmp_vec[0]=largest_net.get_topology().nb_input_units;
        tmp_vec[1]=generate_random_integer_between_range(1, largest_net.get_topology().nb_units_per_hidden_layer);
        tmp_vec[2]=largest_net.get_topology().nb_output_units;
        tmp_vec[3]=generate_random_integer_between_range(1, largest_net.get_topology().nb_hidden_layers);
        // add it to pop
        pop[i]=tmp_vec;
    }
    return pop;
}

/**
 * @brief Evolutionary_trainer::generate_random_topology_genome_population
 * @param quantity pop size
 * @param min_topo smallest possible network architecture
 * @param max_topo biggest possible network architecture
 * @return A pop of random neural nets (represented as
 *         vector : topology desc. followed by params) where
 *         each neural net belong to the same species (between min_topo and max_topo).
 */
vector<vec> Trainer::generate_random_topology_genome_population(unsigned int quantity, net_topology min_topo, net_topology max_topo) {
    // return variable
    vector<vec> pop(quantity);
    for(unsigned int i=0 ; i < quantity ; ++i) {
        // instantiate new random neural net with set topology
        vec tmp_vec(max_topo.get_total_nb_weights() + 4);
        // assign random values [0, 1]
        tmp_vec.randu();
        // make sure topology is valid
        tmp_vec[0]=max_topo.nb_input_units;
        tmp_vec[1]=generate_random_integer_between_range( min_topo.nb_units_per_hidden_layer, max_topo.nb_units_per_hidden_layer);
        tmp_vec[2]=max_topo.nb_output_units;
        tmp_vec[3]=generate_random_integer_between_range(min_topo.nb_hidden_layers, max_topo.nb_hidden_layers);
        // add it to pop
        pop[i]=tmp_vec;
    }
    return pop;
}

void Trainer::evaluate_population(vector<NeuralNet> &pop, data_subset d) {
    for(unsigned int i=0 ; i < pop.size() ; ++i) {
        pop[i].get_f1_score(d);
        nb_err_func_calls++;
    }
    // sort pop according to score
    sort(pop.begin(), pop.end());
}

vector<vec> Trainer::convert_population_to_genomes(vector<NeuralNet> net_pop, net_topology largest_topology){
    vector<vec> genome_pop;
    for(unsigned int i=0; i<net_pop.size(); ++i) {
        genome_pop.push_back(get_genome(net_pop[i], largest_topology));
    }
    return genome_pop;
}

vector<NeuralNet> Trainer::convert_population_to_nets(vector<vec> genome_pop) {
    vector<NeuralNet> pop;
    // convert genome pop into neural network pop
    for(unsigned int i=0; i<genome_pop.size(); ++i){
        pop.push_back(generate_net(genome_pop[i]));
    }
    return pop;
}

NeuralNet Trainer::get_best_model(vector<NeuralNet> pop){
    // sort pop according to score
    sort(pop.begin(), pop.end());
    return pop[0];
}

NeuralNet Trainer::get_best_model(vector<vec> genome_pop) {
    vector<NeuralNet> pop;
    // convert genome pop into neural network pop
    pop=convert_population_to_nets(genome_pop);
    // sort pop according to score
    sort(pop.begin(), pop.end());
    return pop[0];
}

vec Trainer::get_genome(NeuralNet net, net_topology max_topo) {
    // instantiate genome with largest possible size
    vec genome(get_genome_length(max_topo));
    // first four elements contain topology
    genome[0]=net.get_topology().nb_input_units;
    genome[1]=net.get_topology().nb_units_per_hidden_layer;
    genome[2]=net.get_topology().nb_output_units;
    genome[3]=net.get_topology().nb_hidden_layers;
    // others contain params
    for(unsigned int i=0; i<net.get_params().size(); ++i) {
        genome[4+i]=net.get_params()[i];
    }
    return genome;
}

NeuralNet Trainer::generate_net(vec genome){
    NeuralNet net;
    net_topology topology;
    // retrieve topology
    topology.nb_input_units=(unsigned int) genome[0];
    topology.nb_units_per_hidden_layer=(unsigned int) genome[1];
    topology.nb_output_units=(unsigned int) genome[2];
    topology.nb_hidden_layers=(unsigned int) genome[3];
    net.set_topology(topology);
    // retrieve params
    unsigned int size_params=net.get_total_nb_weights();
    vec tmp_params(size_params);
    for(unsigned int i=0; i<size_params; ++i) {
        tmp_params[i]=genome[4+i];
    }
    net.set_params(tmp_params);
    return net;
}

unsigned int Trainer::get_genome_length(net_topology t){
    unsigned int length=0;
    NeuralNet dummyNet;
    dummyNet.set_topology(t);
    length=4 + dummyNet.get_total_nb_weights();
    return length;
}

mat Trainer::get_population_scores(data_subset d){
    mat scores;
    for(unsigned int i=0; i<population.size(); ++i) {
        mat tmp;
        tmp={ population[i].get_f1_score(d) };
        scores=join_horiz(scores, tmp);
    }
    return scores;
}

double Trainer::f_rand(double fMin, double fMax){
    double f=(double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

double Trainer::clip(double x, double min, double max) {
    // only clamp if necessary
    if( (x<min)||(x>max) ){
        double c=x;
        c=((max-min)/2)*((exp(x) - exp(-x))/(exp(x) + exp(-x))) + max - (max-min)/2;
        if(c<min) c=min;
        if(c>max) c=max;
        // check for -nan values
        if(c!=c) c=f_rand(min, max);
        return c;
    }else
        return x;
}


unsigned int Trainer::generate_random_integer_between_range(unsigned int min, unsigned int max) {
    return min + ( std::rand() % ( max - min + 1 ) );
}

























