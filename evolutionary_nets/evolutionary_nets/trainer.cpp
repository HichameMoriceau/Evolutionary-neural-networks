#include "trainer.h"

//user stop flag
unsigned int usf=0;

// used to allow researchers to prematurely stop the experiment
// This automatically triggers averaging and printing of the results
void user_interrupt_handler(int a)
{
    usf=1;
    printf("\nUser interruption caught: (^C)\n");
}

double Trainer::compute_score_variance(vector<NeuralNet> pop){
    double variance=0.0f;
    vec score_values(pop.size());
    for(unsigned int i=0; i< pop.size(); ++i) {
        score_values(i)=pop[i].get_train_score();
        // round after two decimal places
        score_values(i)=(round(score_values(i)) * 100) / 100.0f;
    }
    variance=var(score_values);
    return variance;
}

double Trainer::compute_score_stddev(vector<NeuralNet> pop){
    double std_dev=0.0f;
    vec score_values(pop.size());
    for(unsigned int i=0; i< pop.size(); ++i) {
        score_values(i)=pop[i].get_train_score();
        // round after two decimal places
        score_values(i)=(round(score_values(i)) * 100) / 100.0f;
    }
    std_dev=stddev(score_values);
    return std_dev;
}

double Trainer::compute_score_mean(vector<NeuralNet> pop){
    double mean_pop=0.0f;
    vec score_values(pop.size());
    for(unsigned int i=0; i< pop.size(); ++i) {
        score_values(i)=pop[i].get_train_score();
        // round after two decimal places
        score_values(i)=(round(score_values(i)) * 100) / 100.0f;
    }
    mean_pop=mean(score_values);
    return mean_pop;
}

double Trainer::compute_score_median(vector<NeuralNet> pop){
    double median_pop=0.0f;
    vec score_values(pop.size());
    for(unsigned int i=0; i< pop.size(); ++i) {
        score_values(i)=pop[i].get_train_score();
        // round after two decimal places
        score_values(i)=(round(score_values(i)) * 100) / 100.0f;
    }
    median_pop=median(score_values);
    return median_pop;
}

mat Trainer::generate_metric_line(vector<NeuralNet> population, unsigned int gen){
    mat line;
    line << nb_err_func_calls
         << gen

         << population[0].get_train_acc()
         << population[0].get_train_score()
         << population[0].get_train_mse()

         << population[0].get_test_acc()
         << population[0].get_test_score()
         << population[0].get_test_mse()

         << population[0].get_val_acc()
         << population[0].get_val_score()
         << population[0].get_val_mse()

         << compute_score_variance(population)
         << compute_score_mean(population)
         << population.size()

         << population[0].get_topology().nb_units_per_hidden_layer
         << population[0].get_topology().nb_hidden_layers

         << endr;
    return line;
}

NeuralNet Trainer::train_topology_plus_weights(Data_set data_set, net_topology max_topo, mat &results_score_evolution, unsigned int selected_mutation_scheme) {
    net_topology min_topo;
    min_topo.nb_input_units=max_topo.nb_input_units;
    min_topo.nb_units_per_hidden_layer=1;
    min_topo.nb_output_units=max_topo.nb_output_units;
    min_topo.nb_hidden_layers=1;

    //trained_net=cross_val_training(data_set, min_topo, max_topo, results_score_evolution, test_score, test_acc, selected_mutation_scheme);
    population=generate_random_topology_population(population.size(),min_topo, max_topo);
    NeuralNet trained_net=evolve_through_iterations(data_set, min_topo, max_topo, nb_epochs, results_score_evolution, 1, selected_mutation_scheme, 1);
    return trained_net;
}

NeuralNet Trainer::cross_val_training(Data_set data_set, net_topology min_topo, net_topology max_topo, mat &results_score_evolution, double &test_score, double &test_acc, unsigned int selected_mutation_scheme){
    // Using 'leave-one-out cross-validation' (exhaustive cross validation: particular case of k-fold CV)
    unsigned int nb_folds=data_set.train_set.X.n_rows;
    NeuralNet tmp_net(max_topo), cross_validated_net(max_topo);
    tmp_net.set_topology(max_topo);
    cross_validated_net.set_topology(max_topo);
    mat tmp_results_perfs,perfs_cross_validation;

    unsigned int pop_size=population.size();
    population=generate_random_topology_population(pop_size,min_topo, max_topo);

    unsigned int nb_cv_gens=(nb_epochs)-(nb_epochs/nb_folds);
    //double freq_change_CV_percent = 1;

    // Register signals
    signal(SIGINT, user_interrupt_handler);

    unsigned int passed_gens=nb_cv_gens;

    cout<<"NB CrossValidated gens="<<nb_cv_gens<<endl;
    // for each generation
    for(unsigned int i=0; i<nb_cv_gens/*freq_change_CV_percent*/; i++) {
        unsigned int k=i%nb_folds;
        cout << "Using validation-set" << k << " of" << nb_folds-1 << endl;
        data_set.subdivide_data_cross_validation(k, nb_folds);
        // make sure topology is adequate to data-set
        max_topo.nb_input_units=data_set.train_set.X.n_cols;
        max_topo.nb_output_units=data_set.find_nb_prediction_classes(data_set.data);

        // empty temporary result matrix
        tmp_results_perfs.reset();
        tmp_net=evolve_through_iterations(data_set, min_topo, max_topo, 1/*nb_cv_gens/freq_change_CV_percent*/, tmp_results_perfs, k, selected_mutation_scheme, i);

        if(tmp_net.get_train_score(data_set.val_set)>=cross_validated_net.get_train_score(data_set.val_set)){
            // update best model
            cross_validated_net.set_topology(tmp_net.get_topology());
            cross_validated_net.set_params(tmp_net.get_params());
        }
        // append results for this fold to results to be printed
        perfs_cross_validation=join_vert(perfs_cross_validation, tmp_results_perfs);

        // if user stopped experiment
        if(usf){
            passed_gens=i;
            cout<<"early termination"<<endl;
            break;
        }

    }
    // reset user stopping flag
    usf=0;

    cout << "start last fold" << endl;
    // force last training cycle to do all epochs
    set_epsilon(-1);
    // force last training cycle to make use of entire training set
    data_set.train_set.X=join_vert(data_set.train_set.X,   data_set.train_set.X);
    data_set.train_set.Y=join_vert(data_set.train_set.Y, data_set.train_set.Y);
    // train net
    mat perfs_entire_train_set;
    for(unsigned int i=0;i<nb_epochs/nb_folds;i++){
        tmp_net=evolve_through_iterations(data_set, min_topo, max_topo, 1, perfs_entire_train_set, nb_folds, -1, passed_gens+i);
        // if user stopped experiment
        if(usf){
            cout<<"early termination"<<endl;
            break;
        }
    }
    if(tmp_net.get_train_score(data_set.val_set)>=cross_validated_net.get_train_score(data_set.val_set)){
        // update best model
        cross_validated_net.set_topology(tmp_net.get_topology());
        cross_validated_net.set_params(tmp_net.get_params());
    }

    // return test score as reference
    test_score=cross_validated_net.get_train_score(data_set.test_set);
    // return test accuracy as reference
    test_acc  =cross_validated_net.get_train_acc(data_set.test_set);
    // return result matrix as reference
    results_score_evolution=join_vert(perfs_cross_validation, perfs_entire_train_set);
    // return trained net
    return cross_validated_net;
}

void Trainer::elective_acc(vector<NeuralNet> pop, Data_set data_set, double &ensemble_acc, double &ensemble_score){
    // sort pop by fitness
    sort(pop.begin(), pop.end());
    unsigned int nb_classes=pop[0].get_topology().nb_output_units;
    unsigned int nb_examples=data_set.train_set.Y.n_rows;
    mat elected_votes(data_set.train_set.X.n_rows,1);
    // keeps track of number of votes for each class
    vector<map<unsigned int, unsigned int> >votes(nb_examples);

    // for each indiv
    for(unsigned int i=0;i<pop.size(); i++){
        // perform predictions on data-set
        mat H=pop[i].forward_propagate(data_set.train_set.X);
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
            confusion_matrix(i,j)=count_nb_identicals(i,j, elected_votes, data_set.train_set.Y);
        }
    }

    vec scores(nb_classes);
    // averaged f1 score based on precision and recall
    double computed_score=0;
    double computed_acc=0;
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
    computed_acc=(TP/elected_votes.n_rows)*100;

    // return ensemble accuracy
    ensemble_acc=computed_acc;
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

unsigned int Trainer::count_nb_identicals(unsigned int pred_class, unsigned int actual_class, mat preds, mat actuals){
    unsigned int count=0;
    for(unsigned int i=0; i<preds.n_rows; i++)
        if(preds(i)==pred_class&&actuals(i)==actual_class)
            count++;
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
        formatted_predictions(i)=index;
    }
    return formatted_predictions;
}

vector<NeuralNet> Trainer::generate_population(unsigned int pop_size, net_topology t) {
    vector<NeuralNet> pop(pop_size);
    for(unsigned int i=0;i<pop_size;++i){
        NeuralNet tmp_net(t);
        pop[i]=tmp_net;
    }
    return pop;
}

vector<NeuralNet> Trainer::generate_random_population(unsigned int quantity, NeuralNet largest_net) {
    vector<NeuralNet> pop(quantity);
    for(unsigned int i=0 ; i < quantity ; ++i) {
        net_topology t;
        // make sure genome represents net of same nature (identical nb layers)
        t.nb_input_units=largest_net.get_topology().nb_input_units;
        t.nb_units_per_hidden_layer=generate_random_integer_between_range(1, largest_net.get_topology().nb_units_per_hidden_layer);
        t.nb_output_units=largest_net.get_topology().nb_output_units;
        t.nb_hidden_layers=generate_random_integer_between_range(1, largest_net.get_topology().nb_hidden_layers);
        // add it to pop
        NeuralNet n(t);
        pop[i]=n;
    }
    return pop;
}

vector<vec> Trainer::generate_random_topology_genome_population(unsigned int quantity, NeuralNet largest_net) {
    // return variable
    vector<vec> pop(quantity);
    for(unsigned int i=0;i<quantity;++i) {
        // instantiate new random neural net with set topology
        vec tmp_vec(largest_net.get_topology().get_total_nb_weights() + 4);
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
vector<NeuralNet> Trainer::generate_random_topology_population(unsigned int quantity, net_topology min_topo, net_topology max_topo) {
    // return variable
    vector<NeuralNet> pop(quantity);
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
        pop[i]=to_NeuralNet(tmp_vec);
    }
    return pop;
}

void Trainer::evaluate_population(vector<NeuralNet> &pop, Data_set d, mat& results_score_evolution) {
    // update all performance metrics
    for(unsigned int i=0 ; i < pop.size();++i){
        pop[i].get_fitness_metrics(d);
        nb_err_func_calls++;
        // sort by fittest
        sort(pop.begin(), pop.end());
        // format population performances
        mat line=generate_metric_line(population,0);
        // append result line to result matrix
        results_score_evolution = join_vert(results_score_evolution, line);

    }
    // sort pop according to score
    sort(pop.begin(), pop.end());
}

NeuralNet Trainer::to_NeuralNet(vec p){
    net_topology t;
    // retrieve topology
    t.nb_input_units=(unsigned int) p[0];
    t.nb_units_per_hidden_layer=(unsigned int) p[1];
    t.nb_output_units=(unsigned int) p[2];
    t.nb_hidden_layers=(unsigned int) p[3];
    NeuralNet net(t);
    net.set_params(p);
    return net;
}

mat Trainer::get_population_scores(data_subset d){
    mat scores;
    for(unsigned int i=0; i<population.size(); ++i) {
        mat tmp;
        tmp={ population[i].get_train_score(d) };
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
