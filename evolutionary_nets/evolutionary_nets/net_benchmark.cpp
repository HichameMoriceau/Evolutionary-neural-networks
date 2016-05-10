#include "net_benchmark.h"
#include "data_set.h"

Net_benchmark::Net_benchmark() {
    // set default data-set
    data_set = Data_set();
    //data_set.select_data_set(0);
    // set default max topology
    max_topo.nb_input_units = data_set.training_set.X.n_cols;
    max_topo.nb_units_per_hidden_layer = data_set.training_set.X.n_cols * 4;
    max_topo.nb_output_units = 1;
    max_topo.nb_hidden_layers = 2;

    //  Set default net topology
    unsigned int dataset_nb_features = data_set.training_set.X.n_cols;
    net_topology t;
    t.nb_input_units = dataset_nb_features;
    t.nb_units_per_hidden_layer = dataset_nb_features;
    t.nb_output_units = 1;
    t.nb_hidden_layers = 1;
    set_topology(t);

    // instantiate optimization algorithms
    evo_trainer       = Evolutionary_trainer();

    experiment_file.open("experiment.txt", ios::app);
}

Net_benchmark::~Net_benchmark(){
    experiment_file.close();
}

double Net_benchmark::find_termination_criteria_epsilon(unsigned int many_generations) {
    net_topology min_topo;
    min_topo.nb_input_units = max_topo.nb_input_units;
    min_topo.nb_units_per_hidden_layer = 1;
    min_topo.nb_output_units = max_topo.nb_output_units;
    min_topo.nb_hidden_layers = 1;

    mat results_perfs;
    cout << "search termination criteria epsilon with " << many_generations << " generations"<< endl;

    // initial *generous* run of many generations
    Evolutionary_trainer t;
    // force trainer to perform all epochs
    t.set_epsilon(-1);
    t.initialize_random_population(100, max_topo);

    unsigned int MUTATION_SCHEME_RAND = 0;
    unsigned int MUTATION_SCHEME_BEST = 1;
    t.evolve_through_generations(data_set, min_topo, max_topo,many_generations, results_perfs,0, MUTATION_SCHEME_RAND);

    mat variance_values = results_perfs.col(5);
    double highest_variance = 0;

    double max = -1.0f;
    // find highest variance
    for(unsigned int i=0; i<many_generations; i++){
        if(variance_values[i] > max){
            highest_variance    = variance_values[i];
            max                 = variance_values[i];
        }
    }
    // compute epsilon
    double epsilon = highest_variance / 20;

    cout << "***"
         << "highest variance = " << highest_variance << "\tepsilon = " << epsilon << endl
         << "***" << endl;
    return epsilon;
}

void Net_benchmark::run_benchmark(unsigned int nb_rep) {
    nb_replicates  = nb_rep;
    // set end of search space
    max_topo.nb_input_units             = data_set.training_set.X.n_cols;
    max_topo.nb_output_units            = 1;
    max_topo.nb_hidden_layers           = 2;
    max_topo.nb_units_per_hidden_layer  = max_topo.nb_input_units * 4;

    unsigned int pop_size_GA = 30;
    unsigned int nb_generations_GA = 100;
    unsigned int total_nb_data_sets = 1;

    unsigned int MUTATION_SCHEME_RAND = 0;
    unsigned int MUTATION_SCHEME_BEST = 1;
    unsigned int mutation_scheme = MUTATION_SCHEME_RAND;

    evo_trainer.initialize_random_population(pop_size_GA, max_topo);

    vector<string> data_set_filenames;
    data_set_filenames.push_back("data/iris-data-transformed.csv");
    data_set_filenames.push_back("data/breast-cancer-malignantOrBenign-data-transformed.csv");
    data_set_filenames.push_back("data/breast-cancer-recurrence-data-transformed.csv");
    data_set_filenames.push_back("data/haberman-data-transformed.csv");

    string start_time_str = get_current_date_time();
    auto start_time = system_clock::now();
    // for each data-set
    for(unsigned int i=0; i<total_nb_data_sets; i++) {
        // use data requested by user
        data_set.select_data_set(data_set_filenames[i]);
        // set largest topology
        max_topo.nb_input_units = data_set.training_set.X.n_cols;
        max_topo.nb_units_per_hidden_layer = 10;
        max_topo.nb_output_units = 1;//data_set.find_nb_prediction_classes(data_set.data);
        max_topo.nb_hidden_layers = 1;

        // 500 epochs in total is often more than enough
        double epsilon = find_termination_criteria_epsilon(500);
        // save results of cross-validated training
        train_net_and_save_performances(pop_size_GA, nb_generations_GA, epsilon, mutation_scheme);
    }
    auto end_time = system_clock::now();
    string end_time_str = get_current_date_time();
    auto experiment_duration = duration_cast<std::chrono::minutes>(end_time-start_time).count();

    cout << endl
         << "Training started at  : " << start_time_str << endl
         << "Training finished at : " << end_time_str << " to produce result data "
         << "USING: " << nb_replicates << " replicates and " << total_nb_data_sets << " data sets)" << endl
         << "experiment duration :\t" << experiment_duration << " minutes" << endl;

    experiment_file << "Training started at  : " << start_time_str << endl
                    << "Training finished at : " << end_time_str << " (to produce result data "
                    << "USING: " << nb_replicates << " replicates and " <<total_nb_data_sets << " data sets)" << endl
                    << "experiment duration :\t" << experiment_duration << " minutes" << endl;
}

// returns the net with the best score from a population of nets trained with various topologies
void Net_benchmark::train_topology(NeuralNet &evolved_net){

    // nb of different topologies to search
    unsigned int nb_topology_sizes = 5;
    NeuralNet current_net;
    double current_perf = 0;
    NeuralNet best_net;
    double best_perf = 0;

    /*
     * Use provided net topology as basis
     */
    net_topology t = net.get_topology();

    // search best topology
    for(unsigned int i=0; i<nb_topology_sizes; ++i) {

        // set new topology
        t.nb_units_per_hidden_layer += 1;
        current_net.set_topology(t);
        // optimize weights
        evo_trainer.train( data_set, current_net);
        // record quality of the model
        current_perf = current_net.get_f1_score(data_set.validation_set);

        if(current_perf >= best_perf){
            best_net = current_net;
            best_perf  = current_perf;
        }
    }
    // return net with best topology
    evolved_net = best_net;
}

void Net_benchmark::set_topology(net_topology t){
    net = NeuralNet(t);
}

void Net_benchmark::compute_perfs_test_validation(double &model_score_training_set,
                                                  double &model_prediction_accuracy_training_set,
                                                  double &model_score_validation_set,
                                                  double &model_prediction_accuracy_validation_set) {
    // compute training-set accuracy
    model_prediction_accuracy_training_set   = net.get_accuracy(data_set.training_set);
    // compute training-set score
    model_score_training_set                 = net.get_f1_score(data_set.training_set);
    // compute validation-set accuracy
    model_prediction_accuracy_validation_set = net.get_accuracy(data_set.validation_set);
    // compute validation-set score
    model_score_validation_set               = net.get_f1_score(data_set.validation_set);
}

// returns the number of elements of value 1 in the provided matrix
unsigned int Net_benchmark::count_nb_positive_examples(vec A){
    unsigned int count = 0;
    for(unsigned int i = 0 ; i < A.n_rows ; ++i)
        for(unsigned int j = 0 ; j < A.n_cols ; ++j)
            if(A(i,j) == 1.0f)
                ++count;
    return count;
}

// returns a matrix of results such as :
mat Net_benchmark::evaluate_backprop_general_performances() {
    data_subset training_set   = data_set.training_set;

    // result matrix
    mat results_cost_relative_to_training_set_size;
    // declare tmp variable for used training-set segment
    data_subset training_set_subset;
    // declare variable to record model's performance
    double model_score_training_set      = 0.0f;
    double model_accuracy_training_set   = 0.0f;
    double model_score_validation_set    = 0.0f;
    double model_accuracy_validation_set = 0.0f;

    mat new_line;

    // aggregate score and accuracy values for increasingly large proportion of training-set
    for(unsigned int i = 1 ; i < training_set.X.n_rows ; ++i) {

        model_score_training_set      = 0.0f;
        model_accuracy_training_set   = 0.0f;
        model_score_validation_set    = 0.0f;
        model_accuracy_validation_set = 0.0f;

        // i designates the index limit of the training-set subset
        training_set_subset.X = training_set.X.rows(0, i);
        training_set_subset.Y = training_set.Y.rows(0, i);

        // perform prediction using optimized model on training-subset and record performances
        compute_perfs_test_validation( model_score_training_set,
                                       model_accuracy_training_set,
                                       model_score_validation_set,
                                       model_accuracy_validation_set);

        // create new vector with model's performances on training-set and validation-set
        new_line << i
                 << model_accuracy_training_set   << model_score_training_set
                 << model_accuracy_validation_set << model_score_validation_set
                 << endr;

        // append results for that size to result matrix
        results_cost_relative_to_training_set_size= join_vert(results_cost_relative_to_training_set_size,new_line);
    }
    return results_cost_relative_to_training_set_size;
}


void Net_benchmark::print_results_octave_format(ofstream &result_file, mat recorded_performances, string octave_variable_name){
    // Create header in MATLAB format
    result_file << "# Created by main.cpp, " << get_current_date_time() << endl
                << "# name: " << octave_variable_name << endl
                << "# type: matrix"  << endl
                << "# rows: " << recorded_performances.n_rows << endl
                << "# columns: " << recorded_performances.n_cols << endl;

    // append content of recorded performances into same file
    for(unsigned int i = 0 ; i < recorded_performances.n_rows; ++i){
        for(unsigned int j = 0 ; j < recorded_performances.n_cols; ++j){
            result_file << recorded_performances(i,j) << " ";
        }
        result_file << endl;
    }
    result_file << endl;
}

void Net_benchmark::print_results_octave_format(ofstream &result_file, vector<mat> recorded_performances, string octave_variable_name){
    // Create header in MATLAB format
    result_file << "# name: " << octave_variable_name << endl
                << "# type: cell"  << endl
                << "# rows: 1" << endl
                << "# columns: " << recorded_performances.size() << endl;

    string new_line = "";

    //print as cell array
    for(unsigned int i=0; i<recorded_performances.size(); i++) {
        new_line = "";
        // print cell-element header
        result_file << "# name: <cell-element>" << endl
                    << "# type: matrix" << endl
                    << "# rows: 1" << endl
                    << "# columns: " << recorded_performances[i].n_elem << endl;
        // store each score in population as line
        for(unsigned int e=0; e<recorded_performances[i].n_elem; e++){
            new_line += " " + std::to_string(recorded_performances[i](e));
        }
        // print all scores
        result_file << new_line << endl;

        result_file << endl
                    << endl
                    << endl;
    }
    result_file << endl;
}

double Net_benchmark::corrected_sample_std_dev(mat score_vector){
    // return variable
    double s = 0;
    double N = score_vector.n_rows;
    mat mean_vector = ones(score_vector.size(),1) * mean(score_vector);
    s = (1/(N-1)) * ((double) as_scalar(sum(pow(score_vector - mean_vector, 2))));
    s = sqrt(s);
    return s;
}

void Net_benchmark::train_net_and_save_performances(unsigned int pop_size_GA, unsigned int nb_generations_GA, double epsilon, unsigned int selected_mutation_scheme) {

    set_topology(max_topo);
    evo_trainer.set_nb_epochs(nb_generations_GA);

    // experiment.txt header
    experiment_file << "------------------------------------" << endl
                    << "Data-set\t\t"               << data_set.data_set_filename  << endl
                    << "NB replicates\t\t"          << nb_replicates            << endl
                    << "GA population size\t"       << pop_size_GA              << endl
                    << "Max NB generations\t"       << nb_generations_GA * 11   << endl
                    << "GA Termination criterion"   << "if population score variance < " << epsilon << " then stop" << endl
                    << "NB generations per fold"    << nb_generations_GA     << endl
                    << "End of search space\t"      << max_topo.to_string() << " (nb inputs_units/layer_nb outputs_nb_hid.lay)" << endl
                    << "------------------------------------" << endl
                    << "\n" << "\n";

    //
    // PREPARE PRINT-OUT RESULTS TO FILE
    //

    // declare file for writing-out results
    ofstream result_file;
    result_file.open(data_set.result_filename.c_str(), ios::out);
    // check file successfully open
    if(!result_file.is_open()) {
        cout << "Couldn't open results file, experiment aborted. Is it located in: \"" << data_set.result_filename.c_str() << "\" ?" << endl;
        exit(0);
    }

    //
    // PERFORMANCES DURING TRAINING
    //
    vector<mat> result_matrices_training_perfs;
    mat averaged_performances   = compute_learning_curves_perfs(result_matrices_training_perfs, epsilon, selected_mutation_scheme);
    mat err_perfs               = compute_replicate_error(result_matrices_training_perfs);

    // save results
    averaged_performances = join_horiz(averaged_performances, ones(averaged_performances.n_rows,1) * nb_replicates);
    // save error amonst replicates
    print_results_octave_format(result_file, averaged_performances, data_set.OCTAVE_perfs_VS_nb_epochs);
    print_results_octave_format(result_file, err_perfs, "err_" + data_set.OCTAVE_perfs_VS_nb_epochs);

    //
    // PERFORMANCES AS DATA-SET SIZE INCREASES
    //
    /*
    vector<mat> result_matrices_perfs_data_set_size;
    mat averaged_perfs_data_set_size = compute_learning_curves_dataset_size(result_matrices_perfs_data_set_size, epsilon);
    mat err_perfs_data_set_size = compute_replicate_error(result_matrices_perfs_data_set_size);

    print_results_octave_format(result_file, averaged_perfs_pop_size, octave_variable_name_cost_training_set_size);
    */





    /*
    //
    // PERFORMANCES AS POPULATION SIZE INCREASES
    //
    vector<mat> result_matrices_perfs_pop_size;
    mat pop_sizes;
    mat averaged_perfs_pop_size     = compute_learning_curves_population_size(result_matrices_perfs_pop_size, pop_sizes, epsilon);

    cout << "Score matrix for each replicate" << endl;
    for(unsigned int i=0; i<result_matrices_perfs_pop_size.size(); i++){
        cout << "replicate " << i << " scores:" << endl;
        result_matrices_perfs_pop_size[i].print();
    }
    cout << "POP SIZE - AVERAGE MATRIX :" << endl;
    averaged_perfs_pop_size.print();
    mat err_perfs_pop_size          = compute_pop_size_replicate_error(result_matrices_perfs_pop_size);
    cout << "POP SIZE - ERROR" << endl;
    err_perfs_pop_size.print();
    cout << "for each average score I have an error? " << (err_perfs_pop_size.n_rows == averaged_perfs_pop_size.n_rows) << endl;

    print_results_octave_format(result_file, pop_sizes, "pop_sizes");
    print_results_octave_format(result_file, averaged_perfs_pop_size, octave_variable_name_scores_pop_size);
    print_results_octave_format(result_file, err_perfs_pop_size, "err_" + octave_variable_name_scores_pop_size);
    */
}

void Net_benchmark::training_task(unsigned int i, unsigned int nb_replicates, string data_set_filename, vector<mat> &result_matrices_training_perfs, double epsilon, unsigned int selected_mutation_scheme){

    // --- create copy of attributes ---
    Evolutionary_trainer t;
    t.set_nb_epochs(evo_trainer.get_nb_epochs());
    t.set_epsilon(epsilon);
    t.set_population(evo_trainer.get_population());
    Data_set d = data_set;
    net_topology max_t = max_topo;
    // --- ---

    // result matrices (to be interpreted by Octave script <Plotter.m>)
    mat results_score_evolution;

    cout << endl
         << "***"
         << "\tRUNNING REPLICATE " << i+1 << "/" << nb_replicates << "\t "<< data_set_filename
         << "***"
         << endl;

    // set seed
    unsigned int seed=i*10;
    std::srand(seed);

    // reset net
    #pragma omp critical
    net = NeuralNet();

    // train net up to largest topology
    NeuralNet trained_net = t.train_topology_plus_weights(d, max_t, results_score_evolution, selected_mutation_scheme);
    //NeuralNet trained_net = t.train_topology_plus_weights_PSO(d, max_t, results_score_evolution);

    #pragma omp critical
    net = trained_net;
    // print-out best perfs
    double best_score = results_score_evolution(results_score_evolution.n_rows-1, 3);

    result_matrices_training_perfs.push_back(results_score_evolution);
    cout            << "THREAD" << omp_get_thread_num() << " replicate=" << i << "\tseed=" << seed << "\tbest_score=" << "\ton" << best_score << data_set_filename << endl;
    experiment_file << "THREAD" << omp_get_thread_num() << " replicate=" << i << "\tseed=" << seed << "\tbest_score=" << "\ton" << best_score << data_set_filename << endl;
}

mat Net_benchmark::compute_learning_curves_perfs(vector<mat> &result_matrices_training_perfs, double epsilon, unsigned int selected_mutation_scheme){
    // return variable
    mat averaged_performances;

    // for each replicate
    #pragma omp parallel
    {
        // kick off a single thread
        #pragma omp single
        {
            for(unsigned int i=0; i<nb_replicates; ++i) {
                #pragma omp task
                training_task(i, nb_replicates, data_set.data_set_filename, result_matrices_training_perfs, epsilon, selected_mutation_scheme);
            }
        }
    }
    // reset net
    net = NeuralNet();

    // average PERFS of all replicates
    averaged_performances = average_matrices(result_matrices_training_perfs);
    return averaged_performances;
}

void Net_benchmark::compute_scores_task(unsigned int i, net_topology max_topo, vector<mat> &replicated_results, mat &population_sizes, double epsilon, unsigned int selected_mutation_scheme){

    unsigned int max_pop_size = evo_trainer.get_population_size();
    unsigned int tmp_pop_size=0;
    mat results_score_evolution;

    mat scores;
    mat pop_sizes;

    // for up to twice the size of the original pop
    for(unsigned int p=1; tmp_pop_size<(max_pop_size*4); p++) {

        // --- create copy of attributes ---
        Evolutionary_trainer t;
        t.set_epsilon(epsilon);
        t.set_nb_epochs(evo_trainer.get_nb_epochs());
        t.set_population(evo_trainer.get_population());
        Data_set d = data_set;
        net_topology max_t = max_topo;
        // --- ---

        unsigned int step_size = (max_pop_size/11);
        tmp_pop_size = (p*step_size)+5;

        pop_sizes = join_vert(pop_sizes, to_matrix(tmp_pop_size));

        cout << endl
             << "POPULATION SIZE = " << tmp_pop_size << " up to " << (max_pop_size*2)
             << endl;

        // set seed
        unsigned int seed=i*10;
        std::srand(seed);

        // reset pop and set pop size
        t.initialize_random_population(tmp_pop_size, max_topo);

        // reset net
        #pragma omp critical
        net = NeuralNet();

        // train net up to largest topology
        NeuralNet trained_net = t.train_topology_plus_weights(d, max_t, results_score_evolution, selected_mutation_scheme);
        cout << "REPLICATE" << i << " DONE" << endl;

        #pragma omp critical
        net = trained_net;
        cout << "set net attribute as trained net" << endl;

        scores = join_vert(scores, to_matrix(results_score_evolution(results_score_evolution.n_rows-1,3)));
        cout << "RESULTS SAVED" << endl;
        cout << "RUN NB" << p << endl;
    }
    // save population-size used
    population_sizes = pop_sizes;
    // save results of replica i
    replicated_results.push_back(scores);
}

mat Net_benchmark::compute_learning_curves_population_size(vector<mat> &result_matrices_perfs_pop_size, mat &pop_sizes, double epsilon, unsigned int selected_mutation_scheme){
    // return variable
    mat average_scores;
    // for each replicate
    #pragma omp parallel
    {
        // kick off a single thread
        #pragma omp single
        for(unsigned int i=0; i<nb_replicates; i++) {
            #pragma omp task shared(result_matrices_perfs_pop_size)
            compute_scores_task(i, max_topo, result_matrices_perfs_pop_size, pop_sizes, epsilon, selected_mutation_scheme);
        }
    }

    average_scores = average_matrices(result_matrices_perfs_pop_size);
    // return averaged scores of all populations
    return average_scores;
}

// returns a result matrix containing learning curves
mat Net_benchmark::compute_learning_curves_dataset_size(vector<mat> &result_matrices_perfs_data_set_sizes, unsigned int selected_mutation_scheme) {
    // result matrix
    mat results_cost_relative_to_training_set_size;
    Data_set entire_data_set = data_set;

    // declare variable to record model's performance
    double model_score_training_set      = 0.0f;
    double model_accuracy_training_set   = 0.0f;
    double model_score_validation_set    = 0.0f;
    double model_accuracy_validation_set = 0.0f;

    mat new_line;

    unsigned int m = entire_data_set.training_set.X.n_rows;

    // aggregate score and accuracy values for increasingly large proportion of training-set
    for(unsigned int i = 10 ; i < m ; i=i + (m/10)) {
        //reset net
        net = NeuralNet();

        cout << "Learning curve: " << (i*100) / m << "% done" << endl;
        model_score_training_set      = 0.0f;
        model_accuracy_training_set   = 0.0f;
        model_score_validation_set    = 0.0f;
        model_accuracy_validation_set = 0.0f;

        // where i is the index limit of the training-set subset
        data_set.training_set.X = entire_data_set.training_set.X.rows(0, i);
        data_set.training_set.Y = entire_data_set.training_set.Y.rows(0, i);

        cout << "starting training on partial data-set" << endl;
        // train using cross-validation using current percentage of the data-set
        mat dummy_result_matrix;
        net = evo_trainer.train_topology_plus_weights(data_set, max_topo, dummy_result_matrix, selected_mutation_scheme);

        // perform prediction using optimized model on training-subset and record performances
        compute_perfs_test_validation( model_score_training_set,
                                       model_accuracy_training_set,
                                       model_score_validation_set,
                                       model_accuracy_validation_set);

        // create new vector with model's performances on training-set and validation-set
        new_line << i
                 << model_accuracy_training_set
                 << model_score_training_set
                 << model_accuracy_validation_set
                 << model_score_validation_set
                 << endr;

        // append results for that training-set size to result matrix
        results_cost_relative_to_training_set_size= join_vert(results_cost_relative_to_training_set_size, new_line);
    }

    cout << "after func call, net looks like" << endl
         << net.get_topology().nb_input_units << endl
         << net.get_topology().nb_units_per_hidden_layer << endl
         << net.get_topology().nb_output_units << endl
         << net.get_topology().nb_hidden_layers << endl;

    data_set = entire_data_set;
    return results_cost_relative_to_training_set_size;
}

mat Net_benchmark::average_matrices(vector<mat> results){

    unsigned int smallest_nb_rows = INT_MAX;
    // find lowest and highest nb rows
    for(unsigned int i=0; i<results.size() ;i++){
        if(results[i].n_rows < smallest_nb_rows){
            smallest_nb_rows = results[i].n_rows;
        }
    }
    mat total_results;
    total_results.zeros(smallest_nb_rows, results[0].n_cols);

    // keep only up to the shortest learning curve
    vector<mat> processed_results;
    for(unsigned int i=0; i<results.size(); i++){
        processed_results.push_back( (mat)results[i].rows(0, smallest_nb_rows-1));
    }

    for(unsigned int i=0; i<results.size(); i++){
        total_results += processed_results[i];
    }
    mat averaged_results = total_results / results.size();
    return averaged_results;
}

mat Net_benchmark::to_matrix(double a){
    mat A;
    A << a;
    return A;
}

mat Net_benchmark::compute_replicate_error(vector<mat> results){
    // return variable
    mat err_vec;

    unsigned int smallest_nb_rows = INT_MAX;
    // find lowest and highest nb rows
    for(unsigned int i=0; i<results.size() ;i++){
        if(results[i].n_rows < smallest_nb_rows){
            smallest_nb_rows = results[i].n_rows;
        }
    }
    mat best_scores;
    // for each generation
    for(unsigned int i=0; i<smallest_nb_rows; i++) {
        best_scores.reset();
        // for each replica
        for(unsigned int r=0; r<nb_replicates; r++) {
            // get best score
            best_scores = join_vert(best_scores, to_matrix(results[r](i,3)));
        }
        // append std dev of all best scores for current generation to error vector
        err_vec = join_vert( err_vec, to_matrix(corrected_sample_std_dev(best_scores)));
    }
    return err_vec;
}

mat Net_benchmark::compute_pop_size_replicate_error(vector<mat> result_matrices_perfs_pop_size){
    // return variable
    mat err_vec;
    mat best_scores;

    for(unsigned int i=0; i<result_matrices_perfs_pop_size.size(); i++){
        cout << "matrix" << i << "\t dimensions: " << size(result_matrices_perfs_pop_size[i]) << endl;
    }

    cout << "NB OF DIFF POP SIZES: " << result_matrices_perfs_pop_size[0].n_rows << endl;
    cout << "NB REPs             : " << nb_replicates << endl;
    // for each population size
    for(unsigned int i=0; i<result_matrices_perfs_pop_size[0].n_rows; i++) {
        best_scores.reset();
        // for each replica
        for(unsigned int r=0; r<nb_replicates; r++) {
            // get best score
            best_scores = join_vert(best_scores, to_matrix(result_matrices_perfs_pop_size[r](i,0)));
        }
        // compute std dev of all best scores and append it to error vector
        err_vec = join_vert( err_vec, to_matrix(corrected_sample_std_dev(best_scores)));
    }
    return err_vec;
}

// returns current date/time in format YYYY-MM-DD.HH:mm:ss
const string Net_benchmark::get_current_date_time() {
    time_t     now = time(0);
    struct tm  tstruct;
    char       buffer[80];
    tstruct = *localtime(&now);
    strftime(buffer, sizeof(buffer), "%Y-%m-%d.%X", &tstruct);
    return buffer;
}

