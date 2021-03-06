#include "net_benchmark.h"

Net_benchmark::Net_benchmark() {
    // set default data-set
    data_set = Data_set();
    // set default max topology
    max_topo.nb_input_units = data_set.train_set.X.n_cols;
    max_topo.nb_units_per_hidden_layer = data_set.train_set.X.n_cols * 4;
    max_topo.nb_output_units = 1;
    max_topo.nb_hidden_layers = 2;
    //  Set default net topology
    unsigned int dataset_nb_features = data_set.train_set.X.n_cols;
    net_topology t;
    t.nb_input_units = dataset_nb_features;
    t.nb_units_per_hidden_layer = dataset_nb_features;
    t.nb_output_units = 1;
    t.nb_hidden_layers = 1;
    set_topology(t);
    // instantiate optimization algorithms
    evo_trainer       = Trainer_DE();
    experiment_file.open("random-seeds.txt", ios::app);
}

Net_benchmark::~Net_benchmark(){
    experiment_file.close();
}

void Net_benchmark::run_benchmark(exp_files ef) {
    unsigned int MUTATION_SCHEME_RAND = 0;
    unsigned int MUTATION_SCHEME_BEST = 1;
    unsigned int mutation_scheme = MUTATION_SCHEME_BEST;

    nb_replicates=ef.nb_reps;

    // for each algorithm
    for(unsigned int a=OPTIMIZATION_ALG::DE; a<=OPTIMIZATION_ALG::AIS;a++){
        unsigned int selected_opt_alg = a;
        // measure total time taken for each algorithm
        string start_time_str = get_current_date_time();
        auto start_time = system_clock::now();
        // apply on each data-set
        for(unsigned int i=0; i<ef.dataset_filenames.size(); i++) {
            cout<<"data set: "<<ef.dataset_filenames[i]<<endl;
            // use data requested by user
            data_set.select_data_set(ef.dataset_filenames[i]);
            // set largest topology
            max_topo.nb_input_units = data_set.train_set.X.n_cols;
            max_topo.nb_units_per_hidden_layer = 10;
            max_topo.nb_output_units = data_set.find_nb_prediction_classes(data_set.data);
            max_topo.nb_hidden_layers = 1;

            // 500 epochs in total is often more than enough
            double epsilon = -1;//find_termination_criteria_epsilon(200);
            // save results of cross-validated training
            train_net_and_save_performances(ef.pop_size, ef.max_nb_err_func_calls, selected_opt_alg, epsilon, mutation_scheme);
            cout<<"finished training using all replicates on "<<ef.dataset_filenames[i]<<" data"<<endl;
        }

        auto end_time = system_clock::now();
        string end_time_str = get_current_date_time();
        auto experiment_duration = duration_cast<std::chrono::minutes>(end_time-start_time).count();

        cout << endl
             << "Training started at  : " << start_time_str << endl
             << "Training finished at : " << end_time_str << " to produce result data "
             << "USING: " << nb_replicates << " replicates and " << ef.dataset_filenames.size() << " data sets)" << endl
             << "experiment duration :\t" << experiment_duration << " minutes" << endl;

        experiment_file << "Training started at  : " << start_time_str << endl
                        << "Training finished at : " << end_time_str << " (to produce result data "
                        << "USING: " << nb_replicates << " replicates and " <<ef.dataset_filenames.size()<< " data sets)" << endl
                        << "experiment duration :\t" << experiment_duration << " minutes" << endl;
    }
}

double Net_benchmark::find_termination_criteria_epsilon(unsigned int many_generations) {
    net_topology min_topo;
    min_topo.nb_input_units = max_topo.nb_input_units;
    min_topo.nb_units_per_hidden_layer = 1;
    min_topo.nb_output_units = max_topo.nb_output_units;
    min_topo.nb_hidden_layers = 1;

    mat results_perfs;
    cout<<"search termination criteria epsilon with "<<many_generations<<" generations"<<endl;

    Trainer_DE t;
    // force trainer to perform all epochs
    t.set_epsilon(-1);

    unsigned int MUTATION_SCHEME_RAND = 0;
    unsigned int MUTATION_SCHEME_BEST = 1;
    for(unsigned int i=0;i<many_generations;i++)
        t.evolve_through_iterations(data_set, min_topo, max_topo,1, results_perfs,0, MUTATION_SCHEME_RAND,i);

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

// returns the net with the best score from a population of nets trained with various topologies
void Net_benchmark::train_topology(NeuralNet &evolved_net){

    // nb of different topologies to search
    unsigned int nb_topology_sizes = 5;
    NeuralNet current_net;
    double current_perf = 0;
    NeuralNet best_net;
    double best_perf = 0;
    net_topology t = net.get_topology();

    // search best topology
    for(unsigned int i=0; i<nb_topology_sizes; ++i) {
        // set new topology
        t.nb_units_per_hidden_layer += 1;
        current_net.set_topology(t);
        // optimize weights
        evo_trainer.train( data_set, current_net);
        // record quality of the model
        current_perf = current_net.get_train_score(data_set.val_set);
        // save best network
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

void Net_benchmark::compute_perfs_test_validation(double &model_score_train_set,
                                                  double &model_prediction_acc_train_set,
                                                  double &model_score_val_set,
                                                  double &model_prediction_acc_val_set) {
    // compute training-set accuracy
    model_prediction_acc_train_set   = net.get_train_acc(data_set.train_set);
    // compute training-set score
    model_score_train_set                 = net.get_train_score(data_set.train_set);
    // compute validation-set accuracy
    model_prediction_acc_val_set = net.get_train_acc(data_set.val_set);
    // compute validation-set score
    model_score_val_set               = net.get_train_score(data_set.val_set);
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
    data_subset train_set   = data_set.train_set;
    // result matrix
    mat results_cost_relative_to_train_set_size;
    // declare tmp variable for used training-set segment
    data_subset train_set_subset;
    // declare variable to record model's performance
    double model_score_train_set      = 0.0f;
    double model_acc_train_set   = 0.0f;
    double model_score_val_set    = 0.0f;
    double model_acc_val_set = 0.0f;
    mat new_line;

    // aggregate score and accuracy values for increasingly large proportion of training-set
    for(unsigned int i = 1 ; i < train_set.X.n_rows ; ++i) {
        model_score_train_set      = 0.0f;
        model_acc_train_set   = 0.0f;
        model_score_val_set    = 0.0f;
        model_acc_val_set = 0.0f;
        // i designates the index limit of the training-set subset
        train_set_subset.X = train_set.X.rows(0, i);
        train_set_subset.Y = train_set.Y.rows(0, i);
        // perform prediction using optimized model on training-subset and record performances
        compute_perfs_test_validation( model_score_train_set,
                                       model_acc_train_set,
                                       model_score_val_set,
                                       model_acc_val_set);
        // create new vector with model's performances on training-set and validation-set
        new_line << i
                 << model_acc_train_set   << model_score_train_set
                 << model_acc_val_set << model_score_val_set
                 << endr;
        // append results for that size to result matrix
        results_cost_relative_to_train_set_size= join_vert(results_cost_relative_to_train_set_size,new_line);
    }
    return results_cost_relative_to_train_set_size;
}

void Net_benchmark::print_results_octave_format(ofstream &result_file, mat recorded_performances, string octave_variable_name){
    // Create header in MATLAB format
    result_file << "# Created by main.cpp, " << get_current_date_time() << endl
                << "# name: " << octave_variable_name << endl
                << "# type: matrix"  << endl
                << "# rows: " << recorded_performances.n_rows << endl
                << "# columns: " << recorded_performances.n_cols << endl;
    // append content of recorded performances into same file
    for(unsigned int i=0; i<recorded_performances.n_rows; ++i){
        for(unsigned int j=0; j<recorded_performances.n_cols; ++j)
            result_file << recorded_performances(i,j) << " ";
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
        result_file << new_line
                    << endl
                    << endl
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

void Net_benchmark::train_net_and_save_performances(unsigned int pop_size_GA, unsigned int max_nb_err_func_calls, unsigned int selected_opt_alg, double epsilon, unsigned int selected_mutation_scheme) {
    set_topology(max_topo);
    //evo_trainer.set_nb_epochs(max_nb_gens);
    evo_trainer.set_max_nb_err_func_calls(max_nb_err_func_calls);
    // experiment.txt header
    experiment_file << "------------------------------------" << endl
                    << "Data-set\t\t"               << data_set.data_set_filename  << endl
                    << "NB replicates\t\t"          << nb_replicates            << endl
                    << "GA population size\t"       << pop_size_GA              << endl
                    << "Max NB Calls to err function\t"       << max_nb_err_func_calls * 11   << endl
                    << "GA Termination criterion"   << "if population score variance < " << epsilon << " then stop" << endl
                    << "End of search space\t"      << max_topo.to_string() << " (nb inputs_units/layer_nb outputs_nb_hid.lay)" << endl
                    << "------------------------------------" << endl
                    << "\n" << "\n";

    evo_trainer.set_population(evo_trainer.generate_population(pop_size_GA,max_topo));

    // declare file for writing-out results
    ofstream result_file;
    string result_filename = data_set.result_filename.substr(0,data_set.result_filename.size()-4);
    // produce distinct result file prefixes for each algorithm
    switch(selected_opt_alg){
    case OPTIMIZATION_ALG::DE:
        result_filename+="/DE-";
        break;
    case OPTIMIZATION_ALG::PSO:
        result_filename+="/PSO-";
        break;
    case OPTIMIZATION_ALG::AIS:
        result_filename+="/AIS-";
        break;
    }
    result_filename+="results.mat";
    data_set.result_filename=result_filename;
    result_file.open(result_filename.c_str(), ios::out);
    // check file successfully open
    if(!result_file.is_open()) {
        cout << "Couldn't open results file, experiment aborted. Is it located in: \"" << data_set.result_filename.c_str() << "\" ?" << endl;
        exit(0);
    }

    vector<mat> result_matrices_train_perfs;
    // run experiment iterations
    mat averaged_performances   = compute_learning_curves_perfs(result_matrices_train_perfs, selected_opt_alg, epsilon, selected_mutation_scheme);
    mat err_perfs               = compute_replicate_error(result_matrices_train_perfs);

    // save results
    //averaged_performances = join_horiz(averaged_performances, ones(averaged_performances.n_rows,1) * nb_replicates);
    // save error amonst replicates
    print_results_octave_format(result_file, averaged_performances, "results");
    print_results_octave_format(result_file, err_perfs, "err_results");
    // inform user of where to find result file
    cout<<"results written on "<<data_set.result_filename.c_str()<<endl;
}

void Net_benchmark::training_task(unsigned int i, string data_set_filename, unsigned int selected_opt_algorithm,double epsilon, unsigned int selected_mutation_scheme){
    // return variable
    NeuralNet trained_net;
    Data_set d = data_set;
    net_topology max_t = max_topo;
    // result matrices (to be interpreted by Octave script <Plotter.m>)
    mat results_score_evolution;

    // set seed
    unsigned int seed=i*10;
    std::srand(seed);

    // reset net
    net = NeuralNet();

    // instantiating optimization algorithms
    Trainer_DE trainer_de;
    Trainer_PSO trainer_pso;
    Trainer_AIS trainer_ais;

    // train net up to largest topology using chosen trainer
    switch(selected_opt_algorithm){
    case OPTIMIZATION_ALG::DE:
        cout<<"Differential Evolution on "<<data_set_filename<<endl;
        // initialize optimization algorithm
        trainer_de.set_nb_epochs(evo_trainer.get_nb_epochs());
        trainer_de.set_max_nb_err_func_calls(evo_trainer.max_nb_err_func_calls);
        trainer_de.set_epsilon(epsilon);
        trainer_de.set_population(evo_trainer.get_population());
        trained_net = trainer_de.train_topology_plus_weights(d, max_t, results_score_evolution, selected_mutation_scheme);
        break;
    case OPTIMIZATION_ALG::PSO:
        cout<<"Particle Swarm Optimization on "<<data_set_filename<<endl;
        // initialize optimization algorithm
        trainer_pso.set_nb_epochs(evo_trainer.get_nb_epochs());
        trainer_pso.set_max_nb_err_func_calls(evo_trainer.max_nb_err_func_calls);
        trainer_pso.set_epsilon(epsilon);
        trainer_pso.set_population(evo_trainer.get_population());
        trained_net = trainer_pso.train_topology_plus_weights(d, max_t, results_score_evolution, -1);
        break;
    case OPTIMIZATION_ALG::AIS:
        cout<<"Clonal Selection on "<<data_set_filename<<endl;
        // initialize optimization algorithm
        trainer_ais.set_nb_epochs(evo_trainer.get_nb_epochs());
        trainer_ais.set_max_nb_err_func_calls(evo_trainer.max_nb_err_func_calls);
        trainer_ais.set_epsilon(epsilon);
        trainer_ais.set_population(evo_trainer.get_population());
        trained_net = trainer_ais.train_topology_plus_weights(d, max_t, results_score_evolution, -1);
        break;
    default:
        cout<<"Default Differential Evolution on "<<data_set_filename<<endl;
        // initialize optimization algorithm
        trainer_de.set_nb_epochs(evo_trainer.get_nb_epochs());
        trainer_de.set_max_nb_err_func_calls(evo_trainer.max_nb_err_func_calls);
        trainer_de.set_epsilon(epsilon);
        trainer_de.set_population(evo_trainer.get_population());

        cout<<"***"
            <<endl;
        trained_net = trainer_de.train_topology_plus_weights(d, max_t, results_score_evolution, selected_mutation_scheme);
    }

    // save obtained results locally (binary format)
#pragma omp critical
    results_score_evolution.save("res"+to_string(i)+".mat");
    // memorize trained net
    net = trained_net;
    // print-out best perfs
    double best_score = results_score_evolution(results_score_evolution.n_rows-1, 3);
    cout            << "THREAD"<<omp_get_thread_num()<<" REPLICATE" << i << "\tseed=" << seed << "\ttrain.score=" << "\t" << best_score << " on " << data_set_filename << endl;
    experiment_file << "THREAD"<<omp_get_thread_num()<<" REPLICATE" << i << "\tseed=" << seed << "\ttrain.score=" << "\t" << best_score << " on " << data_set_filename << endl;
}

mat Net_benchmark::compute_learning_curves_perfs(vector<mat> &result_matrices_train_perfs, unsigned int selected_opt_alg, double epsilon, unsigned int selected_mutation_scheme){
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
                {
                training_task(i, data_set.data_set_filename, selected_opt_alg, epsilon, selected_mutation_scheme);
                }
            }
        }
    }

    // aggregate replicates results
    for(unsigned int i=0;i<nb_replicates;i++){
        mat r;
        r.load("res"+to_string(i)+".mat");
        result_matrices_train_perfs.push_back(r);
    }

    // clean-up auto generated result files
    for(unsigned int i=0;i<nb_replicates;i++)
        std::remove(("res"+to_string(i)+".mat").c_str());

    // reset net
    net = NeuralNet();
    // average replicates results
    averaged_performances = average_matrices(result_matrices_train_perfs);
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
        Trainer_DE t;
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

        // reset net
        net = NeuralNet();

        // train net up to largest topology
        NeuralNet trained_net = t.train_topology_plus_weights(d, max_t, results_score_evolution, selected_mutation_scheme);
        cout << "REPLICATE" << i << " DONE" << endl;

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
    mat results_cost_relative_to_train_set_size;
    Data_set entire_data_set = data_set;

    // declare variable to record model's performance
    double model_score_train_set      = 0.0f;
    double model_acc_train_set   = 0.0f;
    double model_score_val_set    = 0.0f;
    double model_acc_val_set = 0.0f;

    mat new_line;

    unsigned int m = entire_data_set.train_set.X.n_rows;

    // aggregate score and accuracy values for increasingly large proportion of training-set
    for(unsigned int i = 10 ; i < m ; i=i + (m/10)) {
        //reset net
        net = NeuralNet();

        cout << "Learning curve: " << (i*100) / m << "% done" << endl;
        model_score_train_set      = 0.0f;
        model_acc_train_set   = 0.0f;
        model_score_val_set    = 0.0f;
        model_acc_val_set = 0.0f;

        // where i is the index limit of the training-set subset
        data_set.train_set.X = entire_data_set.train_set.X.rows(0, i);
        data_set.train_set.Y = entire_data_set.train_set.Y.rows(0, i);

        cout << "starting training on partial data-set" << endl;
        // train using cross-validation using current percentage of the data-set
        mat dummy_result_matrix;
        net = evo_trainer.train_topology_plus_weights(data_set, max_topo, dummy_result_matrix, selected_mutation_scheme);

        // perform prediction using optimized model on training-subset and record performances
        compute_perfs_test_validation( model_score_train_set,
                                       model_acc_train_set,
                                       model_score_val_set,
                                       model_acc_val_set);

        // create new vector with model's performances on training-set and validation-set
        new_line << i
                 << model_acc_train_set
                 << model_score_train_set
                 << model_acc_val_set
                 << model_score_val_set
                 << endr;

        // append results for that training-set size to result matrix
        results_cost_relative_to_train_set_size= join_vert(results_cost_relative_to_train_set_size, new_line);
    }

    cout << "after func call, net looks like" << endl
         << net.get_topology().nb_input_units << endl
         << net.get_topology().nb_units_per_hidden_layer << endl
         << net.get_topology().nb_output_units << endl
         << net.get_topology().nb_hidden_layers << endl;

    data_set = entire_data_set;
    return results_cost_relative_to_train_set_size;
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
    mat err_vec;
    unsigned int smallest_nb_rows = INT_MAX;
    // find lowest and highest nb rows
    for(unsigned int i=0;i<results.size();i++)
        if(results[i].n_rows<smallest_nb_rows)
            smallest_nb_rows = results[i].n_rows;
    mat best_scores;
    // for each generation
    for(unsigned int i=0; i<smallest_nb_rows; i++) {
        best_scores.reset();
        // for each replica: get best score
        for(unsigned int r=0; r<nb_replicates; r++)
            best_scores = join_vert(best_scores, to_matrix(results[r](i,3)));
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

