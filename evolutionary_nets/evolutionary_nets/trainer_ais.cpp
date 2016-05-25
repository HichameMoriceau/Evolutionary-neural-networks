#include "trainer_ais.h"

Trainer_AIS::Trainer_AIS(){
    // default nb generations:
    nb_epochs = 1000;
    // default variance value (convergence treshold for GA stopping criteria)
    epsilon = 1.0f;
    // default topologies for individuals
    net_topology t;
    t.nb_input_units = 1;
    t.nb_units_per_hidden_layer = 5;
    t.nb_output_units = 1;
    t.nb_hidden_layers = 1;
    NeuralNet ann(t);
    initialize_random_population(50, t);

    // default population size: 100
    population = convert_population_to_nets(generate_random_genome_population(100,ann));
}

void Trainer_AIS::train(Data_set data_set, NeuralNet &net){
    mat results_score_evolution;
    train(data_set, net, results_score_evolution);
}

void Trainer_AIS::train(Data_set data_set, NeuralNet &net, mat &results_score_evolution) {
    net = train_topology_plus_weights(data_set, net.get_topology(), results_score_evolution, -1);
}

NeuralNet Trainer_AIS::evolve_through_iterations(Data_set data_set, net_topology min_topo, net_topology max_topo, unsigned int nb_epochs, mat &results_score_evolution, unsigned int index_cross_validation_section, unsigned int selected_mutation_scheme) {
    NeuralNet trained_model = population[0];
    mat new_line;
    // flag alerting that optimization algorithm has had ~ same results for the past 100 generations
    bool plateau = false;
    // flag alerting that the GA has converged
    bool has_converged = false;
    // recorded characteristics
    double prediction_accuracy = 0;
    double score = 0;
    double MSE   = 0;
    double pop_score_variance = 100;
    double pop_score_stddev = 0;
    double pop_score_mean = 0;
    double pop_score_median = 0;
    double ensemble_accuracy = 0;
    double ensemble_score = 0;
    double pop_score = 0;
    double pop_accuracy=0;

    // using vectors as genotype
    vector<vec> genome_population = convert_population_to_genomes(population, max_topo);
    vector<NeuralNet> ensemble = population;

    /**
     *  ALGORITHM:    Clonal Selection
     *
     *
     *  TERMINATION CRITERIA:
     *      If all generations were achieved OR if the GA has already converged
    */
    for(unsigned int i=0; ((i<nb_epochs) && (!has_converged)); ++i) {
        // update individuals
        population = convert_population_to_nets(genome_population);
        // evaluate population
        for(unsigned int s=0; s<population.size() ; ++s) {
            population[s].get_f1_score(data_set.validation_set);
        }
        // sort from fittest
        sort(population.begin(), population.end());
        // get best model
        if(population[0].get_f1_score(data_set.validation_set) >= trained_model.get_f1_score(data_set.validation_set))
            trained_model = population[0];
        // compute accuracy
        elective_accuracy(population, data_set, pop_accuracy, pop_score);

        // get best ensemble
        if(pop_score >= ensemble_score) {
            ensemble = population;
            ensemble_score = pop_score;
            ensemble_accuracy = pop_accuracy;
        }

        genome_population = convert_population_to_genomes(population, max_topo);
        // optimize model params and topology using training-set
        clonal_selection_topology_evolution(genome_population, data_set.training_set, min_topo, max_topo, selected_mutation_scheme);
        // record model performances on new data
        prediction_accuracy =   trained_model.get_accuracy(data_set.validation_set);
        score               =   trained_model.get_f1_score(data_set.validation_set);
        MSE                 =   trained_model.get_MSE(data_set.validation_set);
        pop_score_variance  =   compute_score_variance(genome_population, data_set.validation_set);
        pop_score_stddev    =   compute_score_stddev(genome_population, data_set.validation_set);
        pop_score_mean      =   compute_score_mean(genome_population, data_set.validation_set);
        pop_score_median    =   compute_score_median(genome_population, data_set.validation_set);
        // record results (performances and topology description)
        unsigned int inputs             =   trained_model.get_topology().nb_input_units;
        unsigned int hidden_units       =   trained_model.get_topology().nb_units_per_hidden_layer;
        unsigned int outputs            =   trained_model.get_topology().nb_output_units;
        unsigned int nb_hidden_layers   =   trained_model.get_topology().nb_hidden_layers;
        // format result line
        new_line << i //+ nb_epochs * index_cross_validation_section
                 << MSE
                 << prediction_accuracy
                 << score
                 << pop_score_variance

                 << pop_score_stddev
                 << pop_score_mean
                 << pop_score_median
                 << population.size()
                 << inputs

                 << hidden_units
                 << outputs
                 << nb_hidden_layers
                 << true
                 << selected_mutation_scheme

                 << ensemble_accuracy
                 << ensemble_score
                 << endr;

        // append result line to result matrix
        results_score_evolution = join_vert(results_score_evolution, new_line);
        cout << fixed
             << setprecision(2)
             << "Gen="            << i
             << "\tscore="          << score
             << "  MSE="            << MSE
             << "  acc="            << prediction_accuracy
             << "  score.mean=" << pop_score_mean
             << "  score.var=" << pop_score_variance
             << "\tNB.hid.lay="     << nb_hidden_layers
             << "  NB.hid.units="   << hidden_units
             << "\tens.acc=" << ensemble_accuracy
             << "  ens.score=" << ensemble_score
             << endl;

        // checking for convergence (termination criterion)
        // if 33% of the total_nb_generations have been executed on CV fold
        if(i>(nb_epochs/3)) {
            // if current best score is similar to best score of 100 generations before
            if(score < results_score_evolution(results_score_evolution.n_rows-(nb_epochs/10),3)+1)
                plateau = true;
            else
                plateau = false;
            has_converged = (pop_score_variance<epsilon) && (plateau);
        }else{
            // otherwise always force training on first 10% of total generations
            has_converged = false;
        }
    }
    return trained_model;
}

void Trainer_AIS::clonal_selection_topology_evolution(vector<vec> &pop, data_subset training_set, net_topology min_topo, net_topology max_topo, unsigned int selected_mutation_scheme){
    NeuralNet dummyNet(max_topo);
    unsigned int nb_element_vectorized_Theta = dummyNet.get_total_nb_weights() + 4;
    // Number of individuals retained
    unsigned int selection_size = pop.size()*100/100;
    // Number of random cells incorporated in the population for every generation
    unsigned int nb_rand_cells = (unsigned int)pop.size()*5/100;
    // Scaling factor
    double clone_rate = 0.15;

    // -- Differential Evolution settings (for mutation operation only) --
    // total nb of variables
    unsigned int problem_dimensionality = nb_element_vectorized_Theta;
    // Crossover Rate [0,1]
    double CR = 0.5;
    // differential_weight [0,2]
    double F = 1;
    // -- --

    unsigned int genome_length = get_genome_length(max_topo);

    for(unsigned int j=0; j<pop.size()-1; ++j) {
        // instantiate subpopulations
        vector<vec>selected_indivs = select(selection_size, pop, training_set, max_topo);
        vector<vector<vec>>pop_clones;
        unsigned int nb_clones_array[selection_size];

        // generate clones
        for(unsigned int i=0; i<selection_size; i++) {
            unsigned int nb_clones = compute_nb_clones(clone_rate, selection_size, i+1);
            nb_clones_array[i] = nb_clones;
            pop_clones.push_back(generate_clones(nb_clones, selected_indivs[i]));
        }

        // hyper-mutate (using a DE/RAND/1 mutative-crossover operation)
        for(unsigned int i=0; i<pop_clones.size(); i++) {
            for(unsigned int j=0; j<pop_clones[i].size(); j++){
                // select four random but different individuals from (pop)
                // declare index variables
                unsigned int index_x = generate_random_integer_between_range(1, pop.size() - 1);
                unsigned int index_a;
                unsigned int index_b;
                unsigned int index_c;

                do{
                    index_a = generate_random_integer_between_range(1, pop.size() - 1);
                    // making sure that no two identical indexes are generated
                }while(index_a == index_x);

                do{
                    index_b = generate_random_integer_between_range(1, pop.size() - 1);
                }while(index_b == index_a || index_b == index_x);

                do{
                    index_c = generate_random_integer_between_range(1, pop.size() - 1);
                }while(index_c == index_b || index_c == index_a || index_c == index_x);

                // store corresponding individual in pop
                vec indiv_a = pop[index_a];
                vec indiv_b = pop[index_b];
                vec indiv_c = pop[index_c];

                vec original_model  = pop[index_x];
                vec candidate_model = pop[index_x];
                mutative_crossover(problem_dimensionality, CR, F, genome_length, min_topo, max_topo, original_model, candidate_model,indiv_a, indiv_b, indiv_c);
            }
        }

        // put all solutions in same group
        vector<vec> all_clones = add_all(pop_clones, nb_clones_array);
        // add some random individuals to maintain diversity
        vector<vec> rand_indivs = generate_random_topology_genome_population(nb_rand_cells, min_topo, max_topo);
        all_clones.insert(all_clones.end(), rand_indivs.begin(), rand_indivs.end());

        // select n best solutions
        pop = select(selection_size, all_clones, training_set, max_topo);
    }
    // update population
    population = convert_population_to_nets(pop);
}

vector<vec> Trainer_AIS::select(unsigned int quantity, vector<vec> pop, data_subset training_set, net_topology max_topo){
    if(quantity>pop.size())
        return pop;
    vector<NeuralNet> s_pop;
    vector<NeuralNet> casted_pop = convert_population_to_nets(pop);
    // sort from fittest
    evaluate_population(casted_pop, training_set);
    // select only best indivs
    for(unsigned int i=0; i<quantity; i++){
        s_pop.push_back(casted_pop[i]);
    }
    return convert_population_to_genomes(s_pop, max_topo);
}

vector<vec> Trainer_AIS::generate_clones(unsigned int nb_clones, vec indiv){
    vector<vec> cloned_pop;
    for(unsigned int i=0; i<nb_clones; i++){
        cloned_pop.push_back(indiv);
    }
    return cloned_pop;
}

unsigned int Trainer_AIS::compute_nb_clones(double beta, int pop_size, int index){
    if (round((beta*pop_size)/index)>0)
        return round((beta*pop_size)/index);
    else
        return 1;
}

vector<vec> Trainer_AIS::add_all(vector<vector<vec>> all_populations, unsigned int* nb_clones_array){
    vector<vec> pop;
    for(unsigned int i=0; i<all_populations.size(); i++){
        for(unsigned int j=0; j<nb_clones_array[i]; j++)
        pop.push_back(all_populations[i][j]);
    }
    return pop;
}

void Trainer_AIS::mutative_crossover(unsigned int problem_dimensionality, double CR, double F, unsigned int genome_length, net_topology min_topo, net_topology max_topo, vec original_model, vec &candidate_model, vec indiv_a, vec indiv_b, vec indiv_c){
    vec tmp_genome = original_model;
    // pick random index
    unsigned int R    = generate_random_integer_between_range(1, problem_dimensionality);
    // used to generate random 0 or 1
    unsigned int rand = generate_random_integer_between_range(1, 50);
    // crossover + mutation
    for(unsigned int k=0; k<genome_length; ++k) {
        if( floor(generate_random_integer_between_range(1,problem_dimensionality) == R |
                  (rand%2) < CR)) {
            // protect number of I/Os to be altered
            if( (k!=0) && (k!=2) ) {
                // using a DE/rand/1 scheme
                tmp_genome[k]  = mutation_scheme_DE_rand_1(F, indiv_a[k], indiv_b[k], indiv_c[k]);
                // make sure NB hid. units doesn't break contract
                if(k==1){
                    do{
                        tmp_genome[1] = (((int) abs( mutation_scheme_DE_rand_1(F, indiv_a[1], indiv_b[1], indiv_c[1]))) % max_topo.nb_units_per_hidden_layer) + 1;
                    }while(!(tmp_genome[1]>=min_topo.nb_units_per_hidden_layer && tmp_genome[1]<=max_topo.nb_units_per_hidden_layer));
                }else if(k==3){
                    // make sure NB hid. lay. doesn't break contract
                    do{
                        tmp_genome[3] = (((int) abs(mutation_scheme_DE_rand_1(F, indiv_a[3], indiv_b[3], indiv_c[3])) ) % max_topo.nb_hidden_layers) + 1;
                    }while(!(tmp_genome[3]>=min_topo.nb_hidden_layers && tmp_genome[3]<=max_topo.nb_hidden_layers));
                }
            }

            // once topology is chosen
            if(k==4){
                net_topology candidate_topology;
                candidate_topology.nb_input_units = (unsigned int) tmp_genome[0];
                candidate_topology.nb_units_per_hidden_layer = (unsigned int) tmp_genome[1];
                candidate_topology.nb_output_units = (unsigned int) tmp_genome[2];
                candidate_topology.nb_hidden_layers = (unsigned int) tmp_genome[3];
                genome_length = get_genome_length(candidate_topology);
            }
        }
    }
    // return offspring
    candidate_model = tmp_genome;
}

double Trainer_AIS::mutation_scheme_DE_rand_1(double F, double x_rand_1, double x_rand_2, double x_rand_3){
    return x_rand_1 + F * (x_rand_2 - x_rand_3);
}
// ---
/*
NeuralNet Trainer_AIS::train_topology_plus_weights(Data_set data_set, net_topology max_topo, mat &results_score_evolution, unsigned int selected_mutation_scheme) {
    NeuralNet cross_validated_net;
    net_topology min_topo;
    min_topo.nb_input_units = max_topo.nb_input_units;
    min_topo.nb_units_per_hidden_layer = 1;
    min_topo.nb_output_units = max_topo.nb_output_units;
    min_topo.nb_hidden_layers = 1;

    double avrg_score = 0;
    double avrg_acc   = 0;
    cross_validated_net = cross_validation_training(data_set, min_topo, max_topo, results_score_evolution, avrg_score, avrg_acc, selected_mutation_scheme);
    // append Cross Validation error to result matrix
    mat avrg_CV_score = ones(results_score_evolution.n_rows,1) * avrg_score;
    mat avrg_CV_acc   = ones(results_score_evolution.n_rows,1) * avrg_acc;
    results_score_evolution = join_horiz(results_score_evolution, avrg_CV_score);
    results_score_evolution = join_horiz(results_score_evolution, avrg_CV_acc);
    cout << "average score on all validation-sets = " << avrg_score << endl;
    mat mutation_scheme = ones(results_score_evolution.n_rows,1) * selected_mutation_scheme;
    results_score_evolution = join_horiz(results_score_evolution, mutation_scheme);
    return cross_validated_net;
}

NeuralNet Trainer_AIS::cross_validation_training(Data_set data_set, net_topology min_topo, net_topology max_topo, mat &results_score_evolution, double &avrg_score, double &avrg_acc, unsigned int selected_mutation_scheme){
    unsigned int nb_folds = 10;
    NeuralNet tmp_net(max_topo);
    tmp_net.set_topology(max_topo);
    NeuralNet cross_validated_net;
    cross_validated_net.set_topology(max_topo);
    mat tmp_results_perfs;
    mat perfs_cross_validation;

    unsigned int pop_size = population.size();
    population = convert_population_to_nets(generate_random_topology_genome_population(pop_size,min_topo, max_topo));

    // for each fold
    for(unsigned int k=0; k<nb_folds; ++k) {
        cout << "Using validation-set" << k << " of" << nb_folds-1 << endl;
        data_set.subdivide_data_cross_validation(k, nb_folds);
        // make sure topology is adequate to data-set
        max_topo.nb_input_units = data_set.training_set.X.n_cols;
        //max_topo.nb_output_units = 1;

        // insert model trained on previous CV section in pop
        insert_individual(tmp_net);

        // empty temporary result matrix
        tmp_results_perfs.reset();
        tmp_net = evolve_through_iterations(data_set, min_topo, max_topo, nb_epochs, tmp_results_perfs, k, selected_mutation_scheme);

        // update best model
        cross_validated_net.set_topology(tmp_net.get_topology());
        cross_validated_net.set_params(tmp_net.get_params());

        // append results for this fold to results to be printed
        perfs_cross_validation = join_vert(perfs_cross_validation, tmp_results_perfs);
    }
    cout << "start last fold" << endl;
    // force last training cycle to do all epochs
    set_epsilon(-1);
    // force last training cycle to make use of entire training set
    data_set.training_set.X = data_set.data.cols(0,data_set.data.n_cols-2);
    data_set.training_set.Y = data_set.data.col(data_set.data.n_cols-1);
    // train net
    mat perfs_entire_training_set;
    cross_validated_net = evolve_through_iterations(data_set, min_topo, max_topo, nb_epochs, perfs_entire_training_set, nb_folds, selected_mutation_scheme);

    // compute the average score
    double total_scores=0;
    double total_accuracies=0;
    for(unsigned int i=0; i<nb_folds; i++){
        data_set.subdivide_data_cross_validation(i, nb_folds);
        total_accuracies+=cross_validated_net.get_accuracy(data_set.validation_set);
        total_scores+=cross_validated_net.get_f1_score(data_set.validation_set);
    }
    // force last training cycle to make use of entire training set
    data_set.training_set.X = data_set.data.cols(0,data_set.data.n_cols-2);
    data_set.training_set.Y = data_set.data.col(data_set.data.n_cols-1);
    total_accuracies+=cross_validated_net.get_accuracy(data_set.validation_set);
    total_scores += cross_validated_net.get_f1_score(data_set.validation_set);

    // return average score as reference
    avrg_score = total_scores/(nb_folds+1);
    // return average accuracy as reference
    avrg_acc = total_accuracies/(nb_folds+1);
    // return result matrix as reference
    results_score_evolution = join_vert(perfs_cross_validation, perfs_entire_training_set);
    // return trained net
    return cross_validated_net;
}


double Trainer_AIS::f_rand(double fMin, double fMax){
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

void Trainer_AIS::elective_accuracy(vector<NeuralNet> pop, Data_set data_set, double &ensemble_accuracy, double &ensemble_score){
    // sort pop by fitness
    evaluate_population(pop, data_set.validation_set);
    unsigned int nb_individuals = pop.size();

    mat elected_votes(data_set.validation_set.X.n_rows,1);
    mat pos_votes(data_set.validation_set.X.n_rows,1);
    mat neg_votes(data_set.validation_set.X.n_rows,1);
    for(unsigned int i=0;i<pop.size(); i++){
        // perform predictions on provided data-set
        mat H = pop[i].forward_propagate(data_set.validation_set.X);
        mat Predictions = round(H);

        // retrieve vote (higher fitnesses weight more)
        for(unsigned int p=0; p<Predictions.n_rows; p++){
            if(Predictions[p] == 0){
                neg_votes[p] += (nb_individuals/(p+1));
            }else{
                pos_votes[p] += (nb_individuals/(p+1));
            }
        }
    }

    // apply majority for each example
    for(unsigned int p=0; p<data_set.validation_set.X.n_rows; p++){
        if(pos_votes(p) > neg_votes(p))
            elected_votes(p) = 1;
        else
            elected_votes(p) = 0;
    }

    // compute ensemble score based on <elected_votes>
    // How many selected items are relevant ?
    double precision = 0.0f;
    // How many relevant items are selected ?
    double recall = 0.0f;
    // score is based on precision and recall
    double computed_score = 0.0f;
    unsigned int true_positives  = sum(sum(elected_votes==1 && data_set.validation_set.Y==1));
    unsigned int false_positives = sum(sum(elected_votes==1 && data_set.validation_set.Y==0));
    unsigned int false_negatives = sum(sum(elected_votes==0 && data_set.validation_set.Y==1));
    if( !((true_positives + false_positives)==0 || (true_positives + false_negatives)==0)){
        precision =  ( (double) true_positives) / (true_positives + false_positives);
        recall    =  ( (double) true_positives) / (true_positives + false_negatives);
        // compute score
        computed_score = (2.0f*precision*recall) / (precision + recall);
        // make score of same scale as accuracy (better for plotting)
        computed_score = computed_score * 100;
    }
    // check for -NaN
    if(computed_score != computed_score)
        computed_score = 0;

    // return ensemble accuracy
    ensemble_accuracy = (get_nb_identical_elements(elected_votes.col(0), data_set.validation_set.Y.col(0)) / double(data_set.validation_set.X.n_rows)) * 100;
    // return ensemble score
    ensemble_score = computed_score;
}

unsigned int Trainer_AIS::get_nb_identical_elements(mat A, mat B){
    if(A.n_rows != B.n_rows || A.n_cols != B.n_cols)
        return 0;
    unsigned int count = 0;
    for(unsigned int i = 0 ; i < A.n_rows ; ++i)
        for(unsigned int j = 0 ; j < A.n_cols ; ++j)
            if(A(i,j) == B(i,j))
                ++count;
    return count;
}

void Trainer_AIS::initialize_random_population(unsigned int pop_size, net_topology max_topo){
    if(pop_size < 10) pop_size = 10;
    net_topology min_topo;
    min_topo.nb_input_units = max_topo.nb_input_units;
    min_topo.nb_units_per_hidden_layer = 1;
    min_topo.nb_output_units = max_topo.nb_output_units;
    min_topo.nb_hidden_layers = 1;

    population = convert_population_to_nets(generate_random_topology_genome_population(pop_size,min_topo, max_topo));
}

double Trainer_AIS::compute_score_variance(vector<NeuralNet> pop, data_subset data_set){
    double variance = 0.0f;
    vec score_values(pop.size());
    for(unsigned int i=0; i< pop.size(); ++i) {
        score_values(i) = pop[i].get_f1_score(data_set);
        // round after two decimal places
        score_values(i) = (round(score_values(i)) * 100) / 100.0f;
    }
    variance = var(score_values);
    return variance;
}

double Trainer_AIS::compute_score_variance(vector<vec> pop, data_subset data_set){
    double variance = 0.0f;
    vec score_values(pop.size());
    for(unsigned int i=0; i< pop.size(); ++i) {
        score_values(i) = (generate_net(pop[i])).get_f1_score(data_set);
        // round after two decimal places
        score_values(i) = (round(score_values(i)) * 100) / 100.0f;
    }
    variance = var(score_values);
    return variance;
}

double Trainer_AIS::compute_score_stddev(vector<NeuralNet> pop, data_subset data_set){
    double std_dev = 0.0f;
    vec score_values(pop.size());
    for(unsigned int i=0; i< pop.size(); ++i) {
        score_values(i) = pop[i].get_f1_score(data_set);
        // round after two decimal places
        score_values(i) = (round(score_values(i)) * 100) / 100.0f;
    }
    std_dev = stddev(score_values);
    return std_dev;
}

double Trainer_AIS::compute_score_stddev(vector<vec> pop, data_subset data_set){
    double std_dev = 0.0f;
    vec score_values(pop.size());
    for(unsigned int i=0; i< pop.size(); ++i) {
        score_values(i) = (generate_net(pop[i])).get_f1_score(data_set);
        // round after two decimal places
        score_values(i) = (round(score_values(i)) * 100) / 100.0f;
    }
    std_dev = stddev(score_values);
    return std_dev;
}

double Trainer_AIS::compute_score_mean(vector<NeuralNet> pop, data_subset data_set){
    double mean_pop = 0.0f;
    vec score_values(pop.size());
    for(unsigned int i=0; i< pop.size(); ++i) {
        score_values(i) = pop[i].get_f1_score(data_set);
        // round after two decimal places
        score_values(i) = (round(score_values(i)) * 100) / 100.0f;
    }
    mean_pop = mean(score_values);
    return mean_pop;
}

double Trainer_AIS::compute_score_mean(vector<vec> pop, data_subset data_set){
    double mean_pop = 0.0f;
    vec score_values(pop.size());
    for(unsigned int i=0; i< pop.size(); ++i) {
        score_values(i) = (generate_net(pop[i])).get_f1_score(data_set);
        // round after two decimal places
        score_values(i) = (round(score_values(i)) * 100) / 100.0f;
    }

    // if mean =  NaN
    if(mean(score_values) != mean(score_values)){
        // display details error
        cout << "mean = " << mean(score_values) << endl;
        cout << endl;
        cout << "scores = " ;
        for(unsigned int i=0; i<pop.size(); ++i){
            cout << score_values(i) << " ";
        }
        exit(0);
    }
    mean_pop = mean(score_values);
    return mean_pop;
}

double Trainer_AIS::compute_score_median(vector<NeuralNet> pop, data_subset data_set){
    double median_pop = 0.0f;
    vec score_values(pop.size());
    for(unsigned int i=0; i< pop.size(); ++i) {
        score_values(i) = pop[i].get_f1_score(data_set);
        // round after two decimal places
        score_values(i) = (round(score_values(i)) * 100) / 100.0f;
    }
    median_pop = median(score_values);
    return median_pop;
}

double Trainer_AIS::compute_score_median(vector<vec> pop, data_subset data_set){
    double median_pop = 0.0f;
    vec score_values(pop.size());
    for(unsigned int i=0; i< pop.size(); ++i) {
        score_values(i) = (generate_net(pop[i])).get_f1_score(data_set);
        // round after two decimal places
        score_values(i) = (round(score_values(i)) * 100) / 100.0f;
    }
    median_pop = median(score_values);
    return median_pop;
}

vector<NeuralNet> Trainer_AIS::generate_population(unsigned int pop_size, net_topology t, data_subset training_set) {
    vector<NeuralNet> pop(pop_size);
    for(unsigned int i = 0 ; i < pop_size; ++i) {
        NeuralNet tmp_net(t);
        pop[i] = tmp_net;
    }
    // update all fitness values
    evaluate_population(pop, training_set);
    return pop;
}

vector<vec> Trainer_AIS::generate_random_genome_population(unsigned int quantity, NeuralNet largest_net) {
    vector<vec> pop(quantity);
    for(unsigned int i = 0 ; i < quantity ; ++i) {
        // instantiate new random neural net with set topology
        vec tmp_vec(largest_net.get_total_nb_weights() + 4);
        // assign random params values [0, 1]
        tmp_vec.randu();
        // make sure genome represents net of same nature (identical nb layers)
        tmp_vec[0] = largest_net.get_topology().nb_input_units;
        tmp_vec[1] = generate_random_integer_between_range(1, largest_net.get_topology().nb_units_per_hidden_layer);
        tmp_vec[2] = largest_net.get_topology().nb_output_units;
        tmp_vec[3] = largest_net.get_topology().nb_hidden_layers;
        // add it to pop
        pop[i] = tmp_vec;
    }
    return pop;
}

vector<vec> Trainer_AIS::generate_random_topology_genome_population(unsigned int quantity, NeuralNet largest_net) {
    vector<vec> pop(quantity);
    for(unsigned int i = 0 ; i < quantity ; ++i) {
        // instantiate new random neural net with set topology
        vec tmp_vec(largest_net.get_total_nb_weights() + 4);
        // assign random values [0, 1]
        tmp_vec.randu();
        // make sure topology is valid
        tmp_vec[0] = largest_net.get_topology().nb_input_units;
        tmp_vec[1] = generate_random_integer_between_range(1, largest_net.get_topology().nb_units_per_hidden_layer);
        tmp_vec[2] = largest_net.get_topology().nb_output_units;
        tmp_vec[3] = generate_random_integer_between_range(1, largest_net.get_topology().nb_hidden_layers);
        // add it to pop
        pop[i] = tmp_vec;
    }
    return pop;
}

vector<vec> Trainer_AIS::generate_random_topology_genome_population(unsigned int quantity, net_topology min_topo, net_topology max_topo) {
    vector<vec> pop(quantity);
    for(unsigned int i = 0 ; i < quantity ; ++i) {
        // instantiate new random neural net with set topology
        vec tmp_vec(max_topo.get_total_nb_weights() + 4);
        // assign random values [0, 1]
        tmp_vec.randu();
        // make sure topology is valid
        tmp_vec[0] = max_topo.nb_input_units;
        tmp_vec[1] = generate_random_integer_between_range( min_topo.nb_units_per_hidden_layer, max_topo.nb_units_per_hidden_layer);
        tmp_vec[2] = max_topo.nb_output_units;
        tmp_vec[3] = generate_random_integer_between_range(min_topo.nb_hidden_layers, max_topo.nb_hidden_layers);
        // add it to pop
        pop[i] = tmp_vec;
    }
    return pop;
}

void Trainer_AIS::evaluate_population(vector<NeuralNet> &pop, data_subset d) {
    for(unsigned int i = 0 ; i < pop.size() ; ++i) {
        pop[i].get_f1_score(d);
    }
    // sort pop according to score
    sort(pop.begin(), pop.end());
}

double Trainer_AIS::clip(double x, double min, double max) {
    // only clamp if necessary
    if( (x<min)||(x>max) ){
        double c=x;
        c=((max-min)/2)*((exp(x) - exp(-x))/(exp(x) + exp(-x))) + max - (max-min)/2;
        if(c<min) c=min;
        if(c>max) c=max;
        // check for -nan values
        if(c!=c) c = f_rand(min, max);
        return c;
    }else
        return x;
}

// returns a vector of neural networks corresponding to the provided genomes
vector<NeuralNet> Trainer_AIS::convert_population_to_nets(vector<vec> genome_pop) {
    vector<NeuralNet> pop;
    // convert genome pop into neural network pop
    for(unsigned int i=0; i<genome_pop.size(); ++i){
        pop.push_back(generate_net(genome_pop[i]));
    }
    return pop;
}

vector<vec> Trainer_AIS::convert_population_to_genomes(vector<NeuralNet> net_pop, net_topology largest_topology){
    vector<vec> genome_pop;
    for(unsigned int i=0; i<net_pop.size(); ++i) {
        genome_pop.push_back(get_genome(net_pop[i], largest_topology));
    }
    return genome_pop;
}

NeuralNet Trainer_AIS::get_best_model(vector<NeuralNet> pop){
    // sort pop according to score
    sort(pop.begin(), pop.end());
    return pop[0];
}

NeuralNet Trainer_AIS::get_best_model(vector<vec> genome_pop) {
    vector<NeuralNet> pop;
    // convert genome pop into neural network pop
    pop = convert_population_to_nets(genome_pop);
    // sort pop according to score
    sort(pop.begin(), pop.end());
    return pop[0];
}

vec Trainer_AIS::get_genome(NeuralNet net, net_topology max_topo) {
    // instantiate genome with largest possible size
    vec genome(get_genome_length(max_topo));
    // first four elements contain topology
    genome[0] = net.get_topology().nb_input_units;
    genome[1] = net.get_topology().nb_units_per_hidden_layer;
    genome[2] = net.get_topology().nb_output_units;
    genome[3] = net.get_topology().nb_hidden_layers;
    // others contain params
    for(unsigned int i=0; i<net.get_params().size(); ++i) {
        genome[4+i] = net.get_params()[i];
    }
    return genome;
}

NeuralNet Trainer_AIS::generate_net(vec genome){
    NeuralNet net;
    net_topology topology;
    // retrieve topology
    topology.nb_input_units = (unsigned int) genome[0];
    topology.nb_units_per_hidden_layer = (unsigned int) genome[1];
    topology.nb_output_units = (unsigned int) genome[2];
    topology.nb_hidden_layers = (unsigned int) genome[3];
    net.set_topology(topology);
    // retrieve params
    unsigned int size_params = net.get_total_nb_weights();
    vec tmp_params(size_params);
    for(unsigned int i=0; i<size_params; ++i) {
        tmp_params[i] = genome[4+i];
    }
    net.set_params(tmp_params);
    return net;
}

unsigned int Trainer_AIS::get_genome_length(net_topology t){
    unsigned int length = 0;
    NeuralNet dummyNet;
    dummyNet.set_topology(t);
    length = 4 + dummyNet.get_total_nb_weights();
    return length;
}

unsigned int Trainer_AIS::get_population_size(){   return population.size();   }

mat Trainer_AIS::get_population_scores(data_subset d){
    mat scores;
    for(unsigned int i=0; i<population.size(); ++i) {
        mat tmp;
        tmp = { population[i].get_f1_score(d) };
        scores = join_horiz(scores, tmp);
    }
    return scores;
}
*/

