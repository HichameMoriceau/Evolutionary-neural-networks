#include "trainer_de.h"

Trainer_DE::Trainer_DE(){
    // default nb generations:
    nb_epochs = 1000;
    nb_err_func_calls=0;
    // default variance value (convergence treshold for GA stopping criteria)
    epsilon = 1.0f;
    // default topologies for individuals
    net_topology t;
    t.nb_input_units = 1;
    t.nb_units_per_hidden_layer = 5;
    t.nb_output_units = 2;
    t.nb_hidden_layers = 1;
    NeuralNet ann(t);
    initialize_random_population(50, t);

    // default population size: 100
    population = convert_population_to_nets(generate_random_genome_population(100,ann));
}

void Trainer_DE::train(Data_set data_set, NeuralNet &net){
    mat results_score_evolution;
    train(data_set, net, results_score_evolution);
}

void Trainer_DE::train(Data_set data_set, NeuralNet &net, mat &results_score_evolution) {
    unsigned int MUTATION_SCHEME_RAND = 0;
    unsigned int MUTATION_SCHEME_BEST = 1;
    net = train_topology_plus_weights(data_set, net.get_topology(), results_score_evolution, MUTATION_SCHEME_RAND);
}

NeuralNet Trainer_DE::evolve_through_iterations(Data_set data_set, net_topology min_topo, net_topology max_topo, unsigned int nb_gens, mat &results_score_evolution, unsigned int index_cross_validation_section, unsigned int selected_mutation_scheme, unsigned int current_gen){
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
    vector<genome> genome_population = convert_population_to_genomes(population, max_topo);
    vector<NeuralNet> ensemble = population;

    evaluate_population(population, data_set);
    NeuralNet trained_model=population[0];

    /**
     *  ALGORITHM:    Differential Evolution
     *
     *  Algorithm definition      : https://en.wikipedia.org/wiki/Differential_evolution#Algorithm
     *  Existing mutation schemes : http://www.sciencedirect.com/science/article/pii/S0926985113001845
     *
     *  TERMINATION CRITERIA:
     *      If all generations were achieved OR if the GA has already converged
    */
    for(unsigned int i=0; ((i<nb_gens) && (!has_converged)); i++) {
        // update individuals
        population = convert_population_to_nets(genome_population);
        // sort from fittest
        sort(population.begin(), population.end());

        // get best model
        if(population[0].get_validation_score() >= trained_model.get_validation_score())
            trained_model = population[0];
        // compute accuracy
        elective_accuracy(population, data_set, pop_accuracy, pop_score);
        // get best ensemble
        if(pop_score >= ensemble_score){
            ensemble = population;
            ensemble_score = pop_score;
            ensemble_accuracy = pop_accuracy;
        }

        genome_population = convert_population_to_genomes(population, max_topo);
        // optimize model params and topology using training-set
        differential_evolution_topology_evolution(genome_population, data_set, min_topo, max_topo, selected_mutation_scheme);

        // record model performances on new data
        prediction_accuracy =   trained_model.get_accuracy();
        score               =   trained_model.get_f1_score();
        MSE                 =   trained_model.get_MSE();
        double validation_accuracy=trained_model.get_validation_acc();
        double validation_score=trained_model.get_validation_score();
        // compute stats
        pop_score_variance  =   compute_score_variance(genome_population, data_set.training_set);
        pop_score_stddev    =   compute_score_stddev(genome_population, data_set.training_set);
        pop_score_mean      =   compute_score_mean(genome_population, data_set.training_set);
        pop_score_median    =   compute_score_median(genome_population, data_set.training_set);
        // record results (performances and topology description)
        unsigned int inputs             =   trained_model.get_topology().nb_input_units;
        unsigned int hidden_units       =   trained_model.get_topology().nb_units_per_hidden_layer;
        unsigned int outputs            =   trained_model.get_topology().nb_output_units;
        unsigned int nb_hidden_layers   =   trained_model.get_topology().nb_hidden_layers;
        // format result line
        new_line << i+1 // i + nb_epochs * index_cross_validation_section
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
                 << validation_accuracy
                 << validation_score
                 << nb_err_func_calls

                 << endr;

        // append result line to result matrix
        results_score_evolution = join_vert(results_score_evolution, new_line);

        cout << fixed
             << setprecision(2)
             << "Gen="            << i+1 // i + nb_epochs * index_cross_validation_section
             << "\ttrain.score="          << score
             << "  train.MSE="            << MSE
             << "  train.acc="            << prediction_accuracy
             << "  score.mean=" << pop_score_mean
             << "  score.var=" << pop_score_variance
             << "\tNB.hid.lay="     << nb_hidden_layers
             << "  NB.hid.units="   << hidden_units
             << "\tval.score=" << validation_score
             << " val.acc=" << validation_accuracy
             //<< "\tens.acc=" << ensemble_accuracy
             //<< "  ens.score=" << ensemble_score
             << "  NB err.func.calls="<<nb_err_func_calls
             << endl;

        // checking for convergence (termination criterion)
        // if 33% of the total_nb_generations have been executed on CV fold
        if(i>(nb_gens/3)) {
            // if current best score is similar to best score of 100 generations before
            if(score < results_score_evolution(results_score_evolution.n_rows-(nb_gens/10),3)+1)
                plateau = true;
            else
                plateau = false;
            has_converged = (pop_score_variance<epsilon) && (plateau);
        }else{
            // otherwise always force training on first 10% of total generations
            has_converged = false;
        }
        // added here
    }
    return trained_model;
}

void Trainer_DE::differential_evolution_topology_evolution(vector<genome> &pop, Data_set data_set, net_topology min_topo, net_topology max_topo, unsigned int selected_mutation_scheme){
    NeuralNet dummyNet(max_topo);
    unsigned int nb_element_vectorized_Theta = dummyNet.get_total_nb_weights() + 4;
    unsigned int genome_length = get_genome_length(max_topo);
    // total nb of variables
    unsigned int problem_dimensionality = nb_element_vectorized_Theta;
    // Crossover Rate [0,1]
    double CR = 0.5;
    // differential_weight [0,2]
    double F = 1;

    unsigned int MUTATION_SCHEME_RAND = 0;
    unsigned int MUTATION_SCHEME_BEST = 1;
    for(unsigned int j=0; j<pop.size(); ++j) {
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
        genome original_model  = pop[index_x];
        genome candidate_model = pop[index_x];
        vec indiv_a = pop[index_a].genotype;
        vec indiv_b = pop[index_b].genotype;
        vec indiv_c = pop[index_c].genotype;

        // if user selected a DE/BEST/1 mutation scheme: use best indiv as first indiv
        if(selected_mutation_scheme == MUTATION_SCHEME_BEST)
            indiv_a = pop[0].genotype;

        net_topology candidate_topology;
        candidate_topology.nb_input_units = (unsigned int) candidate_model.genotype[0];
        candidate_topology.nb_units_per_hidden_layer = (unsigned int) candidate_model.genotype[1];
        candidate_topology.nb_output_units = (unsigned int) candidate_model.genotype[2];
        candidate_topology.nb_hidden_layers = (unsigned int) candidate_model.genotype[3];
        genome_length = get_genome_length(candidate_topology);

        double score_best_model         = pop[0].fitness;
        double score_second_best_model  = pop[1].fitness;

        // if the first and second best have identical fitness
        if((score_best_model==score_second_best_model) && score_best_model!=0){
            // force a crossover between the two
            indiv_a = pop[0].genotype;
            indiv_b = pop[1].genotype;
            mutative_crossover(problem_dimensionality, 1, 1, genome_length, min_topo, max_topo, original_model.genotype, candidate_model.genotype, indiv_a, indiv_b, indiv_c);
        }

        // traditional random crossover
        mutative_crossover(problem_dimensionality, CR, F, genome_length, min_topo, max_topo, original_model.genotype, candidate_model.genotype, indiv_a, indiv_b, indiv_c);
        NeuralNet candidate_net=generate_net(candidate_model);
        candidate_net.get_fitness_metrics(data_set);
        candidate_model=get_genome(candidate_net,max_topo);
        nb_err_func_calls++;
        // if candidate outperforms original: replace original by candidate
        if(candidate_model.fitness > original_model.fitness)
            pop[index_x] = candidate_model;
    }
    // update population
    population = convert_population_to_nets(pop);
}

void Trainer_DE::differential_evolution(vector<NeuralNet> &pop, data_subset training_set){
    // total nb weights
    unsigned int nb_element_vectorized_Theta = pop[0].get_total_nb_weights();
    unsigned int problem_dimensionality = nb_element_vectorized_Theta;
    // Crossover Rate [0,1]
    double CR = 0.5;
    // differential_weight [0,2]
    double F = 1;

    for(unsigned int j=0; j<pop.size()-1; ++j) {
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

        // store corresponding individual in (*pop)
        NeuralNet original_model  = pop[index_x];
        NeuralNet candidate_model = pop[index_x];
        NeuralNet indiv_a = pop[index_a];
        NeuralNet indiv_b = pop[index_b];
        NeuralNet indiv_c = pop[index_c];

        // pick random index
        unsigned int R    = generate_random_integer_between_range(1, problem_dimensionality);
        // only used to generate random 0 or 1
        unsigned int rand = generate_random_integer_between_range(1, 50);

        // crossover + mutation
        for(unsigned int k=0; k< nb_element_vectorized_Theta; ++k) {
            if( floor(generate_random_integer_between_range(1,problem_dimensionality) == R | (rand%2) < CR)) {
                vec tmp_params = original_model.get_params();
                // using a DE/rand/1 scheme
                tmp_params[k] = indiv_a.get_params()[k] + F * ( indiv_b.get_params()[k] - indiv_c.get_params()[k]);
                candidate_model.set_params(tmp_params);
            }
        }

        // compute performances
        double candidate_score = 0.0f;
        double original_score  = 0.0f;
        candidate_score = candidate_model.get_f1_score(training_set);
        original_score  = original_model.get_f1_score(training_set);

        bool candidate_is_better_than_original = candidate_score > original_score;

        // selection
        if(candidate_is_better_than_original) {
            // replace original by candidate
            pop[index_x] = candidate_model;
        }
    }
}

void Trainer_DE::train_weights(Data_set data_set, NeuralNet &net, unsigned int nb_epochs, mat &results_score_evolution){
    double cost = 0.0f;
    double prediction_accuracy = 0.0f;
    double score = 0.0f;
    double MSE   = 0.0f;
    double pop_score_variance = 0.0f;
    double pop_score_stddev = 0.0f;
    mat new_line;

    // instantiate a random net with identical topology
    NeuralNet trained_model(net.get_topology());

    population = generate_population(population.size(), net.get_topology(), data_set);

    for(unsigned int i=0; i<nb_epochs; ++i) {
        // optimize model params and topology using training-set
        differential_evolution(population, data_set.training_set);
        trained_model = get_best_model(population);

        // record model performances on new data
        prediction_accuracy = trained_model.get_accuracy(data_set.training_set);
        score               = trained_model.get_f1_score(data_set.training_set);
        MSE                 = trained_model.get_MSE(data_set.training_set);
        pop_score_variance  = compute_score_variance(population, data_set.training_set);
        pop_score_stddev    = compute_score_mean(population, data_set.training_set);

        // record results (performances and topology description)
        unsigned int inputs = net.get_topology().nb_input_units;
        unsigned int hidden_units = net.get_topology().nb_units_per_hidden_layer;
        unsigned int outputs = net.get_topology().nb_output_units;
        unsigned int nb_hidden_layers = net.get_topology().nb_hidden_layers;

        new_line << i
                 << cost
                 << prediction_accuracy
                 << score
                 << MSE
                 << pop_score_variance
                 << pop_score_stddev
                 << population.size()
                 << inputs
                 << hidden_units
                 << outputs
                 << nb_hidden_layers
                 << false // pop *not* heterogeneous (only nets of identical topology)
                 << endr;

        results_score_evolution = join_vert(results_score_evolution, new_line);
        cout << "epoch=" << i << "\tscore=" << score << "\taccuracy=" << prediction_accuracy << endl;
    }

    // return trained model
    net = trained_model;
}
void Trainer_DE::mutative_crossover(unsigned int problem_dimensionality, double CR, double F, unsigned int genome_length, net_topology min_topo, net_topology max_topo, vec original_model, vec &candidate_model, vec indiv_a, vec indiv_b, vec indiv_c){
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

double Trainer_DE::mutation_scheme_DE_rand_1(double F, double x_rand_1, double x_rand_2, double x_rand_3){
    return x_rand_1 + F * (x_rand_2 - x_rand_3);
}
