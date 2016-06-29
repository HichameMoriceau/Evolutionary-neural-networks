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
    // default population size: 100
    population = generate_random_population(100,ann);
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

NeuralNet Trainer_DE::evolve_through_iterations(Data_set data_set, net_topology min_topo, net_topology max_topo, unsigned int nb_gens, mat &results_score_evolution, unsigned int index_cross_val_section, unsigned int selected_mutation_scheme, unsigned int current_gen){
    // flag alerting that optimization algorithm has had ~ same results for the past 100 generations
    bool plateau = false;
    // flag alerting that the GA has converged
    bool has_converged = false;

    // using vectors as genotype
    vector<NeuralNet> ensemble = population;

    evaluate_population(population, data_set,results_score_evolution);
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
        // sort from fittest
        sort(population.begin(), population.end());
        // get best model
        if(population[0].get_val_score()>trained_model.get_val_score())
            trained_model = population[0];

        /*
        // compute accuracy
        elective_acc(population, data_set, pop_acc, pop_score);
        // get best ensemble
        if(pop_score >= ensemble_score){
            ensemble = population;
            ensemble_score = pop_score;
            ensemble_acc = pop_acc;
        }
        */
        // optimize model params and topology using training-set
        differential_evolution_topology_evolution(data_set, min_topo, max_topo, selected_mutation_scheme,results_score_evolution,i);

        /*
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
        */
        // if MAX nb of calls to the error function is reached: stop training
        if(nb_err_func_calls>=max_nb_err_func_calls)break;
    }
    return trained_model;
}

void Trainer_DE::differential_evolution_topology_evolution(Data_set data_set, net_topology min_topo, net_topology max_topo, unsigned int selected_mutation_scheme, mat& results_score_evolution, unsigned int gen){
    // Crossover Rate [0,1]
    double CR = 0.5;
    // differential_weight [0,2]
    double F = 1;
    unsigned int MUTATION_SCHEME_BEST = 1;

    for(unsigned int j=0; j<population.size(); ++j){
        // declare index variables
        unsigned int index_x = generate_random_integer_between_range(1, population.size() - 1);
        unsigned int index_a;
        unsigned int index_b;
        unsigned int index_c;

        // select four random different individuals from pop
        do{
            index_a = generate_random_integer_between_range(1, population.size() - 1);
        }while(index_a == index_x);

        do{
            index_b = generate_random_integer_between_range(1, population.size() - 1);
        }while(index_b == index_a || index_b == index_x);

        do{
            index_c = generate_random_integer_between_range(1, population.size() - 1);
        }while(index_c == index_b || index_c == index_a || index_c == index_x);

        // store corresponding individual in pop
        vec original_model  = population[index_x].get_params();
        vec candidate_model = population[index_x].get_params();
        vec indiv_a = population[index_a].get_params();
        vec indiv_b = population[index_b].get_params();
        vec indiv_c = population[index_c].get_params();

        // if user selected a DE/BEST/1 mutation scheme: use best indiv as first indiv
        if(selected_mutation_scheme == MUTATION_SCHEME_BEST)
            indiv_a = population[0].get_params();

        double score_best_model         = population[0].get_train_score();
        double score_second_best_model  = population[1].get_train_score();

        // if the first and second best have identical fitness
        if((score_best_model==score_second_best_model) && score_best_model!=0){
            // force a crossover between the two
            indiv_a = population[0].get_params();
            indiv_b = population[1].get_params();
            mutative_crossover(1, 1, min_topo, max_topo, original_model,candidate_model, indiv_a, indiv_b, indiv_c);
        }

        // traditional random crossover
        mutative_crossover(CR, F, min_topo, max_topo, original_model,candidate_model, indiv_a, indiv_b, indiv_c);

        NeuralNet candidate_net=to_NeuralNet(candidate_model);
        candidate_net.get_fitness_metrics(data_set);
        nb_err_func_calls++;

        // if candidate outperforms original: replace original by candidate
        if(candidate_net.get_train_score()>population[index_x].get_train_score())
            population[index_x] = candidate_net;

        // format result line
        mat line=generate_metric_line(population, gen);
        // append result line to result matrix
#pragma omp critical
        results_score_evolution = join_vert(results_score_evolution, line);

#ifndef NO_SCREEN_OUT
        cout<<fixed
            <<setprecision(2)
            <<"NB.err.func.calls="<<line[0]<<"\t"
            <<"gen="<<line[1]<<"\t"
            <<"train.mse="<<line[4]<<"\t"
            <<"val.mse="<<line[10]<<"\t"
            <<"test.mse="<<line[7]<<"\t"
            <<"pop.fit.mean="<<line[12]<<"\t"
            <<"NB.hid.units="<<line[14]<<"\t"
            <<"NB.hid.layers="<<line[15]<<"\t"
            <<endl;
#endif
        // if MAX nb of calls to the error function is reached: stop training
        if(nb_err_func_calls>=max_nb_err_func_calls)break;
    }
}

void Trainer_DE::differential_evolution(vector<NeuralNet> &pop, data_subset train_set){
    // total nb weights
    unsigned int nb_element_vectorized_Theta = pop[0].get_topology().get_total_nb_weights();
    unsigned int problem_dimensionality = nb_element_vectorized_Theta;
    // Crossover Rate [0,1]
    double CR = 0.5;
    // differential_weight [0,2]
    double F = 1;

    for(unsigned int j=0; j<pop.size()-1; ++j) {
        // select four random but different individuals from (pop)
        // declare index variables
        unsigned int index_x=generate_random_integer_between_range(1, pop.size() - 1);
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
        NeuralNet original_model =pop[index_x];
        NeuralNet candidate_model=pop[index_x];
        NeuralNet indiv_a=pop[index_a];
        NeuralNet indiv_b=pop[index_b];
        NeuralNet indiv_c=pop[index_c];

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
        candidate_score = candidate_model.get_train_score(train_set);
        original_score  = original_model.get_train_score(train_set);

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
    double prediction_acc = 0.0f;
    double score = 0.0f;
    double MSE   = 0.0f;
    double pop_score_variance = 0.0f;
    double pop_score_stddev = 0.0f;
    mat new_line;

    // instantiate a random net with identical topology
    NeuralNet trained_model(net.get_topology());
    population = generate_population(population.size(), net.get_topology());

    for(unsigned int i=0; i<nb_epochs; ++i) {
        // optimize model params and topology using training-set
        differential_evolution(population, data_set.train_set);
        // sort from fittest
        sort(population.begin(), population.end());
        trained_model = population[0];

        // record model performances on new data
        prediction_acc = trained_model.get_train_acc(data_set.train_set);
        score               = trained_model.get_train_score(data_set.train_set);
        MSE                 = trained_model.get_mse(data_set.train_set);
        pop_score_variance  = compute_score_variance(population);
        pop_score_stddev    = compute_score_mean(population);

        // record results (performances and topology description)
        unsigned int inputs = net.get_topology().nb_input_units;
        unsigned int hidden_units = net.get_topology().nb_units_per_hidden_layer;
        unsigned int outputs = net.get_topology().nb_output_units;
        unsigned int nb_hidden_layers = net.get_topology().nb_hidden_layers;

        new_line << i
                 << cost
                 << prediction_acc
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
        cout << "epoch=" << i << "\tscore=" << score << "\taccuracy=" << prediction_acc << endl;
    }

    // return trained model
    net = trained_model;
}

void Trainer_DE::mutative_crossover(double CR, double F, net_topology min_topo, net_topology max_topo, vec original_model, vec &candidate_model, vec indiv_a, vec indiv_b, vec indiv_c){
    unsigned int problem_dimensionality=max_topo.get_genome_length();
    unsigned int genome_length=problem_dimensionality;
    // clone original to candidate but let the genome be large enough to grow up to max_topo
    vec candidate=randu<vec>(problem_dimensionality);
    for(unsigned int i=0;i<original_model.n_elem;i++)
        candidate(i)=original_model(i);
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
                candidate[k]  = mutation_scheme_DE_rand_1(F, indiv_a[k], indiv_b[k], indiv_c[k]);
                // make sure NB hid. units doesn't break contract
                if(k==1){
                    do{
                        candidate[1] = (((int) abs( mutation_scheme_DE_rand_1(F, indiv_a[1], indiv_b[1], indiv_c[1]))) % max_topo.nb_units_per_hidden_layer) + 1;
                    }while(!(candidate[1]>=min_topo.nb_units_per_hidden_layer && candidate[1]<=max_topo.nb_units_per_hidden_layer));
                }else if(k==3){
                    // make sure NB hid. lay. doesn't break contract
                    do{
                        candidate[3] = (((int) abs(mutation_scheme_DE_rand_1(F, indiv_a[3], indiv_b[3], indiv_c[3])) ) % max_topo.nb_hidden_layers) + 1;
                    }while(!(candidate[3]>=min_topo.nb_hidden_layers && candidate[3]<=max_topo.nb_hidden_layers));
                }
            }

            // once topology is chosen
            if(k==4){
                net_topology candidate_topology;
                candidate_topology.nb_input_units = (unsigned int) candidate[0];
                candidate_topology.nb_units_per_hidden_layer = (unsigned int) candidate[1];
                candidate_topology.nb_output_units = (unsigned int) candidate[2];
                candidate_topology.nb_hidden_layers = (unsigned int) candidate[3];
                //genome_length = candidate_topology.get_total_nb_weights();
            }
        }
    }
    candidate_model=candidate;
}

double Trainer_DE::mutation_scheme_DE_rand_1(double F, double x_rand_1, double x_rand_2, double x_rand_3){
    return x_rand_1 + F * (x_rand_2 - x_rand_3);
}
