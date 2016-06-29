#include "trainer_ais.h"

Trainer_AIS::Trainer_AIS(){
    // default nb generations:
    nb_epochs=1000;
    nb_err_func_calls=0;
    // default variance value (convergence treshold for GA stopping criteria)
    epsilon=1.0f;
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

void Trainer_AIS::train(Data_set data_set, NeuralNet &net){
    mat results_score_evolution;
    train(data_set, net, results_score_evolution);
}

void Trainer_AIS::train(Data_set data_set, NeuralNet &net, mat &results_score_evolution) {
    net=train_topology_plus_weights(data_set, net.get_topology(), results_score_evolution, -1);
}

NeuralNet Trainer_AIS::evolve_through_iterations(Data_set data_set, net_topology min_topo, net_topology max_topo, unsigned int nb_gens, mat &results_score_evolution, unsigned int index_cross_val_section, unsigned int selected_mutation_scheme, unsigned int current_gen) {
    // flag alerting that optimization algorithm has had ~ same results for the past 100 generations
    bool plateau=false;
    // flag alerting that the GA has converged
    bool has_converged=false;

    vector<NeuralNet> ensemble=population;
    evaluate_population(population, data_set,results_score_evolution);
    NeuralNet trained_model=population[0];

    /**
     *  ALGORITHM:    Clonal Selection
     *
     *
     *  TERMINATION CRITERIA:
     *      If all generations were achieved OR if the GA has already converged
    */
    for(unsigned int i=0;(i<nb_gens)&&(!has_converged);i++) {
        // sort from fittest
        sort(population.begin(), population.end());
        // get best model
        if(population[0].get_val_score()>=trained_model.get_val_score())
            trained_model=population[0];

        /*
        // compute accuracy
        elective_acc(ensemble, data_set, pop_acc, pop_score);
        // get best ensemble
        if(pop_score>=ensemble_score){
            ensemble=population;
            ensemble_score=pop_score;
            ensemble_acc=pop_acc;
        }
        */

        // optimize model params and topology using training-set
        clonal_selection_topology_evolution(data_set, min_topo, max_topo, selected_mutation_scheme,results_score_evolution,i);

        /*
        // checking for convergence (termination criterion)
        // if 33% of the total_nb_generations have been executed on CV fold
        if(i>(nb_epochs/3)) {
            // if current best score is similar to best score of 100 generations before
            if(score < results_score_evolution(results_score_evolution.n_rows-(nb_epochs/10),3)+1)
                plateau=true;
            else
                plateau=false;
            has_converged=(pop_score_variance<epsilon) && (plateau);
        }else{
            // otherwise always force training on first 10% of total generations
            has_converged=false;
        }
        */
        // if MAX nb of calls to the error function is reached: stop training
        if(nb_err_func_calls>=max_nb_err_func_calls)break;
    }
    return trained_model;
}

void Trainer_AIS::clonal_selection_topology_evolution(Data_set data_set, net_topology min_topo, net_topology max_topo, unsigned int selected_mutation_scheme,mat& results_score_evolution, unsigned int gen){
    unsigned int pop_size=population.size();
    // Number of individuals retained
    unsigned int selection_size=pop_size*100/100;
    // Number of random cells incorporated in the population for every generation
    unsigned int nb_rand_cells=(unsigned int)pop_size*15/100;
    // Clones scaling factors
    double clone_rate=0.01;
    double clone_scale=0.01;

    // -- Differential Evolution settings (for mutation operation only) --
    // Crossover Rate [0,1]
    double CR=0.5;
    // differential_weight [0,2]
    double F=1;
    // -- --

    for(unsigned int g=0; g<pop_size; ++g) {
        // instantiate subpopulations
        vector<NeuralNet>selected_indivs=select(selection_size, population);
        vector<vector<NeuralNet>>pop_clones;
        unsigned int nb_clones_array[selection_size];

        // generate clones
        for(unsigned int j=0; j<selection_size*clone_scale; j++) {
            unsigned int nb_clones=compute_nb_clones(clone_rate, selection_size,j+1);
            nb_clones_array[j]=nb_clones;
            pop_clones.push_back(generate_clones(nb_clones, selected_indivs[j]));
        }

        // hyper-mutate (using a DE/RAND/1 mutative-crossover operation)
        for(unsigned int i=0; i<pop_clones.size(); i++) {
            for(unsigned int j=0; j<pop_clones[i].size(); j++) {
                // declare index variables
                unsigned int index_x=generate_random_integer_between_range(1, pop_size - 1);
                unsigned int index_a;
                unsigned int index_b;
                unsigned int index_c;

                // select four random but different individuals from (pop)
                // making sure that no two identical indexes are generated
                do{
                    index_a=generate_random_integer_between_range(1,pop_size-1);
                }while(index_a==index_x);

                do{
                    index_b=generate_random_integer_between_range(1,pop_size-1);
                }while(index_b==index_a || index_b==index_x);

                do{
                    index_c=generate_random_integer_between_range(1,pop_size-1);
                }while(index_c==index_b || index_c==index_a || index_c==index_x);

                // store corresponding individual in pop
                vec indiv_a=population[index_a].get_params();
                vec indiv_b=population[index_b].get_params();
                vec indiv_c=population[index_c].get_params();
                vec original_model =population[index_x].get_params();
                vec candidate_model=population[index_x].get_params();

                unsigned int up_to = population[index_x].get_topology().get_genome_length();
                mutative_crossover(CR, F, up_to, min_topo, max_topo, original_model,candidate_model,indiv_a, indiv_b, indiv_c);

                NeuralNet candidate_net=to_NeuralNet(candidate_model);
                // compute offspring's performances
                candidate_net.get_fitness_metrics(data_set);
                nb_err_func_calls++;
                // update clone if interesting
                if(candidate_net.get_train_score()>pop_clones[i][j].get_train_score())
                    pop_clones[i][j]=candidate_net;
                // format result line
                mat line=generate_metric_line(population, gen);
                // append result line to result matrix
                results_score_evolution=join_vert(results_score_evolution, line);

#ifndef NO_SCREEN_OUT
                cout << fixed
                     << setprecision(2)
                     <<"NB.err.func.calls="<<line[0]<<"\t"
                     <<"gen="<<line[1]<<"\t"
                     <<"train.mse="<<line[4]<<"\t"
                     <<"val.mse="<<line[10]<<"\t"
                     <<"test.mse="<<line[7]<<"\t"
                     <<"pop.fit.mean="<<line[12]<<"\t"
                     <<"NB.hid.units="<<line[14]<<"\t"
                     <<"NB.hid.layers="<<line[15]<<"\t"
                     << endl;
#endif

                // if MAX nb of calls to the error function is reached: stop training
                if(nb_err_func_calls>=max_nb_err_func_calls)break;
            }
            // if MAX nb of calls to the error function is reached: stop training
            if(nb_err_func_calls>=max_nb_err_func_calls)break;
        }

        // put all solutions in same group
        vector<NeuralNet> all_clones=add_all(pop_clones, nb_clones_array);
        all_clones.insert(all_clones.end(), population.begin(), population.end());
        // select n best solutions
        population=select(pop_size, all_clones);
        // maintain diversity by forcing random indivs into population
        vector<NeuralNet> rand_indivs=generate_random_topology_population(nb_rand_cells, min_topo, max_topo);
        for(unsigned int k=0;k<nb_rand_cells;k++)
            population[pop_size-nb_rand_cells+k]=rand_indivs[k];

        // if MAX nb of calls to the error function is reached: stop training
        if(nb_err_func_calls>=max_nb_err_func_calls)break;
    }
}

vector<NeuralNet> Trainer_AIS::select(unsigned int n, vector<NeuralNet> pop){
    if(n>pop.size()) return pop;
    sort(pop.begin(),pop.end());
    vector<NeuralNet> s_pop;
    // select best n indivs
    for(unsigned int i=0; i<n; i++)
        s_pop.push_back(pop[i]);
    return s_pop;
}

vector<NeuralNet> Trainer_AIS::generate_clones(unsigned int nb_clones, NeuralNet indiv){
    vector<NeuralNet> cloned_pop;
    for(unsigned int i=0; i<nb_clones; i++)
        cloned_pop.push_back(indiv);
    return cloned_pop;
}

float simple_clip(float x, float lower, float upper) {
  return std::max(lower, std::min(x, upper));
}

unsigned int Trainer_AIS::compute_nb_clones(double beta, int pop_size, int index){
    return ceil(simple_clip(beta*double(pop_size)/double(index), 1, pop_size/10));
}

vector<NeuralNet> Trainer_AIS::add_all(vector<vector<NeuralNet>> all_populations, unsigned int* nb_clones_array){
    vector<NeuralNet> pop;
    for(unsigned int i=0; i<all_populations.size(); i++)
        for(unsigned int j=0; j<nb_clones_array[i]; j++)
            pop.push_back(all_populations[i][j]);
    return pop;
}

void Trainer_AIS::mutative_crossover(double CR, double F, unsigned int genome_length, net_topology min_topo, net_topology max_topo, vec original_model, vec &candidate_model, vec indiv_a, vec indiv_b, vec indiv_c){
    unsigned int problem_dimensionality=max_topo.get_genome_length();
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

double Trainer_AIS::mutation_scheme_DE_rand_1(double F, double x_rand_1, double x_rand_2, double x_rand_3){
    return x_rand_1 + F * (x_rand_2 - x_rand_3);
}
