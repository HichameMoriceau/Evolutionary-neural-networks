#include "trainer_pso.h"

Trainer_PSO::Trainer_PSO(){
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

void Trainer_PSO::train(Data_set data_set, NeuralNet &net){
    mat results_score_evolution;
    train(data_set, net, results_score_evolution);
}

void Trainer_PSO::train(Data_set data_set, NeuralNet &net, mat &results_score_evolution){
    net = train_topology_plus_weights(data_set, net.get_topology(), results_score_evolution, -1);
}

NeuralNet Trainer_PSO::evolve_through_iterations(Data_set data_set, net_topology min_topo, net_topology max_topo, unsigned int nb_gens, mat &results_score_evolution, unsigned int index_cross_validation_section, unsigned int selected_mutation_scheme, unsigned int current_gen) {

    // flag alerting learning stagnation
    bool plateau = false;
    // flag alerting that the GA has converged
    bool has_converged = false;
    // initialize particles
    vector<NeuralNet> ensemble = population;
    // initialize velocity of each particle to random [-1, 1]
    unsigned int nb_params = max_topo.get_total_nb_weights()+4;
    // personal best particle
    vector<NeuralNet> pBests = generate_population(population.size(), max_topo, data_set);

    double pop_score_variance=100;

    evaluate_population(population, data_set);
    NeuralNet trained_model=population[0];

    vector<vec> velocities;
    // init velocities with small random values
    for(unsigned int i=0; i<population.size(); i++){
        vec new_vector(nb_params);
        new_vector = ones(nb_params, 1);
        velocities.push_back(new_vector);
        for(unsigned int j=0; j<nb_params; j++)
            velocities[i][j] = f_rand(0, 1);
        // protect nb I/Os from being altered
        velocities[i][0] = 0;
        velocities[i][2] = 0;
    }

    /**
     *  ALGORITHM:    Particle Swarm Optimization
     *
     *
     *  TERMINATION CRITERIA:
     *      If all generations were achieved OR if the algorithm has converged (variance test)
    */
    for(unsigned int i=0; ((i<nb_gens) && (!has_converged)); ++i) {
        // sort from fittest
        sort(population.begin(), population.end());
        // get best model
        if(population[0].get_f1_score()>=trained_model.get_f1_score())
            trained_model = population[0];

        /*
        // compute accuracy
        elective_accuracy(population, data_set, pop_accuracy, pop_score);
        // get best ensemble
        if(pop_score > ensemble_score){
            ensemble = population;
            ensemble_score = pop_score;
            ensemble_accuracy = pop_accuracy;
        }
        */

        // optimize model params and topology using training-set
        PSO_topology_evolution(velocities, data_set, max_topo, pBests, trained_model, pop_score_variance,results_score_evolution,i);


        /*
        // checking for convergence (termination criterion)
        // if 33% of the total_nb_generations have been executed
        if(i>(nb_epochs/3)) {
            // if current best score is similar to best score of 100 generations before
            if(score < results_score_evolution(results_score_evolution.n_rows-(nb_epochs/10),3)+1)
                plateau = true;
            else
                plateau = false;
            has_converged = (pop_score_variance<epsilon) && (plateau);
        }else// otherwise always force training on first 10% of total generations
            has_converged = false;
        */
    }
    return trained_model;
}

void Trainer_PSO::PSO_topology_evolution(vector<vec> &velocities, Data_set data_set, net_topology max_topo, vector<NeuralNet> &pBest, NeuralNet gBest, double pop_score_variance, mat& results_score_evolution, unsigned int gen){
    unsigned int genome_size = max_topo.get_genome_length();
    // ** PSO settings **
    // velocity weight
    double w = 0.729;
    // importance of personal best (cognitive weight)
    double c1 = 1.494;
    // importance of global best (social weight)
    double c2 = 1.494;
    // ** **

    // learning metrics variables
    double prediction_accuracy = 0;
    double score=0;
    double MSE=0;
    double pop_score_stddev=0;
    double pop_score_mean=0;
    double pop_score_median=0;
    double ensemble_accuracy=0;
    double ensemble_score=0;
    double pop_score=0;
    double pop_accuracy=0;
    mat new_line;

    // update pBest of each particle
    for(unsigned int p=0; p<population.size(); p++){
        // if particle is closer to target than pBest : set particle as <pBest>
        if(population[p].get_f1_score() > pBest[p].get_f1_score())
            pBest[p] = population[p];
    }

    // update gBest
    for(unsigned int p=0; p<population.size(); p++){
        // if particle is closer to target than gBest : set particle as <gBest>
        if(pBest[p].get_f1_score() > gBest.get_f1_score())
            gBest = population[p];
    }

    // for each particle
    for(unsigned int p=0; p<population.size(); p++) {
        for(unsigned int i=0; i<genome_size; i++){
            double r1 = f_rand(0,1);
            double r2 = f_rand(0,1);
            // calculate velocity
            velocities[p][i] = w*velocities[p][i]
                    + c1*r1 * (pBest[p].get_params()[i] - population[p].get_params()[i])
                    + c2*r2 * (gBest.get_params()[i]    - population[p].get_params()[i]);
            velocities[p][i] = clip(velocities[p][i], -5, 5);
        }

        vec particle=population[p].get_params();
        // update particle data
        for(unsigned int i=0; i<particle.n_elem; i++){
            switch(i){
            case 0:
                // protect NB INPUTS from being altered
                particle[0] = data_set.training_set.X.n_cols;
                break;
            case 1:
                // make sure NB HIDDEN UNITS PER LAYER doesn't exceed genome size
                particle[1] = round(clip(particle[i] + velocities[p][i], 2, max_topo.nb_units_per_hidden_layer));
                break;
            case 2:
                // protect NB OUTPUTS from being altered
                particle[2] = max_topo.nb_output_units;
                break;
            case 3:
                // make sure NB HIDDEN LAYERS doesn't exceed genome size
                particle[3] = round(clip(particle[i] + velocities[p][i], 1, max_topo.nb_hidden_layers));
                break;
            default:
                particle[i] = particle[i] + velocities[p][i];
                break;
            }
        }

        NeuralNet candidate=to_NeuralNet(particle);
        candidate.get_fitness_metrics(data_set);
        nb_err_func_calls++;
        if(candidate.get_f1_score()>=population[p].get_f1_score())
            population[p] = candidate;

        // print out learning metrics
        // record model performances on new data
        prediction_accuracy =   population[0].get_accuracy();
        score               =   population[0].get_f1_score();
        MSE                 =   population[0].get_MSE();
        double validation_accuracy=population[0].get_validation_acc();
        double validation_score=population[0].get_validation_score();
        // compute stats
        pop_score_variance  =   compute_score_variance(population);
        pop_score_stddev    =   compute_score_stddev(population);
        pop_score_mean      =   compute_score_mean(population);
        pop_score_median    =   compute_score_median(population);
        // record results (performances and topology description)
        unsigned int inputs             =   population[0].get_topology().nb_input_units;
        unsigned int hidden_units       =   population[0].get_topology().nb_units_per_hidden_layer;
        unsigned int outputs            =   population[0].get_topology().nb_output_units;
        unsigned int nb_hidden_layers   =   population[0].get_topology().nb_hidden_layers;

        // format result line
        new_line << nb_err_func_calls //i+1
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
                 << -1

                 << ensemble_accuracy
                 << ensemble_score
                 << validation_accuracy
                 << validation_score
                 << gen

                 << endr;

        // append result line to result matrix
        results_score_evolution = join_vert(results_score_evolution, new_line);

        cout << fixed
             << setprecision(2)
             << "Gen="            << nb_err_func_calls //i+1
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
             << "  Gen="<<gen
             << endl;
    }
}

double Trainer_PSO::clip(double x, double min, double max) {
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
