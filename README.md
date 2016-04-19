# Evolutionary Neural Networks

Artificial Neural Networks(ANNs) are a very popular technique for supervised learning challenges. This project focuses on the automation of the search for the most adequate neural net architecture and weights for any given use-case. 

Given a maximum size neural network topology (architecture) the population based optimization algorithm(s) autonomously find an appropriate topology and set of weights. The algorithm was tested on 3 data sets: Breast Cancer Malignant (Diagnostic), Breast Cancer Recurrence and Haberman's survival test.

This work contains implementations of the following techniques:
 - Vectorized Neural Network of any topology (using Linear Algebra)
 - Differential Evolution
 - Particle Swarm Optimization
 - N-Fold Cross Validation Method

The program leverages the [Armadillo C++](http://arma.sourceforge.net/) Linear Algebra library and each replicate of the experiment is ran as an [OpenMP](http://openmp.org/wp/) thread.

## Training Algorithms

For anyone interested in implementing a highly reliable and versatile optimization algorithm I would recommend taking a peek at Differential Evolution first since it is simple and powerful.

Optimization algorithm A: [Differential Evolution](https://en.wikipedia.org/wiki/Differential_evolution):

```
// ** Differential Evolution settings **
// Crossover Rate [0,1]
double CR = 0.5;
// differential_weight [0,2]
double F = 1;
// total nb of variables in data-set
unsigned int problem_dimensionality = nb_element_vectorized_Theta;    
// ** **

unsigned int genome_length = get_genome_length(max_topo);

unsigned int MUTATION_SCHEME_RAND = 0;
unsigned int MUTATION_SCHEME_BEST = 1;
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

    // store corresponding individual in pop
    vec original_model  = pop[index_x];
    vec candidate_model = pop[index_x];
    vec indiv_a = pop[index_a];
    vec indiv_b = pop[index_b];
    vec indiv_c = pop[index_c];

    // if user selected a DE/BEST/1 mutation scheme
    if(selected_mutation_scheme == MUTATION_SCHEME_BEST){

        // use the best individual as first individual
        indiv_a = pop[0];
    }

    net_topology candidate_topology;
    candidate_topology.nb_input_units = (unsigned int) candidate_model[0];
    candidate_topology.nb_units_per_hidden_layer = (unsigned int) candidate_model[1];
    candidate_topology.nb_output_units = (unsigned int) candidate_model[2];
    candidate_topology.nb_hidden_layers = (unsigned int) candidate_model[3];
    genome_length = get_genome_length(candidate_topology);

    double score_best_model         = generate_net(pop[0]).get_f1_score(training_set);
    double score_second_best_model  = generate_net(pop[1]).get_f1_score(training_set);

    // if the first and second best have identical fitness
    if((score_best_model==score_second_best_model) && score_best_model!=0){
        // force a crossover between the two
        indiv_a = pop[0];
        indiv_b = pop[1];
        mutative_crossover(problem_dimensionality, 1, 1, genome_length, min_topo, max_topo, original_model, candidate_model, indiv_a, indiv_b, indiv_c);
    }
    // traditional random crossover
    mutative_crossover(problem_dimensionality, CR, F, genome_length, min_topo, max_topo, original_model, candidate_model, indiv_a, indiv_b, indiv_c);

    NeuralNet original_net  = generate_net(original_model);
    NeuralNet candidate_net = generate_net(candidate_model);
    // compute performances
    double original_score  = original_net.get_f1_score(training_set);
    double candidate_score = candidate_net.get_f1_score(training_set);
    // selection
    bool candidate_is_better_than_original = candidate_score > original_score;
    if(candidate_is_better_than_original) {
        // replace original by candidate
        pop[index_x] = candidate_model;
    }
}
// update population
population = convert_population_to_nets(pop);
```



Optimization algorithm B: [Particle Swarm Optimization](https://en.wikipedia.org/wiki/Particle_swarm_optimization):

```
// ** PSO settings (must obey convergence rule)**
// velocity weight
double w = 0.729;
// importance of personal best (cognitive weight)
double c1 = 1.494;
// importance of global best (social weight)
double c2 = 1.494;
// ** **

// update pBest of each particle
for(unsigned int p=0; p<pop.size(); p++){
    // calculate fitness
    double candidate_fitness = generate_net(pop[p]).get_f1_score(training_set);
    // if particle is closer to target than pBest
    if(candidate_fitness > pBest[p].get_f1_score(training_set)){
        // set particle as pBest
        pBest[p] = generate_net(pop[p]);
    }
}

// update gBest
vector<NeuralNet>tmp_pop = convert_population_to_nets(pop);
for(unsigned int p=0; p<pop.size(); p++){
    if(pBest[p].get_f1_score(training_set) > gBest.get_f1_score(training_set)){
        // save best particle as <gBest>
        gBest = tmp_pop[p];
    }
}

// for each particle
for(unsigned int p=0; p<pop.size(); p++) {
    for(unsigned int i=0; i<genome_size; i++){
        double r1 = f_rand(0,1);
        double r2 = f_rand(0,1);
        // calculate velocity
        velocities[p][i] = w*velocities[p][i]
                + c1*r1 * (pBest[p].get_params()[i] - pop[p][i])
                + c2*r2 * (gBest.get_params()[i]    - pop[p][i]);
        velocities[p][i] = clip(velocities[p][i], -5, 5);
    }

    vec particle = pop[p];
    // update particle data
    for(unsigned int i=0; i<genome_size; i++){
        switch(i){
        case 0:
            // protect NB INPUTS from being altered
            particle[0] = training_set.X.n_cols;
            break;
        case 1:
            // make sure NB HIDDEN UNITS PER LAYER doesn't exceed genome size
            particle[1] = round(clip(particle[i] + velocities[p][i], 2, max_topo.nb_units_per_hidden_layer));
            break;
        case 2:
            // protect NB OUTPUTS from being altered
            particle[2] = 1;
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
    if(generate_net(particle).get_f1_score(training_set) >= generate_net(pop[p]).get_f1_score(training_set))
        pop[p] = particle;
}
```

## Documentation

Much more details and the results and conclusions of the experiments can be found in `/dissertation/memoir.pdf`.

