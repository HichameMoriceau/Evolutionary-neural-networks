#include <iostream>
#include <string>
#include "net_benchmark.h"

using namespace std;

// X Simplify the comparison of predictions with labels on multiclass problems (use only 1 dimensional arrays)
// X Apply the simpler method to NEAT
// X Fixed why NEAT didn't improve more that 33%acc
// X Move all replicated methods from TRAINER_AIS, TRAINER_DE and TRAINER_PSO to base class TRAINER.CPP
// X Remove cross-validation from TRAINER_AIS, TRAINER_DE and TRAINER_PSO

// X Update BC starting gene to a simplistic neural network
// X fix the plotting (once CrossValidation was removed)
// X fix PSO calculation of score and accuracy (score=100 and acc=65% is impossible)
// X fix the calculation of the score and accuracy of the ensemble (multiclass problems)
// Fixed the counting of generations
// Add Iris experiment to NEAT application
// Set a fixed number of epochs termination criterion to NEAT
// Fix what's wrong with TRAINER_AIS
// Make sure NEAT is functional with all data sets
// Run NEAT on all data sets
// Plot BENCHMARK results
// Plot NEAT results


// compile:  $ g++ main.cpp -o compiled_executable -larmadillo
// run    :  $ ./evolutionary_nets 64 # for 64 replicates
int main(int argc, char** argv) {
    // default nb replicates
    unsigned int nb_replicates = 1;
    // retrieve nb replicates
    if(argc > 2){
        cout << "More than 1 argument specified. " << endl;
        exit(0);
    }else if(argc == 2){
        unsigned int first_argument = std::atoi(argv[1]);
        nb_replicates = first_argument;
    }

    Net_benchmark bench;
    bench.run_benchmark(nb_replicates);

    cout << "finished" << endl;
    return 0;
}


