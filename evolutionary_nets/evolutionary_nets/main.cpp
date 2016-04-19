#include <iostream>
#include <string>
#include "net_benchmark.h"

using namespace std;

// compile:  g++ main.cpp -o compiled_executable -larmadillo
int main(int argc, char** argv) {
    //
    // BENCHMARKING PERFS OF NEUROEVOLUTIONARY TRAINER
    //

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

/*
    //
    // TESTING BACKPROP
    //

    std::srand(10);

    Data_set D;
    cout << "data-set loaded" << endl;

    string backprop_perfs_vs_epochs = D.octave_variable_name_performances_VS_nb_epochs;
    string backprop_res_file_name = D.result_filename;

    net_topology t;
    t.nb_input_units = D.training_set.X.n_cols;
    t.nb_units_per_hidden_layer = t.nb_input_units*15;
    t.nb_output_units = 1;
    t.nb_hidden_layers = 1;
    NeuralNet nn(t);

    // Gradient Descent/backprop settings
    Backpropagation_trainer backprop;
    backprop.set_nb_epochs(nb_epochs);
    backprop.set_alpha(0.1);

    // train + keep track of metrics
    mat results;
    backprop.train(D, nn, results);

    Net_benchmark bench;
    ofstream backprop_res_file;
    backprop_res_file.open(backprop_res_file_name, ios::out);
    bench.print_results_octave_format(backprop_res_file, results, backprop_perfs_vs_epochs);
*/
    cout << "finished" << endl;
    return 0;
}











