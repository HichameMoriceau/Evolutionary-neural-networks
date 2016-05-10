#include <iostream>
#include <string>
#include "net_benchmark.h"

using namespace std;

// compile:  g++ main.cpp -o compiled_executable -larmadillo
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


