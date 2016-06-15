#include <iostream>
#include <string>
#include "net_benchmark.h"

using namespace std;

// returns number of replicates selected by user (default=1)
unsigned int read_args(int argc, char** argv){
    unsigned int nb_replicates=1;
    if(argc > 2){
        cout << "1 argument is expected." << endl;
        exit(0);
    }else if(argc==2){
        nb_replicates=std::atoi(argv[1]);
    }
    return nb_replicates;
}

/*
   Compile & run:
   $ g++ main.cpp -o compiled_executable -larmadillo
   $ ./evolutionary_nets 64 # for 64 replicates
*/
int main(int argc, char** argv) {
    Net_benchmark bench;
    bench.run_benchmark(read_args(argc,argv));
    cout << "finished" << endl;
    return 0;
}


