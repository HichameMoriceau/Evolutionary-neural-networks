#include <iostream>
#include <string>
#include "net_benchmark.h"

using namespace std;

// returns number of replicates selected by user (default=1)
void read_args(int argc, char** argv,vector<string>& data_set_filenames, unsigned int& nb_reps, unsigned int& nb_gens, unsigned int& pop_size){
    if(argc<(4+1)) {
        cout << "At least 4 arguments expected." << endl;
        exit(0);
    }else{
        unsigned int nb_ds=argc-(3+1);
        cout<<"User has given "<<nb_ds<<" paths to different data_sets"<<endl;
        for(unsigned int i=0;i<nb_ds;i++){
            string s(argv[i+1]);
            data_set_filenames.push_back(s);
        }
        nb_reps=std::atoi(argv[nb_ds+1]);
        nb_gens=std::atoi(argv[nb_ds+2]);
        pop_size=std::atoi(argv[nb_ds+3]);
    }
}

/*
   Compile & run:
   $ g++ main.cpp -o compiled_executable -larmadillo
   $ ./evolutionary_nets 64 # for 64 replicates
*/
int main(int argc, char** argv) {
    Net_benchmark bench;

    vector<string> data_set_filenames;
    unsigned int nb_reps=0,nb_gens=0,pop_size;

    read_args(argc,argv,data_set_filenames, nb_reps,nb_gens,pop_size);
    for(unsigned int i=0; i<data_set_filenames.size();i++)
        cout<<"dataset"<<i<<" : "<<data_set_filenames[i]<<endl;
    cout<<"nb reps="<<nb_reps<<endl;
    cout<<"nb gens="<<nb_gens<<endl;
    cout<<"pop_size="<<pop_size<<endl;

    bench.run_benchmark(data_set_filenames,nb_reps,nb_gens,pop_size);
    cout << "finished" << endl;
    return 0;
}


