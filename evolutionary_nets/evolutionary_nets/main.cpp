#include <iostream>
#include <string>
#include "net_benchmark.h"

using namespace std;

// returns number of replicates selected by user (default=1)
exp_files read_args(int argc, char** argv){
    if(argc<(4+1)) {
        cout<<"At least 4 arguments expected."<<endl;
        cout<<"Arg1: 1 or more data sets."<<endl;
        cout<<"Arg2: Nb of replicates."<<endl;
        cout<<"Arg3: Max number of calls to the error function."<<endl;
        cout<<"Arg4: Algorithm population size."<<endl;
        exit(0);
    }else{
        exp_files ef;
        unsigned int nb_ds=argc-(3+1);
        for(unsigned int i=0;i<nb_ds;i++)
            ef.dataset_filenames.push_back(argv[i+1]);
        ef.nb_reps=std::atoi(argv[nb_ds+1]);
        ef.max_nb_err_func_calls=std::atoi(argv[nb_ds+2]);
        ef.pop_size=std::atoi(argv[nb_ds+3]);
        return ef;
    }
}

/*
   Compile: $ QT Creator build or see http://stackoverflow.com/questions/19206462/compile-a-qt-project-from-command-line for CLI
   Run    : $ ./evolutionary_nets    data/breast-cancer-malignantOrBenign-data-transformed.csv \
                            data/breast-cancer-recurrence-data-transformed.csv \
                            data/iris-data-transformed.csv \
                            data/wine-data-transformed.csv \
                            80 \
                            500 \
                            100
   (List of data set paths followed by: NB REPS, MAX NB CALLS TO ERR FUNCTION, POP SIZE)
*/
int main(int argc, char** argv) {
    Net_benchmark bench;
    bench.run_benchmark(read_args(argc,argv));
    cout << "finished" << endl;
    return 0;
}


