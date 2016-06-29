/*
 Copyright 2001 The University of Texas at Austin

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#include <iostream>
#include <vector>
#include "neat.h"
#include "population.h"
#include "experiments.h"
using namespace std;

// returns number of replicates selected by user (default=1)
exp_files read_args(int argc, char** argv){
    if(argc<(5+1)) {
        cout<<"At least 4 arguments expected."<<endl;
        cout<<"Arg1: 1 or more data sets."<<endl;
        cout<<"Arg2: neuroevolution settings file (*.ne)."<<endl;
        cout<<"Arg3: Nb of replicates."<<endl;
        cout<<"Arg4: Max number of calls to the error function."<<endl;
        cout<<"Arg5: Algorithm population size."<<endl;
        exit(0);
    }else{
        exp_files ef;
        unsigned int nb_ds=argc-(4+1);
        for(unsigned int i=0;i<nb_ds;i++)
            ef.dataset_filenames.push_back(argv[i+1]);
	ef.neuroevolution_settings=argv[nb_ds+1];
        ef.nb_reps=std::atoi(argv[nb_ds+2]);
        ef.max_nb_err_func_calls=std::atoi(argv[nb_ds+3]);
        ef.pop_size=std::atoi(argv[nb_ds+4]);
        return ef;
    }
}

/**
 * Compile: $ make # (No direct use of g++ here)
 * Run    : $ ./neat data/breast-cancer-malignantOrBenign-data-transformed.csv test.ne 4 300 100 #  4 replicates, 300 generations 100 individuals
 */
int main(int argc, char *argv[]) {
  // read CLI args
  exp_files ef=read_args(argc,argv);
  
  //Load in the neuro evolution params
  NEAT::load_neat_params(ef.neuroevolution_settings.c_str(),true);

  // for each data set
  for(unsigned int i=0;i<ef.dataset_filenames.size();i++){
    ef.current_ds=ef.dataset_filenames[i];
    ef.result_file=ef.dataset_filenames[i].substr(0,ef.dataset_filenames[i].size()-4);
    replace(ef.result_file.begin(),ef.result_file.end(),'-','_');
    ef.startgene=ef.result_file+"_startgenes";
    ef.startgene=ef.startgene.substr(5, ef.startgene.size());
    ef.result_file+="_results/NEAT-results.mat";
    // run experiment
    multiclass_test(ef);    
  }

  return 0;
}

