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

/**
 * Compile & Run with:
 * $ make # (No direct use of g++ here)
 * $ ./neat test.ne 0 4 300 # experiment 0, 4 replicates, 300 generations
 */
int main(int argc, char *argv[]) {

  NEAT::Population *p=0;
  srand( (unsigned)time( NULL ) );

  if (argc != (5+1)) {
    cerr << "Invalid arguments:\n\tArg1=NEAT parameters file (.ne file)\n\tArg2=CHOSEN EXPERIMENT INDEX\n\tArg3=NB REPLICATES\n\tArg4=NB GENERATIONS\n\tArg5=POP SIZE\n\t(CLI args override file settings)." << endl;
    return -1;
  }

  //Load in the params
  NEAT::load_neat_params(argv[1],true);

  exp_files ef;
  ef.nb_reps=atoi(argv[3]);
  ef.max_nb_err_func_calls=atoi(argv[4]);
  ef.pop_size=atoi(argv[5]);
  unsigned int choice =atoi(argv[2]);


  //
  // SUPERVISED LEARNING EXPERIMENTS
  //

  switch(choice){
  case 0: // BREAST CANCER MALIGNANCY
    ef.startgene="bcmstartgenes";
    ef.dataset_filename="data/breast-cancer-malignantOrBenign-data-transformed.csv";
    ef.result_file="data/results-neat-bcm.mat";
    multiclass_test(ef);
    break;
  case 1: // IRIS
    ef.startgene="irisstartgenes";
    ef.dataset_filename="data/iris-data-transformed.csv";
    ef.result_file="data/results-neat-iris.mat";
    multiclass_test(ef);
    break;
  case 2: // WINE
    ef.startgene="winestartgenes";
    ef.dataset_filename="data/wine-data-transformed.csv";
    ef.result_file="data/results-neat-wine.mat";
    multiclass_test(ef);
    break;
  case 8: // BREAST CANCER RECURRENCE
    ef.startgene="bcrstartgenes";
    ef.dataset_filename="breast-cancer-recurrence-data-transformed.csv";
    ef.result_file="data/results-neat-bcr.mat";
    multiclass_test(ef);
    break;
  default:
    cout<<"Not an available option."<<endl;
  }

  if (p)
    delete p;
  return 0;
}

