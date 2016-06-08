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
 * $ make
 * $ ./neat test.ne 5 4 300 # experiment 5, 4 replicates, 100 generations
 */
int main(int argc, char *argv[]) {

  int pause;
  NEAT::Population *p=0;
  srand( (unsigned)time( NULL ) );

  if (argc != 5) {
    cerr << "Invalid arguments:\n\tArg1=NEAT parameters file (.ne file)\n\tArg2=CHOSEN EXPERIMENT INDEX\n\tArg3=NB REPLICATES\n\tArg4=NB GENERATIONS" << endl;
    return -1;
  }

  //Load in the params
  NEAT::load_neat_params(argv[1],true);
  unsigned int choice =atoi(argv[2]);
  unsigned int nb_reps=atoi(argv[3]);
  unsigned int nb_gens=atoi(argv[4]);

  exp_files expfiles;

  switch(choice){
  case 1:
    p = pole1_test(nb_gens);
    break;
  case 2:
    p = pole2_test(nb_gens,1);
    break;
  case 3:
    p = pole2_test(nb_gens,0);
    break;
  case 4:
    p=xor_test(nb_gens);
    break;
  case 5: // BREAST CANCER MALIGNANCY
    expfiles.startgene="bcmstartgenes";
    expfiles.dataset_filename="data/breast-cancer-malignantOrBenign-data-transformed.csv";
    expfiles.result_file="data/results-neat-bcm.mat";
    multiclass_test(nb_gens,nb_reps, expfiles);
    break;
  case 6: // IRIS
    expfiles.startgene="irisstartgenes";
    expfiles.dataset_filename="data/iris-data-transformed.csv";
    expfiles.result_file="data/results-neat-iris.mat";
    multiclass_test(nb_gens,nb_reps, expfiles);
    break;
  case 7: // WINE
    expfiles.startgene="winestartgenes";
    expfiles.dataset_filename="data/wine-data-transformed.csv";
    expfiles.result_file="data/results-neat-wine.mat";
    multiclass_test(nb_gens,nb_reps, expfiles);
    break;
  case 8: // BREAST CANCER RECURRENCE
    expfiles.startgene="bcrstartgenes";
    expfiles.dataset_filename="breast-cancer-recurrence-data-transformed.csv";
    expfiles.result_file="data/results-neat-bcr.mat";
    multiclass_test(nb_gens,nb_reps, expfiles);
    break;
  default:
    cout<<"Not an available option."<<endl;
  }

  if (p)
    delete p;
  return 0;
}

