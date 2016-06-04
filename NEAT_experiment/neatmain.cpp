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
//#include <stdlib.h>
//#include <stdio.h>
//#include <iostream>
//#include <iomanip>
//#include <sstream>
//#include <list>
//#include <vector>
//#include <algorithm>
//#include <cmath>
//#include <iostream.h>
//#include "networks.h"
//#include "neat.h"
//#include "genetics.h"
//#include "experiments.h"
//#include "neatmain.h"
#include <iostream>
#include <vector>
#include "neat.h"
#include "population.h"
#include "experiments.h"
using namespace std;

int main(int argc, char *argv[]) {

  int pause;
  NEAT::Population *p=0;
  srand( (unsigned)time( NULL ) );

  if (argc != 5) {
    cerr << "Invalid arguments:\n\tArg1=NEAT parameters file (.ne file)\n\tArg2=CHOSEN EXPERIMENT INDEX\n\tArg3=NB REPLICATES\n\tNB GENERATIONS" << endl;
    return -1;
  }

  //Load in the params
  NEAT::load_neat_params(argv[1],true);

  int choice=atoi(argv[2]);
  unsigned int nb_reps=atoi(argv[3]);
  unsigned int nb_gens=atoi(argv[4]);

  switch ( choice )
    {
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
    case 5:
      /*p=*/bcm_test(nb_gens,nb_reps);
      break;
    default:
      cout<<"Not an available option."<<endl;
    }


  //p = pole1_test(100); // 1-pole balancing
  //p = pole2_test(100,1); // 2-pole balancing, velocity
  //p = pole2_test(100,0); // 2-pole balancing, no velocity (non-markov)

  if (p)
    delete p;

  return(0);
 
}

