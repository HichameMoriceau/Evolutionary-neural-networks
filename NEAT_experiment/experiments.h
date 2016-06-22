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
#ifndef EXPERIMENTS_H
#define EXPERIMENTS_H

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <list>
#include <vector>
#include <algorithm>
#include <cmath>
#include <string>
#include <stdio.h> 
#include <stdlib.h> 
#include "neat.h"
#include "network.h"
#include "population.h"
#include "organism.h"
#include "genome.h"
#include "species.h"
#include <armadillo> // Added by: Hichame Moriceau

using namespace arma; // Added by: Hichame Moriceau
using namespace std;
using namespace NEAT;


// The Breast Cancer Malignancy (BCM) routines *****************************
void multiclass_test(int gens, unsigned int nb_reps,exp_files ef);
void multiclass_epoch(Population *pop,int generation, Organism& best_org, mat &res_mat,unsigned int& nb_calls_err_func,exp_files ef);
// generic multiclass fitness function
void multiclass_evaluate(Organism* org,string dataset_filename, mat &res_mat,unsigned int& nb_calls_err_func,Population *pop, unsigned int generation, Organism& best_org);

// utility routines for Classification problems (BCM, Iris etc.)
void evaluate_perfs(double** data, unsigned int nb_examples, unsigned int nb_attributes_pls_bias, Network* net, double& error, double& fitness, double& accuracy);
double** load_data_array(string dataset_filename,unsigned int &height, unsigned int &width);
unsigned int count_nb_classes(mat labels);
mat generate_conf_mat(unsigned int nb_classes, mat preds,mat labels);
void compute_error_acc_score(mat conf_mat, mat labels,double& error,double& accuracy,double& fitness);
mat compute_learning_curves_perfs(unsigned int gens, unsigned int nb_reps,vector<mat> &result_matrices_training_perfs, exp_files ef);
void multiclass_training_task(unsigned int i, unsigned int nb_replicates, unsigned int gens, vector<mat> &result_matrices_training_perfs, exp_files ef);
unsigned int count_nb_identicals(unsigned int predicted_class, unsigned int expected_class, mat predictions, mat expectations);
mat to_multiclass_format(mat predictions);
mat average_matrices(vector<mat> results);
mat compute_replicate_error(unsigned int nb_reps,vector<mat> results);
void print_results_octave_format(ofstream &result_file, mat recorded_performances, string octave_variable_name);
mat to_matrix(double a);
double corrected_sample_std_dev(mat score_vector);
const string get_current_date_time();

//The XOR evolution routines *****************************************
Population *xor_test(int gens);
bool xor_evaluate(Organism *org);
int xor_epoch(Population *pop,int generation,char *filename, int &winnernum, int &winnergenes,int &winnernodes);

//Single pole balancing evolution routines ***************************
Population *pole1_test(int gens); 
bool pole1_evaluate(Organism *org);
int pole1_epoch(Population *pop,int generation,char *filename);
int go_cart(Network *net,int max_steps,int thresh); //Run input
//Move the cart and pole
void cart_pole(int action, float *x,float *x_dot, float *theta, float *theta_dot);

//Double pole balacing evolution routines ***************************
class CartPole;

Population *pole2_test(int gens,int velocity);
bool pole2_evaluate(Organism *org,bool velocity,CartPole *thecart);
int pole2_epoch(Population *pop,int generation,char *filename,bool velocity, CartPole *thecart,int &champgenes,int &champnodes, int &winnernum, ofstream &oFile);

class CartPole {
public:
  CartPole(bool randomize,bool velocity);
  virtual void simplifyTask();  
  virtual void nextTask();
  virtual double evalNet(Network *net,int thresh);
  double maxFitness;
  bool MARKOV;

  bool last_hundred;
  bool nmarkov_long;  //Flag that we are looking at the champ
  bool generalization_test;  //Flag we are testing champ's generalization

  double state[6];

  double jigglestep[1000];

protected:
  virtual void init(bool randomize);

private:

  void performAction(double output,int stepnum);
  void step(double action, double *state, double *derivs);
  void rk4(double f, double y[], double dydx[], double yout[]);
  bool outsideBounds(); 

  const static int NUM_INPUTS=7;
  const static double MUP = 0.000002;
  const static double MUC = 0.0005;
  const static double GRAVITY= -9.8;
  const static double MASSCART= 1.0;
  const static double MASSPOLE_1= 0.1;

  const static double LENGTH_1= 0.5;		  /* actually half the pole's length */

  const static double FORCE_MAG= 10.0;
  const static double TAU= 0.01;		  //seconds between state updates 

  const static double one_degree= 0.0174532;	/* 2pi/360 */
  const static double six_degrees= 0.1047192;
  const static double twelve_degrees= 0.2094384;
  const static double fifteen_degrees= 0.2617993;
  const static double thirty_six_degrees= 0.628329;
  const static double fifty_degrees= 0.87266;

  double LENGTH_2;
  double MASSPOLE_2;
  double MIN_INC;
  double POLE_INC;
  double MASS_INC;

  //Queues used for Gruau's fitness which damps oscillations
  int balanced_sum;
  double cartpos_sum;
  double cartv_sum;
  double polepos_sum;
  double polev_sum;



};

#endif






