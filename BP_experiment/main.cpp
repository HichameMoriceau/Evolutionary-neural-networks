#include<iostream>
#include<string>
#include<sstream>
#include<vector>
#include<fann.h>
#include<armadillo>
#include<string>
#include<omp.h>
using namespace std;
using namespace arma;

struct exp_files{
  string startgene;
  string dataset_filename;
  string result_file;
};

vector<string> split(const string& str, int delimiter(int) = ::isspace){
  vector<string> result;
  auto e=str.end();
  auto i=str.begin();
  while(i!=e){
    i=find_if_not(i,e, delimiter);
    if(i==e) break;
    auto j=find_if(i,e, delimiter);
    result.push_back(string(i,j));
    i=j;
  }
  return result;
}

int rand_int(int min, int max);
void fixed_topo_exp(int gens, unsigned int nb_reps, exp_files ef);
mat compute_learning_curves_perfs(unsigned int gens, unsigned int nb_reps,vector<mat> &result_matrices_training_perfs, exp_files ef);
void multiclass_fixed_training_task(unsigned int i, unsigned int nb_reps,unsigned int gens,vector<mat> &res_mats_training_perfs, exp_files ef);

unsigned int count_nb_identicals(unsigned int predicted_class, unsigned int expected_class, mat predictions, mat expectations);
mat to_multiclass_format(mat predictions);
mat average_matrices(vector<mat> results);
mat compute_replicate_error(unsigned int nb_reps,vector<mat> results);
void print_results_octave_format(ofstream &result_file, mat recorded_performances, string octave_variable_name);
mat to_matrix(double a);
double corrected_sample_std_dev(mat score_vector);
const string get_current_date_time();
double diverse_topo_exp();

/**
   Experiment: Running multiple instances of a neural network
   trained using Gradient Descent and the Back Propagation algorithm.
   Author: Hichame Moriceau
   Compile & Run with: 
   # g++ -std=c++11 main.cpp -fopenmp -larmadillo -lfann -o runme
   # ./runme 0 30 100 # run 30 replicates of the first experiment
*/
int main(int argc, char * argv []){
  if(argc<(3+1)){ 
    cout<<"Too few args provided. Expected: './runme EXP_INDEX NB_REPLICATES NB GENERATIONS'"<<endl;
    return 0;
  }
  // experiment choice
  unsigned exp_index=atoi(argv[1]);
  // number of replicates
  unsigned int nb_reps=atoi(argv[2]);
  // recorded best error value
  unsigned int nb_gens=atoi(argv[3]);
  exp_files ef;
  ef.startgene=""; // initial gene is irrelevant here

  vector<char*> ds_filenames;
  ds_filenames.push_back((char*)"data/breast-cancer-malignantOrBenign-data-transformed.data");
  ds_filenames.push_back((char*)"data/wine-data-transformed.data"); // multi-class
  ds_filenames.push_back((char*)"data/breast-cancer-recurrence-data-transformed.data");
  ds_filenames.push_back((char*)"data/iris-data-transformed.data"); // multi-class

  switch(exp_index){
  case 0:
    ef.dataset_filename=ds_filenames[0];
    ef.result_file="data/results-bp-fixed-bcm.mat";
    fixed_topo_exp(nb_gens,nb_reps,ef);
    break;
  case 1:
    ef.dataset_filename=ds_filenames[0];
    ef.result_file="data/results-bp-fixed-bcm.mat";
    diverse_topo_exp();
    break;
  default:
    ef.dataset_filename=ds_filenames[0];
    ef.result_file="data/results-bp-fixed-bcm.mat";
    fixed_topo_exp(nb_gens,nb_reps,ef);
  }
  return 0;
}

// generates random integer within range
int rand_int(int min, int max){
  return min + (rand() % (int)(max - min + 1));
}

void fixed_topo_exp(int gens, unsigned int nb_reps, exp_files ef){
  std::ofstream oFile(ef.result_file.c_str(),std::ios::out);
  vector<mat> res_mats_training_perfs;
  mat avrg_mat=compute_learning_curves_perfs(gens,nb_reps,res_mats_training_perfs,ef);
  mat res_mat_err = compute_replicate_error(nb_reps,res_mats_training_perfs);
  // save results
  avrg_mat = join_horiz(avrg_mat, ones(avrg_mat.n_rows,1) * nb_reps);
  // plot results
  print_results_octave_format(oFile,avrg_mat,"results");
  print_results_octave_format(oFile,res_mat_err,"err_results");
}

mat compute_learning_curves_perfs(unsigned int gens, unsigned int nb_reps,vector<mat> &result_matrices_training_perfs, exp_files ef){
  // return variable
  mat averaged_performances;
  // for each replicate
#pragma omp parallel
  {
    // kick off a single thread
#pragma omp single
    {
      for(unsigned int i=0; i<nb_reps; ++i) {
#pragma omp task
	multiclass_fixed_training_task(i, nb_reps,gens,result_matrices_training_perfs,ef);
      }
    }
  }
  // average PERFS of all replicates
  averaged_performances = average_matrices(result_matrices_training_perfs);
  return averaged_performances;
}

double fann_get_accuracy(struct fann* ann, struct fann_train_data *data){
  double acc=0;
  unsigned int count=0;
  unsigned int nb_examples=fann_length_train_data(data);
  unsigned int nb_attributes=fann_num_input_train_data(data);
  unsigned int nb_classes=fann_num_output_train_data(data);

  // model predictions
  float preds[nb_examples];
  // expected predictions
  float labels[nb_examples];

  // obtain model predictions and expected predictions
  for(unsigned int i=0;i<nb_examples;i++){
    fann_type* input=data->input[i];
    labels[i]=data->output[i][0];
    preds[i]=fann_run(ann, input)[0];
    if(preds[i]>=0.5)
      preds[i]=1;
    else
      preds[i]=0;
  }

  for(unsigned int i=0;i<nb_examples;i++)
    if(preds[i]==labels[i])
      count++;
  acc=(count/double(nb_examples))*100;
  return acc;
}

void multiclass_fixed_training_task(unsigned int i, unsigned int nb_reps,unsigned int gens,vector<mat> &res_mats_training_perfs, exp_files ef){
  // result matrices (to be interpreted by Octave script <Plotter.m>)
  mat results_score_evolution;

  cout << endl
       << "***"
       << "\tRUNNING REPLICATE " << i+1 << "/" << nb_reps << "\t ";

  // set seed
  unsigned int seed=i*10;
  std::srand(seed);

  double res=-1;
  unsigned int nb_bp_searches=5;

  mat res_mat;

  // selected data set
  string ds_filename=string(ef.dataset_filename);
  
  // read data set header
  ifstream in(ds_filename);
  if(!in.is_open()){
    cout<<"\nIncorrect data set name."<<endl;
    exit(0);
  }
  string line="";
  getline(in,line);
  vector<string> headers=split(line);
  unsigned int nb_examples=stoi(headers[0]);
  in.close();

  // hyper-params
  const unsigned int nb_inputs=stoi(headers[1]);
  const unsigned int nb_outputs=stoi(headers[2]);
  const unsigned int nb_layers=2;
  const unsigned int nb_hid_units=20;
  const float desired_error=(const float) 0.00f;
  const unsigned int max_epochs=20;
  const unsigned int epochs_between_reports=1;

  fann_type min_weight=-1;
  fann_type max_weight=+1;

  // load training data
  struct fann_train_data *data = fann_read_train_from_file(ds_filename.c_str());
  // randomize examples order
  fann_shuffle_train_data(data);
  // instantiate net
  struct fann*ann=fann_create_standard(nb_layers,nb_inputs, nb_outputs,nb_hid_units);
  struct fann*best_ann=fann_create_standard(nb_layers,nb_inputs, nb_outputs,nb_hid_units);
  fann_randomize_weights(ann,min_weight,max_weight);
  fann_randomize_weights(best_ann,min_weight,max_weight);
  fann_train_epoch(best_ann,data);

  cout<<"setting activation functions"<<endl;
  fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
  fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);

  cout<<"training model"<<endl;

  for(unsigned int b=0;b<nb_bp_searches;b++){
    //reset ann
    fann_randomize_weights(ann,min_weight,max_weight);
    double mse=-1;
    for(unsigned int i=0;i<max_epochs;i++){
      mse=fann_train_epoch(ann,data);

      mat line;
      double pop_score_mean=-1;
      double pop_score_variance=-1;
      double pop_score_stddev=-1;
      double prediction_accuracy=fann_get_accuracy(best_ann, data);
      double score=-1;//fann_get_f1_score(best_ann);
      double validation_accuracy=-1;
      double validation_score=-1;
      double MSE=fann_get_MSE(best_ann);
      unsigned int hidden_units=nb_hid_units;

      std::cout<<"epoch"<<i
	       <<"\tbest.indiv.fitness="<<score
	       <<"\tacc="<<prediction_accuracy
	       <<"\terr="<<MSE
	       <<"\tNB nodes="<<hidden_units
	       <<"\tpop.mean="<<pop_score_mean
	       <<"\tpop.var="<<pop_score_variance
	       <<"\tpop.stddev="<<pop_score_stddev<<endl;

      // format result line (-1 corresponds to irrelevant attributes)
      line << i
	   << MSE
	   << prediction_accuracy
	   << score
	   << pop_score_variance

	   << pop_score_stddev
	   << pop_score_mean
	   << -1//pop_score_median
	   << -1//pop->organisms.size()
	   << nb_inputs // inputs

	   << hidden_units
	   << nb_outputs//outputs
	   << -1//nb_hidden_layers
	   << true
	   << -1//selected_mutation_scheme

	   << -1//ensemble_accuracy
	   << -1//ensemble_score
	   << validation_accuracy
	   << validation_score
	   << -2//nb_calls_err_func
	
	   << endr;
      // Write results on file
      res_mat=join_vert(res_mat,line);
    }
    // memorize best network
    if(fann_get_MSE(ann)<fann_get_MSE(best_ann))
      *best_ann=*ann;
  }
  fann_destroy(ann);
  
  ofstream experiment_file("random-seeds.txt",ios::app);
  // print-out best perfs
  results_score_evolution=join_vert(results_score_evolution, res_mat);
  double best_score = results_score_evolution(results_score_evolution.n_rows-1, 3);
  res_mats_training_perfs.push_back(results_score_evolution);

  cout           <<"THREAD"<<omp_get_thread_num()<<" replicate="<<i<<"\tseed="<<seed<<"\tbest_score="<<"\t"<<best_score<<" on "<<ef.dataset_filename<<endl;
  experiment_file<<"THREAD"<<omp_get_thread_num()<<" replicate="<<i<<"\tseed="<<seed<<"\tbest_score="<<"\t"<<best_score<<" on "<<ef.dataset_filename<<endl;
  experiment_file.close();
}


unsigned int count_nb_identicals(unsigned int predicted_class, unsigned int expected_class, mat predictions, mat expectations){
  unsigned int count=0;
  // for each example
  for(unsigned int i=0; i<predictions.n_rows; i++){
    if(predictions(i)==predicted_class && expectations(i)==expected_class)
      count++;
  }
  return count;
}

mat to_multiclass_format(mat predictions){
  unsigned int nb_classes = predictions.n_cols;
  mat formatted_predictions(predictions.n_rows, 1);
  double highest_activation = 0;
  // for each example
  for(unsigned int i=0; i<predictions.n_rows; i++){
    unsigned int index = 0;
    highest_activation = 0;
    // the strongest activation is considered the prediction
    for(unsigned int j=0; j<nb_classes; j++){
      if(predictions(i,j) > highest_activation){
	highest_activation = predictions(i,j);
	index = j;
      }
    }
    formatted_predictions(i) = index;
  }
  return formatted_predictions;
}

mat average_matrices(vector<mat> results){
  cout<<"AVERAGING RESULTS"<<endl;
  unsigned int smallest_nb_rows = INT_MAX;
  // find lowest and highest nb rows
  for(unsigned int i=0; i<results.size() ;i++){
    if(results[i].n_rows < smallest_nb_rows){
      smallest_nb_rows = results[i].n_rows;
    }
  }
  mat total_results;
  total_results.zeros(smallest_nb_rows, results[0].n_cols);

  // keep only up to the shortest learning curve
  vector<mat> processed_results;
  for(unsigned int i=0; i<results.size(); i++){
    processed_results.push_back( (mat)results[i].rows(0, smallest_nb_rows-1));
  }

  for(unsigned int i=0; i<results.size(); i++){
    total_results += processed_results[i];
  }
  mat averaged_results = total_results / results.size();
  return averaged_results;
}

mat compute_replicate_error(unsigned int nb_reps,vector<mat> results){
  // return variable
  mat err_vec;
  unsigned int smallest_nb_rows = INT_MAX;
  // find lowest and highest nb rows
  for(unsigned int i=0; i<results.size() ;i++){
    if(results[i].n_rows < smallest_nb_rows){
      smallest_nb_rows = results[i].n_rows;
    }
  }
  mat best_scores;
  // for each generation
  for(unsigned int i=0; i<smallest_nb_rows; i++) {
    best_scores.reset();
    // for each replica
    for(unsigned int r=0; r<nb_reps; r++) { // NEAT::num_runs=nb replicates
      // get best score
      best_scores = join_vert(best_scores, to_matrix(results[r](i,3)));
    }
    // append std dev of all best scores for current generation to error vector
    err_vec = join_vert( err_vec, to_matrix(corrected_sample_std_dev(best_scores)));
  }
  return err_vec;
}

void print_results_octave_format(ofstream &result_file, mat recorded_performances, string octave_variable_name){
  // Create header in MATLAB format
  result_file << "# Created by main.cpp, " << get_current_date_time() << endl
	      << "# name: " << octave_variable_name << endl
	      << "# type: matrix"  << endl
	      << "# rows: " << recorded_performances.n_rows << endl
	      << "# columns: " << recorded_performances.n_cols << endl;

  // append content of recorded performances into same file
  for(unsigned int i=0; i<recorded_performances.n_rows; ++i){
    for(unsigned int j=0; j<recorded_performances.n_cols; ++j){
      result_file << recorded_performances(i,j) << " ";
    }
    result_file << endl;
  }
  result_file << endl;
}

mat to_matrix(double a){
  mat A;
  A << a;
  return A;
}

double corrected_sample_std_dev(mat score_vector){
  // return variable
  double s = 0;
  double N = score_vector.n_rows;
  mat mean_vector = ones(score_vector.size(),1) * mean(score_vector);
  s = (1/(N-1)) * ((double) as_scalar(sum(pow(score_vector - mean_vector, 2))));
  s = sqrt(s);
  return s;
}

// returns current date/time in format YYYY-MM-DD.HH:mm:ss
const string get_current_date_time(){
  time_t     now = time(0);
  struct tm  tstruct;
  char       buffer[80];
  tstruct = *localtime(&now);
  strftime(buffer, sizeof(buffer), "%Y-%m-%d.%X", &tstruct);
  return buffer;
}

double diverse_topo_exp(){
  double res=0;
  unsigned int nb_opt_algs=5;
  
  vector<char*> ds_filenames;
  ds_filenames.push_back((char*)"data/wine-data-transformed.data"); // multi-class
  ds_filenames.push_back((char*)"data/breast-cancer-recurrence-data-transformed.data");
  ds_filenames.push_back((char*)"data/iris-data-transformed.data"); // multi-class
  ds_filenames.push_back((char*)"data/breast-cancer-malignantOrBenign-data-transformed.data");

  // selected data set
  string ds_filename=string(ds_filenames[0]);
  
  // read data set header
  ifstream in(ds_filename);
  if(!in.is_open()){
    cout<<"Incorrect data set name."<<endl;
    return 0;
  }

  string line="";
  getline(in,line);
  vector<string> headers=split(line);
  unsigned int nb_examples=stoi(headers[0]);
  in.close();
  
  // hyper-params
  unsigned int num_input=stoi(headers[1]);
  unsigned int num_output=stoi(headers[2]);
  unsigned int num_layers=2;
  unsigned int nb_hid_units=20;
  const float desired_error=(const float) 0.00f;
  const unsigned int max_epochs=10;
  const unsigned int epochs_between_reports=1;

  fann_type min_weight=-1;
  fann_type max_weight=+1;

  struct fann_train_data *data = fann_read_train_from_file(ds_filename.c_str());
  struct fann *ann;

  cout<<"training model"<<endl;
  res=10;
  for(unsigned int b=0;b<nb_opt_algs;b++){
    // randomize depth and size
    num_layers=rand_int(1,3);
    nb_hid_units=rand_int(1,20);
    // encode topology description in array
    vector<unsigned int> desc_vec;
    desc_vec.push_back(num_input);
    for(unsigned int i=0;i<num_layers;i++)
      desc_vec.push_back(nb_hid_units);
    desc_vec.push_back(num_output);
    // cast to array
    unsigned int *desc_array=&desc_vec[0];
    // instantiate model
    ann=fann_create_standard_array(num_layers+2,desc_array); // total NB layers = nb hid layers + input & output layer
    // use sigmoid activation function for all neurons
    fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
    fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);

    cout<<"Net topology:"<<endl
	<<"\tNb inputs            = "<<fann_get_num_input(ann)<<endl
	<<"\tNb hid. units        = "<<nb_hid_units<<endl
	<<"\tNb outputs           = "<<fann_get_num_output(ann)<<endl
	<<"\tNb hid. layers       = "<<fann_get_num_layers(ann)-2<<endl;

    double mse=-1;
    cout<<"running BP"<<b<<endl;
    for(unsigned int i=0;i<max_epochs;i++){
      mse=fann_train_epoch(ann,data);
      cout<<"epoch"<<i<<"\tmse="<<mse<<endl;
    }
    // clear heap
    fann_destroy(ann);
    if(mse<res)
      res=mse;
  }
  cout<<"lowest err obtained after "<<nb_opt_algs<<" != runs of BP for "<<max_epochs<<" epochs is "<<res<<endl<<endl;  
  return res;
}
