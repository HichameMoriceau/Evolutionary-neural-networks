#include<iostream>
#include<string>
#include<sstream>
#include<vector>
#include<fann.h>
#include<armadillo>
#include<string>
#include<omp.h>
#include <iomanip> // setprecision
using namespace std;
using namespace arma;

enum EXP_TYPE{FIXED, CASCADE};

struct exp_details{
  string dataset_filename;
  string result_file;
  EXP_TYPE exp_type;
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
void experiment(int gens, unsigned int nb_reps, exp_details ef);
mat compute_learning_curves_perfs(unsigned int gens, unsigned int nb_reps,vector<mat> &result_matrices_training_perfs, exp_details ef);
struct fann_train_data* fann_instantiate_data(unsigned int nb_examples,unsigned int nb_inputs,unsigned int nb_outputs);
void fann_separate_data(fann_train_data data,fann_train_data* training_data,fann_train_data* validation_data, fann_train_data* test_data);
unsigned int fann_get_nb_hidden_units(fann* best_ann, unsigned int nb_hid_layers);
void multiclass_fixed_training_task(unsigned int i, unsigned int nb_reps,unsigned int gens,vector<mat> &res_mats_training_perfs, exp_details ef);
void multiclass_cascade_training_task(unsigned int i, unsigned int nb_reps,unsigned int gens,vector<mat> &res_mats_training_perfs, exp_details ef);
unsigned int count_nb_classes(mat labels);
mat generate_conf_mat(unsigned int nb_classes, mat preds, mat labels);
void compute_error_acc_score(mat conf_mat, mat labels,double& error,double& accuracy,double& fitness);

unsigned int count_nb_identicals(unsigned int predicted_class, unsigned int expected_class, mat predictions, mat expectations);
mat to_multiclass_format(mat predictions);
mat average_matrices(vector<mat> results);
mat compute_replicate_error(unsigned int nb_reps,vector<mat> results);
void print_results_octave_format(ofstream &result_file, mat recorded_performances, string octave_variable_name);
mat to_matrix(double a);
double corrected_sample_std_dev(mat score_vector);
const string get_current_date_time();

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
  
  exp_details ed;
  // selected experiment
  unsigned int exp_index=atoi(argv[1]);
  // number of replicates
  unsigned int nb_reps=atoi(argv[2]);
  // recorded best error value
  unsigned int nb_gens=atoi(argv[3]);

  vector<char*> ds_filenames;
  ds_filenames.push_back((char*)"data/breast-cancer-malignantOrBenign-data-transformed.data");
  ds_filenames.push_back((char*)"data/wine-data-transformed.data"); // multi-class
  ds_filenames.push_back((char*)"data/breast-cancer-recurrence-data-transformed.data");
  ds_filenames.push_back((char*)"data/iris-data-transformed.data"); // multi-class

  switch(exp_index){
  case 0:
    ed.exp_type=EXP_TYPE::FIXED;
    ed.dataset_filename=ds_filenames[0];
    ed.result_file="data/results-bp-fixed-bcm.mat";
    experiment(nb_gens,nb_reps,ed);
    break;
  case 1:
    ed.exp_type=EXP_TYPE::FIXED;
    ed.dataset_filename=ds_filenames[1];
    ed.result_file="data/results-bp-fixed-wine.mat";
    experiment(nb_gens,nb_reps,ed);
    break;
  case 2:
    ed.exp_type=EXP_TYPE::FIXED;
    ed.dataset_filename=ds_filenames[2];
    ed.result_file="data/results-bp-fixed-bcr.mat";
    experiment(nb_gens,nb_reps,ed);
    break;
  case 3:
    ed.exp_type=EXP_TYPE::FIXED;
    ed.dataset_filename=ds_filenames[3];
    ed.result_file="data/results-bp-fixed-iris.mat";
    experiment(nb_gens,nb_reps,ed);
    break;

  case 4:
    ed.exp_type=EXP_TYPE::CASCADE;
    ed.dataset_filename=ds_filenames[0];
    ed.result_file="data/results-bp-cascade-bcm.mat";
    experiment(nb_gens,nb_reps,ed);
    break;
  case 5:
    ed.exp_type=EXP_TYPE::CASCADE;
    ed.dataset_filename=ds_filenames[1];
    ed.result_file="data/results-bp-cascade-wine.mat";
    experiment(nb_gens,nb_reps,ed);
    break;
  case 6:
    ed.exp_type=EXP_TYPE::CASCADE;
    ed.dataset_filename=ds_filenames[2];
    ed.result_file="data/results-bp-cascade-bcr.mat";
    experiment(nb_gens,nb_reps,ed);
    break;
  case 7:
    ed.exp_type=EXP_TYPE::CASCADE;
    ed.dataset_filename=ds_filenames[3];
    ed.result_file="data/results-bp-cascade-iris.mat";
    experiment(nb_gens,nb_reps,ed);
    break;
  default:
    cout<<"Please use an appropriate experiment index"<<endl;
  }
  return 0;
}

// generates random integer within range
int rand_int(int min, int max){
  return min + (rand() % (int)(max - min + 1));
}

void experiment(int gens, unsigned int nb_reps, exp_details ef){
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

mat compute_learning_curves_perfs(unsigned int gens, unsigned int nb_reps,vector<mat> &result_matrices_training_perfs, exp_details ed){
  // return variable
  mat averaged_performances;
  // for each replicate
#pragma omp parallel
  {
    // kick off a single thread
#pragma omp single
    {
      for(unsigned int i=0; i<nb_reps; ++i) {
	if(ed.exp_type==EXP_TYPE::FIXED){
#pragma omp task
	  multiclass_fixed_training_task(i, nb_reps,gens,result_matrices_training_perfs,ed);
	}else if(ed.exp_type==EXP_TYPE::CASCADE){
	  multiclass_cascade_training_task(i, nb_reps,gens,result_matrices_training_perfs,ed);
	}
      }
    }
  }
  // average PERFS of all replicates
  averaged_performances = average_matrices(result_matrices_training_perfs);
  return averaged_performances;
}

void fann_get_preds_labels(struct fann* ann, struct fann_train_data *data, mat& preds, mat& labels){
  unsigned int nb_examples=fann_length_train_data(data);
  // model predictions
  float _preds[nb_examples];
  // expected predictions
  float _labels[nb_examples];
  // obtain model predictions and expected predictions
  for(unsigned int i=0;i<nb_examples;i++){
    fann_type* input=data->input[i];
    _labels[i]=data->output[i][0];
    _preds[i]=fann_run(ann, input)[0];
    if(_preds[i]>=0.5)
      _preds[i]=1;
    else
      _preds[i]=0;
  }
  // init with values != 0 or 1
  mat p=randu(nb_examples);
  mat l=randu(nb_examples);
  // cast <float[]> to <mat>
  for(unsigned int i=0; i<nb_examples; i++){
    p(i) =_preds[i];
    l(i)=_labels[i];
  }
  preds=p;
  labels=l;
}

unsigned int count_nb_classes(mat labels){
  vector<unsigned int> array;
  bool is_known_class = false;
  // compute nb output units required
  for(unsigned int i=0; i<labels.n_rows; i++) {
    unsigned int current_pred_class = labels(i);
    is_known_class=false;
    // for each known prediction classes: if current output is != from prediction class
    for(unsigned int j=0;j<array.size(); j++) 
      if(current_pred_class==array[j])
	is_known_class = true;
    if(array.empty() || (!is_known_class))
      array.push_back(current_pred_class);
  }
  return array.size();
}

mat generate_conf_mat(unsigned int nb_classes, mat preds, mat labels){
  mat conf_mat=zeros(nb_classes, nb_classes);
  for(unsigned int i=0; i<nb_classes; i++)
    for(unsigned int j=0; j<nb_classes; j++)
      conf_mat(i,j)=count_nb_identicals(i,j,/*to_multiclass_format*/(preds),labels);
  return conf_mat;
}

void compute_error_acc_score(mat conf_mat, mat labels,double& error,double& accuracy,double& fitness){
  unsigned int nb_classes=conf_mat.n_cols;
  // number of class present in the current subset of the data set
  unsigned int nb_local_classes=count_nb_classes(labels);
  unsigned int nb_examples=0;
  for(unsigned int i=0;i<nb_classes;i++)
    nb_examples+=sum(conf_mat.row(i));
  double computed_score=0, computed_acc=0, errsum=0;
  vec scores(nb_classes);
  // computing f1 score for each label
  for(unsigned int i=0; i<nb_classes; i++){
    double TP = conf_mat(i,i);
    double TPplusFN = sum(conf_mat.col(i));
    double TPplusFP = sum(conf_mat.row(i));
    double tmp_precision=TP/TPplusFP;
    double tmp_recall=TP/TPplusFN;
    scores[i] = 2*((tmp_precision*tmp_recall)/(tmp_precision+tmp_recall));
    // prevent -nan
    if(scores[i] != scores[i])
      scores[i] = 0;
    computed_score += scores[i];
  }
  // general f1 score = average of all classes score
  computed_score = (computed_score/nb_local_classes)*100;
  // make sure score doesn't hit 0 (NEAT doesn't seem to like that)
  if(computed_score==0)
    computed_score=0.1;
  // compute accuracy
  double TP =0;
  for(unsigned int i=0; i<nb_classes; i++){
    TP += conf_mat(i,i);
  }
  computed_acc = (TP/double(nb_examples))*100;
  // prevent -nan values
  if(computed_acc!=computed_acc)
    computed_acc=0;
  // compute error
  error=(nb_examples-TP)/nb_examples;
  accuracy=computed_acc;
  fitness=computed_score;
}

struct fann_train_data* fann_instantiate_data(unsigned int nb_examples,unsigned int nb_inputs,unsigned int nb_outputs){
  struct fann_train_data* data;
  // allocate struct memory
  data=(fann_train_data*)malloc(sizeof(fann_train_data));
  // set struct attributes
  data->num_data=nb_examples;
  data->num_input=nb_inputs;
  data->num_output=nb_outputs;
  cout<<"setting num_data to "<<data->num_data<<endl;
  // allocate 2D arrays attributes of struct
  data->input=(fann_type**) malloc(nb_examples*sizeof(fann_type *));
  data->output=(fann_type **) malloc(nb_examples*sizeof(fann_type *));
  for(unsigned int i=0;i<nb_examples;i++)
    data->input[i]=(fann_type *) malloc(nb_inputs*sizeof(fann_type));
  for(unsigned int i=0;i<nb_examples;i++)
    data->output[i]=(fann_type *) malloc(nb_outputs*sizeof(fann_type));
  return data;
}

void fann_separate_data(fann_train_data data,fann_train_data* training_data,fann_train_data* validation_data, fann_train_data* test_data){
  unsigned int nb_inputs=training_data->num_input;
  unsigned int nb_outputs=training_data->num_output;
  unsigned int nb_train_ex=training_data->num_data;
  unsigned int nb_val_ex=validation_data->num_data;
  unsigned int nb_test_ex=test_data->num_data;
  // fill in TRAINING SET
  for(unsigned int i=0;i<nb_train_ex;i++){
    for(unsigned int j=0;j<nb_inputs;j++)
      training_data->input[i][j]=data.input[i][j];
    for(unsigned int j=0;j<nb_outputs;j++)
      training_data->output[i][j]=data.output[i][j];
  }
  // fill in VALIDATION SET
  for(unsigned int i=0;i<nb_val_ex;i++){
    for(unsigned int j=0;j<nb_inputs;j++)
      validation_data->input[i][j]=data.input[i+nb_train_ex][j];
    for(unsigned int j=0;j<nb_outputs;j++)
      validation_data->output[i][j]=data.output[i+nb_train_ex][j];
  }
  // fill in TEST SET
  for(unsigned int i=0;i<nb_test_ex;i++){
    for(unsigned int j=0;j<nb_inputs;j++)
      test_data->input[i][j]=data.input[i+nb_train_ex+nb_val_ex][j];
    for(unsigned int j=0;j<nb_outputs;j++)
      test_data->output[i][j]=data.output[i+nb_train_ex+nb_val_ex][j];
  }
}

unsigned int fann_get_nb_hidden_units(fann* best_ann, unsigned int nb_hid_layers){
  unsigned int count=0;
  unsigned int* layer_array=(unsigned int*)malloc(sizeof(unsigned int)*nb_hid_layers);
  fann_get_layer_array(best_ann,layer_array);
  count=layer_array[1];
  free(layer_array);
  return count;
}

void multiclass_fixed_training_task(unsigned int i, unsigned int nb_reps,unsigned int gens,vector<mat> &res_mats_training_perfs, exp_details ef){
  // result matrices (to be interpreted by Octave script <Plotter.m>)
  mat results_score_evolution;

  cout << endl
       << "***"
       << "\tRUNNING REPLICATE " << i+1 << "/" << nb_reps << "\t "
       << "DATA: " << ef.dataset_filename << endl;

  // set seed
  unsigned int seed=i*10;
  std::srand(seed);

  unsigned int nb_bp_searches=5;
  mat res_mat;

  // load data set
  struct fann_train_data *data=fann_read_train_from_file(ef.dataset_filename.c_str());
  // randomize examples order
  fann_shuffle_train_data(data);

  unsigned int nb_examples=data->num_data;
  unsigned int nb_inputs=data->num_input;
  unsigned int nb_outputs=data->num_output;
  // define data subset proportions
  unsigned int nb_train_ex=nb_examples*60/100;
  unsigned int nb_val_ex=nb_examples*20/100;
  unsigned int nb_test_ex=nb_examples*20/100;

  // allocating memory for all subsets
  fann_train_data*train_data=fann_instantiate_data(nb_train_ex,nb_inputs,nb_outputs);
  fann_train_data*val_data=fann_instantiate_data(nb_val_ex,nb_inputs,nb_outputs);
  fann_train_data*test_data=fann_instantiate_data(nb_test_ex,nb_inputs,nb_outputs);

  // execute segmentation of TRAINING, VALIDATION and TEST sets
  fann_separate_data(*data,train_data,val_data,test_data);
  cout<<endl;

  const unsigned int nb_layers=1;
  const unsigned int nb_hid_units=10;
  const float desired_error=(const float) 0.00f;
  const unsigned int epochs_between_reports=1;

  // encode topology description in array
  vector<unsigned int> desc_vec;
  desc_vec.push_back(nb_inputs);
  for(unsigned int i=0;i<nb_layers;i++)
    desc_vec.push_back(nb_hid_units);
  desc_vec.push_back(nb_outputs);
  // cast to array
  unsigned int *desc_array=&desc_vec[0];

  fann_type min_weight=-1;
  fann_type max_weight=+1;

  unsigned int nb_classes=train_data->num_output;

  // instantiate net (total NB layers = nb hid layers + input & output layer)
  struct fann* ann=fann_create_standard_array(nb_layers+2,desc_array);
  struct fann* best_ann=fann_create_standard_array(nb_layers+2,desc_array);

  fann_randomize_weights(ann,min_weight,max_weight);
  fann_randomize_weights(best_ann,min_weight,max_weight);
  fann_train_epoch(best_ann,train_data);

  fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
  fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);

  // initiate training
  for(unsigned int b=0;b<nb_bp_searches;b++){
    //reset ann
    fann_randomize_weights(ann,min_weight,max_weight);
    for(unsigned int i=0;i<gens;i++){
      fann_train_epoch(ann,train_data);

      double err=0;
      mat line;
      unsigned int epoch=i;
      double pop_score_mean=-1;
      double pop_score_variance=-1;
      double pop_score_stddev=-1;
      double prediction_accuracy=0;//fann_get_accuracy(best_ann, data);
      double score=-1;//fann_get_f1_score(best_ann);
      double validation_accuracy=-1;
      double validation_score=-1;
      unsigned int nb_hid_layers_best=fann_get_num_layers(best_ann); // NB LAYERS IS FIXED
      unsigned int nb_hid_units_best=nb_hid_units; // NB UNITS IS FIXED
      double MSE=fann_get_MSE(best_ann);
      if(nb_classes==1) nb_classes=2;

      mat preds,labels;
      fann_get_preds_labels(best_ann, train_data,preds,labels);
      mat conf_mat=generate_conf_mat(nb_classes,preds,labels);
      compute_error_acc_score(conf_mat, labels, err, prediction_accuracy, score);
      
      std::cout<<"epoch="<<epoch<<"of"<<gens
	       <<"\tBP"<<b
	       <<"\tscore="<<score
	       <<"\tacc="<<prediction_accuracy
	       <<"\tmse="<<MSE
	       <<"\tNB.hid.units="<<nb_hid_units_best
	       <<"\tNB.hid.layers="<<nb_hid_layers_best
	       <<endl;

      // format result line (-1 corresponds to irrelevant attributes)
      line << epoch
	   << MSE
	   << prediction_accuracy
	   << score
	   << pop_score_variance

	   << pop_score_stddev
	   << pop_score_mean
	   << -1//pop_score_median
	   << -1//pop->organisms.size()
	   << nb_inputs // inputs

	   << nb_hid_units_best
	   << nb_outputs//outputs
	   << nb_hid_layers_best
	   << true
	   << -1//selected_mutation_scheme

	   << -1//ensemble_accuracy
	   << -1//ensemble_score
	   << validation_accuracy
	   << validation_score
	   << epoch//nb_calls_err_func

	   << endr;
      // Write results on file
      res_mat=join_vert(res_mat,line);

      // memorize best network
      if(fann_get_MSE(ann)<fann_get_MSE(best_ann))
	best_ann=fann_copy(ann);
    }
    cout<<endl;
  }
  results_score_evolution=join_vert(results_score_evolution, res_mat);

  double test_score=0;
  double test_acc=0;
  double test_err=0;
  // compute ACC and SCORE on TEST SET
  mat test_preds,test_labels;
  fann_get_preds_labels(best_ann, test_data,test_preds,test_labels);
  mat test_conf_mat=generate_conf_mat(nb_classes,test_preds,test_labels);
  compute_error_acc_score(test_conf_mat, test_labels, test_err, test_acc, test_score);
  cout<<"Performances on test set: ACC="<<test_acc<<"\tSCORE="<<test_score<<"\tERR="<<test_err
      <<endl
      <<endl;

  // clean-up memory
  fann_destroy(ann);
  fann_destroy(best_ann);

  for(unsigned int i=0; i<train_data->num_data;i++){
    free(train_data->input[i]);
    free(train_data->output[i]);
  }
  free(train_data->input);
  free(train_data->output);
  free(train_data);

  for(unsigned int i=0; i<val_data->num_data;i++){
    free(val_data->input[i]);
    free(val_data->output[i]);
  }
  free(val_data->input);
  free(val_data->output);
  free(val_data);

  for(unsigned int i=0; i<test_data->num_data;i++){
    free(test_data->input[i]);
    free(test_data->output[i]);
  }
  free(test_data->input);
  free(test_data->output);
  free(test_data);

  // append Cross Validation error to result matrix
  mat test_score_m=ones(results_score_evolution.n_rows,1) * test_score;
  mat test_acc_m  =ones(results_score_evolution.n_rows,1) * test_acc;
  results_score_evolution=join_horiz(results_score_evolution, test_score_m);
  results_score_evolution=join_horiz(results_score_evolution, test_acc_m);
  
  // print-out best perfs
  double best_score = results_score_evolution(results_score_evolution.n_rows-1, 3);
  res_mats_training_perfs.push_back(results_score_evolution);

  ofstream experiment_file("random-seeds.txt",ios::app);
  cout           <<"THREAD"<<omp_get_thread_num()<<" replicate="<<i<<"\tseed="<<seed<<"\tbest_score="<<"\t"<<best_score<<" on "<<ef.dataset_filename<<endl;
  experiment_file<<"THREAD"<<omp_get_thread_num()<<" replicate="<<i<<"\tseed="<<seed<<"\tbest_score="<<"\t"<<best_score<<" on "<<ef.dataset_filename<<endl;
  experiment_file.close();
}


unsigned int count_nb_identicals(unsigned int predicted_class, unsigned int expected_class, mat predictions, mat expectations){
  unsigned int count=0;
  // for each example
  for(unsigned int i=0; i<predictions.n_rows; i++)
    if(predictions(i)==predicted_class && expectations(i)==expected_class)
      count++;
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
  for(unsigned int i=0; i<results.size(); i++)
    processed_results.push_back( (mat)results[i].rows(0, smallest_nb_rows-1));
  // sum up results
  for(unsigned int i=0; i<results.size(); i++)
    total_results += processed_results[i];
  mat averaged_results = total_results / results.size();
  return averaged_results;
}

mat compute_replicate_error(unsigned int nb_reps,vector<mat> results){
  // return variable
  mat err_vec;
  unsigned int smallest_nb_rows = INT_MAX;
  // find lowest and highest nb rows
  for(unsigned int i=0; i<results.size() ;i++)
    if(results[i].n_rows < smallest_nb_rows)
      smallest_nb_rows = results[i].n_rows;
  mat best_scores;
  // for each generation
  for(unsigned int i=0; i<smallest_nb_rows; i++) {
    best_scores.reset();
    // for each replica: get best score
    for(unsigned int r=0; r<nb_reps; r++)
      best_scores = join_vert(best_scores, to_matrix(results[r](i,3)));
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
    for(unsigned int j=0; j<recorded_performances.n_cols; ++j)
      result_file << recorded_performances(i,j) << " ";
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

void multiclass_cascade_training_task(unsigned int i, unsigned int nb_reps,unsigned int gens,vector<mat> &res_mats_training_perfs, exp_details ef){
  // result matrices (to be interpreted by Octave script <Plotter.m>)
  mat results_score_evolution;
  mat res_mat;
  unsigned int nb_opt_algs=5;
  
  cout << endl
       << "***"
       << "\tRUNNING REPLICATE " << i+1 << "/" << nb_reps << "\t "
       << "DATA: " << ef.dataset_filename << endl;

  // set seed
  unsigned int seed=i*10;
  std::srand(seed);

  // load data set
  struct fann_train_data *data=fann_read_train_from_file(ef.dataset_filename.c_str());
  // randomize examples order
  fann_shuffle_train_data(data);

  unsigned int nb_examples=data->num_data;
  unsigned int nb_inputs=data->num_input;
  unsigned int nb_outputs=data->num_output;
  // define data subset proportions
  unsigned int nb_train_ex=nb_examples*60/100;
  unsigned int nb_val_ex=nb_examples*20/100;
  unsigned int nb_test_ex=nb_examples*20/100;

  // allocating memory for all subsets
  fann_train_data*train_data=fann_instantiate_data(nb_train_ex,nb_inputs,nb_outputs);
  fann_train_data*val_data=fann_instantiate_data(nb_val_ex,nb_inputs,nb_outputs);
  fann_train_data*test_data=fann_instantiate_data(nb_test_ex,nb_inputs,nb_outputs);
  // execute segmentation of TRAINING, VALIDATION and TEST sets
  fann_separate_data(*data,train_data,val_data,test_data);

  unsigned int nb_hid_layers=1;
  unsigned int nb_hid_units=1;

  // encode topology description in array
  vector<unsigned int> desc_vec;
  desc_vec.push_back(nb_inputs);
  for(unsigned int i=0;i<nb_hid_layers;i++)
    desc_vec.push_back(nb_hid_units);
  desc_vec.push_back(nb_outputs);
  // cast to array
  unsigned int *desc_array=&desc_vec[0];

  const float desired_error=(const float) 0.00f;
  const unsigned int epochs_between_reports=1;

  fann_type min_weight=-1;
  fann_type max_weight=+1;

  unsigned int nb_classes=train_data->num_output;

  // instantiate net (total NB layers = nb hid layers + input & output layer)
  struct fann* ann=fann_create_standard_array(nb_hid_layers+2,desc_array);
  struct fann* best_ann=fann_create_standard_array(nb_hid_layers+2,desc_array);

  fann_randomize_weights(ann,min_weight,max_weight);
  fann_randomize_weights(best_ann,min_weight,max_weight);
  fann_train_epoch(best_ann,train_data);

  fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
  fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);

  // Layer Growth Factor (<LGF> times the number of input attributes)
  unsigned int LGF=2;

  // initiating training
  for(unsigned int b=0;b<nb_opt_algs;b++){
    cout<<"nb hid layers="<<nb_hid_layers<<endl;
    // increase depth and size
    nb_hid_units++;
    if(nb_hid_units==(nb_inputs*LGF) && b!=0){
      nb_hid_layers++;
      cout<<"INCREASING NB HID LAYERS TO "<<nb_hid_layers<<endl;
    }

    // encode topology description in array
    vector<unsigned int> desc_vec;
    desc_vec.push_back(nb_inputs);
    for(unsigned int i=0;i<nb_hid_layers;i++)
      desc_vec.push_back(nb_hid_units);
    desc_vec.push_back(nb_outputs);
    // cast to array
    unsigned int *desc_array=&desc_vec[0];
    // instantiate model
    ann=fann_create_standard_array(nb_hid_layers+2,desc_array); // total NB layers = nb hid layers + input & output layer

    // use sigmoid activation function for all neurons
    fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
    fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);

    cout<<"Net topology:"<<endl
	<<"\tNb inputs            = "<<fann_get_num_input(ann)<<endl
	<<"\tNb hid. units        = "<<nb_hid_units<<endl
	<<"\tNb outputs           = "<<fann_get_num_output(ann)<<endl
	<<"\tNb hid. layers       = "<<nb_hid_layers<<endl;

    for(unsigned int i=0;i<gens;i++){
      fann_train_epoch(ann,train_data);

      double err=0;
      mat line;
      unsigned int epoch=i;
      double pop_score_mean=-1;
      double pop_score_variance=-1;
      double pop_score_stddev=-1;
      double prediction_accuracy=0;//fann_get_accuracy(best_ann, train_data);
      double score=-1;//fann_get_f1_score(best_ann);
      double validation_accuracy=-1;
      double validation_score=-1;
      unsigned int nb_hid_layers_best=fann_get_num_layers(best_ann)-2;
      unsigned int nb_hid_units_best=fann_get_nb_hidden_units(best_ann,nb_hid_layers_best);

      double MSE=fann_get_MSE(best_ann);

      if(nb_classes==1) nb_classes=2;
      mat preds,labels;
      fann_get_preds_labels(best_ann, train_data,preds,labels);
      mat conf_mat=generate_conf_mat(nb_classes,preds,labels);
      compute_error_acc_score(conf_mat, labels, err, prediction_accuracy, score);

      cout<<"epoch="<<epoch<<"of"<<gens
	  <<"\tBP"<<b
	  <<"\tscore="<<score
	  <<"\tacc="<<prediction_accuracy
	  <<"\tmse="<<MSE
	  <<"\tNB.hid.units="<<nb_hid_units_best
	  <<"\tNB.hid.layers="<<nb_hid_layers_best
	  <<endl;

      // format result line (-1 corresponds to irrelevant attributes)
      line << epoch
	   << MSE
	   << prediction_accuracy
	   << score
	   << pop_score_variance

	   << pop_score_stddev
	   << pop_score_mean
	   << -1//pop_score_median
	   << -1//pop->organisms.size()
	   << nb_inputs

	   << nb_hid_units_best
	   << nb_outputs
	   << nb_hid_layers_best
	   << true
	   << -1//selected_mutation_scheme

	   << -1//ensemble_accuracy
	   << -1//ensemble_score
	   << validation_accuracy
	   << validation_score
	   << epoch//nb_calls_err_func

	   << endr;
      // Write results on file
      res_mat=join_vert(res_mat,line);

      // memorize best network
      if(fann_get_MSE(ann)<fann_get_MSE(best_ann))
	best_ann=fann_copy(ann);
    }
    // clear heap
    fann_destroy(ann);
  }
  results_score_evolution=join_vert(results_score_evolution, res_mat);

  double test_score=0;
  double test_acc=0;
  double test_err=0;
  // compute ACC and SCORE on TEST SET
  mat test_preds,test_labels;
  fann_get_preds_labels(best_ann, test_data,test_preds,test_labels);
  mat test_conf_mat=generate_conf_mat(nb_classes,test_preds,test_labels);
  compute_error_acc_score(test_conf_mat, test_labels, test_err, test_acc, test_score);
  cout<<"Performances on test set: ACC="<<test_acc<<"\tSCORE="<<test_score<<"\tERR="<<test_err
      <<endl
      <<endl;

  // clean-up memory
  fann_destroy(best_ann);
  for(unsigned int i=0; i<train_data->num_data;i++){
    free(train_data->input[i]);
    free(train_data->output[i]);
  }
  free(train_data->input);
  free(train_data->output);
  free(train_data);

  for(unsigned int i=0; i<val_data->num_data;i++){
    free(val_data->input[i]);
    free(val_data->output[i]);
  }
  free(val_data->input);
  free(val_data->output);
  free(val_data);

  for(unsigned int i=0; i<test_data->num_data;i++){
    free(test_data->input[i]);
    free(test_data->output[i]);
  }
  free(test_data->input);
  free(test_data->output);
  free(test_data);

  // append Cross Validation error to result matrix
  mat test_score_m=ones(results_score_evolution.n_rows,1) * test_score;
  mat test_acc_m  =ones(results_score_evolution.n_rows,1) * test_acc;
  results_score_evolution=join_horiz(results_score_evolution, test_score_m);
  results_score_evolution=join_horiz(results_score_evolution, test_acc_m);

  // print-out best perfs
  double best_score = results_score_evolution(results_score_evolution.n_rows-1, 3);
  res_mats_training_perfs.push_back(results_score_evolution);

  ofstream experiment_file("random-seeds.txt",ios::app);
  cout           <<"THREAD"<<omp_get_thread_num()<<" replicate="<<i<<"\tseed="<<seed<<"\tbest_score="<<"\t"<<best_score<<" on "<<ef.dataset_filename<<endl;
  experiment_file<<"THREAD"<<omp_get_thread_num()<<" replicate="<<i<<"\tseed="<<seed<<"\tbest_score="<<"\t"<<best_score<<" on "<<ef.dataset_filename<<endl;
  experiment_file.close();
}
