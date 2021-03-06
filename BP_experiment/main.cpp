#include<iostream>
#include<string>
#include<sstream>
#include<vector>
#include<fann.h>
#include<armadillo>
#include<omp.h>
#include <iomanip> // setprecision
using namespace std;
using namespace arma;

// Comment to display evolution on screen
#define NO_SCREEN_OUT

enum EXP_TYPE{FIXED, CASCADE};

struct exp_files{
  vector<string> dataset_filenames;
  string result_file;
  string current_ds;
  unsigned int max_nb_err_func_calls;
  unsigned int nb_reps;
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
void experiment(exp_files ef);
mat compute_learning_curves_perfs(vector<mat> &result_matrices_training_perfs, exp_files ef);
struct fann_train_data* fann_instantiate_data(unsigned int nb_examples,unsigned int nb_inputs,unsigned int nb_outputs);
void fann_separate_data(fann_train_data data,fann_train_data* training_data,fann_train_data* validation_data, fann_train_data* test_data);
unsigned int fann_get_nb_hidden_units(fann* best_ann, unsigned int nb_hid_layers);
void multiclass_cascade_training_task(unsigned int i, exp_files ef);
unsigned int count_nb_classes(mat labels);
mat generate_conf_mat(unsigned int nb_classes, mat preds, mat labels);
void compute_mse_acc_score(mat preds, mat labels,unsigned int nb_classes,double& mse,double& accuracy,double& fitness);

unsigned int count_nb_identicals(unsigned int predicted_class, unsigned int expected_class, mat predictions, mat expectations);
mat to_multiclass_format(mat predictions);
mat average_matrices(vector<mat> results);
mat compute_replicate_error(unsigned int nb_reps,vector<mat> results);
void print_results_octave_format(ofstream &result_file, mat recorded_performances, string octave_variable_name);
mat to_matrix(double a);
double corrected_sample_std_dev(mat score_vector);
const string get_current_date_time();

exp_files read_args(int argc, char** argv){
  if(argc<(3+1)) {
    cout<<"At least 3 arguments expected."<<endl;
    cout<<"Arg1: 1 or more data sets."<<endl;
    cout<<"Arg2: Nb of replicates."<<endl;
    cout<<"Arg3: Max number of calls to the error function."<<endl;
  }else{
    exp_files ef;
    unsigned int nb_ds=argc-(2+1);
    for(unsigned int i=0;i<nb_ds;i++){
      string s(argv[i+1]);
      // replace .CSV extension by .DATA
      s=s.substr(0,s.size()-3);
      s+="data";
      // save filename
      ef.dataset_filenames.push_back(s);
    }
    ef.nb_reps=std::atoi(argv[nb_ds+1]);
    ef.max_nb_err_func_calls=std::atoi(argv[nb_ds+2]);
    return ef;
  }
}

/**
   Experiment: Running multiple instances of a neural network
   trained using Gradient Descent and the Back Propagation algorithm.
   Author: Hichame Moriceau
   
   Compile: # g++ -std=c++11 main.cpp -fopenmp -larmadillo -lfann -o runme
   Run    : # ./runme data/breast-cancer-malignantOrBenign-data-transformed.csv 1 500 # 1 replicate 500 epochs
*/
int main(int argc, char * argv []){
  // fetch & save CLI args
  exp_files ef=read_args(argc,argv);
  
  // for each data set
  for(unsigned int i=0;i<ef.dataset_filenames.size();i++){
    // set data set filename
    ef.current_ds=ef.dataset_filenames[i];
    // construct appropriate result filename
    ef.result_file=ef.dataset_filenames[i].substr(0,ef.dataset_filenames[i].size()-5);
    replace(ef.result_file.begin(),ef.result_file.end(),'-','_');
    ef.result_file+="_results/BPcascade-results.mat";
    // run experiment
    experiment(ef);
  }

  return 0;
}

// generates random integer within range
int rand_int(int min, int max){
  return min + (rand() % (int)(max - min + 1));
}

void experiment(exp_files ef){
  std::ofstream oFile(ef.result_file.c_str(),std::ios::out);
  vector<mat> res_mats_training_perfs;
  mat avrg_mat=compute_learning_curves_perfs(res_mats_training_perfs,ef);
  mat res_mat_err = compute_replicate_error(ef.nb_reps,res_mats_training_perfs);
  // save results
  avrg_mat = join_horiz(avrg_mat, ones(avrg_mat.n_rows,1) * ef.nb_reps);
  // plot results
  print_results_octave_format(oFile,avrg_mat,"results");
  print_results_octave_format(oFile,res_mat_err,"err_results");
}

mat compute_learning_curves_perfs(vector<mat> &result_matrices_train_perfs, exp_files ef){
  // return variable
  mat averaged_performances;
  // for each replicate
#pragma omp parallel
  {
    // kick off a single thread
#pragma omp single
    {
      for(unsigned int i=0; i<ef.nb_reps; ++i) {
#pragma omp task
	{
	  multiclass_cascade_training_task(i,ef);
	}
      }
    }
  }

  cout<<"gathering results"<<endl;
  // aggregate replicates results
  for(unsigned int i=0;i<ef.nb_reps;i++){
    mat r;
    r.load("res"+to_string(i)+".mat");
    result_matrices_train_perfs.push_back(r);
  }

  // clean-up auto generated result files
  for(unsigned int i=0;i<ef.nb_reps;i++)
    std::remove(("res"+to_string(i)+".mat").c_str());

  // average PERFS of all replicates
  averaged_performances = average_matrices(result_matrices_train_perfs);
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

void compute_mse_acc_score(mat preds, mat labels,unsigned int nb_classes,double& mse,double& accuracy,double& fitness){
  mat conf_mat=generate_conf_mat(nb_classes,preds,labels);
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
  if(computed_score==0) computed_score=0.1;
  // compute accuracy
  double TP =0;
  for(unsigned int i=0; i<nb_classes; i++){
    TP += conf_mat(i,i);
  }
  computed_acc = (TP/double(nb_examples))*100;
  // prevent -nan values
  if(computed_acc!=computed_acc)
    computed_acc=0;
  // compute mse
  mse=as_scalar(sum(square(preds-labels)))/double(nb_examples);
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
  // allocate 2D arrays attributes of struct
  data->input=(fann_type**) malloc(nb_examples*sizeof(fann_type *));
  data->output=(fann_type **) malloc(nb_examples*sizeof(fann_type *));
  for(unsigned int i=0;i<nb_examples;i++)
    data->input[i]=(fann_type *) malloc(nb_inputs*sizeof(fann_type));
  for(unsigned int i=0;i<nb_examples;i++)
    data->output[i]=(fann_type *) malloc(nb_outputs*sizeof(fann_type));
  return data;
}

void standardize(mat &D){
    // for each column
    for(unsigned int i=0; i<D.n_rows; ++i){
        // for each element in this column (do not standardize target attribute)
        for(unsigned int j=0; j<D.n_cols; ++j){
            // apply feature scaling and mean normalization
            D(i,j) = (D(i,j) - mean(D.col(j))) / (max(D.col(j))-min(D.col(j)));
        }
    }
}

mat array2D_to_mat(fann_type** data, unsigned int height,unsigned int width){
  mat m(height,width);
  for(unsigned int i=0;i<height;i++)
    for(unsigned int j=0;j<width;j++)
      m(i,j)=data[i][j];
  return m;
}

fann_type** mat_to_array2D(mat d){
  //double array[d.n_rows][d.n_cols];
  fann_type** array=0;
  array=new fann_type*[d.n_rows];
  for(unsigned int i=0;i<d.n_rows;i++){
    array[i]= new fann_type[d.n_cols];
    for(unsigned int j=0;j<d.n_cols;j++){
      array[i][j]=d(i,j);
    }
  }
  return array;
}

void fann_separate_data(fann_train_data data,fann_train_data* training_data,fann_train_data* validation_data, fann_train_data* test_data){
  unsigned int nb_inputs=training_data->num_input;
  unsigned int nb_outputs=training_data->num_output;
  unsigned int nb_train_ex=training_data->num_data;
  unsigned int nb_val_ex=validation_data->num_data;
  unsigned int nb_test_ex=test_data->num_data;  
  // data pre-processing: apply feature scaling
  mat d=array2D_to_mat(data.input, data.num_data, nb_inputs);
  standardize(d);
  data.input=mat_to_array2D(d);
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

void multiclass_cascade_training_task(unsigned int i, exp_files ef){
  // result matrices (to be interpreted by Octave script <Plotter.m>)
  mat results_score_evolution;
  mat res_mat;
  unsigned int iter=ef.max_nb_err_func_calls;
  unsigned int nb_opt_algs=5;

  cout << endl
       << "***"
       << "\tRUNNING REPLICATE " << i+1 << "/" << ef.nb_reps << "\t "
       << "DATA: " << ef.current_ds << endl;

  // set seed
  unsigned int seed=i*10;
  std::srand(seed);

  // load data set
  struct fann_train_data *data=fann_read_train_from_file(ef.current_ds.c_str());
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
  fann_type min_weight=-1,max_weight=+1;

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

  iter/=nb_opt_algs;
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
    // instantiate model (total NB layers = nb hid layers + input & output layer)
    ann=fann_create_standard_array(nb_hid_layers+2,desc_array);

    // use sigmoid activation function for all neurons
    fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
    fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);

    cout<<"Net topology:"<<endl
	<<"\tNb inputs            = "<<fann_get_num_input(ann)<<endl
	<<"\tNb hid. units        = "<<nb_hid_units<<endl
	<<"\tNb outputs           = "<<fann_get_num_output(ann)<<endl
	<<"\tNb hid. layers       = "<<nb_hid_layers<<endl;

    for(unsigned int i=0;i<iter;i++){
      fann_train_epoch(ann,train_data);

      mat line;
      unsigned int epoch=i+b*iter;
      double pop_score_mean=0,pop_score_var=0,pop_score_stddev=0;
      double train_acc=0,train_score=0;
      double val_acc=0,val_score=0;
      double test_acc=0,test_score=0;
      double val_cur_score=0, val_cur_accuracy=0;
      double train_mse=0,val_mse=0,test_mse=0, val_cur_mse=0;

      unsigned int nb_hid_layers_best=fann_get_num_layers(best_ann)-2;
      unsigned int nb_hid_units_best=fann_get_nb_hidden_units(best_ann,nb_hid_layers_best);

      if(nb_classes==1) nb_classes=2;
      // calculate BEST perfs on TRAINING set
      mat preds,labels;
      fann_get_preds_labels(best_ann, train_data,preds,labels);
      compute_mse_acc_score(preds,labels,nb_classes,train_mse,train_acc,train_score);
      // calculate BEST perfs on VALIDATION set
      mat val_preds,val_labels;
      fann_get_preds_labels(best_ann, val_data,val_preds,val_labels);
      compute_mse_acc_score(val_preds,val_labels,nb_classes,val_mse, val_acc,val_score);
      // calculate BEST perfs on TEST set
      mat test_preds,test_labels;
      fann_get_preds_labels(best_ann, test_data,test_preds,test_labels);
      compute_mse_acc_score(test_preds, test_labels,nb_classes, test_mse, test_acc, test_score);
      // calculate CURRENT on VALIDATION set
      mat val_cur_preds,val_cur_labels;
      fann_get_preds_labels(ann, val_data,val_cur_preds,val_cur_labels);
      compute_mse_acc_score(val_cur_preds, val_cur_labels,nb_classes,val_cur_mse, val_cur_accuracy,val_cur_score);

      // format result line (-1 corresponds to irrelevant attributes)
      line << epoch // = nb_err_func_calls
	   << iter

	   << train_acc
	   << train_score
	   << train_mse

	   << test_acc
	   << test_score
	   << test_mse

	   << val_acc
	   << val_score
	   << val_mse

	   << -1 // instead of: population fitness variance
	   << -1 // instead of: population fitness mean
	   << -1 // instead of: population size

	   << nb_hid_units
	   << nb_hid_layers

	   << endr;

#ifndef NO_SCREEN_OUT
      cout << fixed
	   << setprecision(2)
	   <<"NB.err.func.calls="<<epoch<<"\t"
	//<<"gen="<<epoch<<"\t" // (irrelevant for BP)
	   <<"train.mse="<<train_mse<<"\t"
	   <<"val.mse="<<val_mse<<"\t"
	   <<"test.mse="<<test_mse<<"\t"
	//<<"pop.fit.mean="<<-1<<"\t" // (irrelevant for BP)
	   <<"NB.hid.units="<<nb_hid_units<<"\t"
	   <<"NB.hid.layers="<<nb_hid_layers<<"\t"
	   << endl;
#endif

      // Write results on file
      res_mat=join_vert(res_mat,line);

      // memorize network that best performs on VALIDATION set
      if(val_cur_mse<val_mse)
	best_ann=fann_copy(ann);
    }
    // clear heap
    fann_destroy(ann);
  }
  results_score_evolution=join_vert(results_score_evolution, res_mat);

    // save obtained results locally (binary format)
#pragma omp critical
  results_score_evolution.save("res"+to_string(i)+".mat");

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

  /*
  cout<<"deleting"<<endl;
  for(unsigned int i=0;i<data->num_data;i++)
    delete []data->input[i];
  delete []data->input;

  for(unsigned int i=0;i<data->num_data;i++)
    delete []data->output[i];
  delete []data->output;
  cout<<"deleted"<<endl;
  */

  // print-out best perfs
  double best_score = results_score_evolution(results_score_evolution.n_rows-1, 3);

  ofstream experiment_file("random-seeds.txt",ios::app);
  cout           <<"THREAD"<<omp_get_thread_num()<<" replicate="<<i<<"\tseed="<<seed<<"\tbest_score="<<"\t"<<best_score<<" on "<<ef.current_ds<<endl;
  experiment_file<<"THREAD"<<omp_get_thread_num()<<" replicate="<<i<<"\tseed="<<seed<<"\tbest_score="<<"\t"<<best_score<<" on "<<ef.current_ds<<endl;
  experiment_file.close();
}
