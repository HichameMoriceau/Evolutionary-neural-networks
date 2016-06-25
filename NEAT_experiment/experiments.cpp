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
#include "experiments.h"
#include <cstring>
#include <omp.h>

#define NO_SCREEN_OUT 

void multiclass_test(exp_files ef){
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

void multiclass_epoch(Population* pop,int generation,Organism& best_org, mat &res_mat,unsigned int& nb_calls_err_func, exp_files ef){
  vector<Organism*>::iterator curorg;
  //Evaluate each organism on a test
  for(curorg=(pop->organisms).begin();curorg!=(pop->organisms).end();++curorg)
    multiclass_evaluate(*curorg,ef.dataset_filename,res_mat, nb_calls_err_func, pop,generation, best_org);
  pop->epoch(generation);
}

void multiclass_evaluate(Organism *org, string dataset_filename, mat &res_mat,unsigned int& nb_calls_err_func, Population *pop, unsigned int generation, Organism& best_org) {
  Network *net;
  // update count of calls to the erro function
  nb_calls_err_func++;

  vector<Organism*>::iterator curorg;
  vector<Species*>::iterator curspecies;
  unsigned int nb_examples=-1,nb_attributes=-1;
  double** data=load_data_array(dataset_filename,nb_examples, nb_attributes);
  unsigned int training_height=nb_examples*(double(60)/100);
  unsigned int validation_height=nb_examples*(double(20)/100);
  unsigned int test_height=nb_examples*(double(20)/100);  
  double** training_data=0;
  double** validation_data=0;
  double** test_data=0;  
  net=org->net;
  training_data=new double*[training_height];
  // populate <training_data>
  for(unsigned int i=0;i<training_height;i++){
    training_data[i]=new double[nb_attributes];
    for(unsigned int j=0;j<nb_attributes;j++)
      training_data[i][j]=data[i][j];
  }

  validation_data=new double*[validation_height];
  // populate <validation_data>
  for(unsigned int i=0;i<validation_height;i++){
    validation_data[i]=new double[nb_attributes];
    for(unsigned int j=0;j<nb_attributes;j++)
      validation_data[i][j]=data[training_height+i][j];
  }

  test_data=new double*[test_height];
  // populate <test_data>
  for(unsigned int i=0;i<test_height;i++){
    test_data[i]=new double[nb_attributes];
    for(unsigned int j=0;j<nb_attributes;j++)
      test_data[i][j]=data[training_height+validation_height+i][j];
  }

  // training data is used last for org->error to take precedence over previous runs
  evaluate_perfs(test_data      ,test_height,      nb_attributes,net,org->error, org->test_fitness,       org->test_accuracy);
  evaluate_perfs(validation_data,validation_height,nb_attributes,net,org->error, org->validation_fitness, org->validation_accuracy);
  evaluate_perfs(training_data  ,training_height,  nb_attributes,net,org->error, org->training_fitness,   org->training_accuracy); 
  org->fitness=org->training_fitness;

  std::vector<double> all_fitnesses;
  // find fittest
  for(curorg=(pop->organisms).begin(); curorg!=(pop->organisms).end(); curorg++){
    if((*curorg)->fitness>best_org.fitness){
      // memorize organism
      best_org.error=(*curorg)->error;
      best_org.training_accuracy=(*curorg)->training_accuracy;
      best_org.training_fitness=(*curorg)->training_fitness;
      best_org.validation_accuracy=(*curorg)->validation_accuracy;
      best_org.validation_fitness=(*curorg)->validation_fitness;
      best_org.test_accuracy=(*curorg)->test_accuracy;
      best_org.test_fitness=(*curorg)->test_fitness;
      best_org.fitness=(*curorg)->fitness;
    }
    // memorize organism's fitness
    all_fitnesses.push_back((*curorg)->training_fitness);
  }

  mat line;
  double pop_score_mean=pop->mean(all_fitnesses);
  double pop_score_variance=pop->var(all_fitnesses);
  double pop_score_stddev=pop->stddev(all_fitnesses);
  double prediction_accuracy=best_org.training_accuracy;
  double score=best_org.fitness;
  double validation_accuracy=best_org.validation_accuracy;
  double validation_score=best_org.validation_fitness;
  double MSE=best_org.error;
  unsigned int hidden_units=((best_org.gnome)->nodes).size();

#ifndef NO_SCREEN_OUT
  std::cout<<"NB.err.func.calls="<<nb_calls_err_func
	   <<"\tbest.indiv.fitness="<<score
	   <<"\tacc="<<prediction_accuracy
	   <<"\terr="<<MSE
	   <<"\tNB nodes="<<hidden_units
	   <<"\tpop.mean="<<pop_score_mean
	   <<"\tpop.var="<<pop_score_variance
	   <<"\tpop.stddev="<<pop_score_stddev<<endl;
#endif

  // format result line (-1 = irrelevant attributes)
  line << nb_calls_err_func
       << MSE
       << prediction_accuracy
       << score
       << pop_score_variance

       << pop_score_stddev
       << pop_score_mean
       << -1//pop_score_median
       << pop->organisms.size()
       << best_org.net->inputs.size() // inputs

       << hidden_units
       << best_org.net->outputs.size()//outputs
       << -1//nb_hidden_layers
       << true
       << 0//selected_mutation_scheme

       << -1//ensemble_accuracy
       << -1//ensemble_score
       << validation_accuracy
       << validation_score
       << generation

       << endr;
  // Write results on file
  res_mat=join_vert(res_mat,line);  
  // clean-up memory
  for (unsigned int i=0; i<nb_examples; i++)
    delete [] data[i];
  delete [] data;

  for (unsigned int i=0; i<training_height; i++)
    delete [] training_data[i];
  delete [] training_data;

  for (unsigned int i=0; i<validation_height; i++)
    delete [] validation_data[i];
  delete [] validation_data;

  for (unsigned int i=0; i<test_height; i++)
    delete [] test_data[i];
  delete [] test_data;
}

void evaluate_perfs(double** data,
		    unsigned int nb_examples,
		    unsigned int nb_attributes_pls_bias,
		    Network* net, 
		    double& error, 
		    double& fitness, 
		    double& accuracy){
  bool success;  //Check for successful activation
  //Load and activate the network on each input
  int net_depth=net->max_depth(); //The max depth of the network to be activated
  int relax; //Activates until relaxation
  unsigned int nb_classes=net->outputs.size();
  double out[nb_examples][nb_classes];

  for(unsigned int count=0;count<nb_examples;count++){
    net->load_sensors(data[count]);
    //Relax net and get output
    success=net->activate();
    //use depth to ensure relaxation
    for (relax=0;relax<=net_depth;relax++){
      success=net->activate();
      (*(net->outputs.begin()))->activation;
    }
    for(unsigned int i=0;i<net->outputs.size(); i++)
      out[count][i]=(*(net->outputs[i])).activation;
    net->flush();
  }
  
  if (success) {
    mat preds =zeros(nb_examples,nb_classes);
    mat labels=zeros(nb_examples,1);

    for(unsigned int i=0; i<nb_examples; i++){
      labels(i)=data[i][nb_attributes_pls_bias-2];
      for(unsigned int j=0;j<nb_classes;j++)
	preds(i,j)=out[i][j];
    }

    // generate confusion matrix
    mat conf_mat=generate_conf_mat(nb_classes,preds,labels);
    compute_error_acc_score(conf_mat, labels,error,accuracy,fitness);
  }
  else {
    //The network is flawed (shouldnt happen)
    error   =999.0;
    fitness =0.0000001;
    accuracy=0.0000001;
  }

}
 
double** load_data_array(string dataset_filename,unsigned int &height,unsigned int &width){
  double** data=0;
  string line, field;
  vector< vector<double> > array;  // the 2D array
  vector<double> v;                // array of values for one line only

  ifstream iFile(dataset_filename.c_str(),ios::in);
  while(getline(iFile,line)){ // get next line in file
    v.clear();
    stringstream ss(line);
    while (getline(ss,field,',')){  // break line into comma delimitted fields
      double d = strtod(field.c_str(), NULL);
      v.push_back(d);  // add each field to the 1D array
    }
    array.push_back(v);  // add the 1D array to the 2D array
  }
  iFile.close();

  // randomize the order of the examples
  random_shuffle(array.begin(),array.end());

  height = array.size();
  width  = array[0].size()+1; // +1 for biasing

  data=new double*[height];
  for(unsigned int i=0; i<height; i++){
    data[i]=new double[width];
    for(unsigned int j=0; j<width; j++){
      if(j==0)
	data[i][j] = 1; // biases
      else
	data[i][j] = array[i][j];
    }
  }
  // return 2D array
  return data;
}

unsigned int count_nb_classes(mat labels){
  vector<unsigned int> array;
  bool is_known_class = false;
  // compute nb output units required
  for(unsigned int i=0; i<labels.n_rows; i++) {
    unsigned int current_pred_class = labels(i);
    is_known_class=false;
    // for each known prediction classes
    for(unsigned int j=0;j<array.size(); j++) {
      // if current output is different from prediction class
      if(current_pred_class==array[j]){
	is_known_class = true;
      }
    }
    if(array.empty() || (!is_known_class)){
      array.push_back(current_pred_class);
    }
  }
  return array.size();
}

mat generate_conf_mat(unsigned int nb_classes, mat preds, mat labels){
  mat conf_mat=zeros(nb_classes, nb_classes);
  for(unsigned int i=0; i<nb_classes; i++)
    for(unsigned int j=0; j<nb_classes; j++)
      conf_mat(i,j)=count_nb_identicals(i,j,to_multiclass_format(preds),labels);
  return conf_mat;
}

void compute_error_acc_score(mat conf_mat, mat labels,double& error,double& accuracy,double& fitness){
  unsigned int nb_classes=conf_mat.n_cols;
  // number of class present in the current subset of the data set
  unsigned int nb_local_classes=count_nb_classes(labels);
  unsigned int nb_examples=0;
  for(unsigned int i=0;i<nb_classes;i++)
    nb_examples+=sum(conf_mat.row(i));
  double computed_score=0, computed_acc=0;
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
  error=(nb_examples-TP)/double(nb_examples);
  accuracy=computed_acc;
  fitness=computed_score;
}

mat compute_learning_curves_perfs(vector<mat> &result_matrices_training_perfs, exp_files ef){
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
	      multiclass_training_task(i,result_matrices_training_perfs,ef);
            }
        }
    }
    cout<<"done computing each replicate"<<endl;
    // average PERFS of all replicates
    averaged_performances = average_matrices(result_matrices_training_perfs);
    return averaged_performances;
}

void standardize(mat &D){
    // for each column
    for(unsigned int i=0; i<D.n_rows; ++i){
        // for each element in this column (do not standardize target attribute)
        for(unsigned int j=0; j<D.n_cols-1; ++j){
            // apply feature scaling and mean normalization
            D(i,j) = (D(i,j) - mean(D.col(j))) / (max(D.col(j))-min(D.col(j)));
        }
    }
}

mat array2D_to_mat(double** data, unsigned int height,unsigned int width){
  mat m(height,width);
  for(unsigned int i=0;i<height;i++)
    for(unsigned int j=0;j<width;j++)
      m(i,j)=data[i][j];
  return m;
}

double** mat_to_array2D(mat d){
  //double array[d.n_rows][d.n_cols];
  double** array=0;
  array=new double*[d.n_rows];
  for(unsigned int i=0;i<d.n_rows;i++){
    array[i]= new double[d.n_cols];
    for(unsigned int j=0;j<d.n_cols;j++){
      array[i][j]=d(i,j);
    }
  }
  return array;
}


void multiclass_training_task(unsigned int i,vector<mat>& res_mats_training_perfs,exp_files ef){
  cout<<"RUNNING multiclass_training_task"<<endl;
  // result matrices (to be interpreted by Octave script <Plotter.m>)
  mat results_score_evolution;

  cout << endl
       << "***"
       << "\tRUNNING REPLICATE " << i+1 << "/" << ef.nb_reps << "\t ";

  // set seed
  unsigned int seed=i*10;
  std::srand(seed);

  Population *pop=0;
  Genome *start_genome;

  mat res_mat;

  ifstream iFile(ef.startgene.c_str(),ios::in);

  cout<<"Reading in the start genome"<<endl;
  //Read in the start Genome
  char curword[20];
  int id;
  iFile>>curword;
  iFile>>id;
  start_genome=new Genome(id,iFile);
  iFile.close();

  //Spawn the Population
  pop=new Population(start_genome,ef.pop_size);
  pop->verify();
  Organism best_org(0.0f,start_genome,1);

  vector<Organism*>::iterator curorg;
  unsigned int nb_examples=-1,nb_attributes=-1;
  double** data=load_data_array(ef.dataset_filename,nb_examples, nb_attributes);

  // data pre-processing: apply feature scaling
  mat d=array2D_to_mat(data, nb_examples, nb_attributes);
  standardize(d);
  data=mat_to_array2D(d);

  unsigned int training_height=nb_examples*(double(60)/100);
  unsigned int validation_height=nb_examples*(double(20)/100);
  unsigned int test_height=nb_examples*(double(20)/100);  
  double** training_data=0;
  double** validation_data=0;
  double** test_data=0;
  
  training_data=new double*[training_height];
  // populate <training_data>
  for(unsigned int i=0;i<training_height;i++){
    training_data[i]=new double[nb_attributes];
    for(unsigned int j=0;j<nb_attributes;j++)
      training_data[i][j]=data[i][j];
  }

  validation_data=new double*[validation_height];
  // populate <validation_data>
  for(unsigned int i=0;i<validation_height;i++){
    validation_data[i]=new double[nb_attributes];
    for(unsigned int j=0;j<nb_attributes;j++)
      validation_data[i][j]=data[training_height+i][j];
  }

  test_data=new double*[test_height];
  // populate <test_data>
  for(unsigned int i=0;i<test_height;i++){
    test_data[i]=new double[nb_attributes];
    for(unsigned int j=0;j<nb_attributes;j++)
      test_data[i][j]=data[training_height+validation_height+i][j];
  }

  for(unsigned int i=0;i<nb_examples;i++)
    delete []data[i];
  delete []data;

  // evaluate population
  for(curorg=(pop->organisms).begin(); curorg!=(pop->organisms).end(); curorg++){    
    // update perfs on each data subset
    evaluate_perfs(test_data      ,test_height,      nb_attributes,(*curorg)->net,(*curorg)->error, (*curorg)->test_fitness,       (*curorg)->test_accuracy);
    evaluate_perfs(validation_data,validation_height,nb_attributes,(*curorg)->net,(*curorg)->error, (*curorg)->validation_fitness, (*curorg)->validation_accuracy);
    evaluate_perfs(training_data  ,training_height,  nb_attributes,(*curorg)->net,(*curorg)->error, (*curorg)->training_fitness,   (*curorg)->training_accuracy); 
    (*curorg)->fitness=(*curorg)->training_fitness;
  }


  unsigned int nb_calls_err_func=0;
  std::vector<double> all_fitnesses;
  // accumulate fitness values
  for(curorg=(pop->organisms).begin(); curorg!=(pop->organisms).end(); curorg++)
    all_fitnesses.push_back((*curorg)->training_fitness);

  for(curorg=(pop->organisms).begin(); curorg!=(pop->organisms).end(); curorg++){
    nb_calls_err_func++;
    mat line;
    double pop_score_mean=pop->mean(all_fitnesses);
    double pop_score_variance=pop->var(all_fitnesses);
    double pop_score_stddev=pop->stddev(all_fitnesses);
    double prediction_accuracy=best_org.training_accuracy;
    double score=best_org.fitness;
    double validation_accuracy=best_org.validation_accuracy;
    double validation_score=best_org.validation_fitness;
    double MSE=best_org.error;
    unsigned int hidden_units=((best_org.gnome)->nodes).size();

#ifndef NO_SCREEN_OUT
    std::cout<<"NB.err.func.calls="<<nb_calls_err_func
	     <<"\tbest.indiv.fitness="<<score
	     <<"\tacc="<<prediction_accuracy
	     <<"\terr="<<MSE
	     <<"\tNB nodes="<<hidden_units
	     <<"\tpop.mean="<<pop_score_mean
	     <<"\tpop.var="<<pop_score_variance
	     <<"\tpop.stddev="<<pop_score_stddev<<endl;
#endif

    // format result line (-1 = irrelevant attributes)
    line << nb_calls_err_func
	 << MSE
	 << prediction_accuracy
	 << score
	 << pop_score_variance

	 << pop_score_stddev
	 << pop_score_mean
	 << -1//pop_score_median
	 << pop->organisms.size()
	 << best_org.net->inputs.size() // inputs

	 << hidden_units
	 << best_org.net->outputs.size()//outputs
	 << -1//nb_hidden_layers
	 << true
	 << -1//selected_mutation_scheme

	 << -1//ensemble_accuracy
	 << -1//ensemble_score
	 << validation_accuracy
	 << validation_score
	 << 0

	 << endr;

    // Write results on file
    res_mat=join_vert(res_mat,line);
  }
  //results_score_evolution=join_vert(results_score_evolution, res_mat);

  unsigned int gen=1;
  
  // train until MAX nb of calls to the error function is reached
  for(;nb_calls_err_func<ef.max_nb_err_func_calls;gen++){
    //multiclass_epoch(pop,gen,best_org,res_mat,nb_calls_err_func,ef);    
    vector<Organism*>::iterator curorg;
    // for each individual: evaluate performances
    for(curorg=(pop->organisms).begin();curorg!=(pop->organisms).end();++curorg){
      multiclass_evaluate(*curorg,ef.dataset_filename,res_mat, nb_calls_err_func, pop,gen, best_org);
      // if MAX NB of calls to the error function is reached
      if(nb_calls_err_func>=ef.max_nb_err_func_calls){
	// exit evaluation loop
	break;
      }
    }
    // train for a single generation
    pop->epoch(gen);
    gen++;
  }

  results_score_evolution=join_vert(results_score_evolution, res_mat);

  double test_score=best_org.test_fitness;
  double test_acc=best_org.test_accuracy;

  cout<<"\ttest score ="<<test_score<<endl;
  cout<<"\ttest acc   ="<<test_acc<<endl;

  // append TEST perfs to result matrix
  mat test_score_m=ones(results_score_evolution.n_rows,1) * test_score;
  mat test_acc_m  =ones(results_score_evolution.n_rows,1) * test_acc;
  results_score_evolution=join_horiz(results_score_evolution, test_acc_m);
  results_score_evolution=join_horiz(results_score_evolution, test_score_m);

  // print-out best perfs
  double best_score = results_score_evolution(results_score_evolution.n_rows-1, 3);
  res_mats_training_perfs.push_back(results_score_evolution);

  ofstream experiment_file("random-seeds.txt",ios::app);
  cout           <<"THREAD"<<omp_get_thread_num()<<" replicate="<<i<<"\tseed="<<seed<<"\tbest_score="<<best_score<<" on "<<ef.dataset_filename<<endl;
  experiment_file<<"THREAD"<<omp_get_thread_num()<<" replicate="<<i<<"\tseed="<<seed<<"\tbest_score="<<best_score<<" on "<<ef.dataset_filename<<endl;
  experiment_file.close();
  delete pop;
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
    for(unsigned int i=0; i<results.size() ;i++)
      if(results[i].n_rows < smallest_nb_rows)
	smallest_nb_rows = results[i].n_rows;
    mat total_results;
    total_results.zeros(smallest_nb_rows, results[0].n_cols);
    // keep only up to the shortest learning curve
    vector<mat> processed_results;
    for(unsigned int i=0; i<results.size(); i++)
      processed_results.push_back( (mat)results[i].rows(0, smallest_nb_rows-1));
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
const string get_current_date_time() {
    time_t     now = time(0);
    struct tm  tstruct;
    char       buffer[80];
    tstruct = *localtime(&now);
    strftime(buffer, sizeof(buffer), "%Y-%m-%d.%X", &tstruct);
    return buffer;
}

//Perform evolution on XOR, for gens generations
Population *xor_test(int gens) {
  Population *pop=0;
  Genome *start_genome;
  char curword[20];
  int id;

  ostringstream *fnamebuf;
  int gen;
 
  int evals[NEAT::num_runs];  //Hold records for each run
  int genes[NEAT::num_runs];
  int nodes[NEAT::num_runs];
  int winnernum;
  int winnergenes;
  int winnernodes;
  //For averaging
  int totalevals=0;
  int totalgenes=0;
  int totalnodes=0;
  int expcount;
  int samples;  //For averaging

  memset (evals, 0, NEAT::num_runs * sizeof(int));
  memset (genes, 0, NEAT::num_runs * sizeof(int));
  memset (nodes, 0, NEAT::num_runs * sizeof(int));

  ifstream iFile("xorstartgenes",ios::in);

  cout<<"START XOR TEST"<<endl;

  cout<<"Reading in the start genome"<<endl;
  //Read in the start Genome
  iFile>>curword;
  iFile>>id;
  cout<<"Reading in Genome id "<<id<<endl;
  start_genome=new Genome(id,iFile);
  iFile.close();

  for(expcount=0;expcount<NEAT::num_runs;expcount++) {
    //Spawn the Population
    cout<<"Spawning Population off Genome2"<<endl;

    pop=new Population(start_genome,NEAT::pop_size);
      
    cout<<"Verifying Spawned Pop"<<endl;
    pop->verify();
      
    for (gen=1;gen<=gens;gen++) {

      //This is how to make a custom filename
      fnamebuf=new ostringstream();
      (*fnamebuf)<<"gen_"<<gen<<ends;  //needs end marker

      /*
#ifndef NO_SCREEN_OUT
      cout<<"name of fname: "<<fnamebuf->str()<<endl;
#endif
      */

      char temp[50];
      sprintf (temp, "gen_%d", gen);

      //Check for success
      if (xor_epoch(pop,gen,temp,winnernum,winnergenes,winnernodes)) {
	//	if (xor_epoch(pop,gen,fnamebuf->str(),winnernum,winnergenes,winnernodes)) {
	//Collect Stats on end of experiment
	evals[expcount]=NEAT::pop_size*(gen-1)+winnernum;
	genes[expcount]=winnergenes;
	nodes[expcount]=winnernodes;
	gen=gens;
      }
	
      double max_fitness = -200;

      vector<Species*>::iterator curspecies;
      for(curspecies=(pop->species).begin();curspecies!=(pop->species).end();++curspecies) {
	if((*curspecies)->compute_max_fitness() < max_fitness)
	  max_fitness = (*curspecies)->compute_max_fitness();
      }

      cout << "Epoch " << gen << ", fitness= " << max_fitness << endl;

      double max_fitness_B = -200;

      vector<Organism*>::iterator curorg;
      for(curorg=(pop->organisms).begin();curorg!=(pop->organisms).end();++curorg) {
	if((*curorg)->fitness < max_fitness_B)
	  max_fitness_B = (*curorg)->fitness;
      }

      // ((pop->species))[0] compiles
      cout << "Epoch " << gen << ", fitness= " << max_fitness_B << endl;

      //Clear output filename
      fnamebuf->clear();
      delete fnamebuf;
    }

    if (expcount<NEAT::num_runs-1)
      delete pop;
      
  }

  //Average and print stats
  cout<<"Nodes: "<<endl;
  for(expcount=0;expcount<NEAT::num_runs;expcount++) {
    cout<<nodes[expcount]<<endl;
    totalnodes+=nodes[expcount];
  }
    
  cout<<"Genes: "<<endl;
  for(expcount=0;expcount<NEAT::num_runs;expcount++) {
    cout<<genes[expcount]<<endl;
    totalgenes+=genes[expcount];
  }
    
  cout<<"Evals "<<endl;
  samples=0;
  for(expcount=0;expcount<NEAT::num_runs;expcount++) {
    cout<<evals[expcount]<<endl;
    if (evals[expcount]>0)
      {
	totalevals+=evals[expcount];
	samples++;
      }
  }

  cout<<"Failures: "<<(NEAT::num_runs-samples)<<" out of "<<NEAT::num_runs<<" runs"<<endl;
  cout<<"Average Nodes: "<<(samples>0 ? (double)totalnodes/samples : 0)<<endl;
  cout<<"Average Genes: "<<(samples>0 ? (double)totalgenes/samples : 0)<<endl;
  cout<<"Average Evals: "<<(samples>0 ? (double)totalevals/samples : 0)<<endl;

  return pop;

}

bool xor_evaluate(Organism *org) {
  Network *net;
  double out[4]; //The four outputs
  double this_out; //The current output
  int count;
  double errorsum;

  bool success;  //Check for successful activation
  int numnodes;  /* Used to figure out how many nodes
		    should be visited during activation */

  int net_depth; //The max depth of the network to be activated
  int relax; //Activates until relaxation

  //The four possible input combinations to xor
  //The first number is for biasing
  double in[4][3]={{1.0,0.0,0.0},
		   {1.0,0.0,1.0},
		   {1.0,1.0,0.0},
		   {1.0,1.0,1.0}};
  
  net=org->net;
  numnodes=((org->gnome)->nodes).size();

  net_depth=net->max_depth();

  //TEST CODE: REMOVE
  //cout<<"ACTIVATING: "<<org->gnome<<endl;
  //cout<<"DEPTH: "<<net_depth<<endl;

  //Load and activate the network on each input
  for(count=0;count<=3;count++) {
    net->load_sensors(in[count]);

    //Relax net and get output
    success=net->activate();

    //use depth to ensure relaxation
    for (relax=0;relax<=net_depth;relax++) {
      success=net->activate();
      this_out=(*(net->outputs.begin()))->activation;
    }
    out[count]=(*(net->outputs.begin()))->activation;
    net->flush();
  }
  
  if (success) {
    errorsum=(fabs(out[0])+fabs(1.0-out[1])+fabs(1.0-out[2])+fabs(out[3]));
    org->fitness=pow((4.0-errorsum),2);
    org->error=errorsum;
  }
  else {
    //The network is flawed (shouldnt happen)
    errorsum=999.0;
    org->fitness=0.001;
  }

#ifndef NO_SCREEN_OUT
  cout<<"Org "<<(org->gnome)->genome_id<<"                                     error: "<<errorsum<<"  ["<<out[0]<<" "<<out[1]<<" "<<out[2]<<" "<<out[3]<<"]"<<endl;
  cout<<"Org "<<(org->gnome)->genome_id<<"                                     fitness: "<<org->fitness<<endl;
#endif

  //  if (errorsum<0.05) { 
  //if (errorsum<0.2) {
  if ((out[0]<0.5)&&(out[1]>=0.5)&&(out[2]>=0.5)&&(out[3]<0.5)) {
    org->winner=true;
    return true;
  }
  else {
    org->winner=false;
    return false;
  }

}

int xor_epoch(Population *pop,int generation,char *filename,int &winnernum,int &winnergenes,int &winnernodes) {
  vector<Organism*>::iterator curorg;
  vector<Species*>::iterator curspecies;
  //char cfilename[100];
  //strncpy( cfilename, filename.c_str(), 100 );

  //ofstream cfilename(filename.c_str());

  bool win=false;


  //Evaluate each organism on a test
  for(curorg=(pop->organisms).begin();curorg!=(pop->organisms).end();++curorg) {
    if (xor_evaluate(*curorg)) {
      win=true;
      winnernum=(*curorg)->gnome->genome_id;
      winnergenes=(*curorg)->gnome->extrons();
      winnernodes=((*curorg)->gnome->nodes).size();
      if (winnernodes==5) {
	//You could dump out optimal genomes here if desired
	//(*curorg)->gnome->print_to_filename("xor_optimal");
	//cout<<"DUMPED OPTIMAL"<<endl;
      }
    }
  }
  
  //Average and max their fitnesses for dumping to file and snapshot
  for(curspecies=(pop->species).begin();curspecies!=(pop->species).end();++curspecies) {

    //This experiment control routine issues commands to collect ave
    //and max fitness, as opposed to having the snapshot do it, 
    //because this allows flexibility in terms of what time
    //to observe fitnesses at

    (*curspecies)->compute_average_fitness();
    (*curspecies)->compute_max_fitness();
  }

  //Take a snapshot of the population, so that it can be
  //visualized later on
  //if ((generation%1)==0)
  //  pop->snapshot();

  //Only print to file every print_every generations
  if  (win||
       ((generation%(NEAT::print_every))==0))
    pop->print_to_file_by_species(filename);


  if (win) {
    for(curorg=(pop->organisms).begin();curorg!=(pop->organisms).end();++curorg) {
      if ((*curorg)->winner) {
	cout<<"WINNER IS #"<<((*curorg)->gnome)->genome_id<<endl;
	//Prints the winner to file
	//IMPORTANT: This causes generational file output!
	print_Genome_tofile((*curorg)->gnome,"xor_winner");
      }
    }
    
  }

  pop->epoch(generation);

  if (win) return 1;
  else return 0;

}

//Perform evolution on single pole balacing, for gens generations
Population *pole1_test(int gens) {
  Population *pop=0;
  Genome *start_genome;
  char curword[20];
  int id;

  ostringstream *fnamebuf;
  int gen;

  int expcount;
  int status;
  int runs[NEAT::num_runs];
  int totalevals;
  int samples;  //For averaging

  memset (runs, 0, NEAT::num_runs * sizeof(int));

  ifstream iFile("pole1startgenes",ios::in);

  cout<<"START SINGLE POLE BALANCING EVOLUTION"<<endl;

  cout<<"Reading in the start genome"<<endl;
  //Read in the start Genome
  iFile>>curword;
  iFile>>id;
  cout<<"Reading in Genome id "<<id<<endl;
  start_genome=new Genome(id,iFile);
  iFile.close();
  
  //Run multiple experiments
  for(expcount=0;expcount<NEAT::num_runs;expcount++) {

    cout<<"EXPERIMENT #"<<expcount<<endl;

    cout<<"Start Genome: "<<start_genome<<endl;
      
    //Spawn the Population
    cout<<"Spawning Population off Genome"<<endl;
      
    pop=new Population(start_genome,NEAT::pop_size);
      
    cout<<"Verifying Spawned Pop"<<endl;
    pop->verify();

    for (gen=1;gen<=gens;gen++) {
      cout<<"Generation "<<gen<<endl;
	
      fnamebuf=new ostringstream();
      (*fnamebuf)<<"gen_"<<gen<<ends;  //needs end marker

#ifndef NO_SCREEN_OUT
      cout<<"name of fname: "<<fnamebuf->str()<<endl;
#endif	

      char temp[50];
      sprintf (temp, "gen_%d", gen);

      status=pole1_epoch(pop,gen,temp);
      //status=(pole1_epoch(pop,gen,fnamebuf->str()));
	
      if (status) {
	runs[expcount]=status;
	gen=gens+1;
      }
	
      fnamebuf->clear();
      delete fnamebuf;
	
    }

    if (expcount<NEAT::num_runs-1) delete pop;
  }

  totalevals=0;
  samples=0;
  for(expcount=0;expcount<NEAT::num_runs;expcount++) {
    cout<<runs[expcount]<<endl;
    if (runs[expcount]>0)
      {
        totalevals+=runs[expcount];
        samples++;
      }
  }

  cout<<"Failures: "<<(NEAT::num_runs-samples)<<" out of "<<NEAT::num_runs<<" runs"<<endl;
  cout<<"Average evals: "<<(samples>0 ? (double)totalevals/samples : 0)<<endl;

  return pop;

}

int pole1_epoch(Population *pop,int generation,char *filename) {
  vector<Organism*>::iterator curorg;
  vector<Species*>::iterator curspecies;
  //char cfilename[100];
  //strncpy( cfilename, filename.c_str(), 100 );

  //ofstream cfilename(filename.c_str());

  bool win=false;
  int winnernum;

  //Evaluate each organism on a test
  for(curorg=(pop->organisms).begin();curorg!=(pop->organisms).end();++curorg) {
    if (pole1_evaluate(*curorg)) win=true;
  }
  
  //Average and max their fitnesses for dumping to file and snapshot
  for(curspecies=(pop->species).begin();curspecies!=(pop->species).end();++curspecies) {

    //This experiment control routine issues commands to collect ave
    //and max fitness, as opposed to having the snapshot do it, 
    //because this allows flexibility in terms of what time
    //to observe fitnesses at

    (*curspecies)->compute_average_fitness();
    (*curspecies)->compute_max_fitness();
  }

  //Take a snapshot of the population, so that it can be
  //visualized later on
  //if ((generation%1)==0)
  //  pop->snapshot();

  //Only print to file every print_every generations
  if  (win||
       ((generation%(NEAT::print_every))==0))
    pop->print_to_file_by_species(filename);

  if (win) {
    for(curorg=(pop->organisms).begin();curorg!=(pop->organisms).end();++curorg) {
      if ((*curorg)->winner) {
	winnernum=((*curorg)->gnome)->genome_id;
	cout<<"WINNER IS #"<<((*curorg)->gnome)->genome_id<<endl;
      }
    }    
  }

  //Create the next generation
  pop->epoch(generation);

  if (win) return ((generation-1)*NEAT::pop_size+winnernum);
  else return 0;

}

bool pole1_evaluate(Organism *org) {
  Network *net;

  int numnodes;  /* Used to figure out how many nodes
		    should be visited during activation */
  int thresh;  /* How many visits will be allowed before giving up 
		  (for loop detection) */

  //  int MAX_STEPS=120000;
  int MAX_STEPS=100000;
  
  net=org->net;
  numnodes=((org->gnome)->nodes).size();
  thresh=numnodes*2;  //Max number of visits allowed per activation
  
  //Try to balance a pole now
  org->fitness = go_cart(net,MAX_STEPS,thresh);

#ifndef NO_SCREEN_OUT
  cout<<"Org "<<(org->gnome)->genome_id<<" fitness: "<<org->fitness<<endl;
#endif

  //Decide if its a winner
  if (org->fitness>=MAX_STEPS) { 
    org->winner=true;
    return true;
  }
  else {
    org->winner=false;
    return false;
  }

}

//     cart_and_pole() was take directly from the pole simulator written
//     by Richard Sutton and Charles Anderson.
int go_cart(Network *net,int max_steps,int thresh)
{
  float x,			/* cart position, meters */
    x_dot,			/* cart velocity */
    theta,			/* pole angle, radians */
    theta_dot;		/* pole angular velocity */
  int steps=0,y;

  int random_start=1;

  double in[5];  //Input loading array

  double out1;
  double out2;

  //     double one_degree= 0.0174532;	/* 2pi/360 */
  //     double six_degrees=0.1047192;
  double twelve_degrees=0.2094384;
  //     double thirty_six_degrees= 0.628329;
  //     double fifty_degrees=0.87266;

  vector<NNode*>::iterator out_iter;

  if (random_start) {
    /*set up random start state*/
    x = (lrand48()%4800)/1000.0 - 2.4;
    x_dot = (lrand48()%2000)/1000.0 - 1;
    theta = (lrand48()%400)/1000.0 - .2;
    theta_dot = (lrand48()%3000)/1000.0 - 1.5;
  }
  else 
    x = x_dot = theta = theta_dot = 0.0;
   
  /*--- Iterate through the action-learn loop. ---*/
  while (steps++ < max_steps)
    {
       
      /*-- setup the input layer based on the four iputs --*/
      //setup_input(net,x,x_dot,theta,theta_dot);
      in[0]=1.0;  //Bias
      in[1]=(x + 2.4) / 4.8;;
      in[2]=(x_dot + .75) / 1.5;
      in[3]=(theta + twelve_degrees) / .41;
      in[4]=(theta_dot + 1.0) / 2.0;
      net->load_sensors(in);

      //activate_net(net);   /*-- activate the network based on the input --*/
      //Activate the net
      //If it loops, exit returning only fitness of 1 step
      if (!(net->activate())) return 1;

      /*-- decide which way to push via which output unit is greater --*/
      out_iter=net->outputs.begin();
      out1=(*out_iter)->activation;
      ++out_iter;
      out2=(*out_iter)->activation;
      if (out1 > out2)
	y = 0;
      else
	y = 1;
       
      /*--- Apply action to the simulated cart-pole ---*/
      cart_pole(y, &x, &x_dot, &theta, &theta_dot);
       
      /*--- Check for failure.  If so, return steps ---*/
      if (x < -2.4 || x > 2.4  || theta < -twelve_degrees ||
	  theta > twelve_degrees) 
	return steps;             
    }
   
  return steps;
} 


//     cart_and_pole() was take directly from the pole simulator written
//     by Richard Sutton and Charles Anderson.
//     This simulator uses normalized, continous inputs instead of 
//    discretizing the input space.
/*----------------------------------------------------------------------
  cart_pole:  Takes an action (0 or 1) and the current values of the
  four state variables and updates their values by estimating the state
  TAU seconds later.
  ----------------------------------------------------------------------*/
void cart_pole(int action, float *x,float *x_dot, float *theta, float *theta_dot) {
  float xacc,thetaacc,force,costheta,sintheta,temp;
  
  const float GRAVITY=9.8;
  const float MASSCART=1.0;
  const float MASSPOLE=0.1;
  const float TOTAL_MASS=(MASSPOLE + MASSCART);
  const float LENGTH=0.5;	  /* actually half the pole's length */
  const float POLEMASS_LENGTH=(MASSPOLE * LENGTH);
  const float FORCE_MAG=10.0;
  const float TAU=0.02;	  /* seconds between state updates */
  const float FOURTHIRDS=1.3333333333333;

  force = (action>0)? FORCE_MAG : -FORCE_MAG;
  costheta = cos(*theta);
  sintheta = sin(*theta);
  
  temp = (force + POLEMASS_LENGTH * *theta_dot * *theta_dot * sintheta)
    / TOTAL_MASS;
  
  thetaacc = (GRAVITY * sintheta - costheta* temp)
    / (LENGTH * (FOURTHIRDS - MASSPOLE * costheta * costheta
		 / TOTAL_MASS));
  
  xacc  = temp - POLEMASS_LENGTH * thetaacc* costheta / TOTAL_MASS;
  
  /*** Update the four state variables, using Euler's method. ***/
  
  *x  += TAU * *x_dot;
  *x_dot += TAU * xacc;
  *theta += TAU * *theta_dot;
  *theta_dot += TAU * thetaacc;
}

/* ------------------------------------------------------------------ */
/* Double pole balacing                                               */
/* ------------------------------------------------------------------ */

//Perform evolution on double pole balacing, for gens generations
//If velocity is false, then velocity information will be withheld from the 
//network population (non-Markov)
Population *pole2_test(int gens,int velocity) {
  Population *pop=0;
  Genome *start_genome;
  char curword[20];
  int id;

  ostringstream *fnamebuf;
  int gen;
  CartPole *thecart;

  //Stat collection variables
  int highscore;
  int record[NEAT::num_runs][1000];
  double recordave[1000];
  int genesrec[NEAT::num_runs][1000];
  double genesave[1000];
  int nodesrec[NEAT::num_runs][1000];
  double nodesave[1000];
  int winnergens[NEAT::num_runs];
  int initcount;
  int champg, champn, winnernum;  //Record number of genes and nodes in champ
  int run;
  int curtotal; //For averaging
  int samples;  //For averaging

  ofstream oFile("statout",ios::out);

  champg=0;
  champn=0;

  //Initialize the stat recording arrays
  for (initcount=0;initcount<gens;initcount++) {
    recordave[initcount]=0;
    genesave[initcount]=0;
    nodesave[initcount]=0;
    for (run=0;run<NEAT::num_runs;++run) {
      record[run][initcount]=0;
      genesrec[run][initcount]=0;
      nodesrec[run][initcount]=0;
    }
  }
  memset (winnergens, 0, NEAT::num_runs * sizeof(int));

  char *non_markov_starter="pole2startgenes2";
  char *markov_starter="pole2startgenes1";
  char *startstring;

  if (velocity==0) startstring=non_markov_starter;
  else if (velocity==1) startstring=markov_starter;
  ifstream iFile(startstring,ios::in);
  //ifstream iFile("pole2startgenes",ios::in);

  cout<<"START DOUBLE POLE BALANCING EVOLUTION"<<endl;
  if (!velocity)
    cout<<"NO VELOCITY INPUT"<<endl;

  cout<<"Reading in the start genome"<<endl;
  //Read in the start Genome
  iFile>>curword;
  iFile>>id;
  cout<<"Reading in Genome id "<<id<<endl;
  start_genome=new Genome(id,iFile);
  iFile.close();

  cout<<"Start Genome: "<<start_genome<<endl;

  for (run=0;run<NEAT::num_runs;run++) {
      
    cout<<"RUN #"<<run<<endl;

    //Spawn the Population from starter gene
    cout<<"Spawning Population off Genome"<<endl;
    pop=new Population(start_genome,NEAT::pop_size);
      
    //Alternative way to start off of randomly connected genomes
    //pop=new Population(pop_size,7,1,10,false,0.3);

    cout<<"Verifying Spawned Pop"<<endl;
    pop->verify();
      
    //Create the Cart
    thecart=new CartPole(true,velocity);
      
    for (gen=1;gen<=gens;gen++) {
      cout<<"Epoch "<<gen<<endl;
	
      fnamebuf=new ostringstream();
      (*fnamebuf)<<"gen_"<<gen<<ends;  //needs end marker
      /*
#ifndef NO_SCREEN_OUT
      cout<<"name of fname: "<<fnamebuf->str()<<endl;
#endif
      */

      char temp[50];
      sprintf (temp, "gen_%d", gen);

      highscore=pole2_epoch(pop,gen,temp,velocity, thecart,champg,champn,winnernum,oFile);
      //highscore=pole2_epoch(pop,gen,fnamebuf->str(),velocity, thecart,champg,champn,winnernum,oFile);  
	
      //cout<<"GOT HIGHSCORE FOR GEN "<<gen<<": "<<highscore-1<<endl;
	
      record[run][gen-1]=highscore-1;
      genesrec[run][gen-1]=champg;
      nodesrec[run][gen-1]=champn;
	
      fnamebuf->clear();
      delete fnamebuf;
	
      //Stop right at the winnergen
      if (((pop->winnergen)!=0)&&(gen==(pop->winnergen))) {
	winnergens[run]=NEAT::pop_size*(gen-1)+winnernum;
	gen=gens+1;
      }
	
      //In non-MARKOV, stop right at winning (could go beyond if desired)
      if ((!(thecart->MARKOV))&&((pop->winnergen)!=0))
	gen=gens+1;

#ifndef NO_SCREEN_OUT
      cout<<"gen = "<<gen<<" gens = "<<gens<<endl;
#endif

      if (gen==(gens-1)) oFile<<"FAIL: Last gen on run "<<run<<endl;
      

    }

    if (run<NEAT::num_runs-1) delete pop;
    delete thecart;

  }

  cout<<"Generation highs: "<<endl;
  oFile<<"Generation highs: "<<endl;
  for(gen=0;gen<=gens-1;gen++) {
    curtotal=0;
    for (run=0;run<NEAT::num_runs;++run) {
      if (record[run][gen]>0) {
	cout<<setw(8)<<record[run][gen]<<" ";
	oFile<<setw(8)<<record[run][gen]<<" ";
	curtotal+=record[run][gen];
      }
      else {
	cout<<"         ";
	oFile<<"         ";
	curtotal+=100000;
      }
      recordave[gen]=(double) curtotal/NEAT::num_runs;
	
    }
    cout<<endl;
    oFile<<endl;
  }

  cout<<"Generation genes in champ: "<<endl;
  for(gen=0;gen<=gens-1;gen++) {
    curtotal=0;
    samples=0;
    for (run=0;run<NEAT::num_runs;++run) {
      if (genesrec[run][gen]>0) {
	cout<<setw(4)<<genesrec[run][gen]<<" ";
	oFile<<setw(4)<<genesrec[run][gen]<<" ";
	curtotal+=genesrec[run][gen];
	samples++;
      }
      else {
	cout<<setw(4)<<"     ";
	oFile<<setw(4)<<"     ";
      }
    }
    genesave[gen]=(double) curtotal/samples;

    cout<<endl;
    oFile<<endl;
  }

  cout<<"Generation nodes in champ: "<<endl;
  oFile<<"Generation nodes in champ: "<<endl;
  for(gen=0;gen<=gens-1;gen++) {
    curtotal=0;
    samples=0;
    for (run=0;run<NEAT::num_runs;++run) {
      if (nodesrec[run][gen]>0) {
	cout<<setw(4)<<nodesrec[run][gen]<<" ";
	oFile<<setw(4)<<nodesrec[run][gen]<<" ";
	curtotal+=nodesrec[run][gen];
	samples++;
      }
      else {
	cout<<setw(4)<<"     ";
	oFile<<setw(4)<<"     ";
      }
    }
    nodesave[gen]=(double) curtotal/samples;

    cout<<endl;
    oFile<<endl;
  }

  cout<<"Generational record fitness averages: "<<endl;
  oFile<<"Generational record fitness averages: "<<endl;
  for(gen=0;gen<gens-1;gen++) {
    cout<<recordave[gen]<<endl;
    oFile<<recordave[gen]<<endl;
  }

  cout<<"Generational number of genes in champ averages: "<<endl;
  oFile<<"Generational number of genes in champ averages: "<<endl;
  for(gen=0;gen<gens-1;gen++) {
    cout<<genesave[gen]<<endl;
    oFile<<genesave[gen]<<endl;
  }

  cout<<"Generational number of nodes in champ averages: "<<endl;
  oFile<<"Generational number of nodes in champ averages: "<<endl;
  for(gen=0;gen<gens-1;gen++) {
    cout<<nodesave[gen]<<endl;
    oFile<<nodesave[gen]<<endl;
  }

  cout<<"Winner evals: "<<endl;
  oFile<<"Winner evals: "<<endl;
  curtotal=0;
  samples=0;
  for (run=0;run<NEAT::num_runs;++run) {
    cout<<winnergens[run]<<endl;
    oFile<<winnergens[run]<<endl;
    if (winnergens[run]>0)
      {
        curtotal+=winnergens[run];
        samples++;
      }
  }
  cout<<"Failures: "<<(NEAT::num_runs-samples)<<" out of "<<NEAT::num_runs<<" runs"<<endl;
  oFile<<"Failures: "<<(NEAT::num_runs-samples)<<" out of "<<NEAT::num_runs<<" runs"<<endl;

  cout<<"Average # evals: "<<(samples>0 ? (double) curtotal/samples : 0)<<endl;
  oFile<<"Average # evals: "<<(samples>0 ? (double) curtotal/samples : 0)<<endl;

  oFile.close();

  return pop;

}

//This is used for list sorting of Species by fitness of best organism
//highest fitness first
//Used to choose which organism to test
//bool order_new_species(Species *x, Species *y) {
//
//  return (x->compute_max_fitness() > 
//	  y->compute_max_fitness());
//}

int pole2_epoch(Population *pop,int generation,char *filename,bool velocity,
		CartPole *thecart,int &champgenes,int &champnodes,
		int &winnernum, ofstream &oFile) {
  //char cfilename[100];
  //strncpy( cfilename, filename.c_str(), 100 );

  //ofstream cfilename(filename.c_str());

  vector<Organism*>::iterator curorg;
  vector<Species*>::iterator curspecies;

  vector<Species*> sorted_species;  //Species sorted by max fit org in Species

  int pause;
  bool win=false;

  double champ_fitness;
  Organism *champ;

  //double statevals[5]={-0.9,-0.5,0.0,0.5,0.9};
  double statevals[5]={0.05, 0.25, 0.5, 0.75, 0.95};

  int s0c,s1c,s2c,s3c;

  int score;

  thecart->nmarkov_long=false;
  thecart->generalization_test=false;

  //Evaluate each organism on a test
  for(curorg=(pop->organisms).begin();curorg!=(pop->organisms).end();++curorg) {

    //shouldn't happen
    if (((*curorg)->gnome)==0) {
      cout<<"ERROR EMPTY GEMOME!"<<endl;
      cin>>pause;
    }

    if (pole2_evaluate((*curorg),velocity,thecart)) win=true;

  }

  //Average and max their fitnesses for dumping to file and snapshot
  for(curspecies=(pop->species).begin();curspecies!=(pop->species).end();++curspecies) {

    //This experiment control routine issues commands to collect ave
    //and max fitness, as opposed to having the snapshot do it, 
    //because this allows flexibility in terms of what time
    //to observe fitnesses at

    (*curspecies)->compute_average_fitness();
    (*curspecies)->compute_max_fitness();
  }

  //Take a snapshot of the population, so that it can be
  //visualized later on
  //if ((generation%1)==0)
  //  pop->snapshot();

  //Find the champion in the markov case simply for stat collection purposes
  if (thecart->MARKOV) {
    champ_fitness=0.0;
    for(curorg=(pop->organisms).begin();curorg!=(pop->organisms).end();++curorg) {
      if (((*curorg)->fitness)>champ_fitness) {
	champ=(*curorg);
	champ_fitness=champ->fitness;
	champgenes=champ->gnome->genes.size();
	champnodes=champ->gnome->nodes.size();
	winnernum=champ->gnome->genome_id;
      }
    }
  }

  //Check for winner in Non-Markov case
  if (!(thecart->MARKOV)) {
    
    cout<<"Non-markov case"<<endl;

    //Sort the species
    for(curspecies=(pop->species).begin();curspecies!=(pop->species).end();++curspecies) {
      sorted_species.push_back(*curspecies);
    }

    //sorted_species.sort(order_new_species);
    std::sort(sorted_species.begin(), sorted_species.end(), NEAT::order_new_species);

    //std::sort(sorted_species.begin(), sorted_species.end(), order_species);


    cout<<"Number of species sorted: "<<sorted_species.size()<<endl;

    //First update what is checked and unchecked
    for(curspecies=sorted_species.begin();curspecies!=sorted_species.end();++curspecies) {
      if (((*curspecies)->compute_max_fitness())>((*curspecies)->max_fitness_ever))
	(*curspecies)->checked=false;

    }

    //Now find a species that is unchecked
    curspecies=sorted_species.begin();
    cout<<"Is the first species checked? "<<(*curspecies)->checked<<endl;
    while((curspecies!=(sorted_species.end()))&&
	  ((*curspecies)->checked))
      {
	cout<<"Species #"<<(*curspecies)->id<<" is checked"<<endl;
	++curspecies;
      }

    if (curspecies==(sorted_species.end())) curspecies=sorted_species.begin();

    //Remember it was checked
    (*curspecies)->checked=true;
    cout<<"Is the species now checked? "<<(*curspecies)->checked<<endl;

    //Extract the champ
    cout<<"Champ chosen from Species "<<(*curspecies)->id<<endl;
    champ=(*curspecies)->get_champ();
    champ_fitness=champ->fitness;
    cout<<"Champ is organism #"<<champ->gnome->genome_id<<endl;
    cout<<"Champ fitness: "<<champ_fitness<<endl;
    winnernum=champ->gnome->genome_id;

    cout<<champ->gnome<<endl;

    //Now check to make sure the champ can do 100,000
    thecart->nmarkov_long=true;
    thecart->generalization_test=false;

    //The champ needs tp be flushed here because it may have
    //leftover activation from its last test run that could affect
    //its recurrent memory
    (champ->net)->flush();


    //champ->gnome->print_to_filename("tested");
    
    if (pole2_evaluate(champ,velocity,thecart)) {
      cout<<"The champ passed the 100,000 test!"<<endl;

      thecart->nmarkov_long=false;

      //Given that the champ passed, now run it on generalization tests
      score=0;
      for (s0c=0;s0c<=4;++s0c)
	for (s1c=0;s1c<=4;++s1c)
	  for (s2c=0;s2c<=4;++s2c)
	    for (s3c=0;s3c<=4;++s3c) {
	      thecart->state[0] = statevals[s0c] * 4.32 - 2.16;
	      thecart->state[1] = statevals[s1c] * 2.70 - 1.35;
	      thecart->state[2] = statevals[s2c] * 0.12566304 - 0.06283152;
	      /* 0.06283152 =  3.6 degrees */
	      thecart->state[3] = statevals[s3c] * 0.30019504 - 0.15009752;
	      /* 00.15009752 =  8.6 degrees */
	      thecart->state[4]=0.0;
	      thecart->state[5]=0.0;
	      
	      cout<<"On combo "<<thecart->state[0]<<" "<<thecart->state[1]<<" "<<thecart->state[2]<<" "<<thecart->state[3]<<endl;
	      thecart->generalization_test=true;
	      
	      (champ->net)->flush();  //Reset the champ for each eval

	      if (pole2_evaluate(champ,velocity,thecart)) {
		cout<<"----------------------------The champ passed its "<<score<<"th test"<<endl;
		score++;
	      }
	      
	    }

      if (score>=200) {
	cout<<"The champ wins!!! (generalization = "<<score<<" )"<<endl;
	oFile<<"(generalization = "<<score<<" )"<<endl;
	oFile<<"generation= "<<generation<<endl;
        (champ->gnome)->print_to_file(oFile);
	champ_fitness=champ->fitness;
	champgenes=champ->gnome->genes.size();
	champnodes=champ->gnome->nodes.size();
	winnernum=champ->gnome->genome_id;
	win=true;
      }
      else {
	cout<<"The champ couldn't generalize"<<endl;
	champ->fitness=champ_fitness; //Restore the champ's fitness
      }
    }
    else {
      cout<<"The champ failed the 100,000 test :("<<endl;
      cout<<"made score "<<champ->fitness<<endl;
      champ->fitness=champ_fitness; //Restore the champ's fitness
    }
  }
  
  //Only print to file every print_every generations
  if  (win||
       ((generation%(NEAT::print_every))==0)) {
    cout<<"printing file: "<<filename<<endl;
    pop->print_to_file_by_species(filename);
  }

  if ((win)&&((pop->winnergen)==0)) pop->winnergen=generation;

  //Prints a champion out on each generation
  //IMPORTANT: This causes generational file output!
  print_Genome_tofile(champ->gnome,"champ");

  //Create the next generation
  pop->epoch(generation);

  return (int) champ_fitness;
}

bool pole2_evaluate(Organism *org,bool velocity, CartPole *thecart) {
  Network *net;

  int thresh;  /* How many visits will be allowed before giving up 
		  (for loop detection)  NOW OBSOLETE */

  int pause;

  net=org->net;

  thresh=100;  //this is obsolete

  //DEBUG :  Check flushedness of org
  //org->net->flush_check();

  //Try to balance a pole now
  org->fitness = thecart->evalNet(net,thresh);

#ifndef NO_SCREEN_OUT
  if (org->pop_champ_child)
    cout<<" <<DUPLICATE OF CHAMPION>> ";

  //Output to screen
  cout<<"Org "<<(org->gnome)->genome_id<<" fitness: "<<org->fitness;
  cout<<" ("<<(org->gnome)->genes.size();
  cout<<" / "<<(org->gnome)->nodes.size()<<")";
  cout<<"   ";
  if (org->mut_struct_baby) cout<<" [struct]";
  if (org->mate_baby) cout<<" [mate]";
  cout<<endl;
#endif

  if ((!(thecart->generalization_test))&&(!(thecart->nmarkov_long)))
    if (org->pop_champ_child) {
      cout<<org->gnome<<endl;
      //DEBUG CHECK
      if (org->high_fit>org->fitness) {
	cout<<"ALERT: ORGANISM DAMAGED"<<endl;
	print_Genome_tofile(org->gnome,"failure_champ_genome");
	cin>>pause;
      }
    }

  //Decide if its a winner, in Markov Case
  if (thecart->MARKOV) {
    if (org->fitness>=(thecart->maxFitness-1)) { 
      org->winner=true;
      return true;
    }
    else {
      org->winner=false;
      return false;
    }
  }
  //if doing the long test non-markov 
  else if (thecart->nmarkov_long) {
    if (org->fitness>=99999) { 
      //if (org->fitness>=9000) { 
      org->winner=true;
      return true;
    }
    else {
      org->winner=false;
      return false;
    }
  }
  else if (thecart->generalization_test) {
    if (org->fitness>=999) {
      org->winner=true;
      return true;
    }
    else {
      org->winner=false;
      return false;
    }
  }
  else {
    org->winner=false;
    return false;  //Winners not decided here in non-Markov
  }
}

CartPole::CartPole(bool randomize,bool velocity)
{
  maxFitness = 100000;

  MARKOV=velocity;

  MIN_INC = 0.001;
  POLE_INC = 0.05;
  MASS_INC = 0.01;

  LENGTH_2 = 0.05;
  MASSPOLE_2 = 0.01;

  // CartPole::reset() which is called here
}

//Faustino Gomez wrote this physics code using the differential equations from 
//Alexis Weiland's paper and added the Runge-Kutta himself.
double CartPole::evalNet(Network *net,int thresh)
{
  int steps=0;
  double input[NUM_INPUTS];
  double output;

  int nmarkovmax;  

  double nmarkov_fitness;

  double jiggletotal; //total jiggle in last 100
  int count;  //step counter

  //init(randomize);		// restart at some point
  
  if (nmarkov_long) nmarkovmax=100000;
  else if (generalization_test) nmarkovmax=1000;
  else nmarkovmax=1000;


  init(0);

  if (MARKOV) {
    while (steps++ < maxFitness) {
      
         
      input[0] = state[0] / 4.8;
      input[1] = state[1] /2;
      input[2] = state[2]  / 0.52;
      input[3] = state[3] /2;
      input[4] = state[4] / 0.52;
      input[5] = state[5] /2;
      input[6] = .5;
      
      net->load_sensors(input);
      
      //Activate the net
      //If it loops, exit returning only fitness of 1 step
      if (!(net->activate())) return 1.0;
      
      output=(*(net->outputs.begin()))->activation;
      
      performAction(output,steps);
      
      if (outsideBounds())	// if failure
	break;			// stop it now
    }
    return (double) steps;
  }
  else {  //NON MARKOV CASE

    while (steps++ < nmarkovmax) {
      

      //Do special parameter summing on last hundred
      //if ((steps==900)&&(!nmarkov_long)) last_hundred=true;

      /*
	input[0] = state[0] / 4.8;
	input[1] = 0.0;
	input[2] = state[2]  / 0.52;
	input[3] = 0.0;
	input[4] = state[4] / 0.52;
	input[5] = 0.0;
	input[6] = .5;
      */

      //cout<<"nmarkov_long: "<<nmarkov_long<<endl;

      //if (nmarkov_long)
      //cout<<"step: "<<steps<<endl;

      input[0] = state[0] / 4.8;
      input[1] = state[2]  / 0.52;
      input[2] = state[4] / 0.52;
      input[3] = .5;
      
      net->load_sensors(input);

      //cout<<"inputs: "<<input[0]<<" "<<input[1]<<" "<<input[2]<<" "<<input[3]<<endl;

      //Activate the net
      //If it loops, exit returning only fitness of 1 step
      if (!(net->activate())) return 0.0001;
      
      output=(*(net->outputs.begin()))->activation;

      //cout<<"output: "<<output<<endl;

      performAction(output,steps);

      if (outsideBounds())	// if failure
	break;			// stop it now

      if (nmarkov_long&&(outsideBounds()))	// if failure
	break;			// stop it now
    }

    //If we are generalizing we just need to balance it a while
    if (generalization_test)
      return (double) balanced_sum;
 
    //Sum last 100
    if ((steps>100)&&(!nmarkov_long)) {

      jiggletotal=0;
      cout<<"step "<<steps-99-2<<" to step "<<steps-2<<endl;
      //Adjust for array bounds and count
      for (count=steps-99-2;count<=steps-2;count++)
	jiggletotal+=jigglestep[count];
    }

    if (!nmarkov_long) {
      if (balanced_sum>100) 
	nmarkov_fitness=((0.1*(((double) balanced_sum)/1000.0))+
			 (0.9*(0.75/(jiggletotal))));
      else nmarkov_fitness=(0.1*(((double) balanced_sum)/1000.0));

#ifndef NO_SCREEN_OUTR
      cout<<"Balanced:  "<<balanced_sum<<" jiggle: "<<jiggletotal<<" ***"<<endl;
#endif

      return nmarkov_fitness;
    }
    else return (double) steps;

  }

}

void CartPole::init(bool randomize)
{
  static int first_time = 1;

  if (!MARKOV) {
    //Clear all fitness records
    cartpos_sum=0.0;
    cartv_sum=0.0;
    polepos_sum=0.0;
    polev_sum=0.0;
  }

  balanced_sum=0; //Always count # balanced

  last_hundred=false;

  /*if (randomize) {
    state[0] = (lrand48()%4800)/1000.0 - 2.4;
    state[1] = (lrand48()%2000)/1000.0 - 1;
    state[2] = (lrand48()%400)/1000.0 - 0.2;
    state[3] = (lrand48()%400)/1000.0 - 0.2;
    state[4] = (lrand48()%3000)/1000.0 - 1.5;
    state[5] = (lrand48()%3000)/1000.0 - 1.5;
    }
    else {*/


  if (!generalization_test) {
    state[0] = state[1] = state[3] = state[4] = state[5] = 0;
    state[2] = 0.07; // one_degree;
  }
  else {
    state[4] = state[5] = 0;
  }

  //}
  if(first_time){
    cout<<"Initial Long pole angle = %f\n"<<state[2]<<endl;;
    cout<<"Initial Short pole length = %f\n"<<LENGTH_2<<endl;
    first_time = 0;
  }
}

void CartPole::performAction(double output, int stepnum)
{ 
  
  int i;
  double  dydx[6];

  const bool RK4=true; //Set to Runge-Kutta 4th order integration method
  const double EULER_TAU= TAU/4;
 
  /*random start state for long pole*/
  /*state[2]= drand48();   */
     
  /*--- Apply action to the simulated cart-pole ---*/

  if(RK4){
    for(i=0;i<2;++i){
      dydx[0] = state[1];
      dydx[2] = state[3];
      dydx[4] = state[5];
      step(output,state,dydx);
      rk4(output,state,dydx,state);
    }
  }
  else{
    for(i=0;i<8;++i){
      step(output,state,dydx);
      state[0] += EULER_TAU * dydx[0];
      state[1] += EULER_TAU * dydx[1];
      state[2] += EULER_TAU * dydx[2];
      state[3] += EULER_TAU * dydx[3];
      state[4] += EULER_TAU * dydx[4];
      state[5] += EULER_TAU * dydx[5];
    }
  }

  //Record this state
  cartpos_sum+=fabs(state[0]);
  cartv_sum+=fabs(state[1]);
  polepos_sum+=fabs(state[2]);
  polev_sum+=fabs(state[3]);
  if (stepnum<=1000)
    jigglestep[stepnum-1]=fabs(state[0])+fabs(state[1])+fabs(state[2])+fabs(state[3]);

  if (false) {
    //cout<<"[ x: "<<state[0]<<" xv: "<<state[1]<<" t1: "<<state[2]<<" t1v: "<<state[3]<<" t2:"<<state[4]<<" t2v: "<<state[5]<<" ] "<<
    //cartpos_sum+cartv_sum+polepos_sum+polepos_sum+polev_sum<<endl;
    if (!(outsideBounds())) {
      if (balanced_sum<1000) {
	cout<<".";
	++balanced_sum;
      }
    }
    else {
      if (balanced_sum==1000)
	balanced_sum=1000;
      else balanced_sum=0;
    }
  }
  else if (!(outsideBounds()))
    ++balanced_sum;

}

void CartPole::step(double action, double *st, double *derivs)
{
  double force,costheta_1,costheta_2,sintheta_1,sintheta_2,
    gsintheta_1,gsintheta_2,temp_1,temp_2,ml_1,ml_2,fi_1,fi_2,mi_1,mi_2;

  force =  (action - 0.5) * FORCE_MAG * 2;
  costheta_1 = cos(st[2]);
  sintheta_1 = sin(st[2]);
  gsintheta_1 = GRAVITY * sintheta_1;
  costheta_2 = cos(st[4]);
  sintheta_2 = sin(st[4]);
  gsintheta_2 = GRAVITY * sintheta_2;
    
  ml_1 = LENGTH_1 * MASSPOLE_1;
  ml_2 = LENGTH_2 * MASSPOLE_2;
  temp_1 = MUP * st[3] / ml_1;
  temp_2 = MUP * st[5] / ml_2;
  fi_1 = (ml_1 * st[3] * st[3] * sintheta_1) +
    (0.75 * MASSPOLE_1 * costheta_1 * (temp_1 + gsintheta_1));
  fi_2 = (ml_2 * st[5] * st[5] * sintheta_2) +
    (0.75 * MASSPOLE_2 * costheta_2 * (temp_2 + gsintheta_2));
  mi_1 = MASSPOLE_1 * (1 - (0.75 * costheta_1 * costheta_1));
  mi_2 = MASSPOLE_2 * (1 - (0.75 * costheta_2 * costheta_2));
    
  derivs[1] = (force + fi_1 + fi_2)
    / (mi_1 + mi_2 + MASSCART);
    
  derivs[3] = -0.75 * (derivs[1] * costheta_1 + gsintheta_1 + temp_1)
    / LENGTH_1;
  derivs[5] = -0.75 * (derivs[1] * costheta_2 + gsintheta_2 + temp_2)
    / LENGTH_2;

}

void CartPole::rk4(double f, double y[], double dydx[], double yout[])
{

  int i;

  double hh,h6,dym[6],dyt[6],yt[6];


  hh=TAU*0.5;
  h6=TAU/6.0;
  for (i=0;i<=5;i++) yt[i]=y[i]+hh*dydx[i];
  step(f,yt,dyt);
  dyt[0] = yt[1];
  dyt[2] = yt[3];
  dyt[4] = yt[5];
  for (i=0;i<=5;i++) yt[i]=y[i]+hh*dyt[i];
  step(f,yt,dym);
  dym[0] = yt[1];
  dym[2] = yt[3];
  dym[4] = yt[5];
  for (i=0;i<=5;i++) {
    yt[i]=y[i]+TAU*dym[i];
    dym[i] += dyt[i];
  }
  step(f,yt,dyt);
  dyt[0] = yt[1];
  dyt[2] = yt[3];
  dyt[4] = yt[5];
  for (i=0;i<=5;i++)
    yout[i]=y[i]+h6*(dydx[i]+dyt[i]+2.0*dym[i]);
}

bool CartPole::outsideBounds()
{
  const double failureAngle = thirty_six_degrees; 

  return 
    state[0] < -2.4              || 
	       state[0] > 2.4               || 
    state[2] < -failureAngle     ||
	       state[2] > failureAngle      ||
    state[4] < -failureAngle     ||
	       state[4] > failureAngle;  
}

void CartPole::nextTask()
{

  LENGTH_2 += POLE_INC;   /* LENGTH_2 * INCREASE;   */
  MASSPOLE_2 += MASS_INC; /* MASSPOLE_2 * INCREASE; */
  //  ++new_task;
  cout<<"#Pole Length %2.4f\n"<<LENGTH_2<<endl;
}

void CartPole::simplifyTask()
{
  if(POLE_INC > MIN_INC) {
    POLE_INC = POLE_INC/2;
    MASS_INC = MASS_INC/2;
    LENGTH_2 -= POLE_INC;
    MASSPOLE_2 -= MASS_INC;
    cout<<"#SIMPLIFY\n"<<endl;
    cout<<"#Pole Length %2.4f\n"<<LENGTH_2;
  }
  else
    {
      cout<<"#NO TASK CHANGE\n"<<endl;
    }
}
