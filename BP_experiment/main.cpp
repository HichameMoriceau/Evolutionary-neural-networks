#include<iostream>
#include<string>
#include<sstream>
#include<vector>
#include<fann.h>
#include<armadillo>
using namespace std;
using namespace arma;


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

// generates random integer within range
int rand_int(int min, int max){
  return min + (rand() % (int)(max - min + 1));
}

double fixed_topo_exp(){
  double res=-1;
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
  const unsigned int num_input=stoi(headers[1]);
  const unsigned int num_output=stoi(headers[2]);
  const unsigned int num_layers=2;
  const unsigned int nb_hid_units=20;
  const float desired_error=(const float) 0.00f;
  const unsigned int max_epochs=20;
  const unsigned int epochs_between_reports=1;

  fann_type min_weight=-1;
  fann_type max_weight=+1;

  struct fann_train_data *data = fann_read_train_from_file(ds_filename.c_str());
  struct fann *ann=fann_create_standard(num_layers,
					num_input,
					num_output,
					nb_hid_units);
  fann_randomize_weights(ann,min_weight,max_weight);

  cout<<"setting activation functions"<<endl;
  fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
  fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);

  cout<<"training model"<<endl;
  res=10;
  for(unsigned int b=0;b<nb_opt_algs;b++){
    //reset ann
    fann_randomize_weights(ann,min_weight,max_weight);
    double mse=-1;
    cout<<"running BP"<<b<<endl;
    for(unsigned int i=0;i<max_epochs;i++){
      mse=fann_train_epoch(ann,data);
      cout<<"epoch"<<i<<"\tmse="<<mse<<endl;
    }
    if(mse<res)
      res=mse;
  }
  
  cout<<"lowest err obtained after "<<nb_opt_algs<<" != runs of BP for "<<max_epochs<<" epochs is "<<res<<endl<<endl;

  cout<<"Using the same topology for all networks:"<<endl
      <<"\tNb inputs            = "<<fann_get_num_input(ann)<<endl
      <<"\tTotal nb connections = "<<fann_get_total_connections(ann)<<endl
      <<"\tNb outputs           = "<<fann_get_num_output(ann)<<endl
      <<"\tNb hid. layers       = "<<fann_get_num_layers(ann)-2<<endl;

  //fann_save(ann, "bcm_float.net");
  fann_destroy(ann);
  return res;
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

 /**
   Experiment: Running multiple instances of a neural network
   trained using Gradient Descent and the Back Propagation algorithm.
   Author: Hichame Moriceau
   Compile & Run with: 
   # g++ -std=c++11 main.cpp -larmadillo -lfann -o runme
   # ./runme 0 30 # run 30 replicates of the first experiment
*/
int main(int argc, char * argv []){
  if(argc<(2+1)){ 
    cout<<"Too few args provided. Expected: './runme EXP_INDEX NB_REPLICATES'"<<endl;
    return 0;
  }
  // experiment choice
  unsigned exp_index=atoi(argv[1]);
  // number of replicates
  unsigned int nb_reps=atoi(argv[2]);
  // recorded best error value
  double res=0;

  stringstream ss;
  ss<<"running experiment ";

  switch(exp_index){
  case 0:
    ss<<"IDENTICAL_TOPOLOGY using "<<nb_reps<<" replicates";
    cout<<ss.str()<<endl;
    res=fixed_topo_exp();
    break;
  case 1:
    ss<<"DIVERSE_TOPOLOGY using "<<nb_reps<<" replicates";
    cout<<ss.str()<<endl;
    res=diverse_topo_exp();
    break;
  default:
    ss<<"IDENTICAL_TOPOLOGY using "<<nb_reps<<" replicates";
    cout<<ss.str()<<endl;
    res=fixed_topo_exp();
  }

  cout<<"Finished "<<ss.str()<<". Best MSE="<<res<<endl;

  return 0;
}
