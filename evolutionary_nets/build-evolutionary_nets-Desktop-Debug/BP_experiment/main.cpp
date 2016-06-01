#include<iostream>
#include <fann.h>
#include <vector>
using namespace std;

int main(){  


  const char* ds_filename="data_sets/breast-cancer-recurrence-data-transformed.data";//"data_sets/breast-cancer-malignantOrBenign-data-transformed.data";
  unsigned int nb_opt_algs=5;  

  // hyper-params
  const unsigned int num_input = 9;//9
  const unsigned int num_output = 1;
  const unsigned int num_layers = 2;
  const unsigned int nb_hid_units = 2;
  const float desired_error = (const float) 0.00f;
  const unsigned int max_epochs = 100;
  const unsigned int epochs_between_reports = 1;

  fann_type min_weight=-1;
  fann_type max_weight=+1;

  struct fann_train_data *data = fann_read_train_from_file(ds_filename);
  struct fann *ann=fann_create_standard(num_layers,
					num_input,
					num_output,
					nb_hid_units);
  fann_randomize_weights(ann,min_weight,max_weight);
  cout<<"created net:"<<endl
      <<"nb inputs     ="<<fann_get_num_input(ann)<<endl
      <<"total nb connections ="<<fann_get_total_connections(ann)<<endl
      <<"nb outputs    ="<<fann_get_num_output(ann)<<endl
      <<"nb hid. layers="<<fann_get_num_layers(ann)<<endl;

  cout<<"setting activation functions"<<endl;
  fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
  fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);

//fann_train_on_file(ann,ds_filename,max_epochs,epochs_between_reports, desired_error);
  cout<<"training model"<<endl;
  double min=10;
  for(unsigned int b=0;b<nb_opt_algs;b++){
    //reset ann
    fann_randomize_weights(ann,min_weight,max_weight);
    double mse=-1;
    cout<<"running BP"<<b<<endl;
    for(unsigned int i=0;i<max_epochs;i++){
      mse=fann_train_epoch(ann,data);
      cout<<"epoch"<<i<<"\tmse="<<mse<<endl;
    }
    if(mse<min)
      min=mse;
  }
  
  cout<<"lowest err obtained after"<<nb_opt_algs<<" GD/BP runs for "<<max_epochs<<" epochs is "<<min<<endl;

  fann_save(ann, "bcm_float.net");
  fann_destroy(ann);
  return 0;
}
