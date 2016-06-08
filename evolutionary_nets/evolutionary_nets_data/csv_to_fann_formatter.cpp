#include<iostream>
#include<fstream>
#include<algorithm>
#include<string>
#include<armadillo>
using namespace std;
using namespace arma;

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


unsigned int find_nb_prediction_classes(mat D){
  vector<unsigned int> prediction_classes(0);
  bool is_known_class = false;

  // compute nb output units required
  for(unsigned int i=0; i<D.n_rows; i++) {
    unsigned int current_pred_class = D(i,D.n_cols-1);
    is_known_class=false;
    // for each known prediction classes
    for(unsigned int j=0;j<prediction_classes.size(); j++) {
      // if current output is different from prediction class
      if(current_pred_class==prediction_classes[j]){
	is_known_class = true;
      }
    }
    if(prediction_classes.empty() || (!is_known_class)){
      prediction_classes.push_back(current_pred_class);
    }
  }
  // return nb of types of possible predictions
  return prediction_classes.size();
}

/*
  Transforms a CSV data set into a fann-friendly format. (see fann c++ library)
  Compile & Run with: 

  Author: Hichame Moriceau
  #  g++ csv_to_fann_formatter.cpp -std=c++11 -larmadillo -o runme
  # ./runme "breast-cancer-malignantOrBenign-data-transformed.csv"
*/
int main(int argc, char* argv[]){
  if(argc<1){ 
    cout<<"You must provide a data set to format as CSV."<<endl;
    return 0;
  }
  string in=string(argv[1]);
  string data_set_name_without_suffix=in.substr(0,in.length()-4);
  ifstream csv_file(data_set_name_without_suffix+".csv");
  ofstream data_file;
  data_file.open(data_set_name_without_suffix+".data", ios::out);
  // check if both files are correctly open
  if(csv_file.is_open()&&data_file.is_open()){
    string line;
    // write fann header
    mat data;
    data.load(data_set_name_without_suffix+".csv");
    // feature-scaling
    standardize(data);
    // randomize examples order
    data=shuffle(data);
    unsigned int nb_training_pairs= data.n_rows;
    unsigned int nb_inputs = data.n_cols-1;
    unsigned int nb_outputs= find_nb_prediction_classes(data);
    if(nb_outputs=2)nb_outputs=1;
    data_file<<nb_training_pairs<<" "<<nb_inputs<<" "<<nb_outputs<<endl;
    // write body
    for(unsigned int i=0;i<data.n_rows;i++){
      string inputs="", label="";
      for(unsigned int j=0;j<data.n_cols-1;j++)
	inputs+=to_string(data(i,j))+" ";
      label=to_string(data(i,data.n_cols-1));
      data_file<<inputs<<endl;
      data_file<<label <<endl;
    }
    cout<<"finished."<<endl;
  }else
    cout << "Something went wrong when opening the files." << endl;
  data_file.close();
  csv_file.close();
  return 0;
}
