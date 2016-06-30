#include<iostream>
#include <fstream>
#include<vector>
#include<armadillo>
#include <sstream>
#include <string>

using namespace std;
using namespace arma;

vector<string> read_args(int argc, char** argv,unsigned int& rows){
  if(argc<(2+1)) {
    cout<<"At least 2 arguments expected."<<endl;
    cout<<"Arg   1: size of the summarized table (nb of rows)."<<endl;
    cout<<"Arg(s)2: 1 or more path(s) to result files."<<endl;
    cout<<"Example: ./runme data/breast_cancer_recurrence_data_transformed_results/"<<endl;
    exit(0);
  }else{
    rows=atoi(argv[1]);
    vector<string>filenames;
    unsigned int nb_ds=argc-2;
    for(unsigned int i=0;i<nb_ds;i++)
      filenames.push_back(argv[i+2]);
    return filenames;
  }
}

mat to_matrix(string line){
  mat m(line);
  return m;
}

mat load_res_mat(string input_file){
  std::ifstream infile(input_file.c_str());

  string line;
  mat m;
  while (getline(infile, line)){
    // if the line isn't a comment and has 'many' attributes
    if(!(line.find("#")==0) && line.size()>20)
      m=join_vert(m,to_matrix(line));
  }

  return m;
}

mat summarize(mat m, unsigned int ms_rows){
  unsigned int rows=m.n_rows;
  cout<<"total nb rows = "<<rows<<endl;
  mat ms;
  for(unsigned int i=1;i<=ms_rows;i++){
    unsigned int idx=(rows/ms_rows)*i-1;
    ms=join_vert(ms,m.row(idx));
  }
  return ms;
}

/**
 * Compile: # g++ summarize_results.cpp -larmadillo -o summarize_results
 * Run    : # ./summarize_results 50 data/breast_cancer_malignantOrBenign_data_transformed_results/ data/breast_cancer_recurrence_data_transformed_results/ data/wine_data_transformed_results/
data/iris_data_transformed_results/
 */
int main(int argc, char * argv []){
  // for debug purposes
  cout.precision(2);
  cout.setf(ios::fixed);

  unsigned int nb_rows_summarized_table=0;
  vector<string> res_files=read_args(argc,argv,nb_rows_summarized_table);

  cout<<"rows="<<nb_rows_summarized_table<<endl;

  vector<string> filenames;
  filenames.push_back("AIS-results.mat");
  filenames.push_back("BPcascade-results.mat");
  filenames.push_back("DE-results.mat");
  filenames.push_back("NEAT-results.mat");
  filenames.push_back("PSO-results.mat");

  vector<string> input_files;

  cout<<"Using the following results files:"<<endl;
  for(unsigned int i=0;i<res_files.size();i++){
    for(unsigned int j=0;j<filenames.size();j++){
      input_files.push_back(res_files[i]+filenames[j]);
      cout<<"\t"<<input_files[i]<<endl;
    }
    cout<<endl;
  }

  
  mat m=load_res_mat(input_files[0]);
  mat s_m=summarize(m,nb_rows_summarized_table);
  cout<<"summarized result table for '"<<input_files[0]<<"': "<<endl;
  s_m.raw_print();
  
  cout<<"finished"<<endl;
  return 0;
}
