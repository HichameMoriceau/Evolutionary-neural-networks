#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>

using namespace std;

/*
	It can happen that out of lack of focus, you make the mistake of erasing/writing on a data-set.
	This script allows to convert a <data-set.data> (C++ FANN library format) back into a <data-set.csv>.

	Compile & Run with: 
	# g++ fann_to_csv_formatter.cpp -std=c++11 -o runme
	# ./runme
*/
int main(){
	string data_set_name_without_suffix = "breast-cancer-malignantOrBenign-data-transformed";

	// instantiate and open both files
	ifstream fann_format_file(data_set_name_without_suffix + ".data");
	ofstream recovered_file;
	recovered_file.open(data_set_name_without_suffix + ".csv", ios::out);
	
	if(fann_format_file.is_open() && recovered_file.is_open()){
		string line;
		unsigned int i=0;
		string previous_line = "";

		// parse <data-set.data>
		while(getline(fann_format_file, line)) {
			// discard header
			if(i!=0){
				if(i%2 != 0){
					// retrieve <features>
					//cout << "feature line = " << line << endl;
				}else{
					// retrieve <expected output>
					string tmp 			= previous_line + line;
					string rebuilt_line = "";
					for(unsigned int c=0 ;c<tmp.length(); c++){
						if(tmp[c] == ' ')
							tmp[c] = ',';
					}
					rebuilt_line = tmp;
					//cout << rebuilt_line << endl;
					recovered_file << rebuilt_line << endl;
				}
				previous_line = line;
			}
			i++;
		}
		cout << "Data set " << data_set_name_without_suffix << ".csv was successfully recovered." << endl;
	}else
		cout << "Something went wrong when opening the files." << endl;
	

	fann_format_file.close();
	recovered_file.close();
	return(0);
}