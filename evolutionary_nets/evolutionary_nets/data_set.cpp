#include "data_set.h"

Data_set::Data_set()
{

}

Data_set::Data_set(string full_path)
{
    mat D;
    D.load(full_path);
    // perform feature scaling
    standardize(D);
    // randomize the order of the examples
    D = shuffle(D);
    data = D;
    // divide into training, validation and test sections
    subdivide_data_cross_validation(9,10);
}

Data_set::Data_set(unsigned int data_set_index)
{
    select_data_set(data_set_index);
    // divide into training, validation and test sections
    subdivide_data_cross_validation(9,10);
}

void Data_set::check_fann_format_available(mat D, string data_set_filename){
    string converted_data_set_filename = data_set_filename.substr(0,data_set_filename.find_last_of(".")) + ".data";
    bool file_already_exist;
    ifstream f(converted_data_set_filename.c_str());
    if(f.good()){
        f.close();
        file_already_exist = true;
    }else{
        f.close();
        file_already_exist = false;
    }

    if(!file_already_exist) {

        //
        // transform "csv" to "fann library" format
        //

        ofstream new_data_file;
        new_data_file.open(converted_data_set_filename, ios::out);

        // add "fann library" header
        new_data_file << D.n_rows << " " << D.n_cols-1 << " " << 1 << endl;

        // reformat each line
        string new_line;
        for(unsigned int i=0; i<D.n_rows; i++){
            new_line = "";
            for(unsigned int a=0; a< D.n_cols-1; a++){
                new_line += to_string(D(i,a)) + " ";
            }
            // append inputs
            new_data_file << new_line << endl;
            // append expected output
            new_data_file << D(i,D.n_cols-1) << endl;
        }
    }
}

// load chosen numerical data-set and returns appropriate relative path to result file and octave variable names
void Data_set::select_data_set(unsigned int chosen_data_set_index) {
    switch(chosen_data_set_index) {

    case 0 :
        data_set_filename
                = "data/breast-cancer-malignantOrBenign-data-transformed.csv";
        OCTAVE_perfs_VS_nb_epochs
                = "breastCancer_MalignantOrBenign_results";
        OCTAVE_cost_train_set_size
                = "MalignantOrBenign_results_increasingly_large_train_set";
        OCTAVE_cost_val_set_size
                = "MalignantOrBenign_results_increasingly_large_val_set";
        OCTAVE_scores_pop_size
                = "MalignantOrBenign_results_increasing_pop_size";

        result_filename = "data/breastCancer_MalignantOrBenign-results.mat";
        break;

    case 1 :
        data_set_filename
                = "data/breast-cancer-recurrence-data-transformed.csv";
        OCTAVE_perfs_VS_nb_epochs
                = "breastCancer_RecurrenceOrNot_results";
        OCTAVE_cost_train_set_size
                = "RecurrenceOrNot_results_increasingly_large_train_set";
        OCTAVE_cost_val_set_size
                = "RecurrenceOrNot_results_increasingly_large_val_set";
        OCTAVE_scores_pop_size
                = "RecurrenceOrNot_results_increasing_pop_size";

        result_filename = "data/breastCancer_RecurrenceOrNot-results.mat";
        break;

    case 2 :
        data_set_filename
                = "data/haberman-data-transformed.csv";
        OCTAVE_perfs_VS_nb_epochs
                = "haberman_results";
        OCTAVE_cost_train_set_size
                = "haberman_results_increasingly_large_train_set";
        OCTAVE_cost_val_set_size
                = "haberman_results_increasingly_large_val_set";
        OCTAVE_scores_pop_size
                = "haberman_results_increasing_pop_size";

        result_filename = "data/haberman-results.mat";
        break;

    default:
        invalid_dataset_exception ex;
        throw ex;
    }

    mat D;
    D.load(data_set_filename);
    // perform feature scaling
    standardize(D);
    // randomize the order of the examples
    D = shuffle(D);
    data = D;
    // update nb possible outputs
    find_nb_prediction_classes(D);
    // divide into training, validation and test sections
    subdivide_data_cross_validation(1,10);
}

void Data_set::select_data_set(string filename) {
    string base_name = filename.substr(5,filename.size()-9);
    // set Octave variable names
    data_set_filename = filename;
    OCTAVE_perfs_VS_nb_epochs = base_name + "_results";
    OCTAVE_cost_train_set_size = base_name + "_results_increasingly_large_train_set";
    OCTAVE_cost_val_set_size = base_name + "_results_increasingly_large_val_set";
    OCTAVE_scores_pop_size = base_name + "_results_increasing_pop_size";
    result_filename = "data/" + base_name + "_results.mat";
    // replace all '-' by '_'
    std::replace(OCTAVE_perfs_VS_nb_epochs.begin(), OCTAVE_perfs_VS_nb_epochs.end(), '-','_');
    std::replace(OCTAVE_cost_train_set_size.begin(), OCTAVE_cost_train_set_size.end(), '-','_');
    std::replace(OCTAVE_cost_val_set_size.begin(), OCTAVE_cost_val_set_size.end(), '-','_');
    std::replace(OCTAVE_scores_pop_size.begin(), OCTAVE_scores_pop_size.end(), '-','_');
    std::replace(result_filename.begin(), result_filename.end(), '-','_');
    // load corresponding data set
    mat D;
    D.load(data_set_filename);
    // perform feature scaling
    standardize(D);
    // randomize the order of the examples
    D = shuffle(D);
    data = D;
    // update nb possible outputs
    find_nb_prediction_classes(D);
    // set initial training and test folds
    subdivide_data_cross_validation(1,10);
}

// load chosen data-set and returns appropriate relative path to result file and octave variable names
void Data_set::set_data_set(unsigned int chosen_data_set_index, string &data_set_filename) {
    select_data_set( chosen_data_set_index);
    mat D;
    D.load(data_set_filename);
    // perform feature scaling
    standardize(D);
    // randomize the order of the examples
    D = shuffle(D);
    data = D;
    // divide into training, validation and test sections
    subdivide_data_cross_validation(1,10);
}

// returns number of positive, negative and total examples of <D>
string Data_set::get_data_set_info(mat D) {
  // return variable
  stringstream ss;
  ss << "nb examples = " << D.n_rows << "\n";
  ss << "nb prediction classes = " << find_nb_prediction_classes(D) << "\n";
  return ss.str();
}

unsigned int Data_set::find_nb_prediction_classes(mat D){
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
    // update nb output units
    nb_prediction_classes = prediction_classes.size();
    return nb_prediction_classes;
}

void Data_set::standardize(mat &D){
    // for each column
    for(unsigned int i=0; i<D.n_rows; ++i){
        // for each element in this column (do not standardize target attribute)
        for(unsigned int j=0; j<D.n_cols-1; ++j){
            // apply feature scaling and mean normalization
            D(i,j) = (D(i,j) - mean(D.col(j))) / (max(D.col(j))-min(D.col(j)));
        }
    }
}

unsigned int Data_set::count_nb_elements_equal_to(vec V, double value){
  unsigned int count = 0;
  for(unsigned int i = 0 ; i < V.n_rows ; ++i)
    if(V(i) == value)
      count++;
  return count;
}

void Data_set::subdivide_data_cross_validation(unsigned int index_val_fold, unsigned int nb_folds) {
    if(index_val_fold >= nb_folds) throw new InnapropriateValidationFoldIndex;

    nb_prediction_classes = find_nb_prediction_classes(data);
    unsigned int range=((data.n_rows)*80/100)/nb_folds;//= data.n_rows / nb_folds;
    // build training and validation set while preserving the imbalance ratio
    mat train_section, val_section, test_section;
    for(unsigned int i=0; i<nb_folds; ++i){
        if(i != index_val_fold) {
            //cout<<"training section indices: "<<round(i*range)<<" up to "<<(i+1)*range<<endl;
            train_section = join_vert(train_section, data.rows(round(i*range), (i+1)*range));
        }else{
            //cout<<"validation section indices: "<<round(i*range)<<" up to "<<(i+1)*range<<endl;
            val_section = join_vert(val_section, data.rows(round(i*range), (i+1)*range));
        }
    }
    unsigned int index_low=data.n_rows-(double(data.n_rows)*(20.0/100));
    unsigned int index_high=data.n_rows-1;
    //cout<<"test section indices: "<<index_low<<" up to "<<index_high<<endl;
    test_section=data.rows(index_low, index_high);
    // set training and validation folds
    unsigned int dataset_nb_features = data.n_cols-1;
    unsigned int dataset_last_column_index = data.n_cols-1;
    train_set.X = train_section.cols(0,dataset_nb_features-1);
    train_set.Y = train_section.col(dataset_last_column_index);
    val_set.X = val_section.cols(0,dataset_nb_features-1);
    val_set.Y = val_section.col(dataset_last_column_index);
    test_set.X=test_section.cols(0,dataset_nb_features-1);
    test_set.Y=test_section.col(dataset_last_column_index);
}
