#include "data_set.h"

Data_set::Data_set()
{
    select_data_set(0);
}

Data_set::Data_set(string full_path)
{
    mat D;
    D.load(full_path);
    standardize(D);
    D = shuffle(D);
    data = D;
    subdivide_data_cross_validation(9,10);
}

Data_set::Data_set(unsigned int data_set_index)
{
    select_data_set(data_set_index);
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
        OCTAVE_cost_training_set_size
                = "MalignantOrBenign_results_increasingly_large_training_set";
        OCTAVE_cost_validation_set_size
                = "MalignantOrBenign_results_increasingly_large_validation_set";
        OCTAVE_scores_pop_size
                = "MalignantOrBenign_results_increasing_pop_size";

        result_filename = "data/breastCancer_MalignantOrBenign-results.mat";
        break;

    case 1 :
        data_set_filename
                = "data/breast-cancer-recurrence-data-transformed.csv";
        OCTAVE_perfs_VS_nb_epochs
                = "breastCancer_RecurrenceOrNot_results";
        OCTAVE_cost_training_set_size
                = "RecurrenceOrNot_results_increasingly_large_training_set";
        OCTAVE_cost_validation_set_size
                = "RecurrenceOrNot_results_increasingly_large_validation_set";
        OCTAVE_scores_pop_size
                = "RecurrenceOrNot_results_increasing_pop_size";

        result_filename = "data/breastCancer_RecurrenceOrNot-results.mat";
        break;

    case 2 :
        data_set_filename
                = "data/haberman-data-transformed.csv";
        OCTAVE_perfs_VS_nb_epochs
                = "haberman_results";
        OCTAVE_cost_training_set_size
                = "haberman_results_increasingly_large_training_set";
        OCTAVE_cost_validation_set_size
                = "haberman_results_increasingly_large_validation_set";
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
    standardize(D);
    D = shuffle(D);
    data = D;
    find_nb_prediction_classes(D);
    subdivide_data_cross_validation(1,10);
}

void Data_set::select_data_set(string filename) {

    string base_name = filename.substr(5,filename.size()-9);
    // set Octave variable names
    data_set_filename = filename;
    OCTAVE_perfs_VS_nb_epochs = base_name + "_results";
    OCTAVE_cost_training_set_size = base_name + "_results_increasingly_large_training_set";
    OCTAVE_cost_validation_set_size = base_name + "_results_increasingly_large_validation_set";
    OCTAVE_scores_pop_size = base_name + "_results_increasing_pop_size";
    result_filename = "data/" + base_name + "_results.mat";
    // replace all '-' by '_'
    std::replace(OCTAVE_perfs_VS_nb_epochs.begin(), OCTAVE_perfs_VS_nb_epochs.end(), '-','_');
    std::replace(OCTAVE_cost_training_set_size.begin(), OCTAVE_cost_training_set_size.end(), '-','_');
    std::replace(OCTAVE_cost_validation_set_size.begin(), OCTAVE_cost_validation_set_size.end(), '-','_');
    std::replace(OCTAVE_scores_pop_size.begin(), OCTAVE_scores_pop_size.end(), '-','_');
    std::replace(result_filename.begin(), result_filename.end(), '-','_');
    // load corresponding data set
    mat D;
    D.load(data_set_filename);
    standardize(D);
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
    standardize(D);
    D = shuffle(D);
    data = D;
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
            // round after one decimal places
            //normalized_D(i,j) = round(D(i,j) * 10) / 10.0f;
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

void Data_set::subdivide_data_cross_validation(unsigned int index_validation_fold, unsigned int nb_folds) {
    if(index_validation_fold >= nb_folds) throw new InnapropriateValidationFoldIndex;

    nb_prediction_classes = find_nb_prediction_classes(data);
    if(nb_prediction_classes==2){
        // count how many element each class contains
        unsigned int count_pos=0, count_neg=0;
        for(unsigned int i=0; i<data.n_rows; i++){
            if(data(i,data.n_cols-1) == 0)
                count_neg++;
            else
                count_pos++;
        }

        // determine which class dominates the other
        unsigned int outnumbered_type;
        if(count_pos < count_neg)
            outnumbered_type = 1;
        else
            outnumbered_type = 0;

        // separate data in corresponding subsets
        mat minority_examples, majority_examples;
        for(unsigned int i=0; i<data.n_rows; i++){
            // if current example is one of the outnumbered type
            if(data(i,data.n_cols-1) == outnumbered_type){
                minority_examples = join_vert(minority_examples, data.row(i));
            }else{
                majority_examples = join_vert(majority_examples, data.row(i));
            }
        }

        // compute ratio of how imbalanced the data set is
        unsigned int minority_range = minority_examples.n_rows / nb_folds;
        unsigned int majority_range = majority_examples.n_rows / nb_folds;

        // build training and validation set while preserving the imbalance ratio
        mat training_section, validation_section;
        for(unsigned int i=0; i<nb_folds; ++i) {
            if(i != index_validation_fold) {
                mat majority_section = majority_examples.rows(round(i*majority_range), (i+1)*majority_range);
                mat minority_section = minority_examples.rows(round(i*minority_range), (i+1)*minority_range);
                training_section = join_vert(training_section, majority_section);
                training_section = join_vert(training_section, minority_section);
            }else{
                mat majority_section = majority_examples.rows(round(i*majority_range), (i+1)*majority_range);
                mat minority_section = minority_examples.rows(round(i*minority_range), (i+1)*minority_range);
                validation_section = join_vert(validation_section, majority_section);
                validation_section = join_vert(validation_section, minority_section);
            }
        }
        // set training and validation folds
        unsigned int dataset_nb_features = data.n_cols-1;
        unsigned int dataset_last_column_index = data.n_cols-1;
        training_set.X = training_section.cols(0,dataset_nb_features-1);
        training_set.Y = training_section.col(dataset_last_column_index);
        validation_set.X = validation_section.cols(0,dataset_nb_features-1);
        validation_set.Y = validation_section.col(dataset_last_column_index);
    }else if(nb_prediction_classes>2){
        unsigned int range = data.n_rows / nb_folds;
        range--;
        // build training and validation set while preserving the imbalance ratio
        mat training_section, validation_section;
        for(unsigned int i=0; i<nb_folds; ++i){
            if(i != index_validation_fold) {
                training_section = join_vert(training_section, data.rows(round(i*range), (i+1)*range));
            }else{
                validation_section = join_vert(validation_section, data.rows(round(i*range), (i+1)*range));
            }
        }
        // set training and validation folds
        unsigned int dataset_nb_features = data.n_cols-1;
        unsigned int dataset_last_column_index = data.n_cols-1;
        training_set.X = training_section.cols(0,dataset_nb_features-1);
        training_set.Y = training_section.col(dataset_last_column_index);
        validation_set.X = validation_section.cols(0,dataset_nb_features-1);
        validation_set.Y = validation_section.col(dataset_last_column_index);
    }
}
