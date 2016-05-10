#ifndef DATA_SET_H
#define DATA_SET_H

#include <armadillo>
#include <string>
#include <sstream>
#include <exception>

using namespace std;
using namespace arma;

struct data_subset{
  // features
  mat X;
  // ground truth (labels)
  mat Y;
};

class invalid_dataset_exception : public exception{
    virtual const char* what() const throw(){
        return "This program doesn't know the existence of this data-set";
    }
};

class InnapropriateValidationFoldIndex: public exception{
    virtual const char* what() const throw()
    {
        return "The value provided for identifying the validation fold exceeds the selected number of folds.";
    }
};

class Data_set
{
public:

    mat             data;

    data_subset     training_set;

    data_subset     validation_set;

    string          data_set_filename;

    string          octave_variable_name_performances_VS_nb_epochs;

    string          octave_variable_name_cost_training_set_size;

    string          octave_variable_name_cost_validation_set_size;

    string          octave_variable_name_scores_pop_size;

    string          result_filename;

    unsigned int    nb_prediction_classes;

public:
                    // ctors
                    Data_set();
                    Data_set(string full_path);
                    Data_set(unsigned int data_set_index);

    void            check_fann_format_available(mat D, string data_set_filename);


    void            select_data_set(unsigned int chosen_data_set_index);

    void            set_data_set(unsigned int chosen_data_set_index, string &data_set_filename);

    string          get_data_set_info(mat data_set);

    unsigned int    find_nb_prediction_classes(mat D);

    void            subdivide_data_cross_validation(unsigned int index_validation_fold, unsigned int nb_folds);

private:

    /**
     * @brief standardize
     * @param D data-set to be standardized
     * @return  performs feature scaling and mean normalization
     *          on the given data-set.
     */
    void            standardize(mat &D);

    unsigned int    count_nb_elements_equal_to(vec V, double value);


};

#endif // DATA_SET_H
