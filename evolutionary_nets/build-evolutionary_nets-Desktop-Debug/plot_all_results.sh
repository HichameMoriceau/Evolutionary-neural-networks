#!/bin/bash

(exec "./plotter.m" "data/breast_cancer_malignantOrBenign_data_transformed_results/results.mat" "BC-Malignancy")

(exec "./plotter.m" "data/breast_cancer_recurrence_data_transformed_results/results.mat" "BC-Recurrence")

(exec "./plotter.m" "data/haberman_data_transformed_results/results.mat" "Haberman")
