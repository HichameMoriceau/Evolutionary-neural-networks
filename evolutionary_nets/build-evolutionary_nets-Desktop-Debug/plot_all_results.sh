#!/bin/bash

(exec "./plotter.m" "data/wine_data_transformed_results/results.mat" "Wine")

(exec "./plotter.m" "data/iris_data_transformed_results/results.mat" "Iris")

(exec "./plotter.m" "data/breast_cancer_malignantOrBenign_data_transformed_results/results.mat" "BC-Malignancy")

(exec "./plotter.m" "data/breast_cancer_recurrence_data_transformed_results/results.mat" "BC-Recurrence")
