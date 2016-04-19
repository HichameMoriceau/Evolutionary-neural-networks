#!/usr/bin/octave -qf

fprintf("Running plotter.");

# load experimentation results
load data/breastCancer_MalignantOrBenign-results.mat
load data/breastCancer_RecurrenceOrNot-results.mat
load data/haberman-results.mat
load data/backprop_perfs.csv


# write out plots in image directory
cd 'images/';

population_size	= breastCancer_MalignantOrBenign_results(1,9);
nb_replicates   = breastCancer_MalignantOrBenign_results(1,end);

breastCancer_MalignantOrBenign_AVRG_CV_FITNESS = breastCancer_MalignantOrBenign_results(1,end-3)
breastCancer_RecurrenceOrNot_AVRG_CV_FITNESS = breastCancer_RecurrenceOrNot_results(1,end-3)
haberman_AVRG_CV_FITNESS = haberman_results(1,end-3)

breastCancer_MalignantOrBenign_AVRG_CV_ACC = breastCancer_MalignantOrBenign_results(1,end-2)
breastCancer_RecurrenceOrNot_AVRG_CV_ACC = breastCancer_RecurrenceOrNot_results(1,end-2)
haberman_AVRG_CV_ACC = haberman_results(1,end-2)

#
#
# PLOTTING DATA SET : HABERMAN
#
#

#
# TRAINING PERFS
#
align_right_haberman = haberman_results(end,1) - (haberman_results(end,1)/3) - 26;
errorbar(haberman_results(:,1), haberman_results(:,4), err_haberman_results);
hold on;
grid on;
plot(haberman_results(:,1),haberman_results(:,3), 'k', 'LineWidth', 1);
plot(haberman_results(:,1),haberman_results(:,4), 'r', 'LineWidth', 1);
plot(haberman_results(:,1),haberman_results(:,17), 'm', 'LineWidth', 1);
axis([0, haberman_results(end,1), 0, 100]);
title( strcat("Haberman's survival - population size : ", num2str(population_size), "[", num2str(nb_replicates), " replicates]"));
legend('[Error amongst replicates] Corrected sample standard deviation','Prediction accuracy','F1 score of the best individual', 'F1 score of the ensemble', "location", "southeast");
xlabel('Number of generations');
ylabel('Performance of best individual on each cross-validation set');
text(align_right_haberman, 46 , strcat("Average score on CV sets =", num2str(haberman_AVRG_CV_FITNESS)));
text(align_right_haberman, 42 , strcat("Average acc on CV sets =", num2str(haberman_AVRG_CV_ACC)));
# add topology description
text(align_right_haberman, 38 , "Final topology:");
text(align_right_haberman, 34 , strcat("NB inputs            =", num2str(haberman_results(end,10))) );
text(align_right_haberman, 30 , strcat("NB hidden units  =", num2str(haberman_results(end,11))) );
text(align_right_haberman, 26 , strcat("NB outputs          =", num2str(haberman_results(end,12))) );
text(align_right_haberman, 22 , strcat("NB hidden layers=", num2str(haberman_results(end,13))) );

print -dpng 'haberman-performancesVSepochs.png';
hold off;

plot(haberman_results(:,1),haberman_results(:,5), 'b', 'LineWidth', 1);
hold on;
grid on;
legend('Mean Squared Error', "location", "northeast");
xlabel('Number of generations');
ylabel('MSE of best individual on each cross-validation set');
print -dpng 'haberman-MSEVSepochs.png';
hold off;

#
# POPULATION SCORE STATS
#

plot(haberman_results(:,1), haberman_results(:,5), 'b', 'LineWidth', 1);
grid on;
hold on;
title("Haberman's survival - Variance of individuals's scores against nb generations");
legend('Variance', "location", "northeast");
xlabel('Number of generations');
ylabel('Variance on validation-set');
print -dpng 'haberman-varianceVSepochs.png';
hold off;

plot(haberman_results(:,1), haberman_results(:,6), 'b', 'LineWidth', 1);
hold on;
grid on;
title("Haberman's survival - Standard deviation of individuals's scores against nb generations");
legend('Standard deviation', "location", "northeast");
xlabel('Number of generations');
ylabel('Standard deviation on validation-set');
print -dpng 'haberman-stddevVSepochs.png';
hold off;

plot(haberman_results(:,1), haberman_results(:,7), 'b', 'LineWidth', 1);
hold on;
grid on;
title("Haberman's survival - Mean of individuals's scores against nb generations");
legend('Mean', "location", "southeast");
xlabel('Number of generations');
ylabel('Mean of population score on validation-set');
print -dpng 'haberman-meanVSepochs.png';
hold off;




#
# LEARNING CURVES
#
#{
plot(haberman_results_increasingly_large_training_set(:,1),haberman_results_increasingly_large_training_set(:,2), 'r');
hold on;
title("Haberman's survival - Learning curves");
plot(haberman_results_increasingly_large_training_set(:,1),haberman_results_increasingly_large_training_set(:,4), 'b');
legend('Error training-set', 'Error validation-set');
xlabel('m (training set size)');
ylabel('Error');
print -dpng 'haberman-learning_curve-increasingly_large_training_set.png';
hold off;
#}





#
#
# PLOTTING DATA SET : MALIGNANT OR BENIGN
#
#

align_right_malignant = breastCancer_MalignantOrBenign_results(end,1) - (breastCancer_MalignantOrBenign_results(end,1)/3) - 26;
#
# TRAINING PERFS
#
errorbar(breastCancer_MalignantOrBenign_results(:,1),breastCancer_MalignantOrBenign_results(:,4), err_breastCancer_MalignantOrBenign_results);
hold on;
grid on;
plot(breastCancer_MalignantOrBenign_results(:,1),breastCancer_MalignantOrBenign_results(:,3), 'r', 'LineWidth', 1);
plot(breastCancer_MalignantOrBenign_results(:,1),breastCancer_MalignantOrBenign_results(:,4), 'k', 'LineWidth', 1);
plot(breastCancer_MalignantOrBenign_results(:,1),breastCancer_MalignantOrBenign_results(:,17), 'm', 'LineWidth', 1);
axis([0, breastCancer_MalignantOrBenign_results(end,1), 0, 100]);
title( strcat("Breast cancer (Malignant?) - population size : ", num2str(population_size), "[", num2str(nb_replicates), " replicates]" ));
legend('[Error amongst replicates] Corrected sample standard deviation', 'Prediction accuracy','F1 score of the best individual', 'F1 score of the ensemble', "location", "southeast");
xlabel('Number of generations');
ylabel('Performance of best individual on each cross-validation set');
text(align_right_malignant, 46 , strcat("Average score on CV sets =", num2str(breastCancer_MalignantOrBenign_AVRG_CV_FITNESS)));
text(align_right_malignant, 42 , strcat("Average acc on CV sets =", num2str(breastCancer_MalignantOrBenign_AVRG_CV_ACC)));
# add topology description
text(align_right_malignant, 38 , "Final topology:");
text(align_right_malignant, 34 , strcat("NB inputs            =", num2str(breastCancer_MalignantOrBenign_results(end,10))) );
text(align_right_malignant, 30 , strcat("NB hidden units  =", num2str(breastCancer_MalignantOrBenign_results(end,11))) );
text(align_right_malignant, 25 , strcat("NB outputs          =", num2str(breastCancer_MalignantOrBenign_results(end,12))) );
text(align_right_malignant, 22 , strcat("NB hidden layers=", num2str(breastCancer_MalignantOrBenign_results(end,13))) );
print -dpng 'malignant-performancesVSepochs.png';
hold off;

#backprop_perfs = backprop_perfs(1:(size(breastCancer_MalignantOrBenign_results))(1),:);
plot(breastCancer_MalignantOrBenign_results(:,1),breastCancer_MalignantOrBenign_results(:,2), 'b', 'LineWidth', 1);
hold on;
grid on;
legend('Mean Squared Error', "location", "northeast");
xlabel('Number of generations');
ylabel('MSE of best individual on each cross-validation set');
print -dpng 'malignant-MSEVSepochs.png';
hold off;

#
# POPULATION SCORE STATS
#

plot(breastCancer_MalignantOrBenign_results(:,1), breastCancer_MalignantOrBenign_results(:,5), 'b', 'LineWidth', 1);
hold on;
grid on;
title("Breast cancer (Malignant?) - Variance of individuals's scores against nb generations");
legend('Variance', "location", "northeast");
xlabel('Number of generations');
ylabel('Variance on validation-set');
print -dpng 'malignant-varianceVSepochs.png';
hold off;

plot(breastCancer_MalignantOrBenign_results(:,1), breastCancer_MalignantOrBenign_results(:,6), 'b', 'LineWidth', 1);
hold on;
grid on;
title("Breast cancer (Malignant?) - Standard deviation of individuals's scores against nb generations");
legend('Standard deviation', "location", "northeast");
xlabel('Number of generations');
ylabel('Standard deviation on validation-set');
print -dpng 'malignant-stddevVSepochs.png';
hold off;

plot(breastCancer_MalignantOrBenign_results(:,1), breastCancer_MalignantOrBenign_results(:,7), 'b', 'LineWidth', 1);
hold on;
grid on;
title("Breast cancer (Malignant?) - Mean of individuals's scores against nb generations");
legend('Mean', "location", "southeast");
xlabel('Number of generations');
ylabel('Mean of population score on validation-set');
print -dpng 'malignant-meanVSepochs.png';
hold off;

#{
#
# LEARNING CURVES DATA-SET SIZE
#

plot(MalignantOrBenign_results_increasingly_large_training_set(1,:),MalignantOrBenign_results_increasingly_large_training_set(:,2), 'r');
hold on;
title("Breast cancer (Malignant?) - Data-set Size Learning Curves");
plot(MalignantOrBenign_results_increasingly_large_training_set(:,1),MalignantOrBenign_results_increasingly_large_training_set(:,4), 'b');
legend('Error training-set', 'Error validation-set');
xlabel('m (training set size)');
ylabel('Error');
print -dpng 'malignant-learning_curve-increasingly_large_training_set.png';
hold off;

fprintf('printing learning curves pop size');
errorbar(pop_sizes(:,1), MalignantOrBenign_results_increasing_pop_size(:,1), err_MalignantOrBenign_results_increasing_pop_size);
hold on;
grid on;
axis([0, pop_sizes(end,1), 0, 100]);
title(strcat("Breast cancer (Malignant?) - Population Size Learning Curves", "[", num2str(nb_replicates), " replicates]"));
legend('average F1 score of best individual on validation-set','[Error amongst replicates] Corrected Sample Standard Deviation', "location", "southeast");
print -dpng 'malignant-learning_curve_pop_size.png';
hold off;
#}







#
#
# PLOTTING DATA SET : RECCURENCE OR NOT
#
#



#
# TRAINING PERFS
#

align_right_recurrence = breastCancer_RecurrenceOrNot_results(end,1)- (breastCancer_RecurrenceOrNot_results(end,1)/3) - 26;
errorbar(breastCancer_RecurrenceOrNot_results(:,1),breastCancer_RecurrenceOrNot_results(:,3), err_breastCancer_RecurrenceOrNot_results);
hold on;
grid on;
plot(breastCancer_RecurrenceOrNot_results(:,1),breastCancer_RecurrenceOrNot_results(:,3), 'r', 'LineWidth', 1);
plot(breastCancer_RecurrenceOrNot_results(:,1),breastCancer_RecurrenceOrNot_results(:,4), 'k', 'LineWidth', 1);
plot(breastCancer_RecurrenceOrNot_results(:,1),breastCancer_RecurrenceOrNot_results(:,17), 'm', 'LineWidth', 1);
title( strcat("Breast cancer (Recurrence?) - population size : ", num2str(population_size), "[", num2str(nb_replicates), " replicates]"));
axis([0, breastCancer_RecurrenceOrNot_results(end,1), 0, 100]);
legend('[Error amongst replicates] Corrected sample standard deviation','Prediction accuracy','F1 score of the best individual', 'F1 score of the ensemble', "location", "southeast");
xlabel('Number of generations');
ylabel('Performance of best individual on each cross-validation set');
text(align_right_recurrence, 46 , strcat("Average score on CV sets =", num2str(breastCancer_RecurrenceOrNot_AVRG_CV_FITNESS)));
text(align_right_recurrence, 42 , strcat("Average acc on CV sets =", num2str(breastCancer_RecurrenceOrNot_AVRG_CV_ACC)));
# add topology description
text(align_right_recurrence, 38 , "Final topology:");
text(align_right_recurrence, 34 , strcat("NB inputs            =", num2str(breastCancer_RecurrenceOrNot_results(end,10))) );
text(align_right_recurrence, 30 , strcat("NB hidden units  =", num2str(breastCancer_RecurrenceOrNot_results(end,11))) );
text(align_right_recurrence, 26 , strcat("NB outputs          =", num2str(breastCancer_RecurrenceOrNot_results(end,12))) );
text(align_right_recurrence, 22 , strcat("NB hidden layers=", num2str(breastCancer_RecurrenceOrNot_results(end,13))) );
print -dpng 'recurrence-performancesVSepochs.png';
hold off;



plot(breastCancer_RecurrenceOrNot_results(:,1),breastCancer_RecurrenceOrNot_results(:,5), 'b', 'LineWidth', 1);
hold on;
grid on;
legend('Mean Squared Error', "location", "northeast");
xlabel('Number of generations');
ylabel('MSE of best individual on each cross-validation set');
print -dpng 'recurrence-MSEVSepochs.png';
hold off;

#
# POPULATION SCORE STATS
#

plot(breastCancer_RecurrenceOrNot_results(:,1), breastCancer_RecurrenceOrNot_results(:,5), 'b', 'LineWidth', 1);
hold on;
grid on;
title("Breast cancer (Recurrence?) - Variance of individuals's scores against nb generations");
legend('Variance', "location", "northeast");
xlabel('Number of generations');
ylabel('Variance on validation-set');
print -dpng 'recurrence-varianceVSepochs.png';
hold off;

plot(breastCancer_RecurrenceOrNot_results(:,1), breastCancer_RecurrenceOrNot_results(:,6), 'b', 'LineWidth', 2);
hold on;
grid on;
title("Breast cancer (Recurrence?) - Standard deviation of individuals's scores against nb generations");
legend('Standard deviation', "location", "northeast");
xlabel('Number of generations');
ylabel('Standard deviation on validation-set');
print -dpng 'recurrence-stddevVSepochs.png';
hold off;

plot(breastCancer_RecurrenceOrNot_results(:,1), breastCancer_RecurrenceOrNot_results(:,7), 'b', 'LineWidth', 2);
hold on;
grid on;
title("Breast cancer (Recurrence?) - Mean of individuals's scores against nb generations");
legend('Mean', "location", "southeast");
xlabel('Number of generations');
ylabel('Mean of population score on validation-set');
print -dpng 'recurrence-meanVSepochs.png';
hold off;


#
# LEARNING CURVES
#

plot(RecurrenceOrNot_results_increasingly_large_training_set(:,1),RecurrenceOrNot_results_increasingly_large_training_set(:,2), 'r');
hold on;
grid on;
title("Breast cancer (Recurrence?) - Learning curves");
plot(RecurrenceOrNot_results_increasingly_large_training_set(:,1),RecurrenceOrNot_results_increasingly_large_training_set(:,4), 'b');
legend('Error training-set', 'Error validation-set');
xlabel('m (training set size)');
ylabel('Error');
print -dpng 'recurrence-learning_curve-increasingly_large_training_set.png';
hold off;


fprintf("terminated.\n")


