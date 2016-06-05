#!/usr/bin/octave -qf

arg_list = argv ();
filename = arg_list{1};
data_set_name = arg_list{2};

fprintf("Plotting curves using \n\t->\"%s\"\n" , filename);

# load experimentation results
load("-text",filename);

# write out plots in image directory
cd 'images/';

population_size	= results(1,9);
nb_replicates   = results(1,end);

TEST_SCORE = results(1,end-3);
TEST_ACC = results(1,end-2);


#
# TRAINING PERFS
#
align_right = results(end,1) - (results(end,1)/3) - 26;
errorbar(results(:,1), results(:,4), err_results);
hold on;
grid on;
plot(results(:,1),results(:,4),  'r', 'LineWidth', 1);
plot(results(:,1),results(:,3),  'k', 'LineWidth', 1);
plot(results(:,1),results(:,18), 'm', 'LineWidth', 1);
axis([0, results(end,1), 0, 100]);
title( strcat(data_set_name, " - population size : ", num2str(population_size), " - ", num2str(nb_replicates), " replicates"));
legend('[Error amongst replicates] Corrected sample standard deviation','Best indiv accuracy on training set','Best indiv f1 score on training set', 'Best indiv accuracy on CV set', "location", "southeast");
xlabel('Number of generations');
ylabel('Performance of best individual while training');
text(align_right, 50 , strcat("Score on unseen test data =", num2str(TEST_SCORE)));
text(align_right, 46 , strcat("Score on unseen test data =", num2str(TEST_ACC)));
# add topology description
text(align_right, 42 , "Final topology:");
text(align_right, 38 , strcat("NB inputs            =", num2str(results(end,10))) );
text(align_right, 34 , strcat("NB hidden units  =", num2str(results(end,11))) );
text(align_right, 30 , strcat("NB outputs          =", num2str(results(end,12))) );
text(align_right, 26 , strcat("NB hidden layers=", num2str(results(end,13))) );
plot_name = strcat(data_set_name, "-perfsVSepochs.png");
print('-dpng', '-tiff', plot_name)
hold off;

#
# NB OF CALLS TO ERROR FUNCTION
#
align_right = results(end,20) - (results(end,20)/3) - 26;
errorbar(results(:,20), results(:,4), err_results);
hold on;
grid on;
plot(results(:,20),results(:,4),  'k', 'LineWidth', 1);
plot(results(:,20),results(:,3),  'r', 'LineWidth', 1);
plot(results(:,20),results(:,18), 'm', 'LineWidth', 1);
axis([0, results(end,20), 0, 100]);
title( strcat(data_set_name, " - population size : ", num2str(population_size), " - ", num2str(nb_replicates), " replicates"));
legend('[Error amongst replicates] Corrected sample standard deviation','Best indiv accuracy on training set','Best indiv f1 score on training set', 'Best indiv accuracy on CV set', "location", "southeast");
xlabel('Number of calls to the error function');
ylabel('Performance of best individual while training');
text(align_right, 50 , strcat("Score on unseen test data =", num2str(TEST_SCORE)));
text(align_right, 46 , strcat("Score on unseen test data =", num2str(TEST_ACC)));
# add topology description
text(align_right, 42 , "Final topology:");
text(align_right, 38 , strcat("NB inputs            =", num2str(results(end,10))) );
text(align_right, 34 , strcat("NB hidden units  =", num2str(results(end,11))) );
text(align_right, 30 , strcat("NB outputs          =", num2str(results(end,12))) );
text(align_right, 26 , strcat("NB hidden layers=", num2str(results(end,13))) );
plot_name = strcat(data_set_name, "-perfsVSnbcallstoerrorfunction.png");
print('-dpng', '-tiff', plot_name)
hold off;



plot(results(:,1), results(:,2), 'b', 'LineWidth', 1);
grid on;
hold on;
title(strcat(data_set_name, " - MSE against nb generations"));
legend('MSE', "location", "northeast");
xlabel('Number of generations');
ylabel('Mean Squared Error');
plot_name = strcat(data_set_name, "-MSEVSepochs.png");
print('-dpng', '-tiff', plot_name);
hold off;

#
# POPULATION SCORE STATS
#

plot(results(:,1), results(:,5), 'b', 'LineWidth', 1);
grid on;
hold on;
title(strcat(data_set_name, " - Variance of individuals's scores against nb generations"));
legend('Variance', "location", "northeast");
xlabel('Number of generations');
ylabel('Variance on validation-set');
plot_name = strcat(data_set_name, "-varianceVSepochs.png");
print('-dpng', '-tiff', plot_name);
hold off;

plot(results(:,1), results(:,6), 'b', 'LineWidth', 1);
hold on;
grid on;
title(strcat(data_set_name, " - Standard deviation of individuals's scores against nb generations"));
legend('Standard deviation', "location", "northeast");
xlabel('Number of generations');
ylabel('Standard deviation on validation-set');
plot_name = strcat(data_set_name, "-stddevVSepochs.png");
print('-dpng', '-tiff', plot_name);
hold off;

plot(results(:,1), results(:,7), 'b', 'LineWidth', 1);
hold on;
grid on;
title(strcat(data_set_name, " - Mean of individuals's scores against nb generations"));
legend('Mean', "location", "southeast");
xlabel('Number of generations');
ylabel('Mean of population score on validation-set');
plot_name = strcat(data_set_name, "-meanVSepochs.png");
print('-dpng', '-tiff', plot_name);
hold off;


#
# LEARNING CURVES
#
#{
plot(results_increasingly_large_training_set(:,1),results_increasingly_large_training_set(:,2), 'r');
hold on;
title(strcat(data_set_name, " - Learning curves"));
plot(results_increasingly_large_training_set(:,1),results_increasingly_large_training_set(:,4), 'b');
legend('Error training-set', 'Error validation-set');
xlabel('m (training set size)');
ylabel('Error');
plot_name = strcat(data_set_name, "-learning_curve-increasingly_large_training_set.png");
print('-dpng', '-tiff', plot_name);
hold off;
#}