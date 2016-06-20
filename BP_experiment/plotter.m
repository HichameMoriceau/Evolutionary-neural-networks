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
nb_gens         = results(end,1);

TEST_SCORE = results(1,end-2);
TEST_ACC = results(1,end-1);

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
text(align_right, 50 , strcat("SCORE on unseen test data =", num2str(TEST_SCORE)));
text(align_right, 46 , strcat("ACC on unseen test data =", num2str(TEST_ACC)));
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
x=results(end,20);
y=100;
axis([0, results(end,20), 0, 100]);
title( strcat(data_set_name, " - population size : ", num2str(population_size), " - ", num2str(nb_replicates), " replicates"));
legend('[Error amongst replicates] Corrected sample standard deviation','Best indiv accuracy on training set','Best indiv f1 score on training set', 'Best indiv accuracy on CV set', "location", "southeast");
xlabel('Number of calls to the error function');
ylabel('Performance of best individual while training');
text(align_right, 50 , strcat("SCORE on unseen test data =", num2str(TEST_SCORE)));
text(align_right, 46 , strcat("ACC on unseen test data =", num2str(TEST_ACC)));
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
# RESULT TABLE
#

nb_err_calls=results(end,1)-results(1,1);

# calculate index values of 10 evenly separated rows
indexes=ones(10,1);
for i = [1:10]
    indexes(i)=nb_err_calls*(i*10/100);
endfor
    
indexes=round(indexes);
nb_cols=size(results)(2);

# store each row in result table
table=ones(10,nb_cols);
for i = [1:10]
    table(i,:)=results(indexes(i),:);
endfor

format("compact")
cd ../data/;


table_filename=strcat("BPresults/BP-",data_set_name, "-res-table.csv");
fprintf("Saved as: \n\t->\"%s\"\n" , table_filename);
dlmwrite(table_filename, table, ", ")

fprintf("Columns:\nnb calls to err function, mse, train acc, train score, pop score var, pop score stddev, pop score mean, pop score median, pop size, nb inputs, nb hid units, nb outputs, nb hid layers, is pop diverse?, mutation scheme (0=DE/RAND, 1=DE/BEST), ensemble acc, ensemble score, val acc, val score, gens\n");