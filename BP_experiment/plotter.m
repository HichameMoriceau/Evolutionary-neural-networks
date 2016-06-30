#!/usr/bin/octave -qf

arg_list = argv ();
filename = arg_list{1};
algorithm_name = arg_list{2};
data_set_name = arg_list{3};

alg_data = strcat(algorithm_name, "-", data_set_name);

fprintf("Plotting curves using \n\t->\"%s\"\n" , filename);

# load experimentation results
load("-text",filename);

nb_calls_to_err_func=results(:,1);
generation=results(:,2);
train_acc=results(:,3);
train_score=results(:,4);
train_mse=results(:,5);
test_acc=results(:,6);
test_score=results(:,7);
test_mse=results(:,8);
val_acc=results(:,9);
val_score=results(:,10);
val_mse=results(:,11);
pop_score_var=results(:,12);
pop_score_mean=results(:,13);
pop_size=results(:,14);
nb_units_per_hidden_layer=results(:,15);
nb_hidden_layers=results(:,16);

total_nb_rows=nb_calls_to_err_func(end)-nb_calls_to_err_func(1);

# write out plots in image directory
cd 'images/';

population_size	= results(1,9);
nb_replicates   = results(1,end);
nb_gens         = results(end,1);

TEST_SCORE = results(1,end-3);
TEST_ACC = results(1,end-2);

#
# PERFS AGAINST NB OF CALLS TO ERROR FUNCTION
#
align_right = nb_calls_to_err_func(end) - (nb_calls_to_err_func(end)/3) - 26;
errorbar(nb_calls_to_err_func, results(:,4), err_results);
hold on;
grid on;
plot(nb_calls_to_err_func,train_score,  'k', 'LineWidth', 1);
plot(nb_calls_to_err_func,train_acc,  'r', 'LineWidth', 1);
plot(nb_calls_to_err_func,val_acc, 'm', 'LineWidth', 1);
axis([0, results(end,1), 0, 100]);
title( strcat(alg_data, " - population size : ", num2str(population_size), " - ", num2str(nb_replicates), " replicates"));
legend('[Error amongst replicates] Corrected sample standard deviation','Best indiv f1 score on training set','Best indiv accuracy on training set','Best indiv accuracy on CV set', "location", "southeast");
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
plot_name = strcat(alg_data, "-perfsVSnbcallstoerrorfunction.png");
print('-deps', '-tiff', plot_name)
hold off;
fprintf("Saved as: \n\t->\"%s\"\n" , plot_name);


plot(nb_calls_to_err_func, train_mse, 'b', 'LineWidth', 1);
grid on;
hold on;
title(strcat(alg_data, " - MSE against nb generations"));
legend('MSE', "location", "northeast");
xlabel('Number of generations');
ylabel('Mean Squared Error');
plot_name = strcat(alg_data, "-MSEVSepochs.png");
print('-deps', '-tiff', plot_name);
hold off;
fprintf("Saved as: \n\t->\"%s\"\n" , plot_name);

#
# POPULATION SCORE STATS
#

plot(nb_calls_to_err_func, pop_score_var, 'b', 'LineWidth', 1);
grid on;
hold on;
title(strcat(alg_data, " - Variance of individuals's scores against nb generations"));
legend('Variance', "location", "northeast");
xlabel('Number of generations');
ylabel('Variance on validation-set');
plot_name = strcat(alg_data, "-varianceVSepochs.png");
print('-deps', '-tiff', plot_name);
hold off;
fprintf("Saved as: \n\t->\"%s\"\n" , plot_name);

plot(nb_calls_to_err_func, pop_score_mean, 'b', 'LineWidth', 1);
hold on;
grid on;
title(strcat(alg_data, " - Mean of individuals's scores against nb generations"));
legend('Mean', "location", "southeast");
xlabel('Number of generations');
ylabel('Mean of population score on validation-set');
plot_name = strcat(alg_data, "-meanVSepochs.png");
print('-deps', '-tiff', plot_name);
hold off;
fprintf("Saved as: \n\t->\"%s\"\n" , plot_name);


#
# RESULT SUMMARY
#

# calculate index values of 10 evenly separated rows
indexes=ones(10,1);
for i = [1:10]
    indexes(i)=total_nb_rows*(i*10/100);
endfor
    
indexes=round(indexes);
nb_cols=size(results)(2);

# store each row in result table
table=ones(10,nb_cols);
for i = [1:10]
    table(i,:)=results(indexes(i),:);
endfor

cd '../data/';

table_filename=strcat(algorithm_name,"/",data_set_name, "-table-summary.csv");

# Write out header and table summary
myfile=fopen(table_filename,'wt');
fprintf(myfile,['nb calls to err function, mse, train  acc, train ' ...
                'score, pop score var, pop score stddev, pop score ' ...
                'mean, pop score median, pop size, nb inputs, nb ' ...
                'hid units, nb outputs, nb hid layers, is pop ' ...
                'diverse, mutation scheme, ensemble acc, ensemble ' ...
                'score, val acc, val score, gens, test acc, test ' ...
                'score, mutation scheme, nb replicates\n']);
fclose(myfile);
dlmwrite(table_filename, table, '-append')

fprintf("Saved as: \n\t->\"%s\"\n" , table_filename);

