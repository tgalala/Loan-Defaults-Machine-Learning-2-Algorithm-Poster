% Clear all workspace
clear all; clc; close all;

% Load the data set
data = readtable('cs-training.csv', 'ReadVariableNames',true);

%Print top 8 rows of the training dataset
head(data)

%change data type to double for the two variables...
data.NumberOfDependents = string(data.NumberOfDependents);
data.NumberOfDependents = double(data.NumberOfDependents);
data.MonthlyIncome = string(data.MonthlyIncome);
data.MonthlyIncome = double(data.MonthlyIncome);

%list of features
features = data.Properties.VariableNames(2:end);

%Delete column 1 - "Var1" - this is a redundant column numbering each
%observation
data.Var1 = [];

%Rename Variables 
data.Properties.VariableNames = {'SeriousDlqin2yrs' 'unsecUtil' 'age' 'pastDue30' ...
    'debtRatio' 'monthlyIncome' 'creditLines' 'pastDue90' ...
    'RELoans' 'pastDue60' 'dependents'};

%show decriptive stats of all variables
summary(data)
statsBefore = grpstats(data,'SeriousDlqin2yrs',{'mean','skewness','kurtosis'})


%% Data Pre-Processing

%Replace NaN values with median values for Number of Dependents
data.dependents = fillmissing(data.dependents, 'constant', ...
    median(data.dependents, 'omitnan'));

%delete missing monthly incomes and monthly incomes < 1
data.monthlyIncome = fillmissing(data.monthlyIncome, 'constant', 0);
data(data.monthlyIncome < 1,:) = [];

%Remove outliers:  Loans cannot be utilised more than 100%
%delete rows with loan utilisation > 100%
data(data.unsecUtil > 1,:) = [];

%delete ages less than 18 and over 90
data(data.age > 90,:) = [];
data(data.age < 18,:) = [];

%delete observations where Debt Ratio is zero (ne debt)
data(data.debtRatio == 0,:) = [];

summary(data);

%create pre-processed data file
writetable(data,'preProcessData1811.csv');

% Number of rows and columns
[r c] = size(data);

%Descriptive Stistics after Pre-processing
statsAgain = grpstats(data,'SeriousDlqin2yrs',{'mean','skewness','kurtosis'})

%% Transform Data using log normalisation

%change 0 values by adding 1 
data.pastDue30 = data.pastDue30 + 1;
data.pastDue60 = data.pastDue60 + 1;
data.pastDue90 = data.pastDue90 + 1;
data.creditLines = data.creditLines + 1;
data.RELoans = data.RELoans + 1;
data.dependents = data.dependents + 1;
data.unsecUtil = data.unsecUtil + 1;

%Normalise Data using log transform
logData = log(table2array(data(:, 2:11)));

%convert array to table
newLogData = array2table(logData);

%add SeriousDlqin2yrs
newLogData = [data(:, 1), newLogData];

newLogData.Properties.VariableNames = data.Properties.VariableNames

%writetable(newLogData,'logNormData.csv')

%descriptive statistics:
%statsAfter = grpstats(data,'SeriousDlqin2yrs',{'mean','skewness','kurtosis'})

%% Correlation Matrix using Pearson coefficient

corrTable = corr(table2array(newLogData(:, 1:11)),'type','Pearson');
figure('pos',[100 100 1000 1000])
labels = newLogData.Properties.VariableNames
imagesc(corrTable); % plot the matrix
colorbar;
set(gca,'XTick',[1:11],'xticklabel',labels);
set(gca,'YTick',[1:11],'yticklabel',labels);
xtickangle(45);
set(gca,'CLim',[-1 1]);
title('Pearson Correlation Plot', 'FontSize', 12);
colormap('winter'); % set the colorscheme, you can change the colour if you want
text_s = num2str(corrTable(:), '%0.2f'); 
text_s = strtrim(cellstr(text_s));
[x, y] = meshgrid(1:11);  
set(text(x(:), y(:), text_s(:)), 'color', 'black'); % Set color of text

%% Split Training and Testing Data

%Splitting using Holdout Method
% Splitting the Data (train: 80%, test: 20%)
rng('default');
cvpart = cvpartition(newLogData.SeriousDlqin2yrs, 'HoldOut',0.2);

%Split training which will be undersampled
Xtrain_inter = newLogData(training(cvpart),:);
Ytrain_inter = newLogData(training(cvpart),1);

%Create separate files for test data
Xtest  = newLogData(test(cvpart),:);
writetable(Xtest,'XtestData.csv')
Ytest  = newLogData(test(cvpart),1);
writetable(Ytest,'YtestData.csv')

disp(grpstats(Xtrain_inter(:,{'SeriousDlqin2yrs'}), 'SeriousDlqin2yrs'))
disp(grpstats(Xtest(:,{'SeriousDlqin2yrs'}), 'SeriousDlqin2yrs'))

%% Undersampling the training data

minorityData = Xtrain_inter(Xtrain_inter.SeriousDlqin2yrs==1, :);
majorityData = Xtrain_inter(Xtrain_inter.SeriousDlqin2yrs==0, :);
downsampledMajData = datasample(majorityData, size(minorityData, 1), 'replace', false);
downData = [minorityData; downsampledMajData]

%Shuffle the data
[row,col] = size(downData);
idx = randperm(row);
downSampleData = downData(idx,:);
Xtrain = downSampleData;
Ytrain = downSampleData(:,1);

%create separate files for training data
writetable(Xtrain,'XtrainData.csv')
writetable(Ytrain,'YtrainData.csv')

%% Oversampling using ADASYN - We did not use this method in our final model tuning as this created over 220k observations which made hyperparameter tuning very time consuming

%{
%This method will create new minortiy observations in order to reduce imbalance

%Normalise the training data in order to Oversample
Xtrain_Norm = normalize(Xtrain);

%Apply ADASYN
%1 is the target variable

[new_Xtrain, new_Ytrain] = ADASYN(table2array(Xtrain_inter), table2array(Ytrain_inter), 1, 5, 5);

%Convert back to table
Xtrain2 = array2table(new_Xtrain);
%Add and match column names
Xtrain2.Properties.VariableNames = Xtrain_Norm.Properties.VariableNames;
%Add existing dataset to newly created samples
Xtrain_final =  [Xtrain_Norm; Xtrain2];


%Convert back to table
Ytrain2 = array2table(new_Ytrain);
%Add and match column names
Ytrain2.Properties.VariableNames = Ytrain.Properties.VariableNames;
%Add existing dataset to newly created samples
Ytrain_final = [Ytrain; Ytrain2];

%}

