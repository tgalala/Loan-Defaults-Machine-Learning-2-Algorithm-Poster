%% Basic tree model

%Clear all workspace
clear all; clc; close all;

% Load the data set
Xtrain = readtable('XtrainData.csv', 'ReadVariableNames',true);
Ytrain = readtable('YtrainData.csv', 'ReadVariableNames',true);

% Training Basic ensemble model with 10 fold Cross Validation
basicTreeModel = fitctree(Xtrain,'SeriousDlqin2yrs', 'CrossVal', 'on');

% Basic model loss
loss = kfoldLoss(basicTreeModel);

%Prediction with kfoldpredict
Ypred = kfoldPredict(basicTreeModel);

%Performance metrics
% Confusion Matrix and Cost Loss estimation
[confMatTree costLossTree]  = confusionmat(table2array(Ytrain), Ypred);

% AUC
[Xtree,Ytree,Ttree,AUCtree] = perfcurve(table2array(Ytrain),Ypred,1);

%Precision and Recall metrics
Precision = confMatTree(2,2) / (confMatTree(2,2) + confMatTree(1,2));
Recalltree = confMatTree(2,2) / (confMatTree(2,2) + confMatTree(2,1)); % Sensitivity
% F_score
%Fscore=2*Recall*Precision/(Precision+Recall)

%Display Results of the basic Tree Model
TreeScores =[];
fprintf('__________________________________________________\n\n')
fprintf('___________Basic Classification Tree _____________\n')
fprintf('__________________________________________________\n\n')
fprintf('  Accuracy    Recall   Precision  Fscore    AUC\n')
fprintf('__________________________________________________\n')
TreeScores = [TreeScores; (1-loss)*100,Recalltree,Precision,(2*Recalltree*Precision/(Precision+Recalltree)), AUCtree]
fprintf('__________________________________________________\n')

%% Tuning our ensemble hyperparamters and finding optimized ensemble model

% Creating an an array to capture the hyperparameters results
results_MaxSplit = zeros(1,10);
results_MinLeafSize = zeros(1,20);
results_NumTrees= zeros(1,5);
results_learnRate = zeros(1,5);
results_numBins = zeros(1,5);

method = ["AdaBoostM1", "LogitBoost", "RUSBoost", "GentleBoost"]; % These methods have common hyperparameters
moreMethods = ["RobustBoost", "Bag"]; % Some hyperparameters will not work
% on these two, that's why i seperated them. But they have been tested too.

% Choice of Hyper Parameters
maxSplit = [3,10,43,100,1000]; %  Maximal number of decision splits (3.^(0:m))
minLeaf = [1:20]; % Minimum number of leaf node observations 
numTrees = [10,50,100,200,500]; % Number of trees
learnRate = [0.1,0.25,0.5,0.75]; % Learning rate
numBins = [50,100,110,150]; % Number of bins to increase speed

% Array to capture the results
result_fscore =[];
Scores_EB = [];
tic

%Grid Search to find optimal hyper paramters
for i=1:length(method)

    for k=1:length(maxSplit)

            for m=1:length(minLeaf)

                     for n=1:length(numTrees)

                             for p=1:length(learnRate)

                                     for r=1:length(numBins)

                                                    %Optimizing our model hyperparamters
                                                    template = templateTree('MaxNumSplits', maxSplit(k),'MinLeafSize',minLeaf(m));
                                                    Mdl_Optimized = fitcensemble(Xtrain,'SeriousDlqin2yrs',...
                                                        'Method',method(i),...
                                                        'NumLearningCycles',numTrees(n),...
                                                        'LearnRate',learnRate(p),...
                                                        'NumBins',numBins(r),...
                                                        'Learners', template);

                                                    % Cross-validating the model
                                                    tic % time each cross validated model
                                                    cvMdl_Optimized = crossval(Mdl_Optimized);
                                                    time_EB = toc; % store times

                                                    % Predicting model with kfoldPredict
                                                    y_pred = kfoldPredict(cvMdl_Optimized);

                                                    % Calculating Loss on
                                                    % training set with kfold
                                                    loss = kfoldLoss(cvMdl_Optimized);
                                                    
                                                    %Performance metrics                                                    
                                                    % Confusion matrix
                                                    confMat = confusionmat(Xtrain.SeriousDlqin2yrs, y_pred);

                                                    % Precision and Recall Metrics
                                                    Precision = confMat(2,2) / (confMat(2,2) + confMat(1,2));
                                                    Recall = confMat(2,2) / (confMat(2,2) + confMat(2,1)); % Sensitivity

                                                    % F-Scores
                                                    Fscore=2*Recall*Precision/(Precision+Recall);
                                                    result_fscore=[result_fscore, Fscore];
                                                    
                                                    %Display performance
                                                    %metrics for each model
                                                    fprintf('\n__________________________________________________________________________________________________________________________\n')
                                                    fprintf('\n______________________________Optimized Ensemble Model & Hyper-parameters tuning_________________________________________\n')
                                                    fprintf('\n__________________________________________________________________________________________________________________________\n\n')
                                                    fprintf('    Method         MaxSplit MinLeaf Trees Learn    NumBins   Accuracy     Recall      Precision      Fscore      Time\n')
                                                    fprintf('___________________________________________________________________________________________________________________________\n')
                                                    Scores_EB = [Scores_EB; method(i),maxSplit(k),minLeaf(m),numTrees(n),learnRate(p),numBins(r),(1-loss)*100,Recall,Precision,Fscore, time_EB]
                                                    fprintf('___________________________________________________________________________________________________________________________\n')

                                      end    
                             end   

                     end       
            end
    end
    
end
toc

% Return highest F-score in matrix and the placement
fprintf('    Method         MaxSplit MinLeaf Trees Learn    NumBins   Accuracy     Recall      Precision      Fscore      Time\n')
[maxValue, index] = max(result_fscore);
fprintf('___________________________________________________________________________________________________________________________\n')

% Printing Max Fscore Performance Metrics
bestBoostedModel = Scores_EB(index,:)
