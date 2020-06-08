%% Naive Bayes Optimized Vs. Ensemble LogitBoost Optimized (Final test)

% Clear all workspace
clear all; clc; close all;

% Load the data set
Xtrain = readtable('XtrainData.csv', 'ReadVariableNames',true);
Ytrain = readtable('YtrainData.csv', 'ReadVariableNames',true);
Xtest = readtable('XtestData.csv', 'ReadVariableNames',true);
Ytest = readtable('YtestData.csv', 'ReadVariableNames',true);
Ytest = table2array(Ytest);

rng('default');

%% TRAIN THE OPTIMAL MODELS ON WHOLE TRAINING SET & PREDICT ON TEST SET

%create arrays to store all values
AUC =[];
Fscore = [];
Time = [];
Recall= [];
Precision =[];

% MODEL 1: 

%Ensemble Boosted Trees Optimal hyperparameters
Method_optimal = 'LogitBoost';
MaxSplit_optimal = 3;
MinLeaf_optimal = 18;
nlearn_optimal = 0.25;
NumBins_optimal = 110;
NumTrees_optimal = 100;

% Ensemble Boosted Trees fitting model (training the model on training data)
template = templateTree('MaxNumSplits', MaxSplit_optimal,'MinLeafSize',MinLeaf_optimal);

tic %Time the model
Emdl_optimal = fitcensemble(Xtrain,'SeriousDlqin2yrs',...
    'NumLearningCycles',NumTrees_optimal,...
    'Method','LogitBoost',...
    'LearnRate',nlearn_optimal,...
    'NumBins',NumBins_optimal,...
    'Learners', template);
time_EN = toc; %store the time

% Ensemble Boosted Trees prediction on test set
y_pred_en = predict(Emdl_optimal, Xtest);

% Ensemble Boosted Trees confusion chart (test set)
figure(1)
cm3 = confusionchart(Ytest,y_pred_en);
cm3.Title = 'Boosted Tree Ensemble confusion chart';

% Boosted Tree Ensemble confusion matrix, recall, precision, f-score (test set)
confMat_EN = confusionmat(Ytest, y_pred_en);

% Precision, Recall and FScores
Precision_EN = confMat_EN(2,2) / (confMat_EN(2,2) + confMat_EN(1,2));
Recall_EN = confMat_EN(2,2) / (confMat_EN(2,2) + confMat_EN(2,1)); % Sensitivity
Fscore_EN = 2 * Precision_EN * Recall_EN / (Precision_EN + Recall_EN);

% Ensemble Boosted Trees AUC (test set)
[label, score_rf] = predict(Emdl_optimal, Xtest); % optimized
[Xrf,Yrf,Trf,AUCrf] = perfcurve(Ytest,score_rf(:,2),1); % score vector for positive '1' outcome


% MODEL 2: 

%Naive Bayes Optimal hyper parameters
distribution =  'kernel';
kernel =  'triangle';
width = 0.10;
prior = [0.5, 0.5];


% Naive Bayes fitting model (train set)

tic %time the model
NBmdl_optimal = fitcnb(Xtrain, 'SeriousDlqin2yrs', ...
    'DistributionNames', distribution, ...
    'Kernel', kernel, ...
    'Width', width,...
    'Prior', prior);
time_NB = toc; %store the time value

% Naive Bayes prediction on test set
y_pred_nb = predict(NBmdl_optimal, Xtest);

% Naive Bayes confusion chart (test set)
figure(2)
cm2 = confusionchart(Ytest,y_pred_nb);
cm2.Title = 'Naive Bayes confusion chart';

% Naive Bayes confusion matrix, recall, precision, f-scores (test set)
confMat_NB = confusionmat(Ytest, y_pred_nb);

Precision_NB = confMat_NB(2,2) / (confMat_NB(2,2) + confMat_NB(1,2));
Recall_NB = confMat_NB(2,2) / (confMat_NB(2,2) + confMat_NB(2,1)); % Sensitivity
Fscore_NB = 2 * Precision_NB * Recall_NB / (Precision_NB + Recall_NB);

% Naive Bayes AUC (test set)
[~,score_nb] = predict(NBmdl_optimal, Xtest);
[Xnb,Ynb,Tnb,AUCnb] = perfcurve(Ytest,score_nb(:,2),1);

% Print performance metrics comparision for Naive Bayes & Ensemble LogitBoost
fprintf('______________________________\n\n')
fprintf('    NB        Boosted Ensemble\n')
fprintf('______________________________\n')
Fscore = [Fscore;Fscore_NB,          Fscore_EN]
fprintf('______________________________\n')
AUC = [AUC;AUCnb,                    AUCrf]
fprintf('______________________________\n')
Precion = [Precision;Precision_NB,                   Precision_EN]
fprintf('______________________________\n')
Recall = [Recall;Recall_NB,                    Recall_EN]
fprintf('______________________________\n')
Time = [Time;time_NB,                    time_EN]
fprintf('______________________________\n')


%% PLOT ROC CURVES and Precision-Recall Curves

% Plotting ROC Curves for the two models together for comparision

figure(3)
plot(Xrf,Yrf)
hold on
plot(Xnb,Ynb)
legend('Boosted Ensemble AUC = 0.8414','Naive Bayes AUC = 0.8332','Location','best')
xlabel('False positive rate'); ylabel('True positive rate');
title('ROC Curves Comparison')
hold off

% Plot Precision-Recall Curve
figure(4)
[XN, YN] = perfcurve(Ytest,score_nb(:,2),1, 'XCrit', 'tpr', 'YCrit', 'prec');
[XE, YE] = perfcurve(Ytest,score_rf(:,2),1, 'XCrit', 'tpr', 'YCrit', 'prec');
plot(XE, YE);
hold on
plot(XN, YN)
xlabel('Recall')
ylabel('Precision')
legend('Boosted Ensemble','Naive Bayes','Location','best')
xlim([0, 1])
ylim([0, 1])
title('Recall & Precision Curve Comparison')
hold off
