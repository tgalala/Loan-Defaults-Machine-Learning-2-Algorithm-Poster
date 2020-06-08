%% Load Data
% Clear all workspace
clear all; clc; close all;

% Load the data set
Xtrain = readtable('XtrainData.csv', 'ReadVariableNames',true);
Ytrain = readtable('YtrainData.csv', 'ReadVariableNames',true);

%% Basic Naive Bayes model with [10]-fold CV

features = {'unsecUtil' 'age' 'pastDue30' ...
    'debtRatio' 'monthlyIncome' 'creditLines' 'pastDue90' ...
    'RELoans' 'pastDue60' 'dependents'};

%Run basic model with 10 fold Cross Validation 
basicNBmdl = fitcnb(Xtrain, Ytrain, 'PredictorNames', features,'CrossVal', 'on');

%Calculate training and test errors
trainError = kfoldLoss(basicNBmdl);

%Predict results on training data
mdlScore = kfoldPredict(basicNBmdl);

%Confusion Matrix and Cost Loss estimation
[confMatbasicNB, clnb] = confusionmat(Xtrain.SeriousDlqin2yrs, mdlScore);
cmChart = confusionchart(Xtrain.SeriousDlqin2yrs,mdlScore)

%Performance metrics
% AUC
[XbasicNB,YbasicNB,TbasicNB,AUCbasicNB] = perfcurve(table2array(Ytrain),mdlScore,1);

% Precision and Recall Metrics
Precision_basicNB = confMatbasicNB(2,2) / (confMatbasicNB(2,2) + confMatbasicNB(1,2));
Recall_basicNB = confMatbasicNB(2,2) / (confMatbasicNB(2,2) + confMatbasicNB(2,1)); % Sensitivity
Specificity_basicNB = confMatbasicNB(2,2) / (confMatbasicNB(2,2) + confMatbasicNB(2,1)); % specificity
Fscore_basicNB = 2 * Precision_basicNB * Recall_basicNB / (Precision_basicNB + Recall_basicNB);

%Display results
basicNBScores =[];
fprintf('_________________________________________________\n')
fprintf('_______________Basic NB Model____________________\n')
fprintf('_________________________________________________\n')
fprintf('  Accuracy    Recall   Precision  Fscore    AUC\n')
fprintf('_________________________________________________\n')
basicNBScores = [basicNBScores; (1-trainError),Recall_basicNB,Precision_basicNB,Fscore_basicNB, AUCbasicNB]
fprintf('_________________________________________________\n')

%% Hyperparameter tuning

features = {'unsecUtil' 'age' 'pastDue30' ...
    'debtRatio' 'monthlyIncome' 'creditLines' 'pastDue90' ...
    'RELoans' 'pastDue60' 'dependents'}

distribution = ["normal", "kernel"];
kerneltype = ["normal", "box", "epanechnikov", "triangle"];
kwidth = [0.07 0.08 0.09 0.10 0.11 0.12];

[rows columns] = size(Ytrain);
targetPrior = (sum(Ytrain.SeriousDlqin2yrs==1))/rows;
majorPrior = (rows-sum(Ytrain.SeriousDlqin2yrs==1))/rows;
dataPrior = [targetPrior, majorPrior];
priors = [dataPrior
            0.1 0.9
            0.2 0.8
            0.3 0.7];

result_fscoreNB =[]; % for storing the FScores
kScores = []; % for storing results of each cross validated model

iterations = 10;
counter=1;

tic % time the grid search

%Hyperparameter optimisation for Kernel Distribution

for i=1:iterations %  Total Number of iterations
                      
    for j=1:length(kerneltype) % Loop through the kernel types

        for k=1:length(kwidth) %Loop through Kernel Widths
            
            for m=1:length(priors) % Loop through the priors
                
                nbOptimisedMdl = fitcnb(Xtrain, Ytrain, ...
                    'PredictorNames', features, ...
                    'DistributionNames','kernel', ...
                    'Kernel', kerneltype(j), ...
                    'Width', kwidth(k), ...
                    'Prior', priors(m,:), ...
                    'CrossVal', 'off');
                
                %run the cross validated model
                tic %time each model
                cvNBOptimisedMdl = crossval(nbOptimisedMdl)
                timeNB = toc
 
                % Iteration counter
                fprintf('\n\n******************** Iteration : %f\n', counter);

                
                 % Predicting model with kfoldPredict
                y_predNB = kfoldPredict(cvNBOptimisedMdl);
                
                % Calculating Loss on training set with kfold
                lossNB = kfoldLoss(cvNBOptimisedMdl);
            

                %Performance metrics

                % Confusion matrix
                confMatNB = confusionmat(Xtrain.SeriousDlqin2yrs, y_predNB);

                % AUC
                [XNB,YNB,TNB,AUCNB] = perfcurve(table2array(Ytrain),y_predNB,1);

                % Precision and Recall Scores
                PrecisionNB = confMatNB(2,2) / (confMatNB(2,2) + confMatNB(1,2));
                RecallNB = confMatNB(2,2) / (confMatNB(2,2) + confMatNB(2,1)); % Sensitivity
                SpecificityNB = confMatNB(2,2) / (confMatNB(2,2) + confMatNB(2,1)); % specificity
                FscoreNB = 2 * PrecisionNB * RecallNB / (PrecisionNB + RecallNB);

                result_fscoreNB=[result_fscoreNB, FscoreNB];
                
                %Print results of each cross validated model 
                fprintf('\n\n__________________________________________\n\n')
                fprintf('     Iter   Smoother        width     prior      Accuracy      Recall       Precision    Fscore       time\n')
                fprintf('__________________________________________\n')
                kScores = [kScores; i,kerneltype(j),kwidth(k),priors(m),(1-lossNB)*100,RecallNB,PrecisionNB,FscoreNB, timeNB]
                fprintf('__________________________________________\n')

            end
        end
    end
   counter=counter+1 
end
                                       
toc

% Return highest F-score in matrix and the placement
[maxValue, index] = max(result_fscoreNB);
bestKernelModel = kScores(index,:)

%% Hyperparameter optimisation for Normal distribution
iter = 10
counterN =1
priors = [0.5 0.5
          0.4 0.6
          0.3 0.7
          0.2 0.8
          0.1 0.9]

result_fscore_n =[];
nScores = [];
      
for x=1:iter
    
    for y=1:length(priors)
        
        normMdlPrior = fitcnb(Xtrain, Ytrain, ...
            'PredictorNames', features, ...
            'DistributionNames', 'normal', ...
            'Prior', priors(y,:), ...
            'CrossVal', 'off');
        tic
        cvNormMdlPrior = crossval(normMdlPrior)
        time2 = toc
        % Predicting model with kfoldPredict
        y_pred_norm = kfoldPredict(cvNormMdlPrior);

        % Iteration counter
        fprintf('\n\n******************** Iteration : %f\n', counterN);

        % Calculating Loss on
        % training set with kfold
        lossNormNB = kfoldLoss(cvNormMdlPrior);

        %Performance metrics

        % Confusion matrix
        confMatNormNB = confusionmat(Xtrain.SeriousDlqin2yrs, y_pred_norm);

        % AUC
        [XNB,YNB,TNB,AUCNB] = perfcurve(table2array(Ytrain),y_pred_norm,1);

        % Precision and Recall Metrics
        PrecisionNormNB = confMatNormNB(2,2) / (confMatNormNB(2,2) + confMatNormNB(1,2));
        RecallNormNB = confMatNormNB(2,2) / (confMatNormNB(2,2) + confMatNormNB(2,1)); % Sensitivity
        SpecificityNormNB = confMatNormNB(2,2) / (confMatNormNB(2,2) + confMatNormNB(2,1)); % specificity
        FscoreNormNB = 2 * PrecisionNormNB * RecallNormNB / (PrecisionNormNB + RecallNormNB);
             
        result_fscore_n=[result_fscore_n, FscoreNormNB];

        %Display Results
        fprintf('\n\n__________________________________________\n\n')
        fprintf('     Iter   Prior     Accuracy      Recall       Precision    Fscore       time\n')
        fprintf('__________________________________________\n')
        nScores = [nScores; x,priors(y),(1-lossNormNB),RecallNormNB,PrecisionNormNB,FscoreNormNB, time2]
        fprintf('__________________________________________\n')
    end
    counterN=counterN+1
end

% Return highest F-score in matrix and the placement
[maxValue, index] = max(result_fscore_n);
bestNormModel = nScores(index,:)
            

