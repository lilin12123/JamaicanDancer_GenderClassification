clear all;
close all;
clc;


%% Choice: use angle features or symmetry features
% If you use symmetry features, using FeatureMat, uncommend "load FeatureMat;"

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load FeatureMat;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% If you use angle features, using FeatureMat, uncommend the following part

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load FeatureMat_angle;
% choose perceived gender boundary, those score < boundary is male 
boundary = 0.42; 
FeatureMat((FeatureMat(:,1)<=boundary),1) = 0;
FeatureMat((FeatureMat(:,1)>=(1-boundary)),1) = 2;
FeatureMat((FeatureMat(:,1)>boundary & FeatureMat(:,1)<(1-boundary)),:)=[];
FeatureMat((FeatureMat(:,1)<=boundary),1) = 1;

gender = FeatureMat(:,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%get num of males and females
numMale = size(find(FeatureMat(:,1) == 1),1);
numFemale = size(find(FeatureMat(:,1) == 2),1);



%%%the flow of your code should look like this
Dim = size(FeatureMat,2)-1; %dimension of the feature
countfeat(Dim,2) = 0;
%%countfeat is a Mx2 matrix that keeps track of how many times a feature has been selected, where M is the dimension of the original feature space.
%%The first column of this matrix records how many times a feature has ranked within top 1% during 100 times of feature ranking.
%%The second column of this matrix records how many times a feature was selected by forward feature selection during 100 times.


%% initial
avg_rate_train = 0;
avg_rate_test = 0;
avg_std_test = 0;
avg_ConfMat = [0 0;0 0];
%change num to change loop times, we use 50 times
num = 1;
P_histogram = [];

%% repeating num times

for i = 1:num
    
    % randomly divide into test and traing sets
    % In symmetry features, total 90 male and 69 female in 159 subjects
    % In angle features, total 173 subjects
    [TrainMat, LabelTrain, TestMat, LabelTest]= randomDivideMulti(FeatureMat,[1,2],[floor(numMale*0.8),floor(numFemale*0.8)],[floor(numMale*0.2),floor(numFemale*0.2)]);
%     [TrainMat, LabelTrain, TestMat, LabelTest]= randomDivideMulti(FeatureMat);

    % [h,p_value] = ttest(score_newrate);
    % start feature ranking
    LabelTrain = categorical(LabelTrain);
    LabelTest = categorical(LabelTest);
%     countfeat(topfeatures(:,1),1) =  countfeat(topfeatures(:,1),1) +1;  
    
    
%% the preformance of using the top rate = ?% features, score 1 = VR, score 2 = AVR
    rate = 0.2;
    score = 1;
    
    topfeatures = rankingfeat(TrainMat, LabelTrain, rate, score);

    %try different classifier here
    Model = fitcsvm(TrainMat(:,topfeatures(:,1)'),LabelTrain,'KernelFunction','linear','Standardize',true);  
%     Model = fitcdiscr(TrainMat(:,topfeatures(:,1)'),LabelTrain);
%     Model = fitcknn(TrainMat(:,topfeatures(:,1)'),LabelTrain,'NumNeighbors',10,'Standardize',1);
%     Model = fitctree(TrainMat(:,topfeatures(:,1)'),LabelTrain);

%     mdl = fitPosterior(Model);
    mdl = Model;
    [~,scores] = resubPredict(mdl);
    
    train_pred = predict(Model,TrainMat(:,topfeatures(:,1)'));
    train_ConfMat = confusionmat(LabelTrain,train_pred);
    train_ClassMat = train_ConfMat./(meshgrid(countcats(LabelTrain))');
    train_acc_topfeatures = mean(diag(train_ClassMat));
    
    % if you commented feature selection, also uncomment this.
%     avg_rate_train = avg_rate_train + train_acc_topfeatures;
    
    test_pred = predict(Model,TestMat(:,topfeatures(:,1)'));
    test_ConfMat = confusionmat(LabelTest,test_pred);
    test_ClassMat = test_ConfMat./(meshgrid(countcats(LabelTest))');
    test_acc_topfeatures = mean(diag(test_ClassMat));
    
    % if you commented feature selection, also uncomment this.
%     avg_rate_test = avg_rate_test + test_acc_topfeatures;
    
    if num == 1
        %generate ROC curve of different classifier
        ROC_generater(TrainMat, topfeatures, LabelTrain);
        %plot topfeatures' score
        B = sort(topfeatures(:,2),'descend');
        figure();
        plot(B);
        xlabel('features');
        ylabel('AVR score')
    end

    
    %% start feature selection ---------First uncomment to choose SFS or SBS
%     featureSelected = forwardselection(TrainMat, double(LabelTrain), topfeatures, test_acc_topfeatures);
    featureSelected = backwardselection(TrainMat, double(LabelTrain), topfeatures, test_acc_topfeatures);

    countfeat(featureSelected,2) =  countfeat(featureSelected,2) +1;    
    % for histogram
    P_histogram = [P_histogram featureSelected'];
    % start classification  
    P = featureSelected;
    
    % LDA classifier
%     Model = fitcdiscr(TrainMat(:,P'),LabelTrain);    
    % SVM classifier
%     Model = fitcsvm(TrainMat(:,P'),LabelTrain,'KernelFunction','linear','Standardize',true); 
    % KNN classifier
    Model = fitcknn(TrainMat(:,P'),LabelTrain,'NumNeighbors',10,'Standardize',1);

    train_pred = predict(Model,TrainMat(:,P'));
    train_ConfMat = confusionmat(LabelTrain,train_pred);
    train_ClassMat = train_ConfMat./(meshgrid(countcats(LabelTrain))');
    train_acc = mean(diag(train_ClassMat));
    
    avg_rate_train = avg_rate_train + train_acc;
    
    test_pred = predict(Model,TestMat(:,P'));
    test_ConfMat = confusionmat(LabelTest,test_pred);
    test_ClassMat = test_ConfMat./(meshgrid(countcats(LabelTest))');
    test_acc = mean(diag(test_ClassMat));
    
    avg_rate_test = avg_rate_test + test_acc;
    
    avg_ConfMat = avg_ConfMat + test_ConfMat;
    
    test_std = std(diag(test_ClassMat));
    avg_std_test = avg_std_test + test_std;
   
end
%compute avg acc and std
avg_rate_train = avg_rate_train/num
avg_rate_test = avg_rate_test/num
avg_std_test = avg_std_test/num;
avg_ConfMat = avg_ConfMat/num;

% x stores distinct features in 50 times
% x means those good features
x = unique(P_histogram);
good_features = x;

%% classification using good features----UNCOMMENT to see the results on collected good features
% %generate hist of good feautres
% %it is time consuming because we will generate angle data first
% hist_goodfeatures(x);
% 
% num = 50;
% avg_rate_train_final = 0;
% avg_rate_test_final = 0;
% avg_std_test_final = 0;
% avg_ConfMat_final = [0 0;0 0];
% 
% for i = 1:num
%     
%     % randomly divide into test and traing sets
%     % In symmetry features, total 90 male and 69 female in 159 subjects
%     % In angle features, total 173 subjects
%     [TrainMat, LabelTrain, TestMat, LabelTest]= randomDivideMulti(FeatureMat,[1,2],[floor(numMale*0.8),floor(numFemale*0.8)],[floor(numMale*0.2),floor(numFemale*0.2)]);
% %     [TrainMat, LabelTrain, TestMat, LabelTest]= randomDivideMulti(FeatureMat);
% 
%     % [h,p_value] = ttest(score_newrate);
%     % start feature ranking
%     LabelTrain = categorical(LabelTrain);
%     LabelTest = categorical(LabelTest);
%     Model = fitcknn(TrainMat(:,x),LabelTrain,'NumNeighbors',10,'Standardize',1);
% 
%     train_pred = predict(Model,TrainMat(:,x));
%     train_ConfMat = confusionmat(LabelTrain,train_pred);
%     train_ClassMat = train_ConfMat./(meshgrid(countcats(LabelTrain))');
%     train_acc = mean(diag(train_ClassMat));
%     
%     avg_rate_train_final = avg_rate_train_final + train_acc;
%     
%     test_pred = predict(Model,TestMat(:,x));
%     test_ConfMat = confusionmat(LabelTest,test_pred);
%     test_ClassMat = test_ConfMat./(meshgrid(countcats(LabelTest))');
%     test_acc = mean(diag(test_ClassMat));
%     test_std = std(diag(test_ClassMat));
%     
%     avg_rate_test_final = avg_rate_test_final+test_acc;
%     avg_std_test_final = avg_std_test_final+test_std;
%     avg_ConfMat_final = avg_ConfMat_final+test_ClassMat;
% end
% avg_rate_train_final = avg_rate_train_final/num
% avg_rate_test_final = avg_rate_test_final/num
% avg_std_test_final = avg_std_test_final/num
% avg_ConfMat_final = avg_ConfMat_final/double(num)

%% Unsuprivised Learning on GOOD FEATURES
%reload all subjects
load FeatureMat_angle;
% choose perceived gender boundary, those score < boundary is male 
boundary = 0.5; 
FeatureMat((FeatureMat(:,1)<=boundary),1) = 0;
FeatureMat((FeatureMat(:,1)>=(1-boundary)),1) = 2;
FeatureMat((FeatureMat(:,1)>boundary & FeatureMat(:,1)<(1-boundary)),:)=[];
FeatureMat((FeatureMat(:,1)<=boundary),1) = 1;

gender = FeatureMat(:,1);

%load all subject with only good features
FeatureMat = FeatureMat(:, 2:end);
dataMat = FeatureMat(:, good_features);


Z = linkage(dataMat,'complete','chebychev');
T = cluster(Z,'maxclust',2);
cutoff = median([Z(end-2,3) Z(end-1,3)]);
figure(8);
dendrogram(Z,0,'ColorThreshold',cutoff)

