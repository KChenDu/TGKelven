clear
close all
clc

% read data
mktdata = readtable("datas/ERJ_PBR_VALE_monthly.csv");
%mktdata = readtable("datas/GS_JPM_AXP_HON_AAPL_MSFT_CAT_CVX_MCD_NKE_MMM_TRV_DIS_HD_monthly.csv");

% setup variables
label = "ERJ";
% label = "GS";
inputSteps = 12;
outputSteps = 12;

% split train/test data
n = height(mktdata) - outputSteps;
trainData = mktdata(1 : n, 2 : end);
testData = mktdata(n - inputSteps + 1 : end, 2 : end);
clear mktdata

% build neural network
net = narxnet(1 : inputSteps, 1 : inputSteps, 8);
n_columns = width(trainData);
net.numInputs = n_columns;
net.inputConnect = [ones(1, n_columns); zeros(1, n_columns)];
net.inputs{1}.name = "x1";
net.inputs{2}.name = "y";
for i = 3 : n_columns
    net.inputs{i}.name = strcat("x" + num2str(i - 1));
    net.inputs{i}.processFcns = {'removeconstantrows', 'mapminmax'};
    net.inputWeights{1, i}.delays= 1 : inputSteps;
end
net.divideParam.trainRatio = 0.8;
net.divideParam.valRatio = 0.2;
net.divideParam.testRatio = 0;
% net.trainParam.min_grad = 0;
% view(net)

% prepare inputs and train
X = table2cell(normalize(trainData(:, trainData.Properties.VariableNames ~= label))).';
T = table2array(trainData(:, label));
first = T(end);
T = num2cell(normalize(T)).';
[x, xi, ai, t] = preparets(net, X, {}, T);
net = train(net, x, t, xi, ai);

net = closeloop(net);
% view(net)

X = table2cell(normalize(testData(:, testData.Properties.VariableNames ~= label))).';
T = table2array(testData(:, label));
first = T(13);
train_std = std(T);
T = num2cell(normalize(T)).';
[x, xi, ai, t] = preparets(net, X, {}, T);
y = cell2mat(net(x ,xi, ai));

y = y * 10;
y = y + first - y(1);

writematrix(y, 'results/' + label + '_NARX_prediction.csv');