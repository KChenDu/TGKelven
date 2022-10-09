clear
close all
clc

% read data
mktdata = readtable("datas/ERJ_PBR_VALE_weekly.csv");

% setup variables
label = "ERJ";
inputSteps = 104;
outputSteps = 52;

% split train/test data
n = height(mktdata) - outputSteps;
trainData = mktdata(1 : n, 2 : 4);
testData = mktdata(n - inputSteps + 1 : end, 2 : 4);
clear mktdata

% build neural network
net = narxnet(1 : inputSteps, 1 : inputSteps, 64);
net.numInputs = 3;
net.inputConnect = [1 1 1; 0 0 0];
net.inputs{1}.name = "x1";
net.inputs{3}.name = "x2";
net.inputs{3}.processFcns = {'removeconstantrows', 'mapminmax'};
net.inputWeights{1, 3}.delays= 1 : inputSteps;
net.divideParam.trainRatio = 0.8;
net.divideParam.valRatio = 0.2;
net.divideParam.testRatio = 0;
net.trainParam.min_grad = 0;

% prepare inputs and train
X = table2cell(normalize(trainData(:, trainData.Properties.VariableNames ~= label))).';
T = table2array(trainData(:, label));
first = T(end);
T = num2cell(normalize(T)).';
[x, xi, ai, t] = preparets(net, X, {}, T);
net = train(net, x, t, xi, ai);

% view(net)

net = closeloop(net);
% view(net)

X = table2cell(normalize(testData(:, testData.Properties.VariableNames ~= label))).';
T = table2array(testData(:, label));
train_std = std(T);
T = num2cell(normalize(T)).';
[x, xi, ai, t] = preparets(net, X, {}, T);
y = cell2mat(net(x ,xi, ai));

y = y * train_std;
y = y + first - y(1);

writematrix(y, 'results/' + label + '_NARX_prediction.csv');