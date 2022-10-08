clear
close all
clc

mktdata = readtable("datas/ERJ_PBR_VALE_weekly.csv");

label = "ERJ";
inputSteps = 104;
outputSteps = 52;

n = height(mktdata) - outputSteps;
trainData = mktdata(1 : n, 2 : 4);
testData = mktdata(n - inputSteps + 1 : end, 2 : 4);
clear mktdata

net = narxnet(1 : inputSteps, 1 : inputSteps, 43);
net.numInputs = 3;
net.inputConnect = [1 1 1; 0 0 0];
net.inputs{1}.name = "x1";
net.inputs{3}.name = "x2";
net.inputs{3}.processFcns = {'removeconstantrows', 'mapminmax'};
net.inputWeights{1, 3}.delays= 1 : inputSteps;
net.divideParam.trainRatio = 0.5;
net.divideParam.valRatio = 0.5;
net.divideParam.testRatio = 0;
net.trainParam.min_grad = 0;

X = table2cell(trainData(:, trainData.Properties.VariableNames ~= label)).';
T = table2cell(trainData(:, label)).';
[x, xi, ai, t] = preparets(net, X, {}, T);
net = train(net, x, t, xi, ai);

% view(net)

net = closeloop(net);
% view(net)

X = table2cell(testData(:, testData.Properties.VariableNames ~= label)).';
T = table2cell(testData(:, label)).';
[x, xi, ai, t] = preparets(net, X, {}, T);
y = net(x ,xi, ai);

writecell(y, 'results/' + label + '_NARX_prediction.csv');