[X,T] = simpleseries_dataset;
Y=mat2cell(cellfun(@(x) x*2,X),1,ones(1,100));


net = narxnet(1:2,1:3); %classic narxnet 
net.numInputs = 3; % adding an input
net.inputConnect  =[1 1 1; 0 0 0]; %connecting 3 inputs to the first layer

%defining input 3
net.inputs{3}.name = 'Input3';
net.inputs{3}.processFcns = {'mapminmax'}; %whaterever you want, but remember to set them
net.inputWeights{1,3}.delays= 1:4; % the actual delay value for an input
view(net) %
