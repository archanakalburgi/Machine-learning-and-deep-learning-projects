

load wine_dataset;
x = wineInputs;
t = wineTargets;

% Choose a Training Function
% For a list of all training functions type: help nntrain
% 'trainlm' is usually fastest.
% 'trainbr' takes longer but may be better for challenging problems.
% 'trainscg' uses less memory. Suitable in low memory situations.
trainFcn = 'traingdm';  

clear ('acc');

lr = 0;

% Create a Pattern Recognition Network
hiddenLayerSize = 10;
net = patternnet(hiddenLayerSize, trainFcn);

learning_rate = [0.001, 0.01, 0.04, 0.05, 0.06, 0.07, 0.075, 0.08, 0.085]

% Setup Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 80/100;
net.divideParam.valRatio = 10/100;
net.divideParam.testRatio = 10/100;


for m = 1:length(learning_rate)
    net.trainParam.lr = learning_rate(m);
    
    % Train the Network
    [net,tr] = train(net,x,t);
    
    % Test the Network
    y = net(x);
    e = gsubtract(t,y);
    performance = perform(net,t,y)
    Validation_accuracy = tr.best_vperf
    tind = vec2ind(t);
    yind = vec2ind(y);
    percentErrors = sum(tind ~= yind)/numel(tind);
    acc(m) = (1 - Validation_accuracy) * 100
    if m >= 2 && acc(m) > acc (m-1) 
        lr = learning_rate(m);
    end
end


plot(learning_rate, acc)
xlabel("Learning rate")
ylabel("Accuracy")
fprintf ("The optimal Learning Rate is: ")
lr


%The Optimal Learning rate for the Train/Validation split of 80:20 is
%between 0.04 - 0.07 when the hidden layer contains 10 neurons. 





