

load wine_dataset;
x = wineInputs;
t = wineTargets;

trainFcn = 'traingdm';

clear ('accu');


% Create a Pattern Recognition Network
hiddenLayerSize = 10;
net = patternnet(hiddenLayerSize, trainFcn);

% Training different test sets with the optimal learning rate 
net.trainParam.lr = 0.06;

% Setup Division of Data for Training, Validation, Testing

trainRatio = [10, 20, 30, 40 ,50, 60, 70 ,80, 90, 100]
validationRatio = [45, 40, 35, 30, 25, 20, 15, 10, 5, 0] 
testRatio = [45, 40, 35, 30, 25, 20, 15, 10, 5, 0]

for m = 1:length(trainRatio)
    
    net.divideParam.trainRatio = trainRatio(m)/100
    net.divideParam.valRatio = validationRatio(m)/100
    net.divideParam.testRatio = testRatio(m)/100
    
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
    accu(m) = (1 - Validation_accuracy) * 100
    %figure, plotconfusion(t,y)
end

xlabel("Training set")
ylabel("Accuracy")
title("Training set vs accuracy")
plot(trainRatio, accu)

 


%View the Network
%view(net)

% Plots
% Uncomment these lines to enable various plots.
%figure, plotperform(tr)
%figure, plottrainstate(tr)
%figure, ploterrhist(e)
%figure, plotconfusion(t,y)
%figure, plotroc(t,y)
