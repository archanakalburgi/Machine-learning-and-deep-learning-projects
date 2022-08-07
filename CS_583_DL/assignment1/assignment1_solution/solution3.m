load wine_dataset;

%x = wineInputs;

features = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
clear ('accuracy'); 

for m = 1:length(features)
    
    x = wineInputs(1: features(m), :);
    t = wineTargets;
 
    trainFcn = 'traingdm';
    
    hiddenLayerSize = 10;
    net = patternnet(hiddenLayerSize, trainFcn);

    % Training different test sets with the optimal learning rate 
    net.trainParam.lr = 0.06;
    
    % Setup Division of Data for Training, Validation, Testing
    net.divideParam.trainRatio = 80/100;
    net.divideParam.valRatio = 10/100;
    net.divideParam.testRatio = 10/100;

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
    accuracy(m) = (1 - Validation_accuracy) * 100
    %figure, plotconfusion(t,y)    
end

xlabel("Features")
ylabel("Accuracy")
title("Features vs accuracy")
plot(features, accuracy)


