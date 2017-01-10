function [row] = apples

close all;

% load training data
[curImage, curImagemask, Iapples, IapplesMasks] = LoadApplesScript();

RGBApple = ones(0,3);
RGBNonApple = ones(0,3);

[trainExamples, column] = size(Iapples);

% create training data 
for (i=1:trainExamples)
    curIMask = imread(IapplesMasks{i});
    curIMask = curIMask(:,:,2) > 128;
    curIMask = reshape(curIMask,[],1);
    
    curI = double(imread(Iapples{i})) / 255;
    curI = reshape(curI,[],3);
    
    appleIndices = find(curIMask == 1);
    nonAppleIndices = find(curIMask == 0);
    
    RGBApple = [RGBApple curI(appleIndices,:)'];
    RGBNonApple = [RGBNonApple curI(nonAppleIndices,:)'];
end

n = 60000;
a = randperm(length(RGBApple),n);
b = randperm(length(RGBNonApple),n);

trainApple = RGBApple(:,a);
trainNonApple = RGBNonApple(:,b);

% train mixture of Guassians for distinguishing apple vs. non-apple
mixGauss.d = 3;

% number of Gaussians
nGaussEst = 3;

mixGaussEstApple = fitMixGauss(trainApple,nGaussEst);
mixGaussEstNonApple = fitMixGauss(trainNonApple,nGaussEst);

% priors for whether a pixel is apple or non apple
if (mixGaussEstApple.logLike > mixGaussEstNonApple.logLike)
    priorApple = 0.7;
    priorNonApple = 0.3;
else 
    priorApple = 0.3;
    priorNonApple = 0.7;
end

% run through the pixels in test image and classify them as being apple or
% non apple
testImage = double(imread(Iapples{4})) / 255;
[testRows, testColumns, dimension] = size(testImage);

testMask = imread(IapplesMasks{4});
testMask = testMask(:,:,2) > 128;

posteriorApple = zeros(testRows,testColumns);

truePositive = 0;
trueNegative = 0;
falseNegative = 0;
falsePositive = 0;

for (row = 1:testRows);    
    for (col = 1:testColumns); 
        thisPixelData = squeeze(double(testImage(row,col,:)));

        % likelihood given apple model
        likeApple = getMixGaussLogLike(thisPixelData, mixGaussEstApple);
        %likelihood given non apple model
        likeNonApple = getMixGaussLogLike(thisPixelData, mixGaussEstNonApple);
        
         if (likeApple > likeNonApple)
             likeApple = 1;
             likeNonApple = 0;
         else
             likeApple = 0;
             likeNonApple = 1;
         end    
        
        posteriorApple(row,col) = (likeApple * priorApple)/((likeApple * priorApple)+(likeNonApple * priorNonApple));
        
         if (posteriorApple(row,col) == 1)
             if(posteriorApple(row,col) ~=  testMask(row,col))
                falsePositive = falsePositive + 1;
             else
                truePositive = truePositive + 1;
             end
         else
             if posteriorApple(row,col) ~=  testMask(row,col)
                falseNegative = falseNegative + 1;
             else
                trueNegative = trueNegative + 1;
             end
         end
             
    end;
    
end;

% draw apple posterior
clims = [0, 1];
imagesc(posteriorApple, clims); colormap(gray); axis off; axis image;

fprintf('True positives : %4.3f\n',truePositive);
fprintf('True negatives : %4.3f\n',trueNegative);
fprintf('False positives : %4.3f\n',falsePositive);
fprintf('False negatives : %4.3f\n',falseNegative);

end

%==========================================================================
%==========================================================================

%==========================================================================
%==========================================================================

function logLike = getMixGaussLogLike(data,mixGaussEst);

nData = size(data,2);
logLike = 0;

for(cData = 1:nData)
    thisData = data(:,cData);    
    like = 0;
    for (k = 1:mixGaussEst.k)
        like = like + mixGaussEst.weight(k) * getGaussProb(thisData,mixGaussEst.mean(:,k), mixGaussEst.cov(:,:,k));
    end

    logLike = logLike+log(like);        
end;

end

%==========================================================================
%==========================================================================

function bound = getMixGaussBound(data,mixGaussEst,responsibilities)

nData = size(data,2);
bound = 0;
boundValue = 0;

for(cData = 1:nData)
    thisData = data(:,cData);    
    thisQ = responsibilities(:,cData);
    
    for (k = 1:mixGaussEst.k)
        boundValue = boundValue + thisQ(k) * log(mixGaussEst.weight(k) * getGaussProb(thisData, mixGaussEst.mean(:,k),mixGaussEst.cov(:,:,k))/thisQ(k));
    end

    bound = bound+boundValue;        
end;

end

%==========================================================================
%==========================================================================
%
function mixGaussEst = fitMixGauss(data,k);
        
[nDim nData] = size(data);

postHidden = zeros(k, nData);

mixGaussEst.d = nDim;
mixGaussEst.k = k;
mixGaussEst.weight = (1/k)*ones(1,k);
[idx, mean] = kmeans(data',k);
mixGaussEst.mean = mean';
mixGaussEst.cov = zeros(nDim, nDim, k);
mixGaussEst.logLike = 0;
X = data';
for cGauss = 1:k
    mixGaussEst.cov(:,:,cGauss) = cov(X(idx==k,:));
end

logLike = getMixGaussLogLike(data,mixGaussEst);
fprintf('Log Likelihood Iter 0 : %4.3f\n',logLike);

nIter = 4;
for (cIter = 1:nIter)  
   for (cData = 1:nData)
        thisData = data(:,cData);
        numerator = zeros(k,1);
        for (i = 1:k)
            numerator(i,cData) = mixGaussEst.weight(i) * getGaussProb(thisData, ...
                mixGaussEst.mean(:,i),mixGaussEst.cov(:,:,i));
        end 
        
        for (i = 1:k) 
            postHidden(i,cData) = numerator(i,cData) / sum(numerator(:,cData));
        end
   end;
   
   for (cGauss = 1:k) 
        mixGaussEst.weight(cGauss) = sum(postHidden(cGauss,:))/sum(postHidden(:)); 

        for (dimension = 1:nDim)
            mixGaussEst.mean(dimension, cGauss) = sum(postHidden(cGauss,:) * data(dimension, :)') / sum(postHidden(cGauss,:));
        end
              
        subtractedMean = bsxfun(@minus, data, mixGaussEst.mean(:,cGauss));
        nominator = zeros(3,3);
        for (element = 1:nData)
         nominator = nominator + postHidden(cGauss,element)*(subtractedMean(:,element)*subtractedMean(:,element)');
        end

        mixGaussEst.cov(:,:,cGauss) = nominator / sum(postHidden(cGauss,:));
   end;

   logLike = getMixGaussLogLike(data,mixGaussEst);
   fprintf('Log Likelihood Iter %d : %4.3f\n',cIter,logLike);
   
   mixGaussEst.logLike = logLike;
end;

end

%==========================================================================
%==========================================================================

%subroutine to return gaussian probabilities
function prob = getGaussProb(x,mean,var)

prob = exp(-0.5*((x-mean)'*inv(var)*(x-mean)));
prob = prob/ (2*pi)*det(var).^(1/2);

end