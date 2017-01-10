function [curI, curImask, Iapples, IapplesMasks] = LoadApplesScript
% LoadApplesScript.m
% This optional script may help you get started with loading of photos and masks.

if( ~exist('apples', 'dir') || ~exist('testApples', 'dir') )
    display('Please change current directory to the parent folder of both apples/ and testApples/');
end

% Initialize image data
Iapples = cell(3,1);
Iapples{1} = 'apples/Apples_by_kightp_Pat_Knight_flickr.jpg';
Iapples{2} = 'apples/ApplesAndPears_by_srqpix_ClydeRobinson.jpg';
Iapples{3} = 'apples/bobbing-for-apples.jpg';
Iapples{4} = 'testApples/Bbr98ad4z0A-ctgXo3gdwu8-original.jpg';
Iapples{5} = 'testApples/Apples_by_MSR_MikeRyan_flickr.jpg';
Iapples{6} = 'testApples/audioworm-QKUJj2wmxuI-original.jpg';
Iapples{7} = 'testApples/apple-1589874_640.jpg';
Iapples{8} = 'testApples/fruit-1213041_640.jpg';

% Initialize masks data
IapplesMasks = cell(3,1);
IapplesMasks{1} = 'apples/Apples_by_kightp_Pat_Knight_flickr.png';
IapplesMasks{2} = 'apples/ApplesAndPears_by_srqpix_ClydeRobinson.png';
IapplesMasks{3} = 'apples/bobbing-for-apples.png';
IapplesMasks{4} = 'testApples/Bbr98ad4z0A-ctgXo3gdwu8-original.png';
IapplesMasks{5} = 'testApples/apple-1589874_640_mask.png';
IapplesMasks{6} = 'testApples/fruit-1213041_640_mask.png';

% Loop index
iImage = 1; 
curI = double(imread(  Iapples{iImage}   )) / 255;
% curI is now a double-precision 3D matrix of size (width x height x 3). 
% Each of the 3 color channels is now in the range [0.0, 1.0].
imagesc(curI)


curImask = imread(  IapplesMasks{iImage}   );
% Transform to 1-channel and binary:
% Picked green-channel
curImask = curImask(:,:,2) > 128;  
