% Define the parent input and output folders
parentInputFolder = '/Users/amalkurian/Documents/MATLAB/AVP Data Collection Frames/Louis';
parentOutputFolder = '/Users/amalkurian/Documents/MATLAB/AVP Data Collection GrayScale Images/Louis';

numFolders = 40;

for num = 1:numFolders
    
    inputFolder = fullfile(parentInputFolder, sprintf('Louis_%d_cropped_frames', num));
    outputFolder = fullfile(parentOutputFolder, sprintf('Louis_%d', num));
    imageFiles = dir(fullfile(inputFolder, '*.jpg'));
    
    if ~exist(outputFolder, 'dir')
        mkdir(outputFolder);
    end
    
    for k = 1:length(imageFiles)
        inputFile = imread(fullfile(inputFolder, imageFiles(k).name));
        outputFile = fullfile(outputFolder, ['gray_', imageFiles(k).name]);
        grayImg = rgb2gray(inputFile);
        imwrite(grayImg, outputFile);
    end
    
end
