parentInputFolder = '/Users/amalkurian/Documents/MATLAB/DCT Features/Zachary';
outputParentFolder = '/Users/amalkurian/Documents/MATLAB/Dataset';

numFolders = 40; 

for num = 1:numFolders
   
    inputFolder = fullfile(parentInputFolder, sprintf('Zachary_%d', num));
    videoDCTs = []; 
    dctFiles = dir(fullfile(inputFolder, '*.mat'));
    
    for frameIdx = 1:length(dctFiles) 
        dctData = load(fullfile(inputFolder, dctFiles(frameIdx).name));
        frameDCT = reshape(dctData.reducedDCT, 1, []);
        videoDCTs = [videoDCTs; frameDCT];
    end

    videoFeature = videoDCTs;
    videoLabel = num; 

    % Create the output directory if it does not exist
    outputFolder = fullfile(outputParentFolder, sprintf('Dataset', num));
    if ~exist(outputFolder, 'dir')
        mkdir(outputFolder);
    end
    outputFileName = fullfile(outputFolder, sprintf('Zachary_%d_dataset.mat', num));
    save(outputFileName, 'videoFeature', 'videoLabel');
end
