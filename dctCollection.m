% Parent input and output folders
parentInputFolder = '/Users/amalkurian/Documents/MATLAB/AVP Data Collection GrayScale Images/Amelia';
parentOutputFolder = '/Users/amalkurian/Documents/MATLAB/DCT Features/Amelia';

% Number of folders to process (adjust this as needed)
numFolders = 40;

for num = 1:numFolders
    inputFolder = fullfile(parentInputFolder, sprintf('Amelia_%d', num));
    outputFolder = fullfile(parentOutputFolder, sprintf('Amelia_%d', num));
    
    imageFiles = dir(fullfile(inputFolder, '*.jpg'));
    
    if ~exist(outputFolder, 'dir')
        mkdir(outputFolder);
    end
   
    for k = 1:length(imageFiles)
        inputFile = imread(fullfile(inputFolder, imageFiles(k).name));
        grayImg = im2double(inputFile);
        
        [rows, cols] = size(grayImg);
        padRows = ceil(rows / 8) * 8 - rows;
        padCols = ceil(cols / 8) * 8 - cols;
        paddedImg = padarray(grayImg, [padRows, padCols], 'post');

        T = dctmtx(8);
        dct = @(block_struct) T * block_struct.data * T';
        dctImg = blockproc(paddedImg, [8 8], dct);
        
        %dctImg = dct2(grayImg);
        %reducedDCT = dctImg(1:30, 1:16);
        mask = [1 1 0 0 0 0 0 0;
                1 0 0 0 0 0 0 0;
                0 0 0 0 0 0 0 0;
                0 0 0 0 0 0 0 0;
                0 0 0 0 0 0 0 0;
                0 0 0 0 0 0 0 0;
                0 0 0 0 0 0 0 0;
                0 0 0 0 0 0 0 0];
        reducedDCT = blockproc(dctImg, [8 8], @(block_struct) mask .* block_struct.data);
        reducedDCT = reducedDCT(1:rows, 1:cols);
        % Define the output file name by appending '_dct' to the existing file name
        outputFile = fullfile(outputFolder, [imageFiles(k).name, '_dct.mat']);
        save(outputFile, 'reducedDCT');
    end
    
end
