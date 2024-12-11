setenv('PATH', [getenv('PATH') ':/opt/miniconda3/bin']);

parentInputFolder = '/Users/amalkurian/Documents/MATLAB/DCT Features/Louis';
numFolders = 40; 

% Initialize features and labels
X = []; % Feature matrix (all videos)
y = []; % Labels (all videos)

% Initialize the maximum feature size
maxFeatureSize = 0;

% Loop through each numbered folder
for num = 1:numFolders
    % Define folder paths
    inputFolder = fullfile(parentInputFolder, sprintf('Louis_%d', num));
    
    % Get all .mat files in this video folder
    dctFiles = dir(fullfile(inputFolder, '*.mat'));
    
    if isempty(dctFiles)
        warning('No .mat files found in folder: %s', inputFolder);
        continue;
    end
    
    videoDCTs = []; % Store DCT features for this video

    % Loop through and combine DCT matrices
    for frameIdx = 1:length(dctFiles)
        % Load the DCT matrix from the .mat file
        try
            dctData = load(fullfile(inputFolder, dctFiles(frameIdx).name));
            
            % Check if 'reducedDCT' exists in the loaded data
            if ~isfield(dctData, 'reducedDCT')
                warning('Variable "reducedDCT" not found in file: %s', dctFiles(frameIdx).name);
                continue;
            end
            
            % Get the size of the current DCT frame
            frameDCT = reshape(dctData.reducedDCT, 1, []);
            maxFeatureSize = max(maxFeatureSize, numel(frameDCT)); % Update max size
            
            % Store the frame DCT features
            videoDCTs = [videoDCTs; frameDCT];
        catch ME
            warning('Error loading file: %s\n%s', dctFiles(frameIdx).name, ME.message);
            continue;
        end
    end
    
    if isempty(videoDCTs)
        warning('No valid DCT data found for folder: %s', inputFolder);
        continue;
    end
    
    % Now pad all frames to match the maxFeatureSize
    paddedVideoDCTs = [];
    for frameIdx = 1:size(videoDCTs, 1)
        currentFrameDCT = videoDCTs(frameIdx, :);
        if numel(currentFrameDCT) < maxFeatureSize
            % Pad with zeros to match the max feature size
            currentFrameDCT = [currentFrameDCT, zeros(1, maxFeatureSize - numel(currentFrameDCT))];
        end
        paddedVideoDCTs = [paddedVideoDCTs; currentFrameDCT];
    end
    
    % Flatten the combined DCT features for this video
    videoFeature = reshape(paddedVideoDCTs, 1, []);
    
    % Check if videoFeature has consistent size with the rest of the features
    if size(videoFeature, 2) ~= maxFeatureSize * size(videoDCTs, 1)
        warning('Inconsistent feature size for video %d. Skipping...', num);
        continue;
    end
    
    % Assign label (e.g., folder number as label)
    videoLabel = num;
    
    % Append the feature and label to the dataset
    if isempty(X)
        X = videoFeature; % Initialize X with the first valid video feature
    elseif size(X, 2) == size(videoFeature, 2)
        X = [X; videoFeature]; % Append feature
    else
        warning('Inconsistent feature size for folder: %s. Skipping...', inputFolder);
        continue;
    end
    
    y = [y; videoLabel]; % Append the label
end

% Save the dataset
save('louis_dataset.mat', 'X', 'y');
disp('Dataset saved as louis_dataset.mat');
