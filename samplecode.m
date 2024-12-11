setenv('PATH', [getenv('PATH') ':/opt/miniconda3/bin']);
inputFolder = '/Users/amalkurian/Documents/MATLAB/AVP Data Collection/Zachary'; 
outputFolder = '/Users/amalkurian/Documents/MATLAB/AVP Data Collection Output/Zachary'
files = dir(fullfile(inputFolder, '*.mp4')); % Get all MP4 files in the folder
for k = 1:length(files)
    % Input and output file names
    inputFile = fullfile(inputFolder, files(k).name);
    outputFile = fullfile(outputFolder, sprintf('Zachary_%d_cropped.mp4', k)); % Create output name
    if ~exist(outputFolder)
       mkdir(outputFolder)
    end   
    % ffmpeg command
    command = sprintf(['ffmpeg -i "%s" -vf "crop=150:80:940:500" -c:a copy "%s"'], inputFile, outputFile);
    system(command); % Execute the ffmpeg command
end   

