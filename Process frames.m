setenv('PATH', [getenv('PATH') ':/opt/miniconda3/bin']);
inputFolder = '/Users/amalkurian/Documents/MATLAB/AVP Data Collection Output/Louis';
outputFolder = '/Users/amalkurian/Documents/MATLAB/AVP Data Collection Frames/Louis';
videos = dir(fullfile(inputFolder, '*.mp4'));
features = [];

for k=1:length(videos)
    inputFile = fullfile(inputFolder, videos(k).name);
    outputVideoFolder = fullfile(outputFolder, [videos(k).name(1:end-4), '_frames']);
    mkdir (outputVideoFolder);
    command = sprintf('ffmpeg -i "%s" "%s/%s-%%03d.jpg"', inputFile, outputVideoFolder, videos(k).name(1:end-4));
    system(command);
end


