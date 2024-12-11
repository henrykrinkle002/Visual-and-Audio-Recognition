folderPath = '/Users/amalkurian/Documents/MATLAB/AVP Data Collection/Zachary'; % Specify the folder name
files = dir(fullfile(folderPath, '*.mp4')); % Get all MP4 files in the folder
for k = 1:min(40, length(files)) % Rename up to 40 files
    oldName = fullfile(folderPath, files(k).name); % Full path to the original file
    newName = fullfile(folderPath, sprintf('Zachary_%d.mp4', k)); % Full path to the new name
    movefile(oldName, newName); % Rename the file
end