% 1. Load the Grayscale Image (already in grayscale)
img = imread('/Users/amalkurian/Documents/MATLAB/AVP Data Collection GrayScale Images/Amelia/Amelia_2/gray_Amelia_2_cropped-001.jpg');  % Replace with your image file path
% 1. Load the Grayscale Image
grayImg = double(img);  % Convert to grayscale (if it's not already)

% 2. Get the size of the image
[rows, cols] = size(grayImg);

% 3. Calculate padding for rows and columns to make the size a multiple of 8
padRows = ceil(rows / 8) * 8 - rows;
padCols = ceil(cols / 8) * 8 - cols;

% 4. Pad the image
paddedImg = padarray(grayImg, [padRows, padCols], 'post');  % Pad rows and cols

% 5. Generate the DCT Transformation Matrix for 8x8 Blocks
dctMatrix = dctmtx(8);  % DCT matrix for 8x8 block

% 6. Apply DCT to 8x8 Blocks using blockproc
dct_img = blockproc(paddedImg, [8, 8], @(block_struct) ...
    dctMatrix * block_struct.data * dctMatrix');  % Apply DCT using matrix multiplication

% 7. Display the DCT Transformed Image
imshow(log(abs(dct_img) + 1), []);  % Log scale for visibility
title('Image in Frequency Domain (DCT)');
