function [properties] = extract(img)


%read image into workspace
imrawrgb = imread(img);
imraw = rgb2gray(imrawrgb);
%threshold
level = graythresh(imraw);
%These are the only images from with insect face down
if strcmpi(img,'one.jpg') || strcmpi(img,'two.jpg') || strcmpi(img,'three.jpg')
    im = imbinarize(imraw, level -0.15);
else 
    im = imbinarize(imraw, level);
end

    
im = imcomplement(im);

%filter by area
im = bwareafilt(im, [3000 20000]);

% Remove portions of the image that touch an outside edge.
im = imclearborder(im);

% Fill holes in regions.
im = imfill(im, 'holes');

imshow(im)

% Get properties.
properties = regionprops(im, {'MajorAxisLength', 'MinorAxisLength', 'Area'});
properties = cell2mat(struct2cell(properties));
end