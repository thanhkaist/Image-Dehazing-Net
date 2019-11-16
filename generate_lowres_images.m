% generate all rootfolder/* high res folder to corresponding lr/* low res folder with scale = ...
clear;close all;
rootfolder = 'data';

scale = 0.5;
% get all folder
dirinfo = dir(rootfolder);
dirinfo(~[dirinfo.isdir]) = [];  %remove non-directories
tf = ismember( {dirinfo.name}, {'.', '..','lr'});
dirinfo(tf) = [];  %remove current and parent directory.

for i = 1:length(dirinfo)
    subdir = fullfile( rootfolder,dirinfo(i).name)
    % create lr dir 
    savedir = ['lr/',dirinfo(i).name];
    mkdir(savedir);
    imgs = dir(fullfile(subdir,'*.png'));
    % generate low res img
    for k = 1 : length(imgs)
        name = imgs(k).name;
        image = imread(fullfile(subdir,name));
        lr = imresize(image,scale,'bicubic');
        imwrite(lr, fullfile(savedir,name));
    end
    disp(["Gen %d lr images to %s",sprintf('%d',k),savedir]);
    
end