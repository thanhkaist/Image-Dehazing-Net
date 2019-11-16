close all; clc; clear;
hrDir = 'data/benchmark/B100/HR/';
srDir = 'val/EDSR/B100/';
dataSet = 'B100';
netName = 'EDSR';

scale = 2;
shave = 8;

meanPSNR = 0;
meanSSIM = 0;

hr_imgs = dir(fullfile(hrDir,'*.png'));
sr_imgs = dir(fullfile(srDir,'*.png'));

xlsfiles = {hr_imgs.name};
hr_img_names = sort(xlsfiles);
xlsfiles = {sr_imgs.name};
sr_img_names = sort(xlsfiles);

numImages = length(sr_img_names);

%% Display header 
disp(repmat('-', 1, 80))
disp([repmat('-', 1, 29), 'PSNR & SSIM evaluation', repmat('-', 1, 29)])
disp(repmat('-', 1, 80))
disp(' ')
disp([sprintf('%-25s', 'Model Name'), ' | ', ...
    sprintf('%-10s', 'Set Name'), ' | ', ...
    sprintf('%-5s', 'Scale'), ...
    ' | PSNR / SSIM'])
disp(repmat('-', 1, 80))


%% calculate psnr ssim

for i = 1:numImages
    hr_name = fullfile(hrDir,hr_img_names{i});
    sr_name = fullfile(srDir,sr_img_names{i});
    hrImg = imread(hr_name);
    if length(size(hrImg))<3
        numImages = numImages -1;
        continue
    end
    hrImg = rgb2ycbcr(hrImg);
    hrImg = hrImg(:,:,1);
    srImg = imread(sr_name);
    srImg = rgb2ycbcr(srImg);
    srImg = srImg(:,:,1);
    [h, w, ~] = size(srImg);
    srImg = srImg((1 + shave):(h - shave), (1 + shave):(w - shave), :);
    hrImg = hrImg((1 + shave):(h - shave), (1 + shave):(w - shave), :);
    meanPSNR = meanPSNR + psnr(srImg, hrImg);
    meanSSIM = meanSSIM + ssim(srImg, hrImg);
end

%% Print the result

meanPSNR = meanPSNR / numImages;
meanSSIM = meanSSIM / numImages;

modelNameF = sprintf('%-25s', netName);
setNameF = sprintf('%-10s', dataSet);
scaleF = sprintf('%-5d', scale);
isModelPrint = true;
isSetPrint = true;
disp([modelNameF, ' | ', ...
setNameF, ' | ', ...
scaleF, ...
' | PSNR: ', num2str(meanPSNR, '%.2fdB')])

disp([repmat(' ', 1, 25), ' | ', ...
repmat(' ', 1, 10), ' | ', ...
repmat(' ', 1, 5), ...
' | SSIM: ', num2str(meanSSIM, '%.4f')])


disp(repmat('-', 1, 80))



