% Refer from: https://github.com/limbee/NTIRE2017/blob/master/demo/evaluation.m
close all; clc; clear;
hrDir = 'data/benchmark/Set5/HR/';
srDir = 'result/Net1/';

numImages = 5;
set5Images = ["baby", "bird", "butterfly", "head", "woman"];

scale = 2;
shave = 8;

meanPSNR = 0;
meanSSIM = 0;

disp(repmat('-', 1, 80))
disp([repmat('-', 1, 29), 'PSNR & SSIM evaluation', repmat('-', 1, 29)])
disp(repmat('-', 1, 80))
disp(' ')
disp([sprintf('%-25s', 'Model Name'), ' | ', ...
    sprintf('%-10s', 'Set Name'), ' | ', ...
    sprintf('%-5s', 'Scale'), ...
    ' | PSNR / SSIM'])
disp(repmat('-', 1, 80))

for i = 1:5
    hrImgName = strcat(hrDir, set5Images(i), '.png');
    hrImg = imread(hrImgName{1});
    hrImg = rgb2ycbcr(hrImg);
    hrImg = hrImg(:,:,1);
    srImgName = strcat(srDir, set5Images(i), '.png');
    srImg = imread(srImgName{1});
    srImg = rgb2ycbcr(srImg);
    srImg = srImg(:,:,1);
    [h, w, ~] = size(srImg);
    srImg = srImg((1 + shave):(h - shave), (1 + shave):(w - shave), :);
    hrImg = hrImg((1 + shave):(h - shave), (1 + shave):(w - shave), :);
    meanPSNR = meanPSNR + psnr(srImg, hrImg);
    meanSSIM = meanSSIM + ssim(srImg, hrImg);
end

meanPSNR = meanPSNR / numImages;
meanSSIM = meanSSIM / numImages;

modelNameF = sprintf('%-25s', "Net1");
setNameF = sprintf('%-10s', "Set5");
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


