clc, close all
load("C:\Users\pmans\Documenti\Progetti_local\Pycharm\GAN-PAN\data\W3\" + ...
     "W3_Muni_Urb.mat")
load("C:\Users\pmans\Documenti\Progetti_local\Pycharm\GAN-PAN\results\" + ...
    "W3_full3\W3_1_32_001_mse\" + ...
    "W3_1_32_001_mse_test_3.mat")
addpath([pwd,'/Tools']);

yy = compose_from_patches(gen_decomposed).*2048;
gt = compose_from_patches(gt_decomposed).*2048;
%%
s = size(I_GT);
cut_num = s(1)/2;
L = 11;
Qblocks_size = 32;
flag_cut_bounds = 1;
dim_cut = 21;
thvalues = 0;
printEPS=0;
location1                = [50 70 10 30];  %default: data6: [10 50 1 60]; data7:[140 180 5 60]
location2                = [20 38 10 50]; 
[Q_avg, SAM, ERGAS, SCC, Q] = ...
    indexes_evaluation(...
    yy,gt,ratio,L,Qblocks_size, ...
    flag_cut_bounds,dim_cut,thvalues)
fprintf("Q2n=%.3f Q_avg=%.3f SCC=%.3f SAM=%.3f  ERGAS=%.3f \n", Q, Q_avg, SCC, SAM, ERGAS)

showImage8_zoomin(gt,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L, location1, location2);
showImage8_zoomin(yy,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L, location1, location2);
disp("end")