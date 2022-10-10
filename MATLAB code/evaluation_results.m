clc, close all
load("C:\Users\pmans\Documenti\Progetti_local\Pycharm\GAN-PAN\data\W3\" + ...
    "W3_Muni_Mix.mat")
load("C:\Users\pmans\Documenti\Progetti_local\Pycharm\GAN-PAN\results\W3\" + ...
    "w3_64_max_val_01.mat")
addpath([pwd,'/Tools']);

gg = gt_composed.*2048;
xx = gen_composed;
s = size(I_GT);
cut_num = s(1)/2;
ggg   = I_GT(cut_num+1:end, 1:cut_num, :);
% xx = double(squeeze(g(1,:,:,:)));
gg = pagetranspose(gg);


%yy = (yy .* st)+ m;
yy = (xx.*2048);
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
    yy,gg,ratio,L,Qblocks_size, ...
    flag_cut_bounds,dim_cut,thvalues)
fprintf("Q_avg=%.3f  SAM=%.3f  ERGAS=%.3f  SCC=%.3f  Q=%.3f\n", Q_avg, SAM, ERGAS, SCC, Q)

showImage8_zoomin(ggg,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L, location1, location2);
showImage8_zoomin(gg,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L, location1, location2);
showImage8_zoomin(yy,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L, location1, location2);

disp("end")