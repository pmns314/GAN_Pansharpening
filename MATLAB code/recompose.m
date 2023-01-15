clear, clc, close all
load("C:\Users\pmans\Documenti\Progetti_local\Pycharm\GAN-Pansharpening\data\FR\" + ...
     "GE\" + ...
     "GE_Lond_Urb.mat")
filename=("C:\Users\pmans\Documenti\Progetti_local\Pycharm\GAN-Pansharpening\results\" + ...
    "GANs\GE\" + ...
    "PanColorGan\" + ...
    "pancolorgan_v3.6_test_1.mat");

load(filename);

gt = I_GT ;
gen = double(gen);
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
    gen,gt,ratio,L,Qblocks_size, ...
    flag_cut_bounds,dim_cut,thvalues);
fprintf("Q2n=%.4f Q_avg=%.4f SCC=%.4f SAM=%.4f  ERGAS=%.4f \n", Q, Q_avg, SCC, SAM, ERGAS)
fprintf("%.4f %.4f %.4f %.4f %.4f \n", Q, Q_avg, SAM, ERGAS, SCC)
fprintf("Differenza assoluta medie per canale\n")
disp(reshape(abs(mean(gen, [1,2]) - mean(gt, [1,2])), [1,4]))
showImage4_zoomin(gt,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L, location1, location2);
showImage4_zoomin(gen,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L, location1, location2);
disp("end")