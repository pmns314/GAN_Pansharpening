clc, close all
load("C:\Users\pmans\Documenti\Progetti_local\Pycharm\GAN-PAN\data\W3\" + ...
     "W3_Muni_Urb.mat")
res_path = "C:\Users\pmans\Documenti\Progetti_local\Pycharm\GAN-PAN\results\" + ...
    "W3_full3\";
head = {'Epoch', 'Q', 'Q_avg', 'SAM', 'ERGAS', 'SCC'};
s = size(I_GT);
L = 11;
Qblocks_size = 32;
flag_cut_bounds = 1;
dim_cut = 21;
thvalues = 0;
printEPS=0;
location1                = [50 70 10 30];  %default: data6: [10 50 1 60]; data7:[140 180 5 60]
location2                = [20 38 10 50]; 

%% 
model_name = "W3_1_32_0001_mse";
chk_path = res_path + "/" + model_name;
for test = ["test_0", "test_1"]
    gt = -1;
    test_folder = chk_path+"/"+test;
    file_name_excel = chk_path+"/"+test+".xlsx";
    if exist(file_name_excel, 'file')
       T = readtable(file_name_excel);
       start = T{end,1}+1
       clear T
    else
        start = 1
    end
    stop = length(dir("C:\Users\pmans\Documenti\Progetti_local\Pycharm\GAN-PAN\pytorch_models\trained_models\" + ...
    "W3_all\"+model_name+"/checkpoints"))-2 % escludes the "." and ".." always present
    results = zeros(1, 6);
    for i = start:stop-1
        fprintf("Row number : %d\n", i)
        load(chk_path+"/"+test+"/checkpoint_"+i+".mat")
        gen = compose_from_patches(gen_decomposed).*2048;
        if gt==-1
            gt = compose_from_patches(gt_decomposed).*2048;
        end
        [Q_avg, SAM, ERGAS, SCC, Q] =indexes_evaluation( ...
            gen,gt,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
        %fprintf("Q_avg=%.3f  SAM=%.3f  ERGAS=%.3f  SCC=%.3f  Q=%.3f\n", Q_avg, SAM, ERGAS, SCC, Q)
        row = [i, Q, Q_avg, SAM, ERGAS, SCC];
        results = round(row, 4);
        table_results = array2table(row, 'VariableNames', head);
        writetable(table_results, file_name_excel,'UseExcel',true,'WriteRowNames',true, 'WriteMode','append');
        delete(chk_path+"/"+test+"/checkpoint_"+i+".mat")
    end
end


 
% %% Testing other image
% id_image = 3;
% full_results = cell(1,4);
% i=1;
% 
% for patch_training=[32]
%    
%     for lr=lrs
%         results = zeros(3,5);
%         rn = cell(1,4);
%         cnt=1;
%         for patch_test=[64, 128, 256, 512]
%             file_name="w3_1_"+string(patch_training)+"_"+lr+"_"+string(id_image)+"_"+string(patch_test);
%             load(res_path+file_name+".mat")
%         
%             gen = compose_from_patches(gen_decomposed).*2048;
%             gt = compose_from_patches(gt_decomposed).*2048;
%             [Q_avg, SAM, ERGAS, SCC, Q] =indexes_evaluation( ...
%                 gen,gt,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
%             %fprintf("Q_avg=%.3f  SAM=%.3f  ERGAS=%.3f  SCC=%.3f  Q=%.3f\n", Q_avg, SAM, ERGAS, SCC, Q)
%             results(cnt,:) = [Q, Q_avg, SAM, ERGAS, SCC];
%             rn{cnt} = string(file_name);
%             cnt = cnt+1;
%         end
%         results = round(results, 4);
%         full_results{i} = results;
%         i = i+1;
%         table_results = array2table(results, 'VariableNames', head, 'RowNames',string(rn))
%         writetable(table_results, file_name_excel,'UseExcel',true,'WriteRowNames',true, ...
%             'WriteMode','append' );
%     end
% end
% 
