clc, close all
load("C:\Users\pmans\Documenti\Progetti_local\Pycharm\GAN-PAN\data\W3\" + ...
     "W3_Muni_Urb.mat")
res_path = "C:\Users\pmans\Documenti\Progetti_local\Pycharm\GAN-PAN\results\" + ...
    "W3_full\";
head = {'Q', 'Q_avg', 'SAM', 'ERGAS', 'SCC'};
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
%% Testing Image used for training
id_image = 3;
full_results = cell(1,4);
i=1;
lrs = [ "01", "001", "0001", "-05"];
for patch_training=[32]
    file_name_excel = res_path+"/test_"+string(id_image)+"_"+string(patch_training)+".xlsx";
    if exist(file_name_excel, 'file')
       delete(file_name_excel);
    end
    for lr=lrs
        results = zeros(3,5);
        rn = cell(1,3);
        cnt=1;
        for patch_test=[64, 128, 256]

            file_name="w3_1_"+string(patch_training)+"_"+lr+"_"+string(id_image)+"_"+string(patch_test);
            load(res_path + file_name+".mat")
        
            gen = compose_from_patches(gen_decomposed).*2048;
            gt = compose_from_patches(gt_decomposed).*2048;
            [Q_avg, SAM, ERGAS, SCC, Q] =indexes_evaluation( ...
                gen,gt,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
            %fprintf("Q_avg=%.3f  SAM=%.3f  ERGAS=%.3f  SCC=%.3f  Q=%.3f\n", Q_avg, SAM, ERGAS, SCC, Q)
            results(cnt,:) = [Q, Q_avg, SAM, ERGAS, SCC];
            rn{cnt} = string(file_name);
            cnt = cnt+1;
        end
        results = round(results, 4);
        full_results{i} = results;
        i = i+1;
        table_results = array2table(results, 'VariableNames', head, 'RowNames',string(rn))
        writetable(table_results, file_name_excel,'UseExcel',true,'WriteRowNames',true, ...
            'WriteMode','append' );
    end
end

%% Testing other image
id_image = 3;
full_results = cell(1,4);
i=1;

for patch_training=[32]
    file_name_excel = res_path+"/test_"+string(id_image)+"_"+string(patch_training)+".xlsx";
    if exist(file_name_excel, 'file')
       delete(file_name_excel);
    end
    for lr=lrs
        results = zeros(3,5);
        rn = cell(1,4);
        cnt=1;
        for patch_test=[64, 128, 256, 512]
            file_name="w3_1_"+string(patch_training)+"_"+lr+"_"+string(id_image)+"_"+string(patch_test);
            load(res_path+file_name+".mat")
        
            gen = compose_from_patches(gen_decomposed).*2048;
            gt = compose_from_patches(gt_decomposed).*2048;
            [Q_avg, SAM, ERGAS, SCC, Q] =indexes_evaluation( ...
                gen,gt,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
            %fprintf("Q_avg=%.3f  SAM=%.3f  ERGAS=%.3f  SCC=%.3f  Q=%.3f\n", Q_avg, SAM, ERGAS, SCC, Q)
            results(cnt,:) = [Q, Q_avg, SAM, ERGAS, SCC];
            rn{cnt} = string(file_name);
            cnt = cnt+1;
        end
        results = round(results, 4);
        full_results{i} = results;
        i = i+1;
        table_results = array2table(results, 'VariableNames', head, 'RowNames',string(rn))
        writetable(table_results, file_name_excel,'UseExcel',true,'WriteRowNames',true, ...
            'WriteMode','append' );
    end
end

