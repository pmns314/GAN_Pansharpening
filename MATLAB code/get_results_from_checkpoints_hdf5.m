clc, clear, close all
load("C:\Users\pmans\Documenti\Progetti_local\Pycharm\GAN-PAN\data\W3\" + ...
     "W3_Muni_Urb.mat")
res_path = "C:\Users\pmans\Documenti\Progetti_local\Pycharm\GAN-PAN\results\" + ...
    "W3_full_hdf5";
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
test_data_path = res_path + "/" + model_name;
for test = ["test_0", "test_1"]
    fprintf("%s\n", test);
    test_name =test_data_path+"/"+ test + ".h5";
    filename_excel = test_data_path+"/"+test + ".xlsx";
    if exist(filename_excel, 'file')
       T = readtable(filename_excel);
       start = T{end,1}
       clear T
    else
        start = 2
    end

    gt = h5read(test_name,'/gt');
    gt = permute(gt, [3, 2,1]) .* 2048;
    info = h5info(test_name,'/gen');
    sz = info.Dataspace.Size;
    stop = sz(4);
    count = 64;
   
    data = h5read(test_name, "/gen");
    data = permute(data, [4,3, 2,1]);
  
    for idx=start:stop
        fprintf("Row %d \n", idx)
        img = data(idx, :, :, :);
        img = squeeze(img) .*2048;

        [Q_avg, SAM, ERGAS, SCC, Q] =indexes_evaluation( ...
            img,gt,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
        %fprintf("Q_avg=%.3f  SAM=%.3f  ERGAS=%.3f  SCC=%.3f  Q=%.3f\n", Q_avg, SAM, ERGAS, SCC, Q)
        row = [idx, Q, Q_avg, SAM, ERGAS, SCC]
        results = round(row, 4);
        table_results = array2table(row, 'VariableNames', head);
        writetable(table_results, filename_excel,'UseExcel',true,'WriteRowNames',true, 'WriteMode','append');

    end

end

