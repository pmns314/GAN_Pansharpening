clear, close all, clc
L = 11;
Qblocks_size = 32;
flag_cut_bounds = 1;
dim_cut = 21;
th_values = 0;
ratio = 4.0;
t = zeros(2,3);
%%
satellite = "W3/";
result_folder = "../results/GANs/";

imgs = [];
to_show = [];
names = ["GT","apnn", "PanGan", "PanGan TL", "PanColorGan", "PanColorGan TL",...
             "PSGAN", "PSGAN TL"];
    to_show = ["apnn_v2.5",...
               "pangan_v3.5",...
               "pangan_v3.6",...
               "pancolorgan_v2.2",...
               "pancolorgan_v3.1",...
               "psgan_v3.1",...
               "psgan_v3.5",...
               ];
force_model = "None";
main_title = "W3_Muni_Urb";
index_test = 3;
MatrixResults = zeros(numel(to_show),5);
names_cpy = names;
% names = {"GT", "FR con GT"};
% to_show = ["psgan_v2.0"];

% [status, list] = system( 'dir /B .' );
% result = textscan( list, '%s', 'delimiter', '\n' );
% fileList = result{1}

imgs = zeros(length(names), 512,512,8);
%assert length(names) == length(to_show) == length(imgs)
index = 1;
gt = imread(strcat(result_folder, satellite, "gt_",string(index_test),".tif"));
imgs(index,:,:,:)= gt;
gt = double(gt);
mgt = mean(gt, [1,2]);
[Q_avg, SAM, ERGAS, SCC, Q2n] = indexes_evaluation(gt, gt,...
            ratio, L, Qblocks_size, flag_cut_bounds, dim_cut,th_values);
        MatrixResults(index, :) = [Q2n, Q_avg, SAM, ERGAS, SCC];

%%
index = index + 1;
D = dir(strcat(result_folder,"/",satellite));
for k =3:numel(D)
    type = D(k).name;


    type_init2 = extractBetween(type, 1, 2);
    if  type_init2{1} == "gt"
        continue
    end

    if force_model ~= "None" && type ~= force_model
        continue
    end
    TypeDir = dir(strcat(result_folder,"/",satellite,"/",type));
    for tk = 3:numel(TypeDir)
        img_name = TypeDir(tk).name;
        extracted_id = extract(img_name, length(img_name)-4);
        if extracted_id{1} ~= string(index_test)
                continue
        end
        end_img_name = extractBetween(img_name, 1, strlength(img_name)-11);
        end_img_name =end_img_name{1};
        if length(to_show) > 0
            if ~(any(to_show == end_img_name))
                continue
            else
                index = find(strcmp(to_show, end_img_name)) + 1;
            end
        end
        img_path = strcat(result_folder,"/", satellite,"/", type, "/", img_name);
        ss=strsplit(img_path, ".");
        if ss(length(ss)) == "mat"
            mat = load(img_path);
            gen = mat.gen;
        else
            gen = imread(img_path);
        end
        
        gen = double(gen);
        mgen = mean(gen, [1,2]);
        gen = (gen./mgen).*mgt;
        imgs(index,:,:,:) = gen;
        [Q_avg, SAM, ERGAS, SCC, Q2n] = indexes_evaluation(gen, gt,...
            ratio, L, Qblocks_size, flag_cut_bounds, dim_cut,th_values);
        MatrixResults(index, :) = [Q2n, Q_avg, SAM, ERGAS, SCC];
        %print(names[index],Q2n, Q_avg, SAM,ERGAS)
        if length(to_show) > 0
            img_title = sprintf("%s Q2n:%.4f SAM:%.4f",names(index), Q2n, SAM);
            names(index) = img_title;
        else
            img_title = sprintf("%s Q2n:%.4f SAM:%.4f",end_img_name, Q2n, SAM);
            names(index) = img_title;
            index =index+ 1;
        end
    end
end

sz_imgs = size(imgs);
dim_img = sz_imgs(2);
num_imgs = sz_imgs(1);
num_rows = round(ceil(sqrt(num_imgs)));
if num_imgs <= num_rows^2 - num_rows
    num_cols = num_rows - 1 ;
else
    num_cols = num_rows;
end

% concat = zeros(dim_img,dim_img*num_imgs, 8);
% stop=1;
% for ii=1:num_imgs
%     start = stop;
%     stop = stop + dim_img;
%     concat(:,start:stop-1, :) = imgs(ii,:,:,:);
% end
% lin_stretched = viewimage2(concat);
% close all
% cnt = double(1);
% num_cols = double(num_cols);
% num_rows= double(num_rows);
% 
% f = figure();
% f.WindowState = 'maximized';
% sgtitle(replace(main_title, "_", " "))
% start = 1;
% for i=1:num_imgs
%     subplot(num_cols, num_rows, cnt)
%     stop = start + dim_img-1;
%     img_to_show = lin_stretched(:, start:stop, :);
%     imshow(img_to_show(:,:,3:-1:1))
%     title(names(i))
%     axis('off')
%     cnt = cnt+ 1;
%     start = stop;
% end

f = figure();
f.WindowState = 'maximized';
sgtitle(replace(main_title, "_", " "))
cnt = 1;
for i=1:num_imgs
    subplot(num_cols, num_rows, cnt);
    img_to_show = reshape(imgs(i,:,:,:), [sz_imgs(2), sz_imgs(3), sz_imgs(4)]);
    [img_to_show, t]= linear_stretch(img_to_show, i==1, t);
    imshow(img_to_show(:,:,3:-1:1))
    title(names(i))
    axis('off')
    cnt = cnt+ 1;
end

%%
table_results = array2table(MatrixResults, 'VariableNames', {'Q', 'Q_avg', 'SAM', 'ERGAS', 'SCC'}, 'RowNames',names_cpy);
original_res = readtable("../results/alg_results_"+main_title+".xlsx", "ReadRowNames",true);
total_results = [original_res;table_results(2:end,:)]
disp("Saving in " + main_title + ".xlsx file")
writetable(total_results, main_title+".xlsx",'UseExcel',true,'WriteRowNames',true);
