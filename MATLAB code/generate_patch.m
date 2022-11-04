clear; close all; clc
extract_train = true;
%% Path definition
input_path = "C:\Users\pmans\Documenti\Progetti_local\Pycharm\Gan-Pansharpening\data\RR\";
satellite = "W3/";

satellite_folder = input_path + satellite
output_folder = "C:\Users\pmans\Documenti\Progetti_local\Pycharm\Gan-Pansharpening\datasets\RR\" + satellite

if exist(output_folder,'dir')
    rmdir(output_folder, 's')
end
mkdir(output_folder)


files = dir(satellite_folder+'/*.mat');  
for file=1:length(files)
    scale = 4;
    
    
    % load inpainted image
    img_name = files(file).name;
    command = strcat('load', 32, char(satellite_folder), img_name);
    eval(command)
    
    PAN2 = I_PAN;
    LMS2 = I_MS;
    GT2  = I_GT;
    MS2  = I_MS_LR;    
    
    maxval =  max(PAN2(:));
    
    num_train = 0;
    num_valid = 0;
    size_h = 0;
    size_l = 0;
    
    %% Dataset Creation
    if extract_train == true
        %% Divide l'immagine in training (75%) e testing (25%)
        s = size(GT2);
        cut_num = s(1)/2;
        [a, b, c] = size(GT2);
        GT        = GT2(:, cut_num+1:end,:);  % for training dataset
        GT_val   = GT2(1:cut_num, 1:cut_num, :);     % for validation dataset
        GT_test   = GT2(cut_num+1:end, 1:cut_num, :);     % for testing dataset
        
        PAN       = PAN2(:, cut_num+1:end);
        PAN_val  = PAN2(1:cut_num, 1:cut_num, :);
        PAN_test  = PAN2(cut_num+1:end, 1:cut_num, :);
        
        LMS       = LMS2(:, cut_num+1:end, :);
        LMS_val  = LMS2(:, 1:cut_num, :);
        LMS_test  = LMS2(cut_num+1:end, 1:cut_num, :);
        
        MS        = MS2(:, fix(cut_num/scale)+1:end, :);
        MS_val   = MS2(1:fix(cut_num/scale), 1:fix(cut_num/scale), :);
        MS_test   = MS2(fix(cut_num/scale)+1:end, 1:fix(cut_num/scale), :);
    
        
        %% Train and Validation Patching
        dims = [size(PAN), size(PAN_val)];
        sizes_h = 2.^[5:min(log2(dims))];
        sizes_l = sizes_h./4;
        overlap = 1; % (for traning data: 0<=overlaop<=16)
        fid = fopen(output_folder+'/train_'+string(file)+"_"+'.txt', 'wt' );
        fprintf(fid, "File Used for dataset : %s\n", img_name);
        for i=1:length(sizes_h)
            Pre_NumInd = 1;
            size_l = sizes_l(i);
            size_h = sizes_h(i);
            % Train
            tic
            [gt_Oneimg, pan_Oneimg, ms_Oneimg, lms_Oneimg] = segImg_new(PAN, LMS, GT, MS, size_l, size_h, scale, overlap);
            toc
            [NumInd, ~, ~, ~] = size(gt_Oneimg);
            Post_NumInd = Pre_NumInd + NumInd - 1;
            gt_tmp1 = zeros(size(gt_Oneimg));
            pan_tmp1 = zeros(size(pan_Oneimg));
            ms_tmp1 = zeros(size(ms_Oneimg));
            lms_tmp1 = zeros(size(lms_Oneimg));

            fprintf(['%d-th Img.(patching for training):  ', 'Pre_NumInd = %d;  ', ' Post_NumInd = %d \n'], i, Pre_NumInd, Post_NumInd)
            gt_tmp1(Pre_NumInd: Post_NumInd, :, :, :) = gt_Oneimg;  % gt tensor: Nx64x64x8
            pan_tmp1(Pre_NumInd: Post_NumInd, :, :)   = pan_Oneimg;  % pan tensor: Nx64x64
            ms_tmp1(Pre_NumInd: Post_NumInd, :, :, :) = ms_Oneimg;  % ms tensor: Nx16x16x8
            lms_tmp1(Pre_NumInd: Post_NumInd, :, :, :)= lms_Oneimg;  % lms tensor: Nx64x64x8
            Pre_NumInd = Post_NumInd + 1;
            
            % Val
            tic
            [gt_Oneimg, pan_Oneimg, ms_Oneimg, lms_Oneimg] = segImg_new(PAN_val, LMS_val, GT_val, MS_val, size_l, size_h, scale, overlap);
            toc
            [NumInd, ~, ~, ~] = size(gt_Oneimg);
            Post_NumInd = Pre_NumInd + NumInd - 1;
            fprintf(['%d-th Img.(patching for validation):  ', 'Pre_NumInd = %d;  ', ' Post_NumInd = %d \n'], i, Pre_NumInd, Post_NumInd)
            gt_tmp1(Pre_NumInd: Post_NumInd, :, :, :) = gt_Oneimg;  % gt tensor: Nx64x64x8
            pan_tmp1(Pre_NumInd: Post_NumInd, :, :)   = pan_Oneimg;  % pan tensor: Nx64x64
            ms_tmp1(Pre_NumInd: Post_NumInd, :, :, :) = ms_Oneimg;  % ms tensor: Nx16x16x8
            lms_tmp1(Pre_NumInd: Post_NumInd, :, :, :)= lms_Oneimg;  % lms tensor: Nx64x64x8       
            Pre_NumInd = Post_NumInd + 1;

        
            %% Data Augmentation Increase samples to 10,000 (NxCxHxW's inverse = WxHxCxN)
           
            
            exp_num = size(gt_tmp1, 1);  
            if exp_num < 10000
            
                % Step2: two flips (lr + ud) to add examples
                gt_tmp = zeros(size(gt_tmp1));
                gt_tmp(1:exp_num, :, :, :)             = gt_tmp1;
                gt_tmp(exp_num+1:2*exp_num, :, :, :)   = flip(gt_tmp1, 2);  % two flips (lr + ud) to add examples
                gt_tmp(2*exp_num+1:3*exp_num, :, :, :) = flip(gt_tmp1, 3);
            
                ms_tmp = zeros(size(ms_tmp1));
                ms_tmp(1:exp_num, :, :, :)             = ms_tmp1;
                ms_tmp(exp_num+1:2*exp_num, :, :, :)   = flip(ms_tmp1, 2);  % two flips (lr + ud) to add examples
                ms_tmp(2*exp_num+1:3*exp_num, :, :, :) = flip(ms_tmp1, 3);
            
                lms_tmp = zeros(size(lms_tmp1));
                lms_tmp(1:exp_num, :, :, :)             = lms_tmp1;
                lms_tmp(exp_num+1:2*exp_num, :, :, :)   = flip(lms_tmp1, 2);  % two flips (lr + ud) to add examples
                lms_tmp(2*exp_num+1:3*exp_num, :, :, :) = flip(lms_tmp1, 3);
            
                pan_tmp = zeros(size(pan_tmp1));
                pan_tmp(1:exp_num, :, :)             = pan_tmp1;
                pan_tmp(exp_num+1:2*exp_num, :, :)   = flip(pan_tmp1, 2);  % two flips (lr + ud) to add examples
                pan_tmp(2*exp_num+1:3*exp_num, :, :) = flip(pan_tmp1, 3);
            
                % Step3: only select first 10000 patches for training:
                num_cut = 10000;
                gt_tmp(num_cut+1:end, :, :, :) = []; 
                ms_tmp(num_cut+1:end, :, :, :) = []; 
                lms_tmp(num_cut+1:end, :, :, :) = []; 
                pan_tmp(num_cut+1:end, :, :) = []; 
                
            else
                num_cut = exp_num;
                
                gt_tmp = gt_tmp1;
                ms_tmp = ms_tmp1;
                lms_tmp=lms_tmp1;
                pan_tmp=pan_tmp1;
            end
            
    
            %% Save dataset: 1) training data (90%); 2) validation data (10%); 
           
            sz = size(pan_tmp);
            Post_NumInd = sz(1);
            
            nz_idx    = randperm(Post_NumInd);
            num_train = fix(0.9*Post_NumInd); % # training samples
            num_valid  = Post_NumInd - num_train ; % # validation samples
            
            % save to H5 file (NxCxHxW)
            % generate training dataset:
            gt   = gt_tmp(nz_idx(1:num_train), :, :, :); % NxHxWxC=1x2x3x4
            pan  = pan_tmp(nz_idx(1:num_train), :, :);   % NxHxW = 1x2x3 (PAN)
            ms   = ms_tmp(nz_idx(1:num_train), :, :, :);
            lms  = lms_tmp(nz_idx(1:num_train), :, :, :);
            
            %--- for training data:
            filename_train = output_folder + "/train_"+string(file)+"_"+string(size_h)+".h5";
            
            save_dataset(gt, pan, ms, lms, filename_train);
            
            % generate validation dataset:
            gt   = gt_tmp(nz_idx(num_train+1: num_train+num_valid), :, :, :);
            pan  = pan_tmp(nz_idx(num_train+1: num_train+num_valid), :, :);
            ms   = ms_tmp(nz_idx(num_train+1: num_train+num_valid), :, :, :);
            lms  = lms_tmp(nz_idx(num_train+1: num_train+num_valid), :, :, :);
            
            %--- for valid data:
            filename_valid = output_folder + "/val_"+string(file)+"_"+string(size_h)+".h5";
            
            save_dataset(gt, pan, ms, lms, filename_valid);
            fprintf(fid, "Number of Patches for Training : %d\n", num_train);
            fprintf(fid, "Number of Patches for Validation : %d\n", num_valid);
            fprintf(fid, "Size of Training and Validation Patches: %dx%d\n\n",size_h, size_h );
        
        end
        fclose(fid);
        extract_train = false;
    else
        PAN_test = PAN2;
        LMS_test = LMS2;
        GT_test = GT2;
        MS_test = MS2;    
    end
    
    %% Test Set Patching
    dims = size(PAN_test);
    sizes_h = 2.^[5:min(log2(dims))];
    sizes_l = sizes_h./4;
    fid = fopen(output_folder+'/test_'+string(file)+'.txt', 'wt' );
    fprintf(fid, "File Used for dataset : %s\n", img_name);
    for i=1:length(sizes_h)
        Pre_NumInd_test = 1;
        size_l_test = sizes_l(i);
        size_h_test = sizes_h(i);
        overlap_test = 1; % (for testing data: 0<=overlaop<=64)
        
        tic
        [gt_Oneimg_test, pan_Oneimg_test, ms_Oneimg_test, lms_Oneimg_test] = segImg_new(PAN_test, LMS_test, GT_test, MS_test, size_l_test, size_h_test, scale, overlap_test);
        toc
        
        % save the Imgs into a tensor
        [NumInd_test, ~, ~, ~] = size(gt_Oneimg_test);
        Post_NumInd_test = Pre_NumInd_test + NumInd_test - 1;
        
        fprintf(['%d-th Img. (test):  ', 'Pre_NumInd_test = %d;  ', ' Post_NumInd_test = %d \n'], i, Pre_NumInd_test, Post_NumInd_test)
        % save data
        gt_tmp_test = gt_Oneimg_test;  % gt tensor: Nx512x512x8
        pan_tmp_test = pan_Oneimg_test;  % pan tensor: Nx512x512
        ms_tmp_test = ms_Oneimg_test;  % ms tensor: Nx128x128x8
        lms_tmp_test = lms_Oneimg_test;  % lms tensor: Nx512x512x8
        
        Pre_NumInd_test = Post_NumInd_test + 1; 
            
        sz = size(gt_tmp_test);
        num_test = sz(1);
        gt   = gt_tmp_test(:, :, :, :);
        pan  = pan_tmp_test(:, :, :);
        ms   = ms_tmp_test(:, :, :, :);
        lms  = lms_tmp_test(:, :, :, :);
        
        filename_test = output_folder + "/test_"+string(file)+"_"+string(size_h_test)+".h5";
        save_dataset(gt, pan, ms, lms, filename_test);
    
    
        % Report
    
        fprintf(fid, "Filename = test_"+string(size_h_test)+".h5\n");
        fprintf(fid, "Number of Patches for Test : %d\n", num_test);
        fprintf(fid, "Size of Test Patches: %dx%d\n\n",size_h_test, size_h_test );
       
    end
    fclose(fid);
    close all
end