
clear, clc, close all

addpath([pwd,'/Tools']);
folder_path = 'C:\Users\pmans\Documents\Magistrale\Remote Sensing\Progetto\PAirMax'
output_path = "C:\Users\pmans\Documenti\Progetti_local\Pycharm\Gan-Pansharpening\data\RR\"
ratio = 4;
if exist(output_path, 'dir')
   rmdir(output_path, 's');
end
mkdir(output_path);


folders = dir(folder_path);
folders = setdiff({folders.name},{'.','..'})';
for i = 1:length(folders)
    im_name=folders{i};
    if isfolder(folder_path + "/" + im_name)
        satellite = im_name(1:2);
        switch(satellite)
            case 'W2'
                sensor = 'WV2';
            case 'W3'
                sensor = 'WV3';
            case 'W4'
                sensor = 'WV4';
            case 'GE'
                sensor = 'GeoEye1';
            otherwise
                sensor = 'none';
        end
            
        if ~exist(output_path+"\"+satellite, 'dir')
            mkdir(output_path, satellite);
            index=1;
        else
            index = index+1;
        end

        % Load Images
        I_PAN = double(imread(folder_path + "/" + im_name+"/RR/PAN.tif"));
        I_MS = double(imread(folder_path + "/" + im_name+"/RR/MS.tif"));
        I_MS_LR = double(imread(folder_path + "/" + im_name+"/RR/MS_LR.tif"));
        I_GT = double(imread(folder_path + "/" + im_name+"/RR/GT.tif"));
        
        % Generate Downgraded Version
        % GT
        I_GT = I_MS_LR;
        
        %   Preparation of image to fuse
        [I_MS_LR, I_PAN] = resize_images(I_MS_LR,I_PAN,ratio,sensor);
        
        % Upsampling
        I_MS = interp23tap(I_MS_LR,ratio);
        
        % Save Images to new Folder

        clear i
        save(output_path+"/"+satellite+"/"+im_name+".mat");
               
%         imwrite(ms_LP_d, output_path+"/"+satellite+"/"+index+"-MS_d.tif")
%         imwrite(I_MS, output_path+"/"+satellite+"/"+index+"-GT_MS.tif")
%         imwrite(ms_up, output_path+"/"+satellite+"/"+index+"-MS_up.tif")

    end
end

disp("Images Generated")

% I_MS = double(I_MS);
% I_PAN = double(I_PAN);
% 
% pan_LP = MTF_PAN(I_PAN,sensor,ratio);
% pan_LP_d = pan_LP(3:ratio:end,3:ratio:end);
% 
% ms_orig = imresize(I_MS,1/ratio);
% 
% ms_LP_d = MTF(ms_orig,sensor,ratio);


