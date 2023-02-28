
clear, clc, close all

folder_path = '..\original_images\';
base_output_path = "..\data\";
ratio = 4;

for resolution_folder = ["FR\", "RR\"]
    
    % Create Empty Resolution folder
    output_path = base_output_path + resolution_folder;
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

            if resolution_folder == "RR/"
                % GT
                I_GT = I_MS_LR;
                
                %   Preparation of image to fuse
                [I_MS_LR, I_PAN] = resize_images(I_MS_LR,I_PAN,ratio,sensor);
                
                % Upsampling
                I_MS = interp23tap(I_MS_LR,ratio);
            end
            % Save Images to new Folder
    
            clear i
            save(output_path+"/"+satellite+"/"+im_name+".mat");
        end
    end
end
disp("Images Generated")



