function composed = compose_from_patches(data)

    sz = size(data);
    num_patch = sz(1);
    patch_size = sz(2);
    channels = sz(4); 
    new_dim =patch_size*sqrt(num_patch);
    Gridy = 1:patch_size:new_dim;
    Gridx = 1:patch_size:new_dim;
    
    composed = zeros([new_dim,new_dim,channels]);
    cnt=1;
    
    for i = 1: length(Gridx)
        for j = 1:length(Gridy)
            XX = Gridx(i);
            YY = Gridy(j);  

            composed(XX:XX+patch_size-1, YY:YY+patch_size-1,:) = ...
                data(cnt, :, :, :);
            
            cnt = cnt+1;
          
        end
    end

end