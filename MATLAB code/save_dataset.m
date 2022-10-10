function save_dataset(gt, pan, ms, lms, filename)

% Permute to (NxCxHxW's inverse = WxHxCxN)
gt   = permute(gt,[3 2 4 1]); 
pan_t(1,:,:,:) = pan; 
pan   = permute(pan_t,[4 3 1 2]); 
ms   = permute(ms,[3 2 4 1]); 
lms   = permute(lms,[3 2 4 1]); 

gtsz = size(gt);
mssz = size(ms);
lmssz = size(lms);
pansz =size(pan);

if length(pansz) == 2
    h5create(filename, '/gt', [gtsz(1:end), 1], 'Datatype', 'double'); 
    h5create(filename, '/ms', [mssz(1:end), 1], 'Datatype', 'double'); 
    h5create(filename, '/lms', [lmssz(1:end), 1], 'Datatype', 'double');
    h5create(filename, '/pan', [pansz(1:end), 1], 'Datatype', 'double'); 

    h5write(filename, '/gt', double(gt));
    h5write(filename, '/ms', double(ms));
    h5write(filename, '/lms', double(lms));
    h5write(filename, '/pan', double(pan));
else

    h5create(filename, '/gt', gtsz(1:end), 'Datatype', 'double'); 
    h5create(filename, '/ms', mssz(1:end), 'Datatype', 'double'); 
    h5create(filename, '/lms', lmssz(1:end), 'Datatype', 'double');
    h5create(filename, '/pan', pansz(1:end), 'Datatype', 'double'); 
    
    h5write(filename, '/gt', double(gt), [1,1,1,1], size(gt));
    h5write(filename, '/ms', double(ms), [1,1,1,1], size(ms));
    h5write(filename, '/lms', double(lms), [1,1,1,1], size(lms));
    h5write(filename, '/pan', double(pan), [1,1,1,1], size(pan));

end
clear gt ms lms pan pan_t
end