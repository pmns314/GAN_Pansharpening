function [ImageToView, t]= linear_stretch(ImageToView, calc_limits, t)

[N,M,~] = size(ImageToView);
NM = N*M;
for i=1:3
    b = reshape(double(uint16(ImageToView(:,:,i))),NM,1);
    if calc_limits
        [hb,levelb] = hist(b,max(b)-min(b));
        chb = cumsum(hb);
        t(1, i)=ceil(levelb(find(chb>NM*0.01, 1 )));
        t(2, i)=ceil(levelb(find(chb<NM*0.99, 1, 'last' )));
    end
    t_min = t(1, i);
    t_max = t(2, i);
    b(b<t_min)=t_min;
    b(b>t_max)=t_max;
    b = (b-t_min)/(t_max-t_min);
    ImageToView(:,:,i) = reshape(b,N,M);
end