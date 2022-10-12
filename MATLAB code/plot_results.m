clear, clc, close all
data_path = "C:\Users\pmans\Documenti\Progetti_local\Pycharm\GAN-PAN\results\" + ...
    "W3_full3\"

filenames = ["W3_1_32_01_mse","W3_1_32_001_mse","W3_1_32_0001_mse"];
best_epochs = [];
%best_epochs = [1859 3143 12932];

if length(best_epochs) == length(filenames)
    lrs = ["lr 01","Best lr 01" ,"lr 001", "Best lr 001" ,"lr 0001","Best lr 0001"];
else
    lrs = ["lr 01","lr 001","lr 0001"];
end
colors = ['b', 'r', 'g'];
cnt =1;
for filename = filenames
    T = readtable(data_path + filename+"/test_1.xlsx")
    var_names = T.Properties.VariableNames

    for i=2:6
        figure(i)
        line = table2array(T(:,i));
        indexes = table2array(T(:,1));
        semilogx(indexes,line, colors(cnt))
        title(var_names(i))
        grid
        hold on
        if length(best_epochs) == length(filenames)
            plot(best_epochs(cnt),T{best_epochs(cnt), i}, '.', 'MarkerSize', 15, 'Color',colors(cnt))
        end
        legend(lrs)
    end
    cnt = cnt +1;
end

%% Plot Single CSV
%clear, clc, close all
data_path = "C:\Users\pmans\Documents\Progetti_Local\Pycharm\Gan-Pansharpening\pytorch_models\trained_models\" + ...
    "W3\STPSGAN\Stpsgan\test_1.csv"

cnt =1;

T = readtable(data_path)
var_names = T.Properties.VariableNames
for i=2:5
    figure(i)
    line = table2array(T(:,i));
    indexes = table2array(T(:,1));
    semilogx(indexes,line)
    title(var_names(i))
    grid
    hold on
end
cnt = cnt +1;

