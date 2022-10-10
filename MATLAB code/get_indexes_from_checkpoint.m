function get_indexes_from_checkpoint(i, test_folder, test, chk_path)
        file_name_excel = test_folder+"/"+test+".xlsx";

        fprintf("Row number : %d\n", i)
        load(chk_path+"/"+test+"/checkpoint_"+i+".mat")
        ratio = 4;
        L = 11;
        Qblocks_size = 32;
        flag_cut_bounds = 1;
        dim_cut = 21;
        thvalues = 0;
        head = {'Epoch', 'Q', 'Q_avg', 'SAM', 'ERGAS', 'SCC'};
        gen = compose_from_patches(gen_decomposed).*2048;
        gt = compose_from_patches(gt_decomposed).*2048;
        [Q_avg, SAM, ERGAS, SCC, Q] =indexes_evaluation( ...
            gen,gt,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
        %fprintf("Q_avg=%.3f  SAM=%.3f  ERGAS=%.3f  SCC=%.3f  Q=%.3f\n", Q_avg, SAM, ERGAS, SCC, Q)
        row = [i, Q, Q_avg, SAM, ERGAS, SCC];
        results = round(row, 5);
        table_results = array2table(results, 'VariableNames', head);
        writetable(table_results, file_name_excel,'UseExcel',true,'WriteRowNames',true, 'WriteMode','append');
        
end