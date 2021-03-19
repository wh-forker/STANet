function computecrop(CImgIndex, path, crop_path, rgb_path, txt_path)
imgpath = char(path(CImgIndex, :));
path_len = length(dir(['E:\crop\tcam' imgpath(1:end-10)]));
index = str2num(imgpath(end-9:end-6));
AV_path = ['E:\crop\avcam' imgpath(1:end-10) num2str(index, '%04d') '_c.jpg'];
path_all = strsplit(imgpath, '\');
if ~exist([crop_path path_all{2}],'dir')   mkdir([crop_path path_all{2}]);  end
if ~exist([rgb_path path_all{2}],'dir')   mkdir([rgb_path path_all{2}]);  end
if ~exist([txt_path path_all{2}],'dir')   mkdir([txt_path path_all{2}]);  end
if ~exist([crop_path path_all{2} '\' path_all{3}],'dir')   mkdir([crop_path path_all{2} '\' path_all{3}]);  end
if ~exist([rgb_path path_all{2} '\' path_all{3}],'dir')   mkdir([rgb_path path_all{2}  '\' path_all{3}]);  end
if ~exist([txt_path path_all{2}  '\' path_all{3}],'dir')   mkdir([txt_path path_all{2}  '\' path_all{3}]);  end

if index>=6 & index<path_len-2
    RGB_path = ['G:\AVE-ECCV18-master\AVE_Dataset\img2' imgpath(1:end-6) '.jpg'];
    RGB = im2double(imresize(imread(RGB_path), [356, 356]));
    ALL = zeros(356,356);
    for xmm = -2:2
        T_path = ['E:\crop\tcam' imgpath(1:end-10) num2str(index+xmm, '%04d') '_c.jpg'];
        V_path = ['E:\crop\vcam' imgpath(1:end-10) num2str(index+xmm, '%04d') '_c.jpg'];
        T = MatrixNormalization(im2double(imread(T_path)));
        V = MatrixNormalization(im2double(imread(V_path)));
        VT = MatrixNormalization(V+T+ALL);
        ALL = VT;
    end
    if exist(AV_path,'file')
            AV_path = ['E:\crop\avcam' imgpath(1:end-10) num2str(index, '%04d') '_c.jpg'];
            AV = MatrixNormalization(im2double(imread(AV_path)));
            AVT = MatrixNormalization(VT+AV);
            [row, col]  = find(AVT(:,:) > 2*mean(mean(AVT)));
            max_col = max(col);
            min_col = min(col);
            max_row = max(row);
            min_row = min(row);
            if ~(isempty(max_col)|isempty(min_col)|isempty(max_row)|isempty(min_row))
                result = RGB(min_row:max_row, min_col:max_col,:);
%                 figure(3),imshow(result);
%                 ALL = zeros(356,356,3);
%                 ALL(:,:,1) = AVT;ALL(:,:,2) = AVT;
%                 figure(4),imshow(imadd(RGB*0.5, 0.5*ALL));
                fid=fopen([txt_path path_all{2}  '\' path_all{3} '\' path_all{4}(1:end-6) '.txt'],'w');%写入文件路径
                fprintf(fid,'%d %d %d %d\n', min_row, max_row, min_col, max_col);   %按列输出，若要按行输出：fprintf(fid,'%.4\t',A(jj)); 
                imwrite(result, [crop_path path_all{2}  '\' path_all{3} '\' path_all{4}]);
                imwrite(RGB, [rgb_path path_all{2}  '\' path_all{3} '\' path_all{4}]);
                fclose(fid);
            end
    else
            [row, col]  = find(VT(:,:) > 2*mean(mean(VT)));
            max_col = max(col);
            min_col = min(col);
            max_row = max(row);
            min_row = min(row);
            if ~(isempty(max_col)|isempty(min_col)|isempty(max_row)|isempty(min_row))
                result = RGB(min_row:max_row, min_col:max_col,:);
%                 figure(3),imshow(result);
%                 ALL = zeros(356,356,3);
%                 ALL(:,:,1) = VT;ALL(:,:,2) = VT;
%                 figure(4),imshow(imadd(RGB*0.5, 0.5*ALL));
                fid=fopen([txt_path path_all{2}  '\' path_all{3} '\' path_all{4}(1:end-6) '.txt'],'w');%写入文件路径
                fprintf(fid,'%d %d %d %d\n', min_row, max_row, min_col, max_col);   %按列输出，若要按行输出：fprintf(fid,'%.4\t',A(jj)); s
                imwrite(result, [crop_path path_all{2}  '\' path_all{3} '\' path_all{4}]);
                imwrite(RGB, [rgb_path path_all{2}  '\' path_all{3} '\' path_all{4}]);
                fclose(fid);
            end
    end
end

end