close all;
clear all;
fidp = 'G:\V_ECCV_test\fenlei.txt';
[path, prd, keneng, label] = textread(fidp, '%s%s%s%s', 'delimiter', '+');
ImgIndex = 500001;
LengthFiles = length(path);
crop_path = 'F:\crop\crop\';
rgb_path = 'F:\crop\rgb\';
txt_path = 'F:\crop\txt\';

rect = [];
while(ImgIndex<=600000)
      CImgIndex = ImgIndex;
      if strcmp(label(CImgIndex), prd(CImgIndex)) & str2double(keneng(CImgIndex))>0.99
          if(CImgIndex<=LengthFiles)
               imgpath = char(path(CImgIndex, :));
                path_len = length(dir(['G:\V_ECCV_test\result' imgpath(1:end-10)]));
                index = str2num(imgpath(end-9:end-6));
                path_all = strsplit(imgpath, '\');
                if ~exist([crop_path path_all{2}],'dir')   mkdir([crop_path path_all{2}]);  end
                if ~exist([rgb_path path_all{2}],'dir')   mkdir([rgb_path path_all{2}]);  end
                if ~exist([txt_path path_all{2}],'dir')   mkdir([txt_path path_all{2}]);  end
                if ~exist([crop_path path_all{2} '\' path_all{3}],'dir')   mkdir([crop_path path_all{2} '\' path_all{3}]);  end
                if ~exist([rgb_path path_all{2} '\' path_all{3}],'dir')   mkdir([rgb_path path_all{2}  '\' path_all{3}]);  end
                if ~exist([txt_path path_all{2}  '\' path_all{3}],'dir')   mkdir([txt_path path_all{2}  '\' path_all{3}]);  end

                V_path = ['G:\V_ECCV_test\result' imgpath(1:end-10) num2str(index, '%04d') '_c.jpg'];
                index
                V = MatrixNormalization(im2double(imread(V_path)));
                V = MatrixNormalization(V);
                [row, col]  = find(V(:,:) > 3*mean(mean(V)));
                max_col = max(col);
                min_col = min(col);
                max_row = max(row);
                min_row = min(row);
                if ~(isempty(max_col)|isempty(min_col)|isempty(max_row)|isempty(min_row))
                    rect = [min_row, max_row, min_col, max_col];
                    RGB_path = ['G:\AVE-ECCV18-master\AVE_Dataset\img2' imgpath(1:end-6) '.jpg'];
                    RGB = im2double(imresize(imread(RGB_path), [356, 356]));
                    result = RGB(min_row:max_row, min_col:max_col,:);
    %                 figure(3),imshow(result);
                    %     ALL = zeros(356,356,3);
                    %     ALL(:,:,1) = VT;ALL(:,:,2) = VT;
                    %     figure(4),imshow(imadd(RGB*0.5, 0.5*ALL));
                    fid=fopen([txt_path path_all{2}  '\' path_all{3} '\' path_all{4}(1:end-6) '.txt'],'w');%写入文件路径
                    fprintf(fid,'%d %d %d %d\n', min_row, max_row, min_col, max_col);   %按列输出，若要按行输出：fprintf(fid,'%.4\t',A(jj)); s
                    imwrite(result, [crop_path path_all{2}  '\' path_all{3} '\' path_all{4}]);
                    %     imwrite(RGB, [rgb_path path_all{2}  '\' path_all{3} '\' path_all{4}]);
                    fclose(fid);
                 end
          end
       end
ImgIndex = ImgIndex + 1;
end
fprintf('done!\n');
fclose(fidp);