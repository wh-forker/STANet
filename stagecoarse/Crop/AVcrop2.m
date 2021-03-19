close all;
clear all;
fidp = 'D:\wgt\AVE_master_test\firststage.txt';
[path, prd, keneng, label] = textread(fidp, '%s%s%s%s', 'delimiter', '+');
ImgIndex = 1;
LengthFiles = length(path);
crop_path = 'D:\wgt\AVE_master_test\crop\';
rgb_path = 'D:\wgt\AVE_master_test\rgb\';
txt_path = 'D:\wgt\AVE_master_test\txt\';

rect = [];
for ImgIndex=70000:140000 
      CImgIndex = ImgIndex;
      if strcmp(label(CImgIndex), prd(CImgIndex)) & str2double(keneng(CImgIndex))>0.99
          if(CImgIndex<=LengthFiles)
               imgpath = char(path(CImgIndex, :));
                path_len = length(dir(['D:\wgt\AVE_master_test\result3' imgpath(1:end-10)]));
                index = str2num(imgpath(end-9:end-6));
                path_all = strsplit(imgpath, '\');
                if ~exist([crop_path path_all{2}],'dir')   mkdir([crop_path path_all{2}]);  end
                if ~exist([rgb_path path_all{2}],'dir')   mkdir([rgb_path path_all{2}]);  end
                if ~exist([txt_path path_all{2}],'dir')   mkdir([txt_path path_all{2}]);  end
                if ~exist([crop_path path_all{2} '\' path_all{3}],'dir')   mkdir([crop_path path_all{2} '\' path_all{3}]);  end
                if ~exist([rgb_path path_all{2} '\' path_all{3}],'dir')   mkdir([rgb_path path_all{2}  '\' path_all{3}]);  end
                if ~exist([txt_path path_all{2}  '\' path_all{3}],'dir')   mkdir([txt_path path_all{2}  '\' path_all{3}]);  end

                V_path = ['D:\wgt\AVE_master_test\result3' imgpath(1:end-10) num2str(index, '%04d') '_c.jpg'];
                index
                V = MatrixNormalization(im2double(imread(V_path)));

                [row, col]  = find(V(:,:) > 2*mean(mean(V)));
                max_col = max(col);
                min_col = min(col);
                max_row = max(row);
                min_row = min(row);
                if ~(isempty(max_col)|isempty(min_col)|isempty(max_row)|isempty(min_row))
                    rect = [min_row, max_row, min_col, max_col];
                    RGB_path = ['F:\wgt\AVE\AVE_Dataset\Img2' imgpath(1:end-6) '.jpg'];
                    RGB = im2double(imresize(imread(RGB_path), [224, 224]));
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
end
fprintf('done!\n');