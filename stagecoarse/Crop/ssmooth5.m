close all;
clear all;
imgDataPath = 'D:\wgt\STVIS\test\split1_results\AVAD\';
imgDataDir = dir(imgDataPath); % 遍历所有文件
save_path = 'D:\wgt\STVIS\test\split1_results\AVAD_2\';
V_ALL = [];
for CImgIndex=3:length(imgDataDir)
       imgDir = dir([imgDataPath imgDataDir(CImgIndex).name '/*.jpg']); 
        for index = 3:length(imgDir)
            if ~exist([save_path imgDataDir(CImgIndex).name],'dir')   mkdir([save_path imgDataDir(CImgIndex).name]);  end
            %if ~exist([save_path path_all{2} '\' path_all{3}],'dir')   mkdir([save_path path_all{2} '\' path_all{3}]);  end
            if index == 5
                for xmm = -2:2
                    V_path = [imgDataPath imgDataDir(CImgIndex).name '\' imgDir(index+xmm).name];
                    V = MatrixNormalization(im2double(imread(V_path)));
                    V_ALL{xmm+3} = V;
                end
                VHH = MatrixNormalization((V_ALL{1}+V_ALL{2}+V_ALL{3}+V_ALL{4}+V_ALL{5})./5);
                V_ALL{5} = VHH;
            end
            if index>=6 & index<length(imgDir)-5
                V_path = [imgDataPath imgDataDir(CImgIndex).name '\' imgDir(index+xmm).name];
                index
                V = MatrixNormalization(im2double(imread(V_path)));
                V_ALL{1} = V_ALL{2};V_ALL{2}=V_ALL{3};V_ALL{3}=V_ALL{4};V_ALL{4}=V_ALL{5};V_ALL{5}=V;
                VHH = MatrixNormalization((V_ALL{1}+V_ALL{2}+V_ALL{3}+V_ALL{4}+V_ALL{5})./5);
                V_ALL{5} = VHH;
    %                 figure(3),imshow(VHH);
    %                 ALL = zeros(356,356,3);
    %                 ALL(:,:,1) = VHH;ALL(:,:,2) = VHH;
    %                 figure(4),imshow(imadd(RGB*0.5, 0.5*ALL));
               imwrite(VHH, [save_path imgDataDir(CImgIndex).name '\' imgDir(index+xmm).name]);
            end
        end
end
fprintf('done!\n');