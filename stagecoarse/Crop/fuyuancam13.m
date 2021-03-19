clear all;
close all;
imgDataPath = 'D:\wgt\AVE_master_Crop_test\result\';
savepath = 'F:\wgt\Crop_data\fuyuan\';
imgDataDir = dir(imgDataPath); % 遍历所有文件
count = 0;
for i = 27:28%length(imgDataDir)
    if(isequal(imgDataDir(i).name,'.')||... % 去除系统自带的两个隐文件夹
    isequal(imgDataDir(i).name,'..')||...
    ~imgDataDir(i).isdir) % 去除遍历中不是文件夹的
        continue;
    end
    if ~exist([savepath imgDataDir(i).name],'dir')   mkdir([savepath imgDataDir(i).name]);  end
    imgDir = dir([imgDataPath imgDataDir(i).name]); 
    for j =1:length(imgDir) % 遍历所有图片
        if(isequal(imgDir(j).name,'.')||... % 去除系统自带的两个隐文件夹
        isequal(imgDir(j).name,'..')||...
        ~imgDir(j).isdir) % 去除遍历中不是文件夹的
            continue;
        end
        if ~exist([savepath imgDataDir(i).name '\' imgDir(j).name '\'],'dir')   mkdir([savepath imgDataDir(i).name '\' imgDir(j).name '\']);  end
        img = dir([imgDataPath imgDataDir(i).name '\' imgDir(j).name '\*.jpg']);
        for k = 1:length(img)
            cam = zeros(356, 356, 3); fixation = zeros(356,356);
            if k==1
                im = imresize(im2double(imread(['F:\wgt\AVE\AVE_Dataset\Img2\' imgDataDir(i).name '\' imgDir(j).name '\' img(k).name(1:end-8) '.jpg'])),[356,356]);
            end
            path = [imgDataPath imgDataDir(i).name '\' imgDir(j).name '\' img(k).name];
            txt = load(['F:\wgt\Crop_data\txt\' imgDataDir(i).name '\' imgDir(j).name '\' img(k).name(1:end-8) '.txt']);
            a = txt(2)-txt(1)+1; b = txt(4)-txt(3)+1;
            sal = imresize(im2double(imread(path)), [a,b]);
            fixation(txt(1):txt(2), txt(3):txt(4)) = sal;
            if k==1
                cam(:,:,1) = fixation;cam(:,:,2) = fixation;
%                 figure(1),imshow(imadd(cam*0.5, im*0.5));
                imwrite(imadd(cam*0.5, im*0.5), [savepath imgDataDir(i).name '\' imgDir(j).name '\' img(k).name(1:end-8) '.jpg']);
            end
            imwrite(fixation, [savepath imgDataDir(i).name '\' imgDir(j).name '\' img(k).name(1:end-8) '_c.jpg']);
%             imwrite(G, [savepath imgDataDir(i).name '\' imgDir(j).name '\' img(k).name(1:end-8) '_2.jpg']);
        end
    end
end