import cv2 as cv
import os

path_video = .\Data\AVE\' 
path_save = '.\Data\Img3\' 

for root, dirs, files_video in os.walk(path_video):
    for files_name in files_video:
        print(files_name)
        count = 0
        video_address = path_video + files_name
        os.makedirs(path_save+files_name[:-4] + '\\', exist_ok=True)
        video = cv.VideoCapture(video_address)
        if (video.isOpened() == False):
            print("error opening video stream or file!")
        while (video.isOpened()):
            ret, frame = video.read()
            
            if ret == False:
                break
            # time_stamp_fromvideo = video.get(cv.CAP_PROP_POS_MSEC)
            pic_path = path_save+files_name[:-4] + '\\'
            frame = cv.resize(frame, (400,400))
            cv.imwrite(pic_path + str(count).zfill(4) + '.jpg', frame)
            # print(time_stamp_fromvideo)
            # print("\n")
            # cv.waitKey(1)
            count = count+1
        print(count)

        video.release()
        #print('the current video' + files_name  + ' is done')
