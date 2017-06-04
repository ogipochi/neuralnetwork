import cv2
import numpy as np
import sys
import glob
from optparse import OptionParser


#グローバル変数初期値
im_height=128
im_width=128

projective_rate=80
projective_times=1

Saturation_range=[60,160]
Lightness_range=[60,160]
SL_times=2
contrast_range=[[50,150],[120,220]]
contrast_times=2

resize_range_rate=[50,250]
resize_times=5

degree_range_default=[-70,70]
size_range_default=[-70,70]
degree_times=2

def usage():
    print("usage:preprocess [OPTION]")
    print()
    print("  -d :image directory path")
    sys.exit()

#バックに指定する画素の計算
def back_RGB(image,image_size=128,get_position_per=10,tolerance=10,CHANNEL = 3):
    #取得するピクセルを定義
    percent_point =int( image_size * get_position_per/100)
    a1 = percent_point
    a2=image_size-percent_point
    pixel1 = image[a1][a1]
    pixel2 = image[a1][a2]
    pixel3 = image[a2][a1]
    pixel4 = image[a2][a2]
    #ピクセルの平均との差がtolerate以上なら削除
    pixel_list=[pixel1,pixel2,pixel3,pixel4]
    pixel_mean =[0,0,0]
    def mean_cal(list,num):
        total = 0
        sum   = 0
        for i in list:
            sum+=i[num]
            total+=1
        mean = sum/total
        return mean

    for i in range(CHANNEL):
        pixel_mean[i] = mean_cal(pixel_list,i)
    # for pixel in pixel_list:
    #     if abs(pixel_mean[0]-pixel[0]) > tolerance and abs(pixel_mean[1]-pixel[1]) > tolerance and abs(pixel_mean[2]-pixel[2]) > tolerance:
    #         pixel_list.remove(pixel)
    # if len(pixel_list)==0:
    #     print("recommend tolerate change in back_RGB")
    return pixel_mean
    # else :
    #     returnpix=[0,0,0]
    #     for i in range(CHANNEL):
    #         returnpix[i] = mean_cal(pixel_list,i)
    #     return returnpix
#黒いピクセルをbackで穴埋め
def completion(image ,blank = [255,255,255],CHANNEL = 3):
    height = 0
    for x in image:
        width=0
        for y in x :
            if all((pixel == 0 for pixel in y)):

                image[height][width]=blank
            width+=1
        height+=1
    return image



def Degree(image,back=[255,255,255],complete=True,degree_range=degree_range_default,size_range=size_range_default,times =degree_times,save_dir='result',save_name='',save_format='png'):
    degree_change_once=int((degree_range[1]-degree_range[0])/times)
    size_change_once=int((size_range[1]-size_range[0])/times)
    for sizeMag in range(size_range[0], size_range[1],size_change_once):
        for degMag in range(degree_range[0], degree_range[1],degree_change_once):
            size = tuple([image.shape[1], image.shape[0]])
            center = tuple([int(size[0] / 2), int(size[1] / 2)])
            angle = int(50*degMag/100)
            scale = 1.0 + sizeMag/100
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
            img_rot = cv2.warpAffine(image, rotation_matrix, size, flags=cv2.INTER_CUBIC)
            img_flip = cv2.flip(img_rot, 1)
            if complete==True:
                img_rot=completion(image=img_rot,blank=back)
                img_flip=completion(image=img_flip,blank=back)
            cv2.imwrite(save_dir+'/' + str(degMag) + str(sizeMag) + save_name+'.'+save_format, img_rot)
            cv2.imwrite(save_dir+'/' + 'f' + str(degMag)  + str(sizeMag) + save_name+'.'+save_format, img_flip)

def projectiveTransform(image,back=[255,255,255],real_height=im_height,real_width=im_width,real_initial_position=None,complete=True,savedir='result',save_name='',save_format='png'):
    #グローバル定数をptsに設定
    if real_initial_position == None:
        projective_initial_position = [[int(real_height / 4), int(real_width / 4)],
                                       [int(real_height - real_height / 4), int(real_width / 4)],
                                       [int(real_height - real_height / 4), int(real_width - real_width / 4)],
                                       [int(real_height / 4), int(real_width - real_width / 4)]]

    pts1=np.float32(projective_initial_position)
    change_pix_height = int((real_height/4)*(projective_rate/100))
    change_pix_width = int((real_width/4)*(projective_rate/100))
    change_pix_height_once=int(change_pix_height/projective_times)
    change_pix_width_once = int(change_pix_width/projective_times)
    for h1 in range(-change_pix_height,change_pix_height,change_pix_height_once):
        for w1 in range(-change_pix_width,change_pix_width,change_pix_width_once):
            for h2 in range(-change_pix_height, change_pix_height,change_pix_height_once):
                for w2 in range(-change_pix_width, change_pix_width,change_pix_width_once):
                    for h3 in range(-change_pix_height, change_pix_height,change_pix_height_once):
                        for w3 in range(-change_pix_width, change_pix_width,change_pix_width_once):
                            for h4 in range(-change_pix_height, change_pix_height,change_pix_height_once):
                                for w4 in range(-change_pix_width, change_pix_width,change_pix_width_once):
                                    pts_change = np.float32([[h1, w1], [h2, w2], [h3, w3], [h4, w4]])
                                    pts2 = np.float32([x+y for (x,y) in zip(projective_initial_position,pts_change)])
                                    M = cv2.getPerspectiveTransform(pts1,pts2)
                                    dst_image = cv2.warpPerspective(image,M,(real_height,real_width))
                                    if complete==True:
                                        dst_image = completion(image=dst_image,blank=back)
                                    save_path=savedir+'/'+str(h1)+str(w2)+str(h2)+str(w2)+str(h3)+str(w3)+str(h4)+str(w4)+save_name+'.'+save_format
                                    cv2.imwrite(save_path,dst_image)



def Contrast(image,real_min_range=contrast_range[0],real_max_range=contrast_range[1],times=contrast_times,save_dir='result',save_name='',save_format='png'):
    min_change_once = int((real_min_range[1]-real_min_range[0])/times)
    max_change_once = int((real_max_range[1]-real_max_range[0])/times)

    for min_table in range(real_min_range[0],real_min_range[1],min_change_once):
        for max_table in range(real_max_range[0],real_max_range[1],max_change_once):
            diff_table = max_table - min_table
            look_up_table = np.arange(256, dtype='uint8')
            for i in range(0, min_table):
                look_up_table[i] = 0
            for i in range(min_table, max_table):
                look_up_table[i] = 255 * (i - min_table) / diff_table
            for i in range(max_table, 255):
                look_up_table[i] = 255
            img_contrast = cv2.LUT(image, look_up_table)
            save_path = save_dir+'/'+str(min_table)+str(max_table)+save_name+'.'+save_format
            cv2.imwrite(save_path,img_contrast)
    return

#明度,彩度変換
def Satulation_and_Lightness(image,Saturaion_real_range=None,Lightness_real_range=None,times =None,save_dir='result',save_name='',save_format='png'):
    HSV_image=cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    HSV_result_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if times ==None:
        times = SL_times
    if Saturaion_real_range ==None:
        Saturaion_real_range = Saturation_range
    if Lightness_real_range == None:
        Lightness_real_range =Lightness_range
    Saturation_change_once=int((Saturaion_real_range[1]-Saturaion_real_range[0])/times)
    Lightness_change_once = int((Lightness_real_range[1]-Lightness_real_range[0])/times)
    for saturation in range(Saturaion_real_range[0],Saturation_range[1],Saturation_change_once):
        for lightness in range(Lightness_real_range[0],Lightness_real_range[1],Lightness_change_once):
            for height in range(HSV_image.shape[0]):
                 for width in range(HSV_image.shape[1]):
                     input_s = int((saturation/100)*HSV_image[height][width][1])
                     input_l=int((lightness/100)*HSV_image[height][width][2])
                     if input_s>=255:
                         input_s=255
                     if input_l>=255:
                         input_l=255
                     HSV_result_image[height][width][1]=input_s
                     HSV_result_image[height][width][2]=input_l
            dst_iamge = cv2.cvtColor(HSV_result_image,cv2.COLOR_HSV2BGR)
            save_path = save_dir+'/'+str(saturation)+str(lightness)+save_name+'.'+save_format
            cv2.imwrite(save_path,dst_iamge)

    return
def Resize(image,ratio_lower=resize_range_rate[0],ratio_upper=resize_range_rate[1],times=resize_times,save_dir='result',save_name='',save_format='png'):
    upper_limit=int((image.shape[0]*ratio_upper)/100)
    lower_limit = int((image.shape[0]*ratio_lower)/100)
    change_ratio_once = int((upper_limit-lower_limit)/times)
    for size in range(lower_limit,upper_limit,change_ratio_once):
        dst_image = cv2.resize(image,(size,size))
        save_path=save_dir+'/'+str(size) + save_name+'.'+save_format
        cv2.imwrite(save_path,dst_image)
    return




def main():
    file_list = glob.glob("background/*")

    # for file in file_list:
    #     image_name = file.split('/')[-1]
    #
    #     image = cv2.imread(file)
    #     back_pixel = back_RGB(image=image,image_size=image.shape[0],get_position_per=10)
    #     Degree(image=image,back=back_pixel,save_name=image_name.split('.')[0])
    #     projectiveTransform(image=image,back=back_pixel,real_height=image.shape[0],real_width=image.shape[1],save_name=image_name.split('.')[0])

    file_list = glob.glob('background/*')
    # for file in file_list:
    #     image = cv2.imread(file)
    #     result_image = cv2.resize(image,(256,256))
    #     cv2.imwrite(file,result_image)
    for file in file_list:
        image = cv2.imread(file)
        image_name = file.split('/')[-1]
        print(image_name)
        print(image_name.split('.')[0])
        Contrast(image=image,save_name=image_name.split('.')[0],save_dir='background')
        Satulation_and_Lightness(image=image,save_name=image_name.split('.')[0],save_dir='background')
    

if __name__ == '__main__':
    main()
