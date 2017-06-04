import cv2
import glob
def resize(file_path):
    image = cv2.imread(file_path)
    image_height = image.shape[0]
    image_width = image.shape[1]
    resize_hright = 256
    resize_width = int(image_width*(256/image_height))
    resizediamge = cv2.resize(image,(resize_width,resize_hright))

    cv2.imwrite(file_path,resizediamge)

def main():
    file_list = glob.glob('character/*')
    for file in file_list:
        resize(file)
if __name__ =='__main__':
    main()