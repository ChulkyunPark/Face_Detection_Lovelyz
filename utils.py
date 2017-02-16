import os
import re
import urllib.request
import cv2
from Face_detection_lovelyz.synset import *
def search_url_from_file(file, output_file):
    f = open(file, 'r')
    lines = f.readlines()
    for line in lines:
        re_compile = re.compile('(http[s]?)+[://]+(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9]|[a-zA-Z]))+')
        str_line = re_compile.search(line)
        print(str_line.group(0))
        f_o = open(output_file, 'a')
        f_o.write(str_line.group(0)+'\n')
    f.close()

def download_images(url_file, download_path):
    image_num = 0
    f = open(url_file, 'r')
    data_path = os.path.join("C:/Users\Chulkyun Park\Documents\PycharmProjects\SSD_Detector", download_path)
    url_line = f.readlines()
    for line in url_line:
        image_path = download_path + "_" + str(image_num) + ".jpg"
        full_name = os.path.join(data_path, image_path)
        try:
            urllib.request.urlretrieve(line, full_name)
        except urllib.request.HTTPError as e:
            error_message = e.read()
            print(error_message)
        image_num += 1

# For train.py and test.py
def list_image(directory_name):
    try:
        image_list = []
        filenames = os.listdir(directory_name)
        for filename in filenames:
            full_filename = os.path.join(directory_name, filename)
            if os.path.isdir(full_filename):
                list_image(full_filename)
            else:
                ext = os.path.splitext(full_filename)[-1]
                file_label = os.path.basename(full_filename)
                file_label = file_label.split('_')
                if (ext == '.jpg') | (ext == '.png'):
                    if full_filename in image_list:
                        continue
                    else:
                        label_index = synset_map[file_label[0]]["index"]
                        image_list.append({"label_name":file_label[0],
                                           "label_index":label_index,
                                           "filename":full_filename,
                                           "desc": synset[label_index]})
                else:
                    continue
        return(image_list)
    except PermissionError:
        pass

# For image_download and haar_cascades
def list_image_simple(directory_name):
    try:
        image_list = []
        filenames = os.listdir(directory_name)
        for filename in filenames:
            full_filename = os.path.join(directory_name, filename)
            if os.path.isdir(full_filename):
                list_image(full_filename)
            else:
                ext = os.path.splitext(full_filename)[-1]
                # file_label = os.path.basename(full_filename)
                # file_label = file_label.split('_')
                if (ext == '.jpg') | (ext == '.png'):
                    if full_filename in image_list:
                        continue
                    else:
                        # label_index = synset_map[file_label[0]]["index"]
                        image_list.append(full_filename)
                else:
                    continue
        return(image_list)
    except PermissionError:
        pass



def search_face_from_image(image_list):
    face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    image_num = 0
    for image in image_list:
        image_path = "kei_" + str(image_num) + ".jpg"
        crop_data_path = "C:/Users\Chulkyun Park\Documents\PycharmProjects\SSD_Detector\kei_crop_size"
        crop_full_path = os.path.join(crop_data_path, image_path)

        img = cv2.imread(image)
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(img, 1.3, 5)
        image_num += 1
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            resize_roi_color = cv2.resize(roi_color, (100,100))
            cv2.imwrite(crop_full_path, resize_roi_color)
        # cv2.imshow('img', img)

        # cv2.imwrite(crop_full_path, faces)
        # # image_num += 1

    cv2.waitKey(0)
    cv2.destroyAllWindows()
#
# search_url_from_file('curl_kei.txt', 'url_kei.txt')
# search_url_from_file('curl_jisu.txt', 'url_jisu.txt')
# search_url_from_file('curl_mijoo.txt', 'url_mijoo.txt')
# download_images('url_kei.txt', 'kei')
# download_images('url_jisu.txt', 'jisu')
# download_images('url_mijoo.txt', 'mijoo')
#
#
# image_list = list_image_simple('C:/Users\Chulkyun Park\Documents\PycharmProjects\SSD_Detector\kei')
# # print(image_list)
# search_face_from_image(image_list)