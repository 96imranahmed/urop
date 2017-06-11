import tensorflow as tf
import cv2
from deepgaze.head_pose_estimation import CnnHeadPoseEstimator
import numpy as np
import os
import random 
import xml.etree.ElementTree as ET
import pickle
import cPickle as pickle

sess = tf.Session()
est = CnnHeadPoseEstimator(sess)
op_full_list = []
def pad_image_square(img_in): 
    shape = img_in.shape
    is_grayscale = (len(shape)==2)
    max_dim = max(shape)
    ret_image = None
    if is_grayscale:
        ret_image = np.zeros((max_dim, max_dim))
    else:
        ret_image = np.zeros((max_dim, max_dim, 3))
    ret_image[0:shape[0], 0:shape[1], :] = img_in
    return ret_image

def Hollywood():
    annot_dir =  "HollywoodHeads/Annotations/"
    img_dir =    "HollywoodHeads/JPEGImages/"
    count = 0
    dir_full = os.listdir(annot_dir)
    random.shuffle(dir_full)
    for filename in dir_full:
        if filename.endswith("xml"):
            cur_file = os.path.join(annot_dir, filename)
            cur_size = []
            cur_objects = []
            image = None
            root = ET.parse(cur_file).getroot()
            img_name = root.find('filename').text
            size = root.find("size")
            for elem in size.iter():
                cur_size.append(elem.text)
            del cur_size[0]
            if not cur_size[2] == '3':
                #print('Invalid image, skipping')
                continue   
            for obj in root.findall('object'):
                bbox = obj.find('bndbox')
                difficulty = obj.find('difficult')
                if bbox is None:
                    continue
                cur = {}
                for element in bbox.iter():
                    if element.tag == "bndbox":
                        pass
                    else:
                        cur[element.tag] = element.text
                cur['hard'] = difficulty.text
                cur_objects.append(cur)
            if len(cur_objects) > 1:
                #print('Skipping image, >1 heads')
                continue
            if len(cur_objects)==0:
                #print('Skipping image, no boundary box or error in format')
                continue     
            img_cv = cv2.imread(img_dir+img_name)
            for obj in cur_objects:
                if obj['hard'] == 1: continue
                left = int(float(obj['xmin']))
                right = int(float(obj['xmax']))
                upper = int(float(obj['ymin']))
                lower = int(float(obj['ymax']))
                c = int((left + right)/2)
                r = int((upper + lower)/2)
                s = int(max((right-left), (lower-upper))/2)
                
                nrows = img_cv.shape[0]
                ncols = img_cv.shape[1]
                const = 1.0
                # crop
                r0 = max(int(r - const*s), 0); r1 = min(r + const*s, nrows)
                c0 = max(int(c - const*s), 0); c1 = min(c + const*s, ncols)

                crop_cv = img_cv[int(r0):int(r1), int(c0):int(c1)]
                crop_cv = pad_image_square(crop_cv)

                if s < 70  or np.mean(crop_cv) < 60 or np.var(crop_cv) < 500 or  np.var(crop_cv)/s < 5 or  np.var(crop_cv)/s > 80:
                    continue

                if min(crop_cv.shape[0:1]) < 64: continue
                yaw = est.return_yaw(crop_cv)
                pitch = est.return_pitch(crop_cv)
                # 

                if abs(yaw[0][0][0]) > 60:
                    print(yaw[0][0][0], pitch[0][0][0])
                    cv2.imshow('Img', crop_cv/255.0)
                    cv2.waitKey(500)

def fddb():
    annot_dir =  os.getcwd()+ "/fddb/FDDB-folds/"
    img_dir =  os.getcwd() + "/fddb/"
    cnt = 0
    for f_num in range(1,10+1):
        
            index = str(f_num).zfill(2)
            annot_file = annot_dir + "FDDB-fold-" + index + "-ellipseList.txt"
            
            fp = open(annot_file)
            raw_data = fp.readlines()
            cur_img = 0
            cur_img_url = None
            stage = 0
            for parsed_data in raw_data:                        
                if stage == 0:
                    file_name = parsed_data.rstrip()
                    stage = 1
                elif stage == 1:
                    num_faces = int(parsed_data)
                    file_url = img_dir+file_name +'.jpg'
                    cur_img = cv2.imread(file_url)
                    cur_img_url = file_url
                    if min(np.shape(cur_img)) == 0: 
                        num_faces = 0
                    stage = 2

                elif stage == 2:
                    if num_faces == 0: 
                        stage = 0
                    else:
                        splitted = parsed_data.split()
                        r = int(float(splitted[4]))
                        c = int(float(splitted[3]))
                        s = int(1.1*(float(splitted[0]) + float(splitted[2])))
                                        #                      
                        nrows = cur_img.shape[0]
                        ncols = cur_img.shape[1]
                        const = 1.0
                        # crop
                        r0 = max(int(r - const*s), 0); r1 = min(r + const*s, nrows)
                        c0 = max(int(c - const*s), 0); c1 = min(c + const*s, ncols)

                        crop_cv = cur_img[int(r0):int(r1), int(c0):int(c1)]
                        crop_cv = pad_image_square(crop_cv)
                        
                        skip = False
                        
                        if s < 60  or np.mean(crop_cv) < 50 or np.var(crop_cv) < 500 or  np.var(crop_cv)/s < 5 or  np.var(crop_cv)/s > 80:
                            skip = True

                        if min(crop_cv.shape[0:1]) < 64: skip = True

                        if not skip:
                            yaw = est.return_yaw(crop_cv)
                            pitch = est.return_pitch(crop_cv)

                            if abs(yaw[0][0][0]) > 35:
                                cnt+=1
                                print(cnt, yaw[0][0][0], pitch[0][0][0])
                                cv2.imshow('Img', crop_cv/255.0)
                                cv2.waitKey(30)
                                op_full_list.append([cur_img_url, r, c, s])


                        num_faces -= 1
                        if num_faces == 0:
                            stage = 0

            fp.close()

def wider():
    data_arr = None
    with open("save.p", "rb") as f:
        data_arr = pickle.load(f) 
    random.shuffle(data_arr)
    cnt = 0
    for item in data_arr:
        # construct full image path
        link = int(item[0][:item[0].find('-')])
        path = os.getcwd() + '/' + item[5] +'/images/' + item[0] + '/' + item[1] +'.jpg'
        cur_img = cv2.imread(path)
        bbox = [int(i) for i in item[4]]             
        if not type(item[4]) == list:
            if int(item[4]) == 1:
                continue
        c = bbox[0]
        r = bbox[1]
        s = bbox[2]
        nrows = cur_img.shape[0]
        ncols = cur_img.shape[1]
        const = 1.0
        # crop
        r0 = max(int(r - const*s), 0); r1 = min(r + const*s, nrows)
        c0 = max(int(c - const*s), 0); c1 = min(c + const*s, ncols)

        crop_cv = cur_img[int(r0):int(r1), int(c0):int(c1)]
        crop_cv = pad_image_square(crop_cv)
        
        skip = False
        
        if s < 60  or np.mean(crop_cv) < 50 or np.var(crop_cv) < 500 or  np.var(crop_cv)/s < 5 or  np.var(crop_cv)/s > 80:
            skip = True

        if min(crop_cv.shape[0:1]) < 64: skip = True

        if not skip:
            yaw = est.return_yaw(crop_cv)
            pitch = est.return_pitch(crop_cv)

            if abs(yaw[0][0][0]) > 35:
                cnt+=1
                print(cnt, yaw[0][0][0], pitch[0][0][0])
                cv2.imshow('Img', crop_cv/255.0)
                cv2.waitKey(30)
                op_full_list.append([path, r, c, s])


est.load_yaw_variables("./head_pose/yaw/cnn_cccdd_30k")
est.load_pitch_variables("./head_pose/pitch/cnn_cccdd_30k.tf")
fddb()
wider()
pickle.dump(op_full_list, open("../side_faces.p", "wb"))