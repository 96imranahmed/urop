import cv2
import sys
import numpy as np
import tensorflow as tf
import time
import random
import math
from skimage.util.shape import view_as_windows
from skimage.transform import pyramid_gaussian
import Image
import ImageDraw
from compiler.ast import flatten
import copy
import etc
if sys.version[0] == 2:
    from future import __division__

#Initialise net
sess = tf.InteractiveSession()

#12-net
#conv layer 1
x_12 = tf.placeholder("float", [None, etc.img_size_12 * etc.img_size_12 * etc.input_channel])
W_conv1_12 = etc.weight_variable([3,3,3,16])
b_conv1_12 = etc.bias_variable([16])
x_12_reshaped = tf.reshape(x_12, [-1, etc.img_size_12, etc.img_size_12, etc.input_channel])
h_conv1_12 = tf.nn.relu(etc.conv2d(x_12_reshaped, W_conv1_12) + b_conv1_12)

#pooling layer 1
h_pool1_12 = etc.max_pool_3x3(h_conv1_12)

#fully layer 1
W_fc1_12 = etc.weight_variable([6 * 6 * 16, 16])
b_fc1_12 = etc.bias_variable([16])
h_pool1_12_reshaped = tf.reshape(h_pool1_12, [-1, 6 * 6 * 16])
h_fc1_12 = tf.nn.relu(tf.matmul(h_pool1_12_reshaped, W_fc1_12) + b_fc1_12)

#fully layer2
W_fc2_12 = etc.weight_variable([16, 1])
b_fc2_12 = etc.bias_variable([1])
h_fc2_12 = tf.nn.sigmoid(tf.matmul(h_fc1_12, W_fc2_12) + b_fc2_12)



#24-net
x_24 = tf.placeholder("float", [None, etc.img_size_24 * etc.img_size_24 * etc.input_channel])
#conv layer 1
W_conv1_24 = etc.weight_variable([5,5,3,64])
b_conv1_24 = etc.bias_variable([64])
x_24_reshaped = tf.reshape(x_24, [-1, etc.img_size_24, etc.img_size_24, etc.input_channel])
h_conv1_24 = tf.nn.relu(etc.conv2d(x_24_reshaped, W_conv1_24) + b_conv1_24)

#pooling layer 1
h_pool1_24 = etc.max_pool_3x3(h_conv1_24)

#fully layer 1
W_fc1_24 = etc.weight_variable([12 * 12 * 64, 128])
b_fc1_24 = etc.bias_variable([128])
h_pool1_24_reshaped = tf.reshape(h_pool1_24, [-1, 12 * 12 * 64])
h_fc1_24 = tf.nn.relu(tf.matmul(h_pool1_24_reshaped, W_fc1_24) + b_fc1_24)

#fully layer 2
W_fc2_24 = etc.weight_variable([128 + 16,1])
b_fc2_24 = etc.bias_variable([1])
#h_fc1_12_sliced = tf.slice(h_fc1_12,[0,0,0,0],[-1,1,1,16])
#h_fc1_24_concat = tf.concat(1, [h_fc1_24, tf.reshape(h_fc1_12_sliced,(-1,16))])
h_fc1_24_concat = tf.concat(1,[h_fc1_24,tf.reshape(h_fc1_12,(-1,16))])
h_fc2_24 = tf.nn.sigmoid(tf.matmul(h_fc1_24_concat, W_fc2_24) + b_fc2_24)



#48-net
x_48 = tf.placeholder("float", [None, etc.img_size_48 * etc.img_size_48 * etc.input_channel])
#conv layer 1
W_conv1_48 = etc.weight_variable([5,5,3,64])
b_conv1_48 = etc.bias_variable([64])
x_48_reshaped = tf.reshape(x_48, [-1, etc.img_size_48, etc.img_size_48, etc.input_channel])
h_conv1_48 = tf.nn.relu(etc.conv2d(x_48_reshaped, W_conv1_48) + b_conv1_48)

#pooling layer 1
h_pool1_48 = etc.max_pool_3x3(h_conv1_48)

#normalization layer 1

#conv layer 2
W_conv2_48 = etc.weight_variable([5,5,64,64])
b_conv2_48 = etc.bias_variable([64])
h_conv2_48 = tf.nn.relu(etc.conv2d(h_pool1_48, W_conv2_48) + b_conv2_48)

#normalization layer 2

#pooling layer 2
h_pool2_48 = etc.max_pool_3x3(h_conv2_48)

#fully layer 1
W_fc1_48 = etc.weight_variable([12 * 12 * 64, 256])
b_fc1_48 = etc.bias_variable([256])
h_pool2_48_reshaped = tf.reshape(h_pool2_48, [-1, 12 * 12 * 64])
h_fc1_48 = tf.nn.relu(tf.matmul(h_pool2_48_reshaped, W_fc1_48) + b_fc1_48)

#fully layer 2
W_fc2_48 = etc.weight_variable([256 + 128 + 16, 1])
b_fc2_48 = etc.bias_variable([1])
h_fc1_48_concat = tf.concat(1, [h_fc1_24_concat, h_fc1_48])
h_fc2_48 = tf.sigmoid(tf.matmul(h_fc1_48_concat, W_fc2_48) + b_fc2_48)




#########calib-net
#12-net
#conv layer 1
x_12_cali = tf.placeholder("float", [None, etc.img_size_12 * etc.img_size_12 * etc.input_channel])
W_conv1_12_cali = etc.weight_variable([3,3,3,16],'calib_wc1_12')
b_conv1_12_cali = etc.bias_variable([16],'calib_bc1_12')
x_12_reshaped_cali = tf.reshape(x_12_cali, [-1, etc.img_size_12, etc.img_size_12, etc.input_channel])
h_conv1_12_cali = tf.nn.relu(etc.conv2d(x_12_reshaped_cali, W_conv1_12_cali) + b_conv1_12_cali)

#pooling layer 1
h_pool1_12_cali = etc.max_pool_3x3(h_conv1_12_cali)

#fully layer 1
W_fc1_12_cali = etc.weight_variable([6 * 6 * 16, 128],'calib_wfc1_12')
b_fc1_12_cali = etc.bias_variable([128],'calib_bfc1_12')
h_pool1_12_reshaped_cali = tf.reshape(h_pool1_12_cali, [-1, 6 * 6 * 16])
h_fc1_12_cali = tf.nn.relu(tf.matmul(h_pool1_12_reshaped_cali, W_fc1_12_cali) + b_fc1_12_cali)

#fully layer2
W_fc2_12_cali = etc.weight_variable([128, etc.cali_patt_num],'calib_wfc2_12')
b_fc2_12_cali = etc.bias_variable([etc.cali_patt_num],'calib_bfc2_12')
h_fc2_12_cali = tf.nn.softmax(tf.matmul(h_fc1_12_cali, W_fc2_12_cali) + b_fc2_12_cali)



#24-net
x_24_cali = tf.placeholder("float", [None, etc.img_size_24 * etc.img_size_24 * etc.input_channel])
#conv layer 1
W_conv1_24_cali = etc.weight_variable([5,5,3,32],'calib_wc1_24')
b_conv1_24_cali = etc.bias_variable([32],'calib_bc1_24')
x_24_reshaped_cali = tf.reshape(x_24_cali, [-1, etc.img_size_24, etc.img_size_24, etc.input_channel])
h_conv1_24_cali = tf.nn.relu(etc.conv2d(x_24_reshaped_cali, W_conv1_24_cali) + b_conv1_24_cali)

#pooling layer 1
h_pool1_24_cali = etc.max_pool_3x3(h_conv1_24_cali)

#fully layer 1
W_fc1_24_cali = etc.weight_variable([12 * 12 * 32, 64],'calib_wfc1_24')
b_fc1_24_cali = etc.bias_variable([64],'calib_bfc1_24')
h_pool1_24_reshaped_cali = tf.reshape(h_pool1_24_cali, [-1, 12 * 12 * 32])
h_fc1_24_cali = tf.nn.relu(tf.matmul(h_pool1_24_reshaped_cali, W_fc1_24_cali) + b_fc1_24_cali)

#fully layer 2
W_fc2_24_cali = etc.weight_variable([64,etc.cali_patt_num],'calib_wfc2_24')
b_fc2_24_cali = etc.bias_variable([etc.cali_patt_num],'calib_bfc2_24')
h_fc2_24_cali = tf.nn.softmax(tf.matmul(h_fc1_24_cali, W_fc2_24_cali) + b_fc2_24_cali)

#48-net
x_48_cali = tf.placeholder("float", [None, etc.img_size_48 * etc.img_size_48 * etc.input_channel])
#conv layer 1
W_conv1_48_cali = etc.weight_variable([5,5,3,64],'calib_wc1_48')
b_conv1_48_cali = etc.bias_variable([64],'calib_bc1_48')
x_48_reshaped_cali = tf.reshape(x_48_cali, [-1, etc.img_size_48, etc.img_size_48, etc.input_channel])
h_conv1_48_cali = tf.nn.relu(etc.conv2d(x_48_reshaped_cali, W_conv1_48_cali) + b_conv1_48_cali)

#pooling layer 1
h_pool1_48_cali = etc.max_pool_3x3(h_conv1_48_cali)

#normalization layer 1

#conv layer 2
W_conv2_48_cali = etc.weight_variable([5,5,64,64],'calib_wc2_48')
b_conv2_48_cali = etc.bias_variable([64],'calib_bc2_48')
h_conv2_48_cali = tf.nn.relu(etc.conv2d(h_pool1_48_cali, W_conv2_48_cali) + b_conv2_48_cali)

#normalization layer 2


#fully layer 1
W_fc1_48_cali = etc.weight_variable([24 * 24 * 64, 256],'calib_wfc1_48')
b_fc1_48_cali = etc.bias_variable([256],'calib_bfc1_48')
h_conv2_48_reshaped_cali = tf.reshape(h_conv2_48_cali, [-1, 24 * 24 * 64])
h_fc1_48_cali = tf.nn.relu(tf.matmul(h_conv2_48_reshaped_cali, W_fc1_48_cali) + b_fc1_48_cali)

#fully layer 2
W_fc2_48_cali = etc.weight_variable([256, etc.cali_patt_num],'calib_wfc2_48')
b_fc2_48_cali = etc.bias_variable([etc.cali_patt_num],'calib_bfc2_48')
h_fc2_48_cali = tf.nn.softmax(tf.matmul(h_fc1_48_cali, W_fc2_48_cali) + b_fc2_48_cali)

sess.run(tf.initialize_all_variables())


saver = tf.train.Saver({"Variable":W_conv1_12, "Variable_1":b_conv1_12, "Variable_2":W_fc1_12, "Variable_3":b_fc1_12, "Variable_4":W_fc2_12, "Variable_5":b_fc2_12, "Variable_6": W_conv1_24, "Variable_7":b_conv1_24, "Variable_8":W_fc1_24, "Variable_9":b_fc1_24, "Variable_10":W_fc2_24, "Variable_11":b_fc2_24})
saver.restore(sess,etc.save_dir+"24-net.ckpt")

saver = tf.train.Saver({"Variable_12":W_conv1_48, "Variable_13":b_conv1_48, "Variable_14":W_conv2_48, "Variable_15":b_conv2_48, "Variable_16":W_fc1_48, "Variable_17":b_fc1_48, "Variable_18":W_fc2_48,"Variable_19":b_fc2_48})
saver.restore(sess,etc.save_dir+"48-net.ckpt")


saver = tf.train.Saver({"calib_wc1_12":W_conv1_12_cali, "calib_bc1_12":b_conv1_12_cali, "calib_wfc1_12":W_fc1_12_cali, "calib_bfc1_12":b_fc1_12_cali, "calib_wfc2_12":W_fc2_12_cali, "calib_bfc2_12":b_fc2_12_cali})
saver.restore(sess, etc.save_dir + "12-net_calib.ckpt")


saver = tf.train.Saver({"calib_wc1_24": W_conv1_24_cali, "calib_bc1_24":b_conv1_24_cali, "calib_wfc1_24":W_fc1_24_cali, "calib_bfc1_24":b_fc1_24_cali, "calib_wfc2_24":W_fc2_24_cali, "calib_bfc2_24":b_fc2_24_cali})
saver.restore(sess, etc.save_dir + "24-net_calib.ckpt")


saver = tf.train.Saver({"calib_wc1_48":W_conv1_48_cali, "calib_bc1_48":b_conv1_48_cali, "calib_wc2_48":W_conv2_48_cali, "calib_bc2_48":b_conv2_48_cali, "calib_wfc1_48":W_fc1_48_cali, "calib_bfc1_48":b_fc1_48_cali, "calib_wfc2_48":W_fc2_48_cali,"calib_bfc2_48":b_fc2_48_cali})
saver.restore(sess,etc.save_dir+"48-net_calib.ckpt")

thr_12 = 0.99       
thr_24 = 0.99

def procimg(img): 
    if len(np.shape(np.asarray(img))) < 3:
        arr = np.asarray(img)
        img = np.zeros((img.size[1],img.size[0],etc.input_channel),np.uint8)
        img[:,:,0] = arr
        img[:,:,1] = arr
        img[:,:,2] = arr
        img = Image.fromarray(img)
    
    #12-net
    result_box = etc.slidingW_Test(img,thr_12,x_12,h_fc2_12) 

    if len(result_box) > 0:
        
        #12-calib net
        test_db_12 = np.zeros((len(result_box),etc.dim_12),np.float32)
        for id_,box in enumerate(result_box):
            resized_img_12 = etc.img2array(box[5],etc.img_size_12) 
            test_db_12[id_,:] = resized_img_12

        result = h_fc2_12_cali.eval(feed_dict={x_12_cali: test_db_12})
        result_box = etc.calib_run(result_box,result,img)

        #NMS_fast for each scale
        scale_cur = result_box[0][6]
        scale_box = []
        final_box = []
        for id_,box in enumerate(result_box):
            if box[6] == scale_cur:
                scale_box.append(box)
            if box[6] != scale_cur or id_ == len(result_box)-1:
                scale_box.sort(key=lambda x :x[4])
                scale_box.reverse()
                supp_box_id = etc.NMS_fast(scale_box)
                final_box += [scale_box[i] for i in supp_box_id]
                scale_cur += 1
                scale_box = [box]
        result_box = final_box
        final_box = []                     
        

    if len(result_box) > 0:
        #24-net
        test_db_12 = np.zeros((len(result_box),etc.dim_12),np.float32)
        test_db_24 = np.zeros((len(result_box),etc.dim_24),np.float32)
        for id_,box in enumerate(result_box):
            resized_img_12 = etc.img2array(box[5],etc.img_size_12)
            resized_img_24 = etc.img2array(box[5],etc.img_size_24)
           
            test_db_12[id_,:] = resized_img_12
            test_db_24[id_,:] = resized_img_24
        

        final_box = []
        test_db = np.empty((0,etc.dim_24),np.float32)
        for tit in xrange(len(test_db_12)/etc.test_bench_num+1):
            if tit == len(test_db_12)/etc.test_bench_num:
                test_12 = test_db_12[tit*etc.test_bench_num:,:]
                test_24 = test_db_24[tit*etc.test_bench_num:,:]
                box_tmp = result_box[tit*etc.test_bench_num:]
            else:
                test_12 = test_db_12[tit*etc.test_bench_num:(tit+1)*etc.test_bench_num,:]
                test_24 = test_db_24[tit*etc.test_bench_num:(tit+1)*etc.test_bench_num,:]
                box_tmp = result_box[tit*etc.test_bench_num:(tit+1)*etc.test_bench_num]

            result = h_fc2_24.eval(feed_dict={x_12: test_12, x_24: test_24})
            result_id = np.where(result > thr_24)[0]
            test_db = np.append(test_db,test_24[result_id,:],axis=0)
            final_box += [box_tmp[i] for i in result_id]
        result_box = final_box
        final_box = [] 
        
     
    if len(result_box) > 0:
        #24-net_calib
        test_db_24 = test_db
        result = h_fc2_24_cali.eval(feed_dict={x_24_cali: test_db_24})
        result_box = etc.calib_run(result_box,result,img)
        

        #NMS_fast for each scale
        scale_cur = result_box[0][6]
        scale_box = []
        final_box = []
        for id_,box in enumerate(result_box):
            if box[6] == scale_cur:
                scale_box.append(box)
            if box[6] != scale_cur or id_ == len(result_box)-1:
                scale_box.sort(key=lambda x :x[4])
                scale_box.reverse()
                supp_box_id = etc.NMS_fast(scale_box)
                final_box += [scale_box[i] for i in supp_box_id]
                scale_cur += 1
                scale_box = [box]
        result_box = final_box
        final_box = []                     
        
    test_db_12 = np.zeros((len(result_box),etc.dim_12),np.float32)
    test_db_24 = np.zeros((len(result_box),etc.dim_24),np.float32)
    test_db_48 = np.zeros((len(result_box),etc.dim_48),np.float32)
    for id_,box in enumerate(result_box):
        resized_img_12 = etc.img2array(box[5],etc.img_size_12)
        resized_img_24 = etc.img2array(box[5],etc.img_size_24)
        resized_img_48 = etc.img2array(box[5],etc.img_size_48)

        test_db_12[id_,:] = resized_img_12
        test_db_24[id_,:] = resized_img_24
        test_db_48[id_,:] = resized_img_48
    
    for thr_idx in xrange(etc.thr_num): 
        
        #thr_ = 1e-3 + thr_idx * 1e-3  
        thr_ = thr_idx * 1e-2
        final_box = []
        if len(result_box) > 0: 
            #48-net
            result_box_copy = [] 
            result_box_copy += [copy.deepcopy([box[0], box[1], box[2], box[3], box[4]]) + [box[5].copy()] + copy.deepcopy([box[6], box[7]]) for box in result_box]
            
            final_box = []
            for tit in xrange(len(test_db_12)/etc.test_bench_num+1):
                if tit == len(test_db_12)/etc.test_bench_num:
                    test_12 = test_db_12[tit*etc.test_bench_num:,:]
                    test_24 = test_db_24[tit*etc.test_bench_num:,:]
                    test_48 = test_db_48[tit*etc.test_bench_num:,:]
                    box_tmp = result_box_copy[tit*etc.test_bench_num:]
                else:
                    test_12 = test_db_12[tit*etc.test_bench_num:(tit+1)*etc.test_bench_num,:]
                    test_24 = test_db_24[tit*etc.test_bench_num:(tit+1)*etc.test_bench_num,:]
                    test_48 = test_db_48[tit*etc.test_bench_num:(tit+1)*etc.test_bench_num,:]
                    box_tmp = result_box_copy[tit*etc.test_bench_num:(tit+1)*etc.test_bench_num]

                result = h_fc2_48.eval(feed_dict={x_12: test_12, x_24: test_24, x_48: test_48})
                result_id = np.where(result > thr_)[0] 
                final_box += [box_tmp[i] for i in result_id]
        
        #global_NMS_fast
        final_box.sort(key=lambda x :x[4])
        final_box.reverse()
        supp_box_id = etc.NMS_fast(final_box)
        final_box = [final_box[i] for i in supp_box_id]
        
        if len(final_box) > 0:
            
            #48-net_calib
            test_db_48_calib = np.zeros((len(final_box),etc.dim_48),np.float32)
            for id_,box in enumerate(final_box):
                resized_img_48 = etc.img2array(box[5],etc.img_size_48)
                test_db_48_calib[id_,:] = resized_img_48            
            result = h_fc2_48_cali.eval(feed_dict={x_48_cali: test_db_48_calib})
            final_box = etc.calib_run(final_box,result,img)
            
            for box in final_box:
                h = box[3]-box[1]
                box[1] -= 0.2*h
                box[1] = max(0,box[1])
                ImageDraw.Draw(img).rectangle((box[0], box[1], box[2], box[3]), outline = (0,0,255))
            final_box = [] 
    return img


def main(argv):
    import time
    vid_in = cv2.VideoCapture(0)
    ret = True
    cnt = 0 
    cnt_max = 20
    cur_t = time.time()
    while (ret == True):
        cnt+=1
        if cnt > cnt_max:
            cnt = 0
            diff = time.time() - cur_t
            print('FPS ', cnt_max/diff)
            cur_t = time.time()
        ret, frame = vid_in.read()
        pil_im = procimg(Image.fromarray(frame).convert('RGB'))
        cv_im = np.array(pil_im)
        #cv_im = cv2.cvtColor(cv_im, cv2.cv.CV_BGR2RGB)
        cv2.imshow('Webcam', cv_im)
        cv2.waitKey(10)
    vid_in.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(sys.argv)