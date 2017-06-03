from __future__ import division
import sys
import cv2
import numpy as np
import time
import argparse
import cypico
import multiprocessing
import os

face_settings = { 'confidence': 1, 'orientations': [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875], \
                    'scale': 1.1, 'stride':0.2, 'min_size': 30  }

face_suppress_settings = {'round_to_val': 30, 'radii_round': 50, 'stack_length': 3, 'positive_thresh': 4, \
 'remove_thresh': -1, 'step_add': 1, 'step_subtract': -2, 'coarse_scale': 8.0, 'coarse_radii_scale': 3.0}

test_path = os.getcwd() + '/face.hex'
feed_list = [(True ,test_path, face_settings, face_suppress_settings), (False, 'faces', face_settings, face_suppress_settings)] 
#Please put fastest cascade fist, or it may risk mucking up the quality of the updates

class Process(object):
    vid_in = None
    cur_frame = None
    setings = {}
    access_lock = multiprocessing.Lock()
    instances = []

    def setup_input(self):
        self.vid_in = cv2.VideoCapture('ISS.mp4')
        self.vid_in.set(cv2.CAP_PROP_POS_FRAMES,400)
    
    def update_frame(self):
        self.access_lock.acquire() #Locks to prevent simultaneous R/W
        _, frm = self.vid_in.read() #Need to change as required
        self.cur_frame = frm
        self.access_lock.release()
        return frm

    def read_frame(self)
        self.access_lock.acquire() #Locks to prevent simultaneous R/W§
        ret_val = self.cur_frame
        self.access_lock.release()
        return ret_val

    def master(self):
        self.setup_capture() #Required if using OpenCV (sets video read parameters)
        while (True):
            frm = self.update_frame()

    def worker(self, id):
        while (self.cur_frame == None ):
            pass #Wait until a valid frame can be read ('burn-in')
        while(True):
            frm = self.read_frame()


    
    def __init__(self, feed_list):
        has_manual = False
        num_cascades = len(feed_list)
        for __indx in range(num_cascades):
            __inst = feed_list[__indx]
            if __inst[0]:
                if (has_manual):
                    raise Exception('Multiple run-time cascades loads are not allowed, please install all but \
                    one cascade in the appropriate package')
                else:
                    has_manual = True
                    cypico.load_cascade(_inst[1]) #Loads cascade into memory
                    self.settings[__indx] = ('manual', __inst[2], __inst[3])
            else:
                self.settings[__indx] = (__inst[1], __inst[2], __inst[3])
        for __indx in range(num_cascades):
            p = None
            if __indx == 0:
                p = multiprocessing.Process(target=self.master, args=(self))
            else:
                p = multiprocessing.Process(target=self.worker, args=(self, __indx))
            self.instances.append(p)
            p.start()
        


            



def main():
    vid_in = None
    #OpenCV inits
    color_stream = None
    cypico.load_cascade(test_path)
    vid_in = cv2.VideoCapture('ISS.mp4')
    vid_in.set(cv2.CAP_PROP_POS_FRAMES,400)
    ret = True
    cur_time = time.time()
    loop_lim = 10
    cur_loop = 0
    fpr_buffer = {}
    while (ret == True):
        cur_loop+=1
        if cur_loop > loop_lim:
            cur_loop = 0
            chk = time.time() - cur_time
            cur_time = time.time()
            print(loop_lim/chk)
        _, frame = vid_in.read()
        if frame is None: 
            print('Error reading frame') 
            return
        width_des = 640.0
        r = width_des / frame.shape[1]
        try: 
            if r < 1:
                dim = (int(width_des), int(frame.shape[0] * r))
                frame = cv2.resize(frame, dim, interpolation = cv2.INTER_LANCZOS4)
        except:
            pass
        data = lighting_balance(frame, True, False)
        data = np.asarray(data, dtype='uint8')
        det = None
        if open_ni:
            det = cypico.detect(data, face_settings, 'manual')
        else:
            det = cypico.detect(data, face_settings,  'manual')
        chk = cypico.remove_overlap(det)
        fpr_buffer, chk = clean_fpr(fpr_buffer, chk, face_suppress_settings)
        for cur in chk:
            cv2.circle(frame, (int(cur[1][1]), int(cur[1][0])), int(cur[2]/2), (0,0,255), 3)
        cv2.imshow('Camera', frame)
        cv2.waitKey(1)
    vid_in.release()
    cv2.destroyAllWindows()

def lighting_balance(img, should_CLAHE, should_blur):
    if should_CLAHE:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        final = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
    else:
        final = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if should_blur:
         final = cv2.GaussianBlur(final,(5,5),0.5)
    return final

def clean_fpr(fpr_buffer, new_detections, input_settings):
    #Constants to set as required
    round_to_val = radii_round = stack_length = positive_thresh = remove_thresh = None
    step_add = step_subtract = coarse_round = coarse_radii_round = None
    try:
        round_to_val = input_settings['round_to_val']
    except KeyError:
        round_to_val = 30 #Degree to which a value (x, y) coordinate is rounded to
    try:
        radii_round = input_settings['radii_round']
    except KeyError:
        radii_round = 50  #Degree to which the radius r is rounded to
    try:
        stack_length = input_settings['stack_length']
    except KeyError:
        stack_length = 5 #Limit the number of stored data points for comparison
    try:
        positive_thresh = input_settings['positive_thresh']
    except KeyError:
        positive_thresh = 4 #Threshold score at which a detection is marked as true
    try:
        remove_thresh = input_settings['remove_thresh']
    except KeyError:
        remove_thresh = -1 #Threshold score at which a detection is marked as a false positive
    try:
        step_add = input_settings['step_add']
    except KeyError:
        step_add = 1 #If there is a match, the degree to whcih the score is incremented
    try:
        step_subtract = input_settings['step_subtract']
    except KeyError:
        step_subtract = -2 #If there isn't a match, the degree to which the score is decreased
    try:
        coarse_round = round_to_val*input_settings['coarse_scale']
    except KeyError:
        coarse_round = round_to_val*8.0 #Coarse degree to which a value (x, y) is rounded to
    try:
        coarse_radii_round = radii_round*input_settings['coarse_radii_scale']
    except KeyError:
        coarse_radii_round = radii_round*3.0 #Coarse degree to which the radius r is rounded to
 
    check_change = {}
    check_add = {}
    to_remove = []

    #Check new detections
    for detect in new_detections:
        key = (__round(detect[1][1], round_to_val), __round(detect[1][0], round_to_val),  __round(detect[2], radii_round))
        if key in fpr_buffer:
            check_change[key] = (detect[2], detect[3], detect[0], (detect[1][1], detect[1][0]))
        else:
            check_add[key] = (detect[2], detect[3], detect[0], (detect[1][1], detect[1][0])) #Radius, Orientations, Confidences, X,Y

    #Compare with existing buffer and update existing keys
    for buffer_key in fpr_buffer:
        if buffer_key in check_change:
            cur = check_change[buffer_key]
            fpr_val = fpr_buffer[buffer_key]
            fpr_val[0] += step_add
            fpr_val[1] = __add_stack(fpr_val[1], cur[0], stack_length)
            fpr_val[2] = __add_stack(fpr_val[2], cur[1], stack_length)
            fpr_val[3] = __add_stack(fpr_val[3], cur[2], stack_length)
            fpr_val[4] = (__add_stack(fpr_val[4][0], cur[3][0], stack_length), 
                          __add_stack(fpr_val[4][1], cur[3][1], stack_length))
            fpr_buffer[buffer_key] = fpr_val
        else:
            #Stores most recent orientations, radii - no need to delete
            fpr_buffer[buffer_key][0] += step_subtract
            
            #Delete from fpr_buffer if too low
            if fpr_buffer[buffer_key][0] < remove_thresh:
                to_remove.append(buffer_key)

    #Delete any "dead" keys from fpr buffer
    for del_key in to_remove:
        del fpr_buffer[del_key]        
    
    #Add new keys as required 
    for add_key in check_add:
        cur = check_add[add_key]
        fpr_buffer[add_key] = [step_add, __add_stack([], cur[0], stack_length), 
                                         __add_stack([], cur[1], stack_length),
                                         __add_stack([], cur[2], stack_length),
                                         (__add_stack([], cur[3][0], stack_length),
                                         __add_stack([], cur[3][1], stack_length))]

    #Remove mergers and acquisitions
    cur_activations = []
    for chk_out in fpr_buffer:
        cur = fpr_buffer[chk_out]
        if cur[0] > positive_thresh:
            recent = False
            if chk_out in check_change:
                recent = True
            cur_activations.append((chk_out, recent)) 
    
    active_dict = {}
    active_to_remove = []
    if len(cur_activations) > 1:
        # print(cur_activations)
        for active in cur_activations:
            key_orig = active[0]
            is_recent = active[1]
            key = (__round(key_orig[1], coarse_round), __round(key_orig[0], coarse_round),  __round(key_orig[2], coarse_radii_round))
            if key in active_dict:
                if active_dict[key][1] == False and is_recent:
                    #Delete key from fpr buffer
                    active_to_remove.append(active_dict[key][0])
                    #TODO: Merge data from that key into this one
                if active_dict[key][1] == True and not is_recent:
                    #Merge current key in active_dict list
                    active_to_remove.append(key_orig)
            else:
                active_dict[key] = (key_orig, is_recent)
    
    for key in active_to_remove:
        del fpr_buffer[key]
    
    clean_detections = []
    #Loop through to find valid items
    for chk_out in fpr_buffer:
        cur = fpr_buffer[chk_out]
        if cur[0] > positive_thresh:
            clean_detections.append([__mean(cur[3]), (__mean(cur[4][1]), __mean(cur[4][0])), 
                                     __mean(cur[1]), __mean(cur[2])])

    # print(fpr_buffer.keys())
    return (fpr_buffer, clean_detections)

def __round(input, round_to_val):
    return round(float(input)/round_to_val)*round_to_val

def __add_stack(input, item, max_n):
    if len(input) > max_n:
        del input[0]
        input.append(item)
    else:
        input.append(item)
    return input

def __mean(input):
    return int(float(sum(input))/len(input))

if __name__ == "__main__":
    main()
  