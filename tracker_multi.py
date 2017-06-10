from __future__ import division
import sys
import cv2
import numpy as np
import time
import argparse
import cypico
import multiprocessing
import os

face_settings = { 'confidence': 3, 'orientations': [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875], \
                    'scale': 1.2, 'stride':0.2, 'min_size': 20  }
face_suppress_settings = {'round_to_val': 30, 'radii_round': 50, 'stack_length': 3, 'positive_thresh': 4, \
 'remove_thresh': -1, 'step_add': 1, 'step_subtract': -2, 'coarse_scale': 8.0, 'coarse_radii_scale': 3.0}

test_path = os.getcwd() + '/face.hex'
feed_list = [(False, 'bkp', face_settings, face_suppress_settings), (False, 'bkp', face_settings, face_suppress_settings)] 
#Please put fastest cascade first, or it may risk ruining the quality of the updates

class Process(object):
    #Parameters set by config
    settings = {}
    preproc_CLAHE = False
    preproc_BLUR = False
    vid_shape = None
    vid_update_method = None
    vid_op_method = None
    num_cascades = 0

    #Multiprocessing Specific
    m = multiprocessing.Manager()
    results_q = m.Queue()
    pipe_main, pipe_master = multiprocessing.Pipe()
    access_lock = multiprocessing.Lock()
    instances = []

    #Constants
    loop_lim = 10

    
    def __init__(self, feed_list, settings):
        has_manual = False
        self.num_cascades = len(feed_list)
        try:
            self.preproc_CLAHE = settings['CLAHE']
        except KeyError:
            self.preproc_CLAHE = False #Whether to run CLAHE Histogram Equalisation on input image
        try:
            self.preproc_BLUR = settings['Blur']
        except KeyError:
            self.preproc_BLUR = False #Whether to run CLAHE Histogram Equalisation on input image
        try:
            self.vid_shape = settings['Shape']
        except KeyError:
            raise Exception('No shape passed as part of config')
        try:
            self.vid_update_method = settings['Update_Method']
        except KeyError:
            raise Exception('No frame update get method passed as part of config')
        try:
            self.vid_op_method = settings['Output_Method']
        except KeyError:
            raise Exception('No output method passed as part of config')
        for __indx in range(self.num_cascades):
            __inst = feed_list[__indx]
            if __inst[0]:
                if (has_manual):
                    raise Exception('Multiple run-time cascades loads are not allowed, please install all but \
                    one cascade in the appropriate package')
                else:
                    has_manual = True
                    cypico.load_cascade(__inst[1]) #Loads cascade into memory
                    self.settings[__indx] = ('manual', __inst[2], __inst[3])
            else:
                self.settings[__indx] = (__inst[1], __inst[2], __inst[3])

    def run(self):
        c_frm = self.m.Value('Frm', 0) #Stores current frame
        for __indx in range(self.num_cascades):
            p = None
            if __indx == 0:
                p = multiprocessing.Process(target=self.master, args=(self.access_lock, self.settings[0], c_frm, self.pipe_master, self.results_q))
            else:
                p = multiprocessing.Process(target=self.worker, args=(self.access_lock, __indx, c_frm, self.results_q))
            self.instances.append(p)
            p.start()
        pipe_event = False
        while True:
            if self.pipe_main.poll():
                pipe_event = self.pipe_main.recv() #Receives a request from the master
            if type(pipe_event) == int and pipe_event == 1:
                self.access_lock.acquire()
                c_frm.value = self.update_frame()
                self.pipe_main.send(0)
                self.access_lock.release()
            try:
                results = self.results_q.get(False)
                self.vid_op_method(results[0], results[1])
            except Exception as ex:
                pass #Results Q is empty ah well
            pipe_event = None #reset
        [p.join() for p in self.instances]
    
    def master(self, lock, set_tup, shared_frame, master_pipe, q_out):
        cur_time = time.time()
        cur_loop = 0 #FOR FPS CALC
        fpr_buffer = {} #For false positive clean

        while (True):
            master_pipe.send(1)
            frm = None
            event = master_pipe.recv() #Wait to receive a response from Pipe
            if type(event) == int and event == 0:            
                lock.acquire()
                frm = shared_frame.value
                lock.release()
            if type(frm) == int: 
                continue #If its an integer, then it hasn't initialised yet
            #COMPUTE FPS#
            cur_loop+=1
            if cur_loop > self.loop_lim:
                cur_loop = 0
                chk = time.time() - cur_time
                cur_time = time.time()
                print('Master FPS = ' + str(self.loop_lim/chk))
            #COMPUTE FPS END#
            det = cypico.detect(frm, set_tup[1], set_tup[0])
            det = cypico.remove_overlap(det)
            fpr_buffer, det = self.clean_fpr(fpr_buffer, det, set_tup[2])
            q_out.put((det, 0))

    def worker(self, lock, worker_id, shared_frame, q_out):
        cur_time = time.time()
        cur_loop = 0 #FOR FPS CALC
        fpr_buffer = {} #For false positive clean
        set_tup = self.settings[worker_id]
        while(True):
            lock.acquire(timeout = 0.5) #Timeout if not switched in 1/2 second
            frm = shared_frame.value
            lock.release() #Only release if locked
            if type(frm) == int: continue 

            #COMPUTE FPS#
            cur_loop+=1
            if cur_loop > self.loop_lim:
                cur_loop = 0
                chk = time.time() - cur_time
                cur_time = time.time()
                print('Worker ' + str(worker_id) + ' FPS = ' + str(self.loop_lim/chk))
            #COMPUTE FPS END#

            det = cypico.detect(frm, set_tup[1], set_tup[0])
            det = cypico.remove_overlap(det)
            fpr_buffer, det = self.clean_fpr(fpr_buffer, det, set_tup[2])
            q_out.put((det, worker_id))


     
    def update_frame(self):
        frm = self.vid_update_method()
        frm = self.lighting_balance(frm, self.preproc_CLAHE, self.preproc_BLUR)
        frm = np.asarray(frm, dtype='uint8')
        return frm
   
    @staticmethod
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

    @staticmethod
    def __round(input, round_to_val):
        return round(float(input)/round_to_val)*round_to_val

    @staticmethod
    def __add_stack(input, item, max_n):
        if len(input) > max_n:
            del input[0]
            input.append(item)
        else:
            input.append(item)
        return input

    @staticmethod
    def __mean(input):
        return int(float(sum(input))/len(input))

    def clean_fpr(self, fpr_buffer, new_detections, input_settings):
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
            stack_length = 3 #Limit the number of stored data points for comparison
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
            key = (self.__round(detect[1][1], round_to_val), self.__round(detect[1][0], round_to_val),  self.__round(detect[2], radii_round))
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
                fpr_val[1] = self.__add_stack(fpr_val[1], cur[0], stack_length)
                fpr_val[2] = self.__add_stack(fpr_val[2], cur[1], stack_length)
                fpr_val[3] = self.__add_stack(fpr_val[3], cur[2], stack_length)
                fpr_val[4] = (self.__add_stack(fpr_val[4][0], cur[3][0], stack_length), 
                            self.__add_stack(fpr_val[4][1], cur[3][1], stack_length))
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
            fpr_buffer[add_key] = [step_add, self.__add_stack([], cur[0], stack_length), 
                                            self.__add_stack([], cur[1], stack_length),
                                            self.__add_stack([], cur[2], stack_length),
                                            (self.__add_stack([], cur[3][0], stack_length),
                                            self.__add_stack([], cur[3][1], stack_length))]

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
                key = (self.__round(key_orig[1], coarse_round), self.__round(key_orig[0], coarse_round),  self.__round(key_orig[2], coarse_radii_round))
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
                clean_detections.append([self.__mean(cur[3]), (self.__mean(cur[4][1]), self.__mean(cur[4][0])), 
                                        self.__mean(cur[1]), self.__mean(cur[2])])

        # print(fpr_buffer.keys())
        return (fpr_buffer, clean_detections)


vid_in = None
file = 'ISS.mp4'
cv2.setNumThreads(0) #Done for OpenCV
cur_frame = None

def update_frame(width_desired = 640.0):  #Change as required
    global vid_in, cur_frame
    ret, frm = vid_in.read()
    if ret == False: raise Exception('Video update failure')
    r = width_desired / frm.shape[1]
    try: 
        if r < 1:
            dim = (int(width_desired), int(frm.shape[0] * r))
            frm = cv2.resize(frm, dim, interpolation = cv2.INTER_LANCZOS4)
    except Exception as ex:
        print(ex)
    cur_frame = frm
    return frm

def setup_video_input():  #Change as required
    global vid_in, file
    vid_in = cv2.VideoCapture(file)
    vid_in.set(cv2.CAP_PROP_POS_FRAMES,500)

def get_vid_shape(width_desired = 640.0):  #Change as required
    global vid_in, file
    _, frm = vid_in.read() 
    r = width_desired / frm.shape[1]
    try: 
        if r < 1:
            dim = (int(width_desired), int(frm.shape[0] * r))
            frm = cv2.resize(frm, dim, interpolation = cv2.INTER_LANCZOS4)
    except Exception as ex:
        print(ex)
    return np.shape(frm)

def process_detection(detections, id):
    global cur_frame
    if cur_frame is None: return #Hasn't initialised yet, return
    id_color_map = {0:(0,0,255), 1:(0,255,0), 2:(255,0,0)}
    for cur in detections:
        cv2.circle(cur_frame, (int(cur[1][1]), int(cur[1][0])), int(cur[2]/2), id_color_map[id], 3)
    cv2.imshow('Camera', cur_frame)
    cv2.waitKey(1)

def main():
    setup_video_input()
    config = {'CLAHE': True, 'Blur': False, 'Shape': get_vid_shape(), 'Update_Method': update_frame, 'Output_Method': process_detection}
    cur_proc = Process(feed_list = feed_list, settings = config)
    cur_proc.run()

if __name__ == "__main__":
    main()
  