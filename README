# UROP
**The /training folder contains the dataset scripts to train pico**
**This repository contains the detection code using various methods**

The methods used are as follows:
1. tracker_haar - a very simple haar implementation. Very slow, not in-plane rotation invariant
2. tracker_slow - a CNN cascade implementation - again, very slow (although accurate). Not in-plane rotation invariant and models needs retraining (refer to paper)
3. tracker_pico - a single instance of the pico-based detector (only works for one model at any given time)
4. **tracker_multi** - a multi-instance pico-based detector (works for multiple models concurrently - note: uses multiple cores!)

To work with the multi-object tracker:
    1. Need to define functions to get a new frame upon update, and to handle results
        (I have defined a couple of functions myself here as an example, using a video and OpenCV)
    2. Need to define settings for different models and collect them in a dictionary
        Refer to your cypico installation to see how the name corresponds to the model
    3. All the multiproceessing code is specified within the internals of the Process() class and do not need to be touched.
        Instead, refer to remaining code to get a better idea of how to modify it to work with your requirements

Sample usage (see *tracker_multi.py* for a full example):     

```python
face_settings = { 'confidence': 3, 'orientations': [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875], \
                    'scale': 1.2, 'stride':0.2, 'min_size': 20  }
face_suppress_settings = {'round_to_val': 30, 'radii_round': 50, 'stack_length': 5, 'positive_thresh': 4, \
 'remove_thresh': -1, 'step_add': 1, 'step_subtract': -2, 'coarse_scale': 8.0, 'coarse_radii_scale': 3.0}
test_path = os.getcwd() + '/face.hex'
feed_list = [(False, 'bkp', face_settings, face_suppress_settings), (True, test_path, face_settings, face_suppress_settings)] 
config = {'CLAHE': True, 'Blur': False, 'Update_Method': update_frame, 'Output_Method': process_detection}

cur_proc = Process(feed_list = feed_list, settings = config)
```
