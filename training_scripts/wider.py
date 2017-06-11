#
#
#

#
import sys
import random
import numpy
from scipy import misc
from PIL import Image
from PIL import ImageOps
import struct
import argparse
import os
import pickle
import psutil
import time
import random
#
plot = 0
debug = False

if plot:
    import matplotlib.pyplot
    import matplotlib.image
    import matplotlib.cm


#

#
def write_rid(im):
    #
    # raw intensity data
    #

    #
    global debug
    if debug:
        chk = Image.fromarray(im)
        chk.show()
        time.sleep(1)
        for proc in psutil.process_iter():
            if proc.name() == "display":
                proc.kill()
    h = im.shape[0]
    w = im.shape[1]

    #
    hw = struct.pack('ii', h, w)

    tmp = [None]*w*h
    for y in range(0, h):
        for x in range(0, w):
            tmp[y*w + x] = im[y, x]

    #
    pixels = struct.pack('%sB' % w*h, *tmp)

    #
    sys.stdout.buffer.write(hw)
    sys.stdout.buffer.write(pixels)

#
def export(im, r, c, s):
    #
    nrows = im.shape[0]
    ncols = im.shape[1]

    # crop
    r0 = max(int(r - 0.75*s), 0); r1 = min(r + 0.75*s, nrows)
    c0 = max(int(c - 0.75*s), 0); c1 = min(c + 0.75*s, ncols)

    im = im[r0:r1, c0:c1]

    nrows = im.shape[0]
    ncols = im.shape[1]

    r = r - r0
    c = c - c0

    # resize, if needed
    maxwsize = 192.0
    wsize = max(nrows, ncols)

    ratio = maxwsize/wsize

    if ratio<1.0:
        im = numpy.asarray( Image.fromarray(im).resize((int(ratio*ncols), int(ratio*nrows))) )

        r = ratio*r
        c = ratio*c
        s = ratio*s

    #
    nrands = 7;

    lst = []

    for i in range(0, nrands):
        #
        stmp = s*random.uniform(0.9, 1.1)

        rtmp = r + s*random.uniform(-0.05, 0.05)
        ctmp = c + s*random.uniform(-0.05, 0.05)

        #
        if plot:
            matplotlib.pyplot.cla()

            matplotlib.pyplot.plot([ctmp-stmp/2, ctmp+stmp/2], [rtmp-stmp/2, rtmp-stmp/2], 'b', linewidth=3)
            matplotlib.pyplot.plot([ctmp+stmp/2, ctmp+stmp/2], [rtmp-stmp/2, rtmp+stmp/2], 'b', linewidth=3)
            matplotlib.pyplot.plot([ctmp+stmp/2, ctmp-stmp/2], [rtmp+stmp/2, rtmp+stmp/2], 'b', linewidth=3)
            matplotlib.pyplot.plot([ctmp-stmp/2, ctmp-stmp/2], [rtmp+stmp/2, rtmp-stmp/2], 'b', linewidth=3)

            matplotlib.pyplot.imshow(im, cmap=matplotlib.cm.Greys_r)

            matplotlib.pyplot.show()

        lst.append( (int(rtmp), int(ctmp), int(stmp)) )

    #
    # check = Image.fromarray(im).show()
    # print(numpy.mean(im), s, numpy.var(im)/s)
    # time.sleep(1)
    # for proc in psutil.process_iter():
    #     if proc.name == "display":
    #         proc.kill()
    write_rid(im)
    sys.stdout.buffer.write( struct.pack('i', nrands) )

    for i in range(0, nrands):
        sys.stdout.buffer.write( struct.pack('iii', lst[i][0], lst[i][1], lst[i][2]) )

def mirror_and_export(im, r, c, s):
    #
    # exploit mirror symmetry of the face
    #

    # flip image
    im = numpy.asarray(ImageOps.mirror(Image.fromarray(im)))

    # flip column coordinate of the object
    c = im.shape[1] - c

    # export
    export(im, r, c, s)

# image list
#
data_arr = None
limit = 10000
with open("save.p", "rb") as f:
    data_arr = pickle.load(f) 
random.shuffle(data_arr)

count = 0
for item in data_arr:
    # construct full image path
    link = int(item[0][:item[0].find('-')])
    path = os.getcwd() + '/' + item[5] +'/images/' + item[0] + '/' + item[1] +'.jpg'
    cur_img = Image.open(path).convert('L')
    bbox = [int(i) for i in item[4]]             
    if not type(item[4]) == list:
        if int(item[4]) == 1:
            continue

    c = bbox[0]
    r = bbox[1]
    s = bbox[2]
   
    im = numpy.asarray(im)
    if s < 60  or np.mean(im) < 50 or np.var(im) < 500 or  np.var(im)/s < 5 or  np.var(im)/s > 80:
        continue

    #
    count+=1
    if count == limit: break
    #
    export(im, r, c, s)

    # faces are symmetric and we exploit this here
    mirror_and_export(im, r, c, s)