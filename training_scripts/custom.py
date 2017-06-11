import sys
import random
import numpy
from scipy import misc
from PIL import Image
from PIL import ImageOps
from PIL import ImageStat
import struct
import argparse
import os
import pickle
import psutil
import time
import random
import cv2
#
plot = 0
debug = False

if plot:
    import matplotlib.pyplot
    import matplotlib.image
    import matplotlib.cm


def write_rid(im):
    #
    # raw intensity data
    #

    #
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

    im = im[int(r0):int(r1), int(c0):int(c1)]

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
with open("side_faces.p", "rb") as f:
    data_arr = pickle.load(f, encoding='latin1')
 
random.shuffle(data_arr)
count = 0
limit = 1000000000
for item in data_arr:
    # construct full image path
    cur_img = None
    try:
        cur_img = Image.open(item[0]).convert('L')
    except:
        #print('Error loading image!')
        continue

    r = int(item[1])
    c = int(item[2])
    s = int(item[3])

    im = numpy.asarray(cur_img)
    if s < 60  or numpy.mean(im) < 50 or numpy.var(im) < 500 or  numpy.var(im)/s < 5 or  numpy.var(im)/s > 80:
        continue


    #
    count+=1
    if count == limit: break
    #
    export(im, r, c, s)

    # faces are symmetric and we exploit this here
    mirror_and_export(im, r, c, s)