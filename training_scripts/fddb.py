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
import time
import xml.etree.ElementTree as ET
import cv2
import psutil
#
plot = 0

if plot:
    import matplotlib.pyplot
    import matplotlib.image
    import matplotlib.cm

#

#
 
def CLAHE(img):
    return Image.fromarray(img).convert('L')
    # lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # l, a, b = cv2.split(lab)
    # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    # cl = clahe.apply(l)
    # limg = cv2.merge((cl,a,b))
    # final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    # final = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
    # return Image.fromarray(final).convert('L')

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
    if min(numpy.shape(im)) <= 0:
        return
    stats = ImageStat.Stat(Image.fromarray(im))
    if stats.mean[0] < 50: return

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
n = 0
MAX_COUNT = 6000

annot_dir =  os.getcwd()+ "/fddb/FDDB-folds/"
img_dir =  os.getcwd() + "/fddb/"
for f_num in range(1,10+1):
      
        index = str(f_num).zfill(2)
        annot_file = annot_dir + "FDDB-fold-" + index + "-ellipseList.txt"
        
        fp = open(annot_file)
        raw_data = fp.readlines()
        cur_img = 0

        stage = 0
        for parsed_data in raw_data:                        
            if stage == 0:
                file_name = parsed_data.rstrip()
                stage = 1
            elif stage == 1:
                num_faces = int(parsed_data)
                file_url = img_dir+file_name +'.jpg'
                cur_img = cv2.imread(file_url)
                im = CLAHE(cur_img)
                im = numpy.asarray(im)
                if min(numpy.shape(im)) == 0: 
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
            
                    # if s < 70  or numpy.mean(im) < 60 or numpy.var(im) < 500 or  numpy.var(im)/s < 5 or  numpy.var(im)/s > 80:
                    #     continue
                    #
                    if s > 30:
                        export(im, r, c, s)
                        # # faces are symmetric and we exploit this here
                        mirror_and_export(im, r, c, s)
                    num_faces -= 1
                    if num_faces == 0:
                        stage = 0

        fp.close()
