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
        im = CLAHE(img_cv)
        for obj in cur_objects:
            if obj['hard'] == 1: continue
            left = int(float(obj['xmin']))
            right = int(float(obj['xmax']))
            upper = int(float(obj['ymin']))
            lower = int(float(obj['ymax']))
            c = int((left + right)/2)
            r = int((upper + lower)/2)
            s = int(max((right-left), (lower-upper))/2)
            if count > MAX_COUNT:
                break
            #
            im = numpy.asarray(im)
            if s < 70  or numpy.mean(im) < 60 or numpy.var(im) < 500 or  numpy.var(im)/s < 5 or  numpy.var(im)/s > 80:
                continue
            #
            export(im, r, c, s)

            # # faces are symmetric and we exploit this here
            mirror_and_export(im, r, c, s)
            count+=1
        if count > MAX_COUNT:
            break
