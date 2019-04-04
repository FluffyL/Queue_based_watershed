from __future__ import print_function
from functools import reduce

import sys
import os
import numpy as np
import cv2
import math
import queue
import scipy.signal as signal
import SimpleITK as sitk
import skimage.io as io
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

sys.setrecursionlimit(100000)

def read_img(path):
 data = sitk.ReadImage(path,sitk.sitkFloat32)
 data = sitk.GetArrayFromImage(data)
 return data

def img_show(img):
 for i in range(img.shape[0]):
  io.imshow(img[i,:,:],cmap='gray')
  print(i)
  io.show()

def label_show(img):
 for i in range(img.shape[0]):
  io.imshow(img[i,:,:])
  print(i)
  io.show()

def sort_label(label,position):
#sort d_label and d_position refering to ascended d_label
 l = len(label)
 temp = 0
 for j in range(0,l-1):
  count = j
  for i in range(j,l-1):
   if label[count] > label[i+1]:
    count = i + 1
  label[j],label[count] = label[count],label[j]
  position[j],position[count] = position[count],position[j]

 return label,position

def recursive_extent(i,j,k):
#recursive procedure to extent the starting regions in the boring step

 global img
 global img_init
 global simg
 global label
 global init
 global queue_num

 queue_name = globals()
 center_grey = img[i][j][k]
 same_num = 0
 d_label = []
 d_position = []
 if i-1 >= 0: 
  if img[i-1][j][k] == center_grey:
   same_num = same_num + 1
   d_label.append(label[i-1][j][k])
   d_position.append('a')
  else:
   label[i-1][j][k] = (-1)*abs(img_init[i-1][j][k]-img_init[i][j][k])
   if -int(label[i-1][j][k]) in queue_num:
    #global queue_name['q%s'%(label[i-1][j][k])]
    queue_name['q%s'%(-int(label[i-1][j][k]))].put([i-1,j,k])
   else:
    queue_num.append(-int(label[i-1][j][k]))
    queue_name['q%s'%(-int(label[i-1][j][k]))] = queue.Queue()
    queue_name['q%s'%(-int(label[i-1][j][k]))].put([i-1,j,k])
 
 if i+1 < simg[0]:
  if img[i+1][j][k] == center_grey:
   same_num = same_num + 1
   d_label.append(label[i+1][j][k])
   d_position.append('b')
  else:
   label[i+1][j][k] = (-1)*abs(img_init[i+1][j][k]-img_init[i][j][k])
   if -int(label[i+1][j][k]) in queue_num:
    #global queue_name['q%s'%(label[i+1][j][k])]
    queue_name['q%s'%(-int(label[i+1][j][k]))].put([i+1,j,k])
   else:
    queue_num.append(-int(label[i+1][j][k]))
    queue_name['q%s'%(-int(label[i+1][j][k]))] = queue.Queue()
    queue_name['q%s'%(-int(label[i+1][j][k]))].put([i+1,j,k])
 
 if j-1 >= 0: 
  if img[i][j-1][k] == center_grey:
   same_num = same_num + 1
   d_label.append(label[i][j-1][k])
   d_position.append('c')
  else:
   label[i][j-1][k] = (-1)*abs(img_init[i][j-1][k]-img_init[i][j][k])
   if -int(label[i][j-1][k]) in queue_num:
    #global queue_name['q%s'%(label[i][j-1][k])]
    queue_name['q%s'%(-int(label[i][j-1][k]))].put([i,j-1,k])
   else:
    queue_num.append(-int(label[i][j-1][k]))
    queue_name['q%s'%(-int(label[i][j-1][k]))] = queue.Queue()
    queue_name['q%s'%(-int(label[i][j-1][k]))].put([i,j-1,k])

 if j+1 < simg[1]:
  if img[i][j+1][k] == center_grey:
   same_num = same_num + 1
   d_label.append(label[i][j+1][k])
   d_position.append('d')
  else:
   label[i][j+1][k] = (-1)*abs(img_init[i][j+1][k]-img_init[i][j][k])
   if -int(label[i][j+1][k]) in queue_num:
    #global queue_name['q%s'%(label[i][j+1][k])]
    queue_name['q%s'%(-int(label[i][j+1][k]))].put([i,j+1,k])
   else:
    queue_num.append(-int(label[i][j+1][k]))
    queue_name['q%s'%(-int(label[i][j+1][k]))] = queue.Queue()
    queue_name['q%s'%(-int(label[i][j+1][k]))].put([i,j+1,k])

 if k-1 >= 0:
  if img[i][j][k-1] == center_grey:
   same_num = same_num + 1
   d_label.append(label[i][j][k-1])
   d_position.append('e')
  else:
   label[i][j][k-1] = (-1)*abs(img_init[i][j][k-1]-img_init[i][j][k])
   if -int(label[i][j][k-1]) in queue_num:
    #global queue_name['q%s'%(label[i-1][j][k+1])]
    queue_name['q%s'%(-int(label[i][j][k-1]))].put([i,j,k-1])
   else:
    queue_num.append(-int(label[i][j][k-1]))
    queue_name['q%s'%(-int(label[i][j][k-1]))] = queue.Queue()
    queue_name['q%s'%(-int(label[i][j][k-1]))].put([i,j,k-1])

 if k+1 < simg[2]:
  if img[i][j][k+1] == center_grey:
   same_num = same_num + 1
   d_label.append(label[i][j][k+1])
   d_position.append('f')
  else:
   label[i][j][k+1] = (-1)*abs(img_init[i][j][k+1]-img_init[i][j][k])
   if -int(label[i][j][k+1]) in queue_num:
    #global queue_name['q%s'%(label[i-1][j][k+1])]
    queue_name['q%s'%(-int(label[i][j][k+1]))].put([i,j,k+1])
   else:
    queue_num.append(-int(label[i][j][k+1]))
    queue_name['q%s'%(-int(label[i][j][k+1]))] = queue.Queue()
    queue_name['q%s'%(-int(label[i][j][k+1]))].put([i,j,k+1])

 #d_label = np.array(d_label)
 if same_num == 1:

  if d_position[0] == 'a':
   if label[i-1][j][k] == 0:
    init = init + 1
    label[i-1][j][k] = init
    label[i][j][k] = init
   else:
    label[i][j][k] = label[i-1][j][k]

  if d_position[0] == 'b':
   if label[i+1][j][k] == 0:
    init = init + 1
    label[i+1][j][k] = init
    label[i][j][k] = init
   else:
    label[i][j][k] = label[i+1][j][k]

  if d_position[0] == 'c':
   if label[i][j-1][k] == 0:
    init = init + 1
    label[i][j-1][k] = init
    label[i][j][k] = init
   else:
    label[i][j][k] = label[i][j-1][k]

  if d_position[0] == 'd':
   if label[i][j+1][k] == 0:
    init = init + 1
    label[i][j+1][k] = init
    label[i][j][k] = init
   else:
    label[i][j][k] = label[i][j+1][k]

  if d_position[0] == 'e':
   if label[i][j][k-1] == 0:
    init = init + 1
    label[i][j][k-1] = init
    label[i][j][k] = init
   else:
    label[i][j][k] = label[i][j][k-1]

  if d_position[0] == 'f':
   if label[i][j][k+1] == 0:
    init = init + 1
    label[i][j][k+1] = init
    label[i][j][k] = init
   else:
    label[i][j][k] = label[i][j][k+1]
    
 elif same_num > 1:

  d_label,d_position = sort_label(d_label,d_position)
  flag = -1

  for num in range(0,len(d_label)):
   if d_label[num] > 0 and flag == -1:
    flag = num

  if flag == -1:
   init = init + 1
   label[i][j][k] = init
   for num in range(0,len(d_label)):
    if d_position[num] == 'a':
     label[i-1][j][k] = init
    if d_position[num] == 'b':
     label[i+1][j][k] = init
    if d_position[num] == 'c':
     label[i][j-1][k] = init
    if d_position[num] == 'd':
     label[i][j+1][k] = init
    if d_position[num] == 'e':
     label[i][j][k-1] = init
    if d_position[num] == 'f':
     label[i][j][k+1] = init

  else:
   label[i][j][k] = flag
   for num in range(0,len(d_label)):
    if d_position[num] == 'a':
     label[i-1][j][k] = flag
    if d_position[num] == 'b':
     label[i+1][j][k] = flag
    if d_position[num] == 'c':
     label[i][j-1][k] = flag
    if d_position[num] == 'd':
     label[i][j+1][k] = flag
    if d_position[num] == 'e':
     label[i][j][k-1] = flag
    if d_position[num] == 'f':
     label[i][j][k+1] = flag

def recursive_empty(i,j,k,qn):
 #recursivily empty those queues
 global label
 global img
 global simg
 
 if label[i][j][k] < 0:

  if i-1 >= 0:
   if qn == int(abs(img_init[i-1][j][k]-img_init[i][j][k])):
    new_label = label[i-1][j][k]

  if i+1 <= simg[0]-1:
   if qn == int(abs(img_init[i+1][j][k]-img_init[i][j][k])):
    new_label = label[i+1][j][k]

  if j-1 >= 0:
   if qn == int(abs(img_init[i][j-1][k]-img_init[i][j][k])):
    new_label = label[i][j-1][k]

  if j+1 <= simg[1]-1:
   if qn == int(abs(img_init[i][j+1][k]-img_init[i][j][k])):
    new_label = label[i][j+1][k]
  if k-1 >= 0:

   if qn == int(abs(img_init[i][j][k-1]-img_init[i][j][k])):
    new_label = label[i][j][k-1]

  if k+1 <= simg[2]-1:
   if qn == int(abs(img_init[i][j][k+1]-img_init[i][j][k])):
    new_label = label[i][j][k+1]

 else:
  new_label = label[i][j][k]

 return new_label
 
  


def watershed(denoise_value,flood_time,threshold):
 '''
 A queue-based region growing algorithm for accurate segmentation of multi-dimensional digital images

 denoise_value:delete voxel whose luminance lower than denoise_value*max_luminance
 flood_time:empty queues whose number is less than flood_time
 threshold:a starting zone is made of points whose grey level difference is lower than a given threshold to reduce over estimation(equivalent to a thresholding of the gradient image.)
 '''
 global img
 global img_init
 global simg
 global queue_num
 #get rid of noise affects the background
 smax = np.max(img)
 img = np.maximum(img,smax*denoise_value)

 #median filter 3x3x3
 img = signal.medfilt(img,[3,3,3])
 img_init = img

 #boring step
 #choose the start points
 
 img = np.floor(img/threshold)*threshold
 img_init = np.floor(img_init)
 label_num = 0
 for i in range(0,simg[0]):
  for j in range(0,simg[1]):
   for k in range(0,simg[2]):
    if label[i][j][k] == 0:
     recursive_extent(i,j,k)

 #flooding step
 #empty queues in order

 queue_name = globals()
 queue_num = sorted(queue_num)
 flood_num = 0
 for qn in queue_num:
  flood_num = flood_num + 1
  if flood_num <= flood_time:
   while not queue_name['q%s'%(qn)].empty():
    position = queue_name['q%s'%(qn)].get()
    i = position[0]
    j = position[1]
    k = position[2]
    label[i][j][k] = recursive_empty(i,j,k,qn)
   


if __name__ == '__main__':

 filename = ""#your file path(nii.gz)
 img = read_img(filename)
 img_init = img
 init = 0
 queue_num = []
 simg = np.shape(img)
 label = np.zeros(simg)
 
 seg_img=watershed(denoise_value=0.08,flood_time=300,threshold=200)
 img_show(label)

 '''
 simg = np.shape(img)
 smax = np.max(img)
 smin = np.min(img)
 for j in range(0,simg[0]):
  for k in range(0,simg[1]):
   for l in range(0,simg[2]): 
    img[j][k][l] = round(img[j][k][l]*255/smax)
 '''
