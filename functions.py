#!/usr/bin/env python
# coding: utf-8

# # Functions
# 
# ### All functions and utilities are reserved in this file

# In[2]:


import glob
import os
import re #regex library
from platform import python_version

import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from PIL import Image
import cv2

import shutil
from shutil import copyfile


# In[3]:


all_path = []
for path in glob.glob("../data_all/mouse_pos/*.jpg"):
    all_path.append(path)


# In[4]:


data = pd.read_csv("../data_all/mouse_body_pos.csv")


# In[5]:


c1_path = "../data_all/initial_class/c1/"
c2_path = "../data_all/initial_class/c2/"
c3_path = "../data_all/initial_class/c3/"
c4_path = "../data_all/initial_class/c4/"
c5_path = "../data_all/initial_class/c5/"
c6_path = "../data_all/initial_class/c6/"
c7_path = "../data_all/initial_class/c7/"
c8_path = "../data_all/initial_class/c8/"
c9_path = "../data_all/initial_class/c9/"
exception1_path = "../data_all/initial_class/exception_1/"
exception2_path = "../data_all/initial_class/exception_2/"
no_nose_path = "../data_all/initial_class/no_nose/"


# In[6]:


# function 1

def pathSorting(all_path):
    """
    This function creates the sorted list of the initial all_path list
    Result contains the sorted list of image paths (str)
    """
    fnum_str_list=[]
    # read the file and sort by numbers
    for path in all_path:    
        split_path=os.path.split(path)
        fname=split_path[1]
        fnum_included_str = re.findall(r'[0-9]+', fname)
        fnum_str = fnum_included_str[0]
        fnum_str_list.append(fnum_str)
        fnum_str_list.sort()
    sorted_path = [os.path.join(split_path[0], "img_"+fnum+".jpg") for fnum in fnum_str_list]
    return sorted_path


# In[7]:


def frame2Path(fr_num, all_path):
    """
    This function converts the frame no. (int) to 4-digit fr_num (str)
    and returns the fr_num and complete the image path (str) of f_num
    
    e.g.) 
    
       ('0940', '../data_all/mouse_pos/img_0940.jpg') = frame2Path(940, all_path)
    
    """
    fr_num=str(fr_num)    
    if len(fr_num)==1:
        fr_num = '000'+fr_num
    elif len(fr_num)==2:
        fr_num = '00'+fr_num
    elif len(fr_num)==3:
        fr_num = '0'+fr_num         
    split_path = os.path.split(all_path[0]) # an arbitrary path from all paths 
    path = os.path.join(split_path[0], "img_"+fr_num+".jpg")
    return fr_num, path


# In[8]:


def data_num2fr_num(df):
    """
    This function returns the integer extracted from the file name
    Input: e.g.) df (as a single row data strip of data)
    Output: integer
    """
    file_name=df.iloc[0][0]   # e.g.) 'img_0061.jpg'
    num_as_string=re.findall(r'[0-9]+', file_name)[0]   # e.g.) '0061'
    regex = "^0+(?!$)"
    num_as_string = re.sub(regex, "", num_as_string) # e.g.) 61 (None type)
    result = int(num_as_string) # 61 as a string
    return result


# In[9]:


def showImage(fr_num, all_path):
    """
    This function simply shows the image of frame no.
    desgnated by the argument'fr_num'
    """
    fr_num, path = frame2Path(fr_num, all_path)
    img=cv2.imread(path) 
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = Image.open(path)
    plt.imshow(img) 
    plt.axis('off') 
    plt.show()


# In[10]:


def showData(fr_num):
    """
    This fuction returns the data strip of the target frame
    """
    single_df=data.iloc[[fr_num-1]]
    return single_df


# In[11]:


def showMarks(fr_num, data): 
    """
    This function returns two results:
    (1) image (ndarray) 
    (2) matrix of marked body parts (adarray of integer)
         [[ ear_rx, ear_ry ],
          [ ear_lx, ear_ly ],
          [ ear_cx, ear_cy ],
          [ eye_rx, eye_ry ],
          [ eye_lx, eye_ly ],
          [ eye_cx, eye_cy ],
          [   n_x ,   n_y  ]]
    If the data contains NaN value, it raises an error.
    """
    # Path 
    __, img_path=frame2Path(fr_num, all_path)
#     data_path = '../data_all/mouse_pos'
#     img_path = os.path.join(data_path,"img_0"+str(fr_num)+'.jpg')
    single_df=data.iloc[[fr_num-1]]
    if not single_df.isnull().values.any():
        # Retrieve the coordinates of the body parts to define variables
        ear_rx, ear_ry = int(single_df[['ear_r_x']].squeeze()), int(single_df[['ear_r_y']].squeeze())
        ear_lx, ear_ly = int(single_df[['ear_l_x']].squeeze()), int(single_df[['ear_l_y']].squeeze())
        eye_rx, eye_ry = int(single_df[['eye_r_x']].squeeze()), int(single_df[['eye_r_y']].squeeze())
        eye_lx, eye_ly = int(single_df[['eye_l_x']].squeeze()), int(single_df[['eye_l_y']].squeeze())
        n_x, n_y = int(single_df[['nose_x']].squeeze()), int(single_df[['nose_y']].squeeze())
        # calculate the centers
        ear_cx, ear_cy = int((ear_rx+ear_lx)/2), int((ear_ry+ear_ly)/2)
        eye_cx, eye_cy = int((eye_rx+eye_lx)/2), int((eye_ry+eye_ly)/2)
        
        all_list = np.empty((7,2))
        all_list[:] = np.nan
        all_list[0] = ear_rx, ear_ry
        all_list[1] = ear_lx, ear_ly
        all_list[2] = ear_cx, ear_cy
        all_list[3] = eye_rx, eye_ry
        all_list[4] = eye_lx, eye_ly
        all_list[5] = eye_cx, eye_cy
        all_list[6] = n_x, n_y
        
        radius = 1
        cyan=[0,255,255]
        yellow=[255,255,0]
        magenta=[255,0,255]
        font=cv2.FONT_HERSHEY_SIMPLEX
        font_scale=0.5

        img=cv2.imread(img_path) 
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_marked=img

        # plot the detected body parts
        img_marked=cv2.circle(img_marked, (ear_rx, ear_ry), radius=radius, color=cyan, thickness=5)
        img_marked=cv2.circle(img_marked, (ear_lx, ear_ly), radius=radius, color=cyan, thickness=5)
        img_marked=cv2.circle(img_marked, (eye_rx, eye_ry), radius=radius, color=yellow, thickness=5)
        img_marked=cv2.circle(img_marked, (eye_lx, eye_ly), radius=radius, color=yellow, thickness=5)
        img_marked=cv2.circle(img_marked, (n_x, n_y), radius=radius, color=magenta, thickness=5)

        # mark the center position
        img_marked=cv2.putText(img_marked, 'x', (ear_cx, ear_cy), font, font_scale, color=cyan, thickness=2)
        img_marked=cv2.putText(img_marked, 'x', (eye_cx, eye_cy), font, font_scale, color=yellow, thickness=2)
    #     return plt.imshow(img_marked), print('test')
        return img_marked, all_list.astype(int)
    else:
        return print('Check whether the data strip contains NaN values.')


# In[12]:


def showMarkInFct(df): 
    """
    This function is utilized in the function showClass 
    Input: data strip (single row dataframe)
    
    And it returns two results:
    (1) image (ndarray) 
    (2) matrix of marked body parts (adarray of integer)
         [[ ear_rx, ear_ry ],
          [ ear_lx, ear_ly ],
          [ ear_cx, ear_cy ],
          [ eye_rx, eye_ry ],
          [ eye_lx, eye_ly ],
          [ eye_cx, eye_cy ],
          [   n_x ,   n_y  ]]
    If the data contains NaN value, it raises an error.
    """
    # Path 
    fr_num = data_num2fr_num(df)
    __, img_path=frame2Path(fr_num, all_path)
    
#     data_path = '../data_all/mouse_pos'
#     img_path = os.path.join(data_path,"img_0"+str(fr_num)+'.jpg')
    single_df=df
    if not single_df.isnull().values.any():
        # Retrieve the coordinates of the body parts to define variables
        ear_rx, ear_ry = int(single_df[['ear_r_x']].squeeze()), int(single_df[['ear_r_y']].squeeze())
        ear_lx, ear_ly = int(single_df[['ear_l_x']].squeeze()), int(single_df[['ear_l_y']].squeeze())
        eye_rx, eye_ry = int(single_df[['eye_r_x']].squeeze()), int(single_df[['eye_r_y']].squeeze())
        eye_lx, eye_ly = int(single_df[['eye_l_x']].squeeze()), int(single_df[['eye_l_y']].squeeze())
        n_x, n_y = int(single_df[['nose_x']].squeeze()), int(single_df[['nose_y']].squeeze())
        # calculate the centers
        ear_cx, ear_cy = int((ear_rx+ear_lx)/2), int((ear_ry+ear_ly)/2)
        eye_cx, eye_cy = int((eye_rx+eye_lx)/2), int((eye_ry+eye_ly)/2)
        
        all_list = np.empty((7,2))
        all_list[:] = np.nan
        all_list[0] = ear_rx, ear_ry
        all_list[1] = ear_lx, ear_ly
        all_list[2] = ear_cx, ear_cy
        all_list[3] = eye_rx, eye_ry
        all_list[4] = eye_lx, eye_ly
        all_list[5] = eye_cx, eye_cy
        all_list[6] = n_x, n_y
        
        radius = 1
        cyan=[0,255,255]
        yellow=[255,255,0]
        magenta=[255,0,255]
        font=cv2.FONT_HERSHEY_SIMPLEX
        font_scale=0.5

        img=cv2.imread(img_path) 
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_marked=img

        # plot the detected body parts
        img_marked=cv2.circle(img_marked, (ear_rx, ear_ry), radius=radius, color=cyan, thickness=5)
        img_marked=cv2.circle(img_marked, (ear_lx, ear_ly), radius=radius, color=cyan, thickness=5)
        img_marked=cv2.circle(img_marked, (eye_rx, eye_ry), radius=radius, color=yellow, thickness=5)
        img_marked=cv2.circle(img_marked, (eye_lx, eye_ly), radius=radius, color=yellow, thickness=5)
        img_marked=cv2.circle(img_marked, (n_x, n_y), radius=radius, color=magenta, thickness=5)

        # mark the center position
        img_marked=cv2.putText(img_marked, 'x', (ear_cx, ear_cy), font, font_scale, color=cyan, thickness=2)
        img_marked=cv2.putText(img_marked, 'x', (eye_cx, eye_cy), font, font_scale, color=yellow, thickness=2)
    #     return plt.imshow(img_marked), print('test')
        return img_marked, all_list.astype(int)
    else:
        return print('Check whether the data strip contains NaN values.')


# In[13]:


def showClass(data):
    """
    This function shows the pose class according to the approx. pan angle as followings:
    right(1) - (2) - (3) - (4) - front face(5) - (6)- (7) - (8) - left(9)
    Input: dataframe (only a little part of the total data is recommended)
    Output: image, class, and no. of exception cases
    """ 
    
    n1=0 # to calculate the num of exception in case 1
    n2=0 # to calculate the num of exception in case 2
    
    # retrieve data strip
    for data_num in range(len(data)):  # need to take an absolute coords!
#         fr_num=data_num+1
        df = data.iloc[[data_num]]
        fr_num = data_num2fr_num(df)
        # Basically, the pan angle class starts from the right to the left
        # 
    
        ########################################################
        # case 1: if there exists at least one NaN in the data #
        ########################################################

        if df.isnull().values.any(): # dealing with the case where there's any NaN

            # remove when there is no nose pose info  (This removes 89 cases)     
            if df.isnull()['nose_x'].item():
                print("fr_num: {} = no nose".format(fr_num))
                showImage(fr_num, all_path)

            # case 1-1: head turned to the right approx. over 45 degrees
            elif df.isnull()['eye_r_x'].item()&df.notnull()['ear_r_x'].item()&df.notnull()['ear_l_x'].item():
                # case 1-1-1: approx. 45 to 90 
                if df['ear_l_x'].item()-df['ear_r_x'].item() > 0:
                    print("fr_num: {} 'CLASS 2' = turning to the right (approx. 45 to 90 degrees)".format(fr_num))
                    showImage(fr_num, all_path)
                # case 1-1-2: approx. over 90  # ref. frame 30
                else:
                    print("fr_num: {} 'CLASS 1' = turning to the right (approx. over 90 degrees)".format(fr_num))
                    showImage(fr_num, all_path)

            # case 1-2: head turned to the left approx. over 45 degrees
            elif df.isnull()['eye_l_x'].item()&df.notnull()['ear_r_x'].item()&df.notnull()['ear_l_x'].item():
                    # case 1-2-1: approx. 45 to 90
                if df['ear_l_x'].item()-df['ear_r_x'].item() > 0:
                    print("fr_num: {} 'CLASS 8' =turning to the left (approx. 45 to 90 degrees)".format(fr_num))
                    showImage(fr_num, all_path)
                else:
                    print("fr_num: {} 'CLASS 9' = turning to the left (approx. over 90 degrees)".format(fr_num))
                    showImage(fr_num, all_path)
            else:    
                n1 += 1                
                print("fr_num: {} = Exceptions reported from case 1".format(fr_num))
                showImage(fr_num, all_path)
    #       ----  

        ##############################################
        # case 2: all parts are detected without NaN #
        ##############################################

        else: 
            img, mark = showMarkInFct(df)
            ear_rx, ear_ry = mark[0]
            ear_lx, ear_ly = mark[1]
            ear_cx, ear_cy = mark[2]
            eye_rx, eye_ry = mark[3]
            eye_lx, eye_ly = mark[4]
            eye_cx, eye_cy = mark[5]
            n_x, n_y = mark[6]

            # nose is located between eyes (then we determine using proportion)
            if eye_rx<n_x & n_x<eye_lx:
                hori_eyes=abs(eye_lx-eye_rx)
                n_r_dev_raw=abs(eye_rx-n_x)
                n_l_dev_raw=abs(eye_lx-n_x)
                assert n_r_dev_raw+n_l_dev_raw == hori_eyes
                # how much the nose deviates: 
                n_r_dev=n_r_dev_raw/hori_eyes #if smaller than 0.5 -> turning to the right
                n_l_dev=n_l_dev_raw/hori_eyes # if smaller than 0.5 -> turning to the left
                print('|R_eye---({:.2f})---N---({:.2f})---L_eye|'.format(n_r_dev, n_l_dev))
                # define the front face when n_r_dev and n_l_dev is larger than 0.25 
                # to be a front face, those numbers should be between 0.5 w.r.t. the center
                if (n_r_dev >0.25) & (n_l_dev>0.25):
                    print("fr_num: {} 'CLASS 5' = front face".format(fr_num))
                    showImage(fr_num, all_path)
                elif n_r_dev >0.25:
                    print("fr_num: {} 'CLASS 6' = turning to the left within approx. 22.5 degrees".format(fr_num))
                    showImage(fr_num, all_path)
                else:
                    print("fr_num: {} 'CLASS 4' = turning to the right within approx. 22.5 degrees".format(fr_num))
                    showImage(fr_num, all_path)

            # nose is located outside of the right eye 
            elif eye_rx>=n_x & n_x<eye_lx:
                print("fr_num: {} 'CLASS 3' = turned to the right (approx. 22.5 to 45 degrees)".format(fr_num))
                showImage(fr_num, all_path)

            elif eye_rx<n_x & n_x>=eye_lx:
                print("fr_num: {} 'CLASS 7' = turned to the left (approx. 22.5 to 45 degrees)".format(fr_num))
                showImage(fr_num, all_path)

            else:
                n2+=1               
                print("fr_num: {} = Exceptions reported from case 2".format(fr_num))
                showImage(fr_num, all_path)
                
    return print('''Exceptions from case 1 = {} 
                 from case 2 = {}'''.format(n1, n2))


# In[14]:


def saveClass_reserve(data):
    
    """
    This function save image file according to the result form showClass:
    
    right(1) - (2) - (3) - (4) - front face(5) - (6)- (7) - (8) - left(9)
    Input: dataframe (only a little part of the total data is recommended)
    Output: no. of all cases saved
    """ 
    print("""
    
    Start saving ...
    
    """)
    
    nc1, nc2, nc3, nc4, nc5, nc6, nc7, nc8, nc9 = 0,0,0,0,0,0,0,0,0
    nn=0 # to calculate the num of no nose cases
    ne1=0 # to calculate the num of exception in case 1
    ne2=0 # to calculate the num of exception in case 2 
    
    # retrieve data strip
    for data_num in range(len(data)):  # need to take an absolute coords!
        df = data.iloc[[data_num]]
        fr_num = data_num2fr_num(df)
        
        fr_num_str = frame2Path(fr_num, all_path)[0]
        img_path = frame2Path(fr_num, all_path)[1] # this might hinder showClass of showMarksInFct, let's see
        
    
        ########################################################
        # case 1: if there exists at least one NaN in the data #
        ########################################################

        if df.isnull().values.any(): # dealing with the case where there's any NaN

            # remove when there is no nose pose info  (This removes 89 cases)     
            if df.isnull()['nose_x'].item():
                nn+=1
                copyfile(img_path, no_nose_path+"img_"+fr_num_str+".jpg")
                
            # case 1-1: head turned to the right approx. over 45 degrees
            elif df.isnull()['eye_r_x'].item()&df.notnull()['ear_r_x'].item()&df.notnull()['ear_l_x'].item():
                # case 1-1-1: approx. 45 to 90 
                if df['ear_l_x'].item()-df['ear_r_x'].item() > 0:
                    nc2+=1
                    copyfile(img_path, c2_path+"img_"+fr_num_str+".jpg")
                # case 1-1-2: approx. over 90  # ref. frame 30
                else:
                    nc1+=1
                    copyfile(img_path, c1_path+"img_"+fr_num_str+".jpg")

            # case 1-2: head turned to the left approx. over 45 degrees
            elif df.isnull()['eye_l_x'].item()&df.notnull()['ear_r_x'].item()&df.notnull()['ear_l_x'].item():
                    # case 1-2-1: approx. 45 to 90
                if df['ear_l_x'].item()-df['ear_r_x'].item() > 0:
                    nc8+=1
                    copyfile(img_path, c8_path+"img_"+fr_num_str+".jpg")
                else:
                    nc9+=1
                    copyfile(img_path, c9_path+"img_"+fr_num_str+".jpg")
            else:    
                ne1 += 1                
                copyfile(img_path, exception1_path+"img_"+fr_num_str+".jpg")

        ##############################################
        # case 2: all parts are detected without NaN #
        ##############################################

        else: 
            img, mark = showMarkInFct(df)
            ear_rx, ear_ry = mark[0]
            ear_lx, ear_ly = mark[1]
            ear_cx, ear_cy = mark[2]
            eye_rx, eye_ry = mark[3]
            eye_lx, eye_ly = mark[4]
            eye_cx, eye_cy = mark[5]
            n_x, n_y = mark[6]

            # nose is located between eyes (then we determine using proportion)
            if eye_rx<n_x & n_x<eye_lx:
                hori_eyes=abs(eye_lx-eye_rx)
                n_r_dev_raw=abs(eye_rx-n_x)
                n_l_dev_raw=abs(eye_lx-n_x)
                assert n_r_dev_raw+n_l_dev_raw == hori_eyes
                # how much the nose deviates: 
                n_r_dev=n_r_dev_raw/hori_eyes #if smaller than 0.5 -> turning to the right
                n_l_dev=n_l_dev_raw/hori_eyes # if smaller than 0.5 -> turning to the left
#                 print('|R_eye---({:.2f})---N---({:.2f})---L_eye|'.format(n_r_dev, n_l_dev))
                # define the front face when n_r_dev and n_l_dev is larger than 0.25 
                # to be a front face, those numbers should be between 0.5 w.r.t. the center
                if (n_r_dev >0.25) & (n_l_dev>0.25):
                    nc5+=1
                    copyfile(img_path, c5_path+"img_"+fr_num_str+".jpg")
                elif n_r_dev >0.25:
                    nc6+=1
                    copyfile(img_path, c6_path+"img_"+fr_num_str+".jpg")
                else:
                    nc4+=1
                    copyfile(img_path, c4_path+"img_"+fr_num_str+".jpg")

            # nose is located outside of the right eye 
            elif eye_rx>=n_x & n_x<eye_lx:
                nc3+=1
                copyfile(img_path, c3_path+"img_"+fr_num_str+".jpg")

            elif eye_rx<n_x & n_x>=eye_lx:
                nc7+=1
                copyfile(img_path, c7_path+"img_"+fr_num_str+".jpg")

            else:
                ne2+=1               
                copyfile(img_path, exception2_path+"img_"+fr_num_str+".jpg")
    
    class_no_list =[nc1, nc2, nc3, nc4, nc5, nc6, nc7, nc8, nc9, nn, ne1, ne2]
    class_out = pd.DataFrame(class_no_list, 
                             index=['nc1', 'nc2', 'nc3', 'nc4', 'nc5', 'nc6', 'nc7', 'nc8', 'nc9', 'nn', 'ne1', 'ne2'], 
                             columns=['counts'])

    print("    ... Done!")
    return class_out


# In[15]:


def removeAllFiles():
    """
    This function removes all the files in class 
    from c1 to c9, and nn, ne1, ne2 as well
    """
    print(""" 
    
    Start removing ...
    
    """)
    saved_path_all = [c1_path, c2_path, c3_path, c4_path,
                      c5_path, c6_path, c7_path, c8_path,
                      c9_path, exception1_path, exception2_path,
                      no_nose_path]
    for path in saved_path_all:
        files = glob.glob(path+"*")
        for f in files:
            os.remove(f)
    return print("    ... Done!")


# In[16]:


def panCheck(img, mark):
    """
    Run this function after executing 'showMarks(fr_num, data)'
    to acquire the argurments (img, mark).
    This function simply presents the status of pan angle-wise pose with marked image. 
    An error is raised when there is NaN value in the 'mark' argument.
    """ 
    if not np.isnan(np.sum(mark)):
        ear_rx, ear_ry = mark[0]
        ear_lx, ear_ly = mark[1]
        ear_cx, ear_cy = mark[2]
        eye_rx, eye_ry = mark[3]
        eye_lx, eye_ly = mark[4]
        eye_cx, eye_cy = mark[5]
        n_x, n_y = mark[6]
        # initial check
        if ear_rx < eye_rx & eye_rx < n_x:
            print('right side clear')
        if ear_lx > eye_lx & eye_lx > n_x:
            print('left side clear')
         # Hypothesis 1: critical part is the relation between the nose and eyes
        hori_eyes=abs(eye_lx-eye_rx) # distance between the eyes (horizontal)
        n_r_dev_raw=abs(eye_rx-n_x)
        n_l_dev_raw=abs(eye_lx-n_x)
        assert n_r_dev_raw+n_l_dev_raw == hori_eyes
        # how much the nose deviates: 
        n_r_dev=n_r_dev_raw/hori_eyes #if smaller than 0.5 -> turning to the right
        n_l_dev=n_l_dev_raw/hori_eyes # if smaller than 0.5 -> turning to the left
#         print(f'|R_eye---{n_r_dev}---N---{n_l_dev}---L_eye|')
        print('|R_eye---({:.2f})---N---({:.2f})---L_eye|'.format(n_r_dev, n_l_dev))
        
        plt.imshow(img)
        plt.axis('off')
        
        # define the front face when n_r_dev and n_l_dev is larger than 0.25 
        # to be a front face, those numbers should be between 0.5 w.r.t. the center
        if (n_r_dev >0.25) & (n_l_dev>0.25):
            print('front face in terms of x-coords') # (1)
        elif n_r_dev >0.25:
            print('turning to the left') # (2)
        else:
            print('turning to the right') #(3)
        
    else:
        return print('Check whether the datastrip contains NaN values.')
    


# In[17]:


def tiltCheck(img, mark):   #deprecated or not use for the moment due to erronous 
    """
    Run this function after executing 'showMarks(fr_num, data)'
    to acquire the argurments (img, mark).
    This function simply presents the status of tilt angle-wise pose with marked image. 
    An error is raised when there is NaN value in the 'mark' argument.
    """ 
    if not np.isnan(np.sum(mark)):
        ear_rx, ear_ry = mark[0]
        ear_lx, ear_ly = mark[1]
        ear_cx, ear_cy = mark[2]
        eye_rx, eye_ry = mark[3]
        eye_lx, eye_ly = mark[4]
        eye_cx, eye_cy = mark[5]
        n_x, n_y = mark[6]
        
        # initial check   
        # Hypothesis 1: comparison between the distances (nose to eyes, eyes to ears) may imply important info.
        eye2ear_vert_raw=abs(eye_cy-ear_cy)
        eye2n_vert_raw=abs(eye_cy-n_y)
        ear2n_vert_raw=abs(ear_cy-n_y)
        assert eye2ear_vert_raw+eye2n_vert_raw == ear2n_vert_raw
        
        # which is larger and how much is the difference?
        eye2ear_vert_pro=round(eye2ear_vert_raw/ear2n_vert_raw,2) # round at 2nd decimal
        eye2n_vert_pro=round(eye2n_vert_raw/ear2n_vert_raw,2)
        result_m='''   
        --Ear center--
              |
            {:.2f}
              |
         -Eye center-
              |
            {:.2f}
              |
           --Nose--
        '''
        print(result_m.format(eye2ear_vert_pro, eye2n_vert_pro))
#         print('vertical distance from [Eye to Ear] : [Eye to Nose] = {:.2f} : {:.2f}'.format(eye2ear_vert_pro, eye2n_vert_pro))

        plt.imshow(img)
        plt.axis('off')
        
        # Hypothesis 2: Define the tilt-wise front face when the proportion is within 0.4 - 0.6
        if (eye2ear_vert_pro <= 0.6) &  (eye2ear_vert_pro >=0.4):
            print('front face in terms of y-coords') # (1)
        elif eye2ear_vert_pro > 0.6:
            print('tilting down')
        else:
            print('tiliting up')    
    
    else:
        return print('Check whether the datastrip contains NaN values.') 


# In[18]:


def showClass_old_erroneous(data):
    """
    This function shows the pose class according to the approx. pan angle as followings:
    right(1) - (2) - (3) - (4) - front face(5) - (6)- (7) - (8) - left(9)
    Input: dataframe (only a little part of the total data is recommended)
    Output: image, class, and no. of exception cases
    """ 
    
    n1=0 # to calculate the num of exception in case 1
    n2=0 # to calculate the num of exception in case 2
    
    # retrieve data strip
    for data_num in range(len(data)):
        fr_num=data_num+1
        df = data.iloc[[data_num]]
        
        # Basically, the pan angle class starts from the right to the left
        # 
    
        ########################################################
        # case 1: if there exists at least one NaN in the data #
        ########################################################

        if df.isnull().values.any(): # dealing with the case where there's any NaN

            # remove when there is no nose pose info  (This removes 89 cases)     
            if df.isnull()['nose_x'].item():
                print("fr_num: {} = no nose".format(fr_num))
                showImage(fr_num, all_path)

            # case 1-1: head turned to the right approx. over 45 degrees
            elif df.isnull()['eye_r_x'].item()&df.notnull()['ear_r_x'].item()&df.notnull()['ear_l_x'].item():
                # case 1-1-1: approx. 45 to 90 
                if df['ear_l_x'].item()-df['ear_r_x'].item() > 0:
                    print("fr_num: {} 'CLASS 2' = turning to the right (approx. 45 to 90 degrees)".format(fr_num))
                    showImage(fr_num, all_path)
                # case 1-1-2: approx. over 90  # ref. frame 30
                else:
                    print("fr_num: {} 'CLASS 1' = turning to the right (approx. over 90 degrees)".format(fr_num))
                    showImage(fr_num, all_path)

            # case 1-2: head turned to the left approx. over 45 degrees
            elif df.isnull()['eye_l_x'].item()&df.notnull()['ear_r_x'].item()&df.notnull()['ear_l_x'].item():
                    # case 1-2-1: approx. 45 to 90
                if df['ear_l_x'].item()-df['ear_r_x'].item() > 0:
                    print("fr_num: {} 'CLASS 8' =turning to the left (approx. 45 to 90 degrees)".format(fr_num))
                    showImage(fr_num, all_path)
                else:
                    print("fr_num: {} 'CLASS 9' = turning to the left (approx. over 90 degrees)".format(fr_num))
                    showImage(fr_num, all_path)

            else:    
                n1 += 1                
                print("fr_num: {} = Exceptions reported from case 1".format(fr_num))
                showImage(fr_num, all_path)
    #       ----  

        ##############################################
        # case 2: all parts are detected without NaN #
        ##############################################

        else: 
            img, mark = showMarks(fr_num, data)
            ear_rx, ear_ry = mark[0]
            ear_lx, ear_ly = mark[1]
            ear_cx, ear_cy = mark[2]
            eye_rx, eye_ry = mark[3]
            eye_lx, eye_ly = mark[4]
            eye_cx, eye_cy = mark[5]
            n_x, n_y = mark[6]

            # nose is located between eyes (then we determine using proportion)
            if eye_rx<n_x & n_x<eye_lx:
                hori_eyes=abs(eye_lx-eye_rx)
                n_r_dev_raw=abs(eye_rx-n_x)
                n_l_dev_raw=abs(eye_lx-n_x)
                assert n_r_dev_raw+n_l_dev_raw == hori_eyes
                # how much the nose deviates: 
                n_r_dev=n_r_dev_raw/hori_eyes #if smaller than 0.5 -> turning to the right
                n_l_dev=n_l_dev_raw/hori_eyes # if smaller than 0.5 -> turning to the left
                print('|R_eye---({:.2f})---N---({:.2f})---L_eye|'.format(n_r_dev, n_l_dev))
                # define the front face when n_r_dev and n_l_dev is larger than 0.25 
                # to be a front face, those numbers should be between 0.5 w.r.t. the center
                if (n_r_dev >0.25) & (n_l_dev>0.25):
                    print("fr_num: {} 'CLASS 5' = front face".format(fr_num))
                    showImage(fr_num, all_path)
                elif n_r_dev >0.25:
                    print("fr_num: {} 'CLASS 6' = turning to the left within approx. 22.5 degrees".format(fr_num))
                    showImage(fr_num, all_path)
                else:
                    print("fr_num: {} 'CLASS 4' = turning to the right within approx. 22.5 degrees".format(fr_num))
                    showImage(fr_num, all_path)

            # nose is located outside of the right eye 
            elif eye_rx>=n_x & n_x<eye_lx:
                print("fr_num: {} 'CLASS 3' = turned to the right (approx. 22.5 to 45 degrees)".format(fr_num))
                showImage(fr_num, all_path)

            elif eye_rx<n_x & n_x>=eye_lx:
                print("fr_num: {} 'CLASS 7' = turned to the left (approx. 22.5 to 45 degrees)".format(fr_num))
                showImage(fr_num, all_path)

            else:
                n2+=1               
                print("fr_num: {} = Exceptions reported from case 2".format(fr_num))
                showImage(fr_num, all_path)
                
    return print("Exceptions from case 1: {} and from case 2: {}".format(n1, n2))


# In[ ]:





# In[ ]:




