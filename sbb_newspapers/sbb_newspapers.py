#! /usr/bin/env python3

__version__ = '1.0'

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sys import getsizeof
import random
from tqdm import tqdm
from keras.models import model_from_json
from keras.models import load_model
import math
from shapely import geometry
from sklearn.cluster import KMeans
import gc
from keras import backend as K
import tensorflow as tf
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import xml.etree.ElementTree as ET
import warnings
import click
import time
from multiprocessing import Process, Queue, cpu_count
from matplotlib import pyplot, transforms
import matplotlib.patches as mpatches
import imutils

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

__doc__ = \
    """
    tool to extract table form data from alto xml data
    """


class sbb_newspapers:
    def __init__(self, image_dir, f_name,dir_out, dir_models):
        self.image_dir = image_dir  # XXX This does not seem to be a directory as the name suggests, but a file
        self.dir_out = dir_out
        self.f_name = f_name
        if self.f_name is None:
            try:
                self.f_name = image_dir.split('/')[len(image_dir.split('/')) - 1]
                self.f_name = self.f_name.split('.')[0]
            except:
                self.f_name = self.f_name.split('.')[0]
        self.dir_models = dir_models
        self.kernel = np.ones((5, 5), np.uint8)
        self.model_page_dir = dir_models + '/model_page_mixed_best.h5'
        self.model_region_dir_p = dir_models +'/model_layout_newspapers.h5'#'/model_main_home_5_soft_new.h5'#'/model_home_soft_5_all_data.h5' #'/model_main_office_long_soft.h5'#'/model_20_cat_main.h5'
        self.model_textline_dir = dir_models + '/model_textline_newspapers.h5'#'/model_hor_ver_home_trextline_very_good.h5'# '/model_hor_ver_1_great.h5'#'/model_curved_office_works_great.h5'


    def filter_contours_area_of_image_tables(self,image,contours,hirarchy,max_area,min_area):
        found_polygons_early = list()

        jv=0
        for c in contours:
            if len(c) < 3:  # A polygon cannot have less than 3 points
                continue

            polygon = geometry.Polygon([point[0] for point in c])
            #area = cv2.contourArea(c)
            area = polygon.area
            ##print(np.prod(thresh.shape[:2]))
            # Check that polygon has area greater than minimal area
            #print(hirarchy[0][jv][3],hirarchy )
            if area >=min_area*np.prod(image.shape[:2]) and area <=max_area*np.prod(image.shape[:2]):#and hirarchy[0][jv][3]==-1 :
                #print(c[0][0][1])
                found_polygons_early.append(
                    np.array(  [ [point] for point in polygon.exterior.coords] , dtype=np.int32) )
            jv+=1
        return found_polygons_early
    def find_polygons_size_filter(self, contours, median_area, scaler_up=1.2, scaler_down=0.8):
        found_polygons_early = list()

        for c in contours:
            if len(c) < 3:  # A polygon cannot have less than 3 points
                continue

            polygon = geometry.Polygon([point[0] for point in c])
            area = polygon.area
            # Check that polygon has area greater than minimal area
            if area >= median_area * scaler_down and area <= median_area * scaler_up:
                found_polygons_early.append(
                    np.array([point for point in polygon.exterior.coords], dtype=np.uint))
        return found_polygons_early

    def filter_contours_area_of_image(self, image, contours, hirarchy, max_area, min_area):
        found_polygons_early = list()

        jv = 0
        for c in contours:
            if len(c) < 3:  # A polygon cannot have less than 3 points
                continue

            polygon = geometry.Polygon([point[0] for point in c])
            area = polygon.area
            if area >= min_area * np.prod(image.shape[:2]) and area <= max_area * np.prod(
                    image.shape[:2]) and hirarchy[0][jv][3] == -1 :  # and hirarchy[0][jv][3]==-1 :
                found_polygons_early.append(
                    np.array([ [point] for point in polygon.exterior.coords], dtype=np.uint))
            jv += 1
        return found_polygons_early

    def filter_contours_area_of_image_interiors(self, image, contours, hirarchy, max_area, min_area):
        found_polygons_early = list()

        jv = 0
        for c in contours:
            if len(c) < 3:  # A polygon cannot have less than 3 points
                continue

            polygon = geometry.Polygon([point[0] for point in c])
            area = polygon.area
            if area >= min_area * np.prod(image.shape[:2]) and area <= max_area * np.prod(image.shape[:2]) and \
                    hirarchy[0][jv][3] != -1:
                # print(c[0][0][1])
                found_polygons_early.append(
                    np.array([point for point in polygon.exterior.coords], dtype=np.uint))
            jv += 1
        return found_polygons_early

    def resize_image(self, img_in, input_height, input_width):
        return cv2.resize(img_in, (input_width, input_height), interpolation=cv2.INTER_NEAREST)

    def resize_ann(self, seg_in, input_height, input_width):
        return cv2.resize(seg_in, (input_width, input_height), interpolation=cv2.INTER_NEAREST)
    
    def rotatedRectWithMaxArea(self,w, h, angle):
        if w <= 0 or h <= 0:
            return 0,0

        width_is_longer = w >= h
        side_long, side_short = (w,h) if width_is_longer else (h,w)

        # since the solutions for angle, -angle and 180-angle are all the same,
        # if suffices to look at the first quadrant and the absolute values of sin,cos:
        sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
        if side_short <= 2.*sin_a*cos_a*side_long or abs(sin_a-cos_a) < 1e-10:
            # half constrained case: two crop corners touch the longer side,
            #   the other two corners are on the mid-line parallel to the longer line
            x = 0.5*side_short
            wr,hr = (x/sin_a,x/cos_a) if width_is_longer else (x/cos_a,x/sin_a)
        else:
            # fully constrained case: crop touches all 4 sides
            cos_2a = cos_a*cos_a - sin_a*sin_a
            wr,hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a

        return wr,hr

    def get_one_hot(self, seg, input_height, input_width, n_classes):
        seg = seg[:, :, 0]
        seg_f = np.zeros((input_height, input_width, n_classes))
        for j in range(n_classes):
            seg_f[:, :, j] = (seg == j).astype(int)
        return seg_f


    def color_images(self, seg, n_classes):
        ann_u = range(n_classes)
        if len(np.shape(seg)) == 3:
            seg = seg[:, :, 0]

        seg_img = np.zeros((np.shape(seg)[0], np.shape(seg)[1], 3)).astype(np.uint8)
        colors = sns.color_palette("hls", n_classes)

        for c in ann_u:
            c = int(c)
            segl = (seg == c)
            seg_img[:, :, 0] = segl * c
            seg_img[:, :, 1] = segl * c
            seg_img[:, :, 2] = segl * c
        return seg_img

    def color_images_diva(self, seg, n_classes):
        ann_u = range(n_classes)
        if len(np.shape(seg)) == 3:
            seg = seg[:, :, 0]

        seg_img = np.zeros((np.shape(seg)[0], np.shape(seg)[1], 3)).astype(float)
        # colors=sns.color_palette("hls", n_classes)
        colors = [[1, 0, 0], [8, 0, 0], [2, 0, 0], [4, 0, 0]]

        for c in ann_u:
            c = int(c)
            segl = (seg == c)
            seg_img[:, :, 0][seg == c] = colors[c][0]  # segl*(colors[c][0])
            seg_img[:, :, 1][seg == c] = colors[c][1]  # seg_img[:,:,1]=segl*(colors[c][1])
            seg_img[:, :, 2][seg == c] = colors[c][2]  # seg_img[:,:,2]=segl*(colors[c][2])
        return seg_img

    def rotate_image(self, img_patch, slope):
        (h, w) = img_patch.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, slope, 1.0)
        return cv2.warpAffine(img_patch, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    def rotyate_image_different(self,img,slope):
        #img = cv2.imread('images/input.jpg')
        num_rows, num_cols = img.shape[:2]

        rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), slope, 1)
        img_rotation = cv2.warpAffine(img, rotation_matrix, (num_cols, num_rows) ) 
        return img_rotation

    def cleaning_probs(self, probs: np.ndarray, sigma: float) -> np.ndarray:
        # Smooth
        if sigma > 0.:
            return cv2.GaussianBlur(probs, (int(3 * sigma) * 2 + 1, int(3 * sigma) * 2 + 1), sigma)
        elif sigma == 0.:
            return cv2.fastNlMeansDenoising((probs * 255).astype(np.uint8), h=20) / 255
        else:  # Negative sigma, do not do anything
            return probs

    def crop_image_inside_box(self, box, img_org_copy):
        image_box = img_org_copy[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
        return image_box, [box[1], box[1] + box[3], box[0], box[0] + box[2]]

    def otsu_copy(self, img):
        img_r = np.zeros(img.shape)
        img1 = img[:, :, 0]
        img2 = img[:, :, 1]
        img3 = img[:, :, 2]
        # print(img.min())
        # print(img[:,:,0].min())
        # blur = cv2.GaussianBlur(img,(5,5))
        # ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        retval1, threshold1 = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        retval2, threshold2 = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        retval3, threshold3 = cv2.threshold(img3, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        img_r[:, :, 0] = threshold1
        img_r[:, :, 1] = threshold1
        img_r[:, :, 2] = threshold1
        return img_r

    def otsu_copy_binary(self,img):
        img_r=np.zeros((img.shape[0],img.shape[1],3))
        img1=img[:,:,0]

        retval1, threshold1 = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        

        img_r[:,:,0]=threshold1
        img_r[:,:,1]=threshold1
        img_r[:,:,2]=threshold1

        img_r=img_r/float(np.max(img_r))*255
        return img_r
    def get_image_and_scales(self):
        
        self.image = cv2.imread(self.image_dir)
        self.image_org=np.copy(self.image)
        self.height_org = self.image.shape[0]
        self.width_org = self.image.shape[1]
        """
        if self.image.shape[0] < 1000:
            self.img_hight_int = 2800
            self.img_width_int = int(self.img_hight_int * self.image.shape[1] / float(self.image.shape[0]))
        
        elif self.image.shape[0] < 2000 and self.image.shape[0] >= 1000:
            self.img_hight_int = 3500
            self.img_width_int = int(self.img_hight_int * self.image.shape[1] / float(self.image.shape[0]))

        elif self.image.shape[0] < 3000 and self.image.shape[0] >= 2000:
            self.img_hight_int = 5500
            self.img_width_int = int(self.img_hight_int * self.image.shape[1] / float(self.image.shape[0]))

        elif self.image.shape[0] < 4000 and self.image.shape[0] >= 3000:
            self.img_hight_int = 6500
            self.img_width_int = int(self.img_hight_int * self.image.shape[1] / float(self.image.shape[0]))
        
        else:
            self.img_hight_int = self.image.shape[0]
            self.img_width_int = self.image.shape[1]
        """
    
        self.img_hight_int =int(self.image.shape[0]*1)
        self.img_width_int = int(self.image.shape[1]*1)
        self.scale_y = self.img_hight_int / float(self.image.shape[0])
        self.scale_x = self.img_width_int / float(self.image.shape[1])

        self.image = self.resize_image(self.image, self.img_hight_int, self.img_width_int)
        
        
    def get_image_and_scales_deskewd(self,img_deskewd):
        
        self.image = img_deskewd
        self.image_org=np.copy(self.image)
        self.height_org = self.image.shape[0]
        self.width_org = self.image.shape[1]
        """
        if self.image.shape[0] < 1000:
            self.img_hight_int = 2800
            self.img_width_int = int(self.img_hight_int * self.image.shape[1] / float(self.image.shape[0]))
        
        elif self.image.shape[0] < 2000 and self.image.shape[0] >= 1000:
            self.img_hight_int = 3500
            self.img_width_int = int(self.img_hight_int * self.image.shape[1] / float(self.image.shape[0]))

        elif self.image.shape[0] < 3000 and self.image.shape[0] >= 2000:
            self.img_hight_int = 5500
            self.img_width_int = int(self.img_hight_int * self.image.shape[1] / float(self.image.shape[0]))

        elif self.image.shape[0] < 4000 and self.image.shape[0] >= 3000:
            self.img_hight_int = 6500
            self.img_width_int = int(self.img_hight_int * self.image.shape[1] / float(self.image.shape[0]))
        
        else:
            self.img_hight_int = self.image.shape[0]
            self.img_width_int = self.image.shape[1]
        """
    
        self.img_hight_int =int(self.image.shape[0]*1)
        self.img_width_int = int(self.image.shape[1]*1)
        self.scale_y = self.img_hight_int / float(self.image.shape[0])
        self.scale_x = self.img_width_int / float(self.image.shape[1])

        self.image = self.resize_image(self.image, self.img_hight_int, self.img_width_int)

    def start_new_session_and_model(self, model_dir):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        session = tf.InteractiveSession()
        model = load_model(model_dir, compile=False)

        return model, session
    
    def return_bonding_box_of_contours(self,cnts):
        boxes_tot=[]
        for i in range(len(cnts)):
            x,y,w,h = cv2.boundingRect(cnts[i])

            box=[x,y,w,h]
            boxes_tot.append(box)
        return boxes_tot
    
    def find_features_of_lines(self,contours_main):
        
        areas_main=np.array([cv2.contourArea(contours_main[j]) for j in range(len(contours_main))])
        M_main=[cv2.moments(contours_main[j]) for j in range(len(contours_main))]
        cx_main=[(M_main[j]['m10']/(M_main[j]['m00']+1e-32)) for j in range(len(M_main))]
        cy_main=[(M_main[j]['m01']/(M_main[j]['m00']+1e-32)) for j in range(len(M_main))]
        x_min_main=np.array([np.min(contours_main[j][:,0,0]) for j in range(len(contours_main))])
        x_max_main=np.array([np.max(contours_main[j][:,0,0]) for j in range(len(contours_main))])

        y_min_main=np.array([np.min(contours_main[j][:,0,1]) for j in range(len(contours_main))])
        y_max_main=np.array([np.max(contours_main[j][:,0,1]) for j in range(len(contours_main))])

        slope_lines=[]
        
        for kk in range(len(contours_main)):
            [vx,vy,x,y] = cv2.fitLine(contours_main[kk], cv2.DIST_L2,0,0.01,0.01)
            slope_lines.append( ( (vy/vx)/np.pi*180 )[0]  )
                    
        slope_lines_org=slope_lines
        slope_lines=np.array(slope_lines)
        slope_lines[(slope_lines<10) & (slope_lines>-10)]=0
        
        slope_lines[(slope_lines<-200) | (slope_lines>200)]=1
        slope_lines[ (slope_lines!=0) &  (slope_lines!=1)]=2
        
        dis_x=np.abs(x_max_main-x_min_main)
        return slope_lines,dis_x, x_min_main ,x_max_main ,np.array(cy_main),np.array(slope_lines_org),y_min_main ,y_max_main,np.array(cx_main)
        
    def return_parent_contours(self,contours,hierarchy):
        contours_parent=[ contours[i] for i in range(len(contours) ) if hierarchy[0][i][3]==-1  ]
        return contours_parent
    
    def isNaN(self,num):
        return num != num
    def early_deskewing_slope_calculation_based_on_lines(self,region_pre_p):
        # lines are labels by 6 in this model
        seperators_closeup=( (region_pre_p[:,:,:]==6))*1
        
        seperators_closeup=seperators_closeup.astype(np.uint8)
        imgray = cv2.cvtColor(seperators_closeup, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 0, 255, 0)

        contours_lines,hierachy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        slope_lines,dist_x, x_min_main ,x_max_main ,cy_main,slope_lines_org,y_min_main, y_max_main, cx_main=self.find_features_of_lines(contours_lines)

        slope_lines_org_hor=slope_lines_org[slope_lines==0]
        args=np.array( range(len(slope_lines) ))
        len_x=seperators_closeup.shape[1]/4.0

        args_hor=args[slope_lines==0]
        dist_x_hor=dist_x[slope_lines==0]
        x_min_main_hor=x_min_main[slope_lines==0]
        x_max_main_hor=x_max_main[slope_lines==0]
        cy_main_hor=cy_main[slope_lines==0]

        args_hor=args_hor[dist_x_hor>=len_x/2.0]
        x_max_main_hor=x_max_main_hor[dist_x_hor>=len_x/2.0]
        x_min_main_hor=x_min_main_hor[dist_x_hor>=len_x/2.0]
        cy_main_hor=cy_main_hor[dist_x_hor>=len_x/2.0]
        slope_lines_org_hor=slope_lines_org_hor[dist_x_hor>=len_x/2.0]


        slope_lines_org_hor=slope_lines_org_hor[np.abs(slope_lines_org_hor)<1.2]
        slope_mean_hor=np.mean(slope_lines_org_hor)

        if np.abs(slope_mean_hor)>1.2:
            slope_mean_hor=0

        #deskewed_new=rotate_image(image_regions_eraly_p[:,:,:],slope_mean_hor)


        args_ver=args[slope_lines==1]
        y_min_main_ver=y_min_main[slope_lines==1]
        y_max_main_ver=y_max_main[slope_lines==1]
        x_min_main_ver=x_min_main[slope_lines==1]
        x_max_main_ver=x_max_main[slope_lines==1]
        cx_main_ver=cx_main[slope_lines==1]
        dist_y_ver=y_max_main_ver-y_min_main_ver
        len_y=seperators_closeup.shape[0]/3.0
    
        return slope_mean_hor,cx_main_ver,dist_y_ver
        
    def return_contours_of_interested_region(self,region_pre_p,pixel):
        
        # pixels of images are identified by 5
        if len(region_pre_p.shape)==3:
            cnts_images=(region_pre_p[:,:,0]==pixel)*1
        else:
            cnts_images=(region_pre_p[:,:]==pixel)*1
        cnts_images=cnts_images.astype(np.uint8)
        cnts_images=np.repeat(cnts_images[:, :, np.newaxis], 3, axis=2)
        imgray = cv2.cvtColor(cnts_images, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 0, 255, 0)
        contours_imgs,hiearchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        contours_imgs=self.return_parent_contours(contours_imgs,hiearchy)
        contours_imgs=self.filter_contours_area_of_image_tables(thresh,contours_imgs,hiearchy,max_area=1,min_area=0.0002)
        return contours_imgs
    
    def return_contours_of_interested_textline(self,region_pre_p,pixel):
        
        # pixels of images are identified by 5
        if len(region_pre_p.shape)==3:
            cnts_images=(region_pre_p[:,:,0]==pixel)*1
        else:
            cnts_images=(region_pre_p[:,:]==pixel)*1
        cnts_images=cnts_images.astype(np.uint8)
        cnts_images=np.repeat(cnts_images[:, :, np.newaxis], 3, axis=2)
        imgray = cv2.cvtColor(cnts_images, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 0, 255, 0)
        contours_imgs,hiearchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        contours_imgs=self.return_parent_contours(contours_imgs,hiearchy)
        contours_imgs=self.filter_contours_area_of_image_tables(thresh,contours_imgs,hiearchy,max_area=1,min_area=0.000000003)
        return contours_imgs
    def find_images_contours_and_replace_table_and_graphic_pixels_by_image(self,region_pre_p):
        
        # pixels of images are identified by 5
        cnts_images=(region_pre_p[:,:,0]==5)*1
        cnts_images=cnts_images.astype(np.uint8)
        cnts_images=np.repeat(cnts_images[:, :, np.newaxis], 3, axis=2)
        imgray = cv2.cvtColor(cnts_images, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 0, 255, 0)
        contours_imgs,hiearchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        contours_imgs=self.return_parent_contours(contours_imgs,hiearchy)
        #print(len(contours_imgs),'contours_imgs')
        contours_imgs=self.filter_contours_area_of_image_tables(thresh,contours_imgs,hiearchy,max_area=1,min_area=0.0003)
        
        #print(len(contours_imgs),'contours_imgs')
        
        boxes_imgs=self.return_bonding_box_of_contours(contours_imgs)
        
        for i in range(len(boxes_imgs)):
            x1=int(boxes_imgs[i][0] )
            x2=int(boxes_imgs[i][0]+ boxes_imgs[i][2])
            y1=int(boxes_imgs[i][1] )
            y2=int(boxes_imgs[i][1]+ boxes_imgs[i][3])
            region_pre_p[y1:y2,x1:x2,0][region_pre_p[y1:y2,x1:x2,0]==8]=5
            region_pre_p[y1:y2,x1:x2,0][region_pre_p[y1:y2,x1:x2,0]==7]=5
        return region_pre_p
        
    def do_prediction(self,patches,img,model):
        
        img_height_model = model.layers[len(model.layers) - 1].output_shape[1]
        img_width_model = model.layers[len(model.layers) - 1].output_shape[2]
        n_classes = model.layers[len(model.layers) - 1].output_shape[3]
        


        if patches:
            if img.shape[0]<img_height_model:
                img=self.resize_image(img,img_height_model,img.shape[1])
                
            if img.shape[1]<img_width_model:
                img=self.resize_image(img,img.shape[0],img_width_model)
                
            margin = int(0.1 * img_width_model)

            width_mid = img_width_model - 2 * margin
            height_mid = img_height_model - 2 * margin


            img = img / float(255.0)

            img_h = img.shape[0]
            img_w = img.shape[1]

            prediction_true = np.zeros((img_h, img_w, 3))
            mask_true = np.zeros((img_h, img_w))
            nxf = img_w / float(width_mid)
            nyf = img_h / float(height_mid)

            if nxf > int(nxf):
                nxf = int(nxf) + 1
            else:
                nxf = int(nxf)

            if nyf > int(nyf):
                nyf = int(nyf) + 1
            else:
                nyf = int(nyf)

            for i in range(nxf):
                for j in range(nyf):

                    if i == 0:
                        index_x_d = i * width_mid
                        index_x_u = index_x_d + img_width_model
                    elif i > 0:
                        index_x_d = i * width_mid
                        index_x_u = index_x_d + img_width_model

                    if j == 0:
                        index_y_d = j * height_mid
                        index_y_u = index_y_d + img_height_model
                    elif j > 0:
                        index_y_d = j * height_mid
                        index_y_u = index_y_d + img_height_model

                    if index_x_u > img_w:
                        index_x_u = img_w
                        index_x_d = img_w - img_width_model
                    if index_y_u > img_h:
                        index_y_u = img_h
                        index_y_d = img_h - img_height_model
                        
                    

                    img_patch = img[index_y_d:index_y_u, index_x_d:index_x_u, :]

                    label_p_pred = model.predict(
                        img_patch.reshape(1, img_patch.shape[0], img_patch.shape[1], img_patch.shape[2]))

                    seg = np.argmax(label_p_pred, axis=3)[0]

                    seg_color = np.repeat(seg[:, :, np.newaxis], 3, axis=2)

                    if i==0 and j==0:
                        seg_color = seg_color[0:seg_color.shape[0] - margin, 0:seg_color.shape[1] - margin, :]
                        seg = seg[0:seg.shape[0] - margin, 0:seg.shape[1] - margin]

                        mask_true[index_y_d + 0:index_y_u - margin, index_x_d + 0:index_x_u - margin] = seg
                        prediction_true[index_y_d + 0:index_y_u - margin, index_x_d + 0:index_x_u - margin,
                        :] = seg_color
                        
                    elif i==nxf-1 and j==nyf-1:
                        seg_color = seg_color[margin:seg_color.shape[0] - 0, margin:seg_color.shape[1] - 0, :]
                        seg = seg[margin:seg.shape[0] - 0, margin:seg.shape[1] - 0]

                        mask_true[index_y_d + margin:index_y_u - 0, index_x_d + margin:index_x_u - 0] = seg
                        prediction_true[index_y_d + margin:index_y_u - 0, index_x_d + margin:index_x_u - 0,
                        :] = seg_color
                        
                    elif i==0 and j==nyf-1:
                        seg_color = seg_color[margin:seg_color.shape[0] - 0, 0:seg_color.shape[1] - margin, :]
                        seg = seg[margin:seg.shape[0] - 0, 0:seg.shape[1] - margin]

                        mask_true[index_y_d + margin:index_y_u - 0, index_x_d + 0:index_x_u - margin] = seg
                        prediction_true[index_y_d + margin:index_y_u - 0, index_x_d + 0:index_x_u - margin,
                        :] = seg_color
                        
                    elif i==nxf-1 and j==0:
                        seg_color = seg_color[0:seg_color.shape[0] - margin, margin:seg_color.shape[1] - 0, :]
                        seg = seg[0:seg.shape[0] - margin, margin:seg.shape[1] - 0]

                        mask_true[index_y_d + 0:index_y_u - margin, index_x_d + margin:index_x_u - 0] = seg
                        prediction_true[index_y_d + 0:index_y_u - margin, index_x_d + margin:index_x_u - 0,
                        :] = seg_color
                        
                    elif i==0 and j!=0 and j!=nyf-1:
                        seg_color = seg_color[margin:seg_color.shape[0] - margin, 0:seg_color.shape[1] - margin, :]
                        seg = seg[margin:seg.shape[0] - margin, 0:seg.shape[1] - margin]

                        mask_true[index_y_d + margin:index_y_u - margin, index_x_d + 0:index_x_u - margin] = seg
                        prediction_true[index_y_d + margin:index_y_u - margin, index_x_d + 0:index_x_u - margin,
                        :] = seg_color
                        
                    elif i==nxf-1 and j!=0 and j!=nyf-1:
                        seg_color = seg_color[margin:seg_color.shape[0] - margin, margin:seg_color.shape[1] - 0, :]
                        seg = seg[margin:seg.shape[0] - margin, margin:seg.shape[1] - 0]

                        mask_true[index_y_d + margin:index_y_u - margin, index_x_d + margin:index_x_u - 0] = seg
                        prediction_true[index_y_d + margin:index_y_u - margin, index_x_d + margin:index_x_u - 0,
                        :] = seg_color
                        
                    elif i!=0 and i!=nxf-1 and j==0:
                        seg_color = seg_color[0:seg_color.shape[0] - margin, margin:seg_color.shape[1] - margin, :]
                        seg = seg[0:seg.shape[0] - margin, margin:seg.shape[1] - margin]

                        mask_true[index_y_d + 0:index_y_u - margin, index_x_d + margin:index_x_u - margin] = seg
                        prediction_true[index_y_d + 0:index_y_u - margin, index_x_d + margin:index_x_u - margin,
                        :] = seg_color
                        
                    elif i!=0 and i!=nxf-1 and j==nyf-1:
                        seg_color = seg_color[margin:seg_color.shape[0] - 0, margin:seg_color.shape[1] - margin, :]
                        seg = seg[margin:seg.shape[0] - 0, margin:seg.shape[1] - margin]

                        mask_true[index_y_d + margin:index_y_u - 0, index_x_d + margin:index_x_u - margin] = seg
                        prediction_true[index_y_d + margin:index_y_u - 0, index_x_d + margin:index_x_u - margin,
                        :] = seg_color

                    else:
                        seg_color = seg_color[margin:seg_color.shape[0] - margin, margin:seg_color.shape[1] - margin, :]
                        seg = seg[margin:seg.shape[0] - margin, margin:seg.shape[1] - margin]

                        mask_true[index_y_d + margin:index_y_u - margin, index_x_d + margin:index_x_u - margin] = seg
                        prediction_true[index_y_d + margin:index_y_u - margin, index_x_d + margin:index_x_u - margin,
                        :] = seg_color

            prediction_true = prediction_true.astype(np.uint8)
                
        if not patches:
            img_h_page=img.shape[0]
            img_w_page=img.shape[1]
            img = img /float( 255.0)
            img = self.resize_image(img, img_height_model, img_width_model)

            label_p_pred = model.predict(
                img.reshape(1, img.shape[0], img.shape[1], img.shape[2]))

            seg = np.argmax(label_p_pred, axis=3)[0]
            seg_color =np.repeat(seg[:, :, np.newaxis], 3, axis=2)
            prediction_true = self.resize_image(seg_color, img_h_page, img_w_page)
            prediction_true = prediction_true.astype(np.uint8)
        return prediction_true
            
        

    def extract_page(self):
        patches=False
        model_page, session_page = self.start_new_session_and_model(self.model_page_dir)
        ###img = self.otsu_copy(self.image)
        for ii in range(1):
            img = cv2.GaussianBlur(self.image, (5, 5), 0)

        
        img_page_prediction=self.do_prediction(patches,img,model_page)
        
        imgray = cv2.cvtColor(img_page_prediction, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(imgray, 0, 255, 0)

        thresh = cv2.dilate(thresh, self.kernel, iterations=3)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cnt_size = np.array([cv2.contourArea(contours[j]) for j in range(len(contours))])

        cnt = contours[np.argmax(cnt_size)]

        x, y, w, h = cv2.boundingRect(cnt)

        box = [x, y, w, h]

        croped_page, page_coord = self.crop_image_inside_box(box, self.image)
        
        self.cont_page=[]
        self.cont_page.append( np.array( [ [ page_coord[2] , page_coord[0] ] , 
                                                    [ page_coord[3] , page_coord[0] ] ,
                                                    [ page_coord[3] , page_coord[1] ] ,
                                                [ page_coord[2] , page_coord[1] ]] ) )

        session_page.close()
        del model_page
        del session_page
        del contours
        del thresh
        del img

        gc.collect()
        return croped_page, page_coord
    


    def extract_text_regions(self, img,patches,cols):
        img_height_h=img.shape[0]
        img_width_h=img.shape[1]

        if patches and cols>=3 :
            model_region, session_region = self.start_new_session_and_model(self.model_region_dir_p)
        if not patches:
            model_region, session_region = self.start_new_session_and_model(self.model_region_dir_np)
            
        if patches and cols==2 :
            model_region, session_region = self.start_new_session_and_model(self.model_region_dir_p_2col)
            
        if patches and cols==1 :
            model_region, session_region = self.start_new_session_and_model(self.model_region_dir_p_2col)
        

            
        if patches and cols>=2:
            
            img = self.otsu_copy_binary(img)#self.otsu_copy(img)
            img = img.astype(np.uint8)
            
            
        if patches and cols==1:
            
            img = self.otsu_copy_binary(img)#self.otsu_copy(img)
            img = img.astype(np.uint8)
            img= self.resize_image(img, int(img_height_h*1), int(img_width_h*1) )
            

        
        prediction_regions=self.do_prediction(patches,img,model_region)
        prediction_regions=self.resize_image(prediction_regions, img_height_h, img_width_h )
        
        
        session_region.close()
        del model_region
        del session_region
        gc.collect()
        return prediction_regions
    
    def extract_only_text_regions(self, img,patches):
        
        model_region, session_region = self.start_new_session_and_model(self.model_only_text)
        img = self.otsu_copy_binary(img)#self.otsu_copy(img)
        img = img.astype(np.uint8)
        img_org=np.copy(img)
        
        img_h=img_org.shape[0]
        img_w=img_org.shape[1]
        
        img= self.resize_image(img_org, int(img_org.shape[0]*1), int(img_org.shape[1]*1))
        

        prediction_regions1=self.do_prediction(patches,img,model_region)
        
        prediction_regions1= self.resize_image(prediction_regions1, img_h, img_w)
        

        #prediction_regions1 = cv2.dilate(prediction_regions1, self.kernel, iterations=4)
        #prediction_regions1 = cv2.erode(prediction_regions1, self.kernel, iterations=7)
        #prediction_regions1 = cv2.dilate(prediction_regions1, self.kernel, iterations=2)
        
        
        img= self.resize_image(img_org, int(img_org.shape[0]*1), int(img_org.shape[1]*1))
        
        
        prediction_regions2=self.do_prediction(patches,img,model_region)
        
        prediction_regions2= self.resize_image(prediction_regions2, img_h, img_w)
        
    
        #prediction_regions2 = cv2.dilate(prediction_regions2, self.kernel, iterations=2)
        prediction_regions2 = cv2.erode(prediction_regions2, self.kernel, iterations=2)
        prediction_regions2 = cv2.dilate(prediction_regions2, self.kernel, iterations=2)
        
        
        #prediction_regions=(  (prediction_regions2[:,:,0]==1) & (prediction_regions1[:,:,0]==1) )
        #prediction_regions=(prediction_regions1[:,:,0]==1) 
        
        session_region.close()
        del model_region
        del session_region
        gc.collect()
        return prediction_regions1[:,:,0]
    
    def extract_binarization(self, img,patches):
        
        model_bin, session_bin = self.start_new_session_and_model(self.model_binafrization)

        
        img_h=img.shape[0]
        img_w=img.shape[1]
        
        img= self.resize_image(img, int(img.shape[0]*1), int(img.shape[1]*1))
        

        prediction_regions=self.do_prediction(patches,img,model_bin)
        
        res=(prediction_regions[:,:,0]!=0)*1

        

        
        img_fin=np.zeros((res.shape[0],res.shape[1],3) )
        res[:,:][res[:,:]==0]=2
        res=res-1
        res=res*255
        img_fin[:,:,0]=res
        img_fin[:,:,1]=res
        img_fin[:,:,2]=res
        
        session_bin.close()
        del model_bin
        del session_bin
        gc.collect()
        #plt.imshow(img_fin[:,:,0])
        #plt.show()
        return img_fin

    def get_text_region_contours_and_boxes(self, image):
        rgb_class_of_texts = (1, 1, 1)
        mask_texts = np.all(image == rgb_class_of_texts, axis=-1)

        image = np.repeat(mask_texts[:, :, np.newaxis], 3, axis=2) * 255
        image = image.astype(np.uint8)

        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, self.kernel)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, self.kernel)


        imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(imgray, 0, 255, 0)

        contours, hirarchy = cv2.findContours(thresh.copy(), cv2.cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        main_contours = self.filter_contours_area_of_image(thresh, contours, hirarchy, max_area=1, min_area=0.00001)
        self.boxes = []
        
        for jj in range(len(main_contours)):
            x, y, w, h = cv2.boundingRect(main_contours[jj])
            self.boxes.append([x, y, w, h])
            

        return main_contours
    def boosting_headers_by_longshot_region_segmentation(self,textregion_pre_p,textregion_pre_np,img_only_text):
        textregion_pre_p_org=np.copy(textregion_pre_p)
        # 4 is drop capitals
        headers_in_longshot= (textregion_pre_np[:,:,0]==2)*1
        textregion_pre_p[:,:,0][headers_in_longshot[:,:] ==1]=2
        textregion_pre_p[:,:,0][textregion_pre_p[:,:,0]==1]=0
        #textregion_pre_p[:,:,0][( img_only_text[:,:]==1) & (textregion_pre_p[:,:,0]!=7)  & (textregion_pre_p[:,:,0]!=2)]=1 # eralier it was so, but by this manner the drop capitals are alse deleted
        textregion_pre_p[:,:,0][( img_only_text[:,:]==1) & (textregion_pre_p[:,:,0]!=7)  & (textregion_pre_p[:,:,0]!=4) & (textregion_pre_p[:,:,0]!=2)]=1
        return textregion_pre_p
    
    def boosting_text_only_regions_by_header(self,textregion_pre_np,img_only_text):
        result= (( img_only_text[:,:]==1) | (textregion_pre_np[:,:,0]==2) ) *1
        return result
        
    def get_all_image_patches_coordination(self, image_page):
        self.all_box_coord=[]
        for jk in range(len(self.boxes)):
            _,crop_coor=self.crop_image_inside_box(self.boxes[jk],image_page)
            self.all_box_coord.append(crop_coor) 
        

    def textline_contours(self, img,patches,scaler_h,scaler_w):
        
        
        
        if patches:
            model_textline, session_textline = self.start_new_session_and_model(self.model_textline_dir)
        if not patches:
            model_textline, session_textline = self.start_new_session_and_model(self.model_textline_dir_np)
            
        ##img = self.otsu_copy(img)
        ##img = img.astype(np.uint8)
        
        img_org=np.copy(img)
        img_h=img_org.shape[0]
        img_w=img_org.shape[1]
        
        img= self.resize_image(img_org, int(img_org.shape[0]*scaler_h), int(img_org.shape[1]*scaler_w))
        
        prediction_textline=self.do_prediction(patches,img,model_textline)
        
        prediction_textline= self.resize_image(prediction_textline, img_h, img_w)
        
        patches=False
        prediction_textline_longshot=self.do_prediction(patches,img,model_textline)
        
        prediction_textline_longshot_true_size= self.resize_image(prediction_textline_longshot, img_h, img_w)
        
        
        #scaler_w=1.5
        #scaler_h=1.5
        #patches=True
        #img= self.resize_image(img_org, int(img_org.shape[0]*scaler_h), int(img_org.shape[1]*scaler_w))
        
        #prediction_textline_streched=self.do_prediction(patches,img,model_textline)
        
        #prediction_textline_streched= self.resize_image(prediction_textline_streched, img_h, img_w)
        
        ##plt.imshow(prediction_textline_streched[:,:,0])
        ##plt.show()
        
        #sys.exit()
        session_textline.close()

        del model_textline
        del session_textline
        gc.collect()
        return prediction_textline[:,:,0],prediction_textline_longshot_true_size[:,:,0]

    def get_textlines_for_each_textregions(self, textline_mask_tot, boxes):
        textline_mask_tot = cv2.erode(textline_mask_tot, self.kernel, iterations=1)
        self.area_of_cropped = []
        self.all_text_region_raw = []
        for jk in range(len(boxes)):
            crop_img, crop_coor = self.crop_image_inside_box(boxes[jk],
                                                             np.repeat(textline_mask_tot[:, :, np.newaxis], 3, axis=2))
            crop_img=crop_img.astype(np.uint8)
            self.all_text_region_raw.append(crop_img[:, :, 0])
            self.area_of_cropped.append(crop_img.shape[0] * crop_img.shape[1])


    def seperate_lines_new_inside_teils(self, img_path, thetha):
        (h, w) = img_path.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, -thetha, 1.0)
        x_d = M[0, 2]
        y_d = M[1, 2]

        thetha = thetha / 180. * np.pi
        rotation_matrix = np.array([[np.cos(thetha), -np.sin(thetha)], [np.sin(thetha), np.cos(thetha)]])


        x_min_cont = 0
        x_max_cont = img_path.shape[1]
        y_min_cont = 0
        y_max_cont = img_path.shape[0]

        xv = np.linspace(x_min_cont, x_max_cont, 1000)

        mada_n = img_path.sum(axis=1)
        
        ##plt.plot(mada_n)
        ##plt.show()

        first_nonzero = 0  # (next((i for i, x in enumerate(mada_n) if x), None))

        y = mada_n[:]  # [first_nonzero:last_nonzero]
        y_help = np.zeros(len(y) + 40)
        y_help[20:len(y) + 20] = y
        x = np.array(range(len(y)))

        peaks_real, _ = find_peaks(gaussian_filter1d(y, 3), height=0)
        if len(peaks_real)<=2 and len(peaks_real)>1:
            sigma_gaus=10
        else:
            sigma_gaus=5
    
    
        z= gaussian_filter1d(y_help, sigma_gaus)
        zneg_rev=-y_help+np.max(y_help)
        zneg=np.zeros(len(zneg_rev)+40)
        zneg[20:len(zneg_rev)+20]=zneg_rev
        zneg= gaussian_filter1d(zneg, sigma_gaus)

        peaks, _ = find_peaks(z, height=0)
        peaks_neg, _ = find_peaks(zneg, height=0)
        
        for nn in range(len(peaks_neg)):
            if peaks_neg[nn]>len(z)-1:
                peaks_neg[nn]=len(z)-1
            if peaks_neg[nn]<0:
                peaks_neg[nn]=0
        
        diff_peaks=np.abs( np.diff(peaks_neg) )
        
        cut_off=20
        peaks_neg_true=[]
        forest=[]
        
        for i in range(len(peaks_neg)):
            if i==0:
                forest.append(peaks_neg[i])
            if i<(len(peaks_neg)-1):
                if diff_peaks[i]<=cut_off:
                    forest.append(peaks_neg[i+1])
                if diff_peaks[i]>cut_off:
                    #print(forest[np.argmin(z[forest]) ] )
                    if not self.isNaN(forest[np.argmin(z[forest]) ]):
                        peaks_neg_true.append(forest[np.argmin(z[forest]) ])
                    forest=[]
                    forest.append(peaks_neg[i+1])
            if i==(len(peaks_neg)-1):
                #print(print(forest[np.argmin(z[forest]) ] ))
                if not self.isNaN(forest[np.argmin(z[forest]) ]):
                    peaks_neg_true.append(forest[np.argmin(z[forest]) ])
                    
                    
        diff_peaks_pos=np.abs( np.diff(peaks) )
        
        cut_off=20
        peaks_pos_true=[]
        forest=[]
        
        for i in range(len(peaks)):
            if i==0:
                forest.append(peaks[i])
            if i<(len(peaks)-1):
                if diff_peaks_pos[i]<=cut_off:
                    forest.append(peaks[i+1])
                if diff_peaks_pos[i]>cut_off:
                    #print(forest[np.argmin(z[forest]) ] )
                    if not self.isNaN(forest[np.argmax(z[forest]) ]):
                        peaks_pos_true.append(forest[np.argmax(z[forest]) ])
                    forest=[]
                    forest.append(peaks[i+1])
            if i==(len(peaks)-1):
                #print(print(forest[np.argmin(z[forest]) ] ))
                if not self.isNaN(forest[np.argmax(z[forest]) ]):
                    peaks_pos_true.append(forest[np.argmax(z[forest]) ])
        
        #print(len(peaks_neg_true) ,len(peaks_pos_true) ,'lensss')
        
        if len(peaks_neg_true)>0:
            peaks_neg_true=np.array(peaks_neg_true)
            """
            #plt.figure(figsize=(40,40))
            #plt.subplot(1,2,1)
            #plt.title('Textline segmentation von Textregion')
            #plt.imshow(img_path)
            #plt.xlabel('X')
            #plt.ylabel('Y')
            #plt.subplot(1,2,2)
            #plt.title('Dichte entlang X')
            #base = pyplot.gca().transData
            #rot = transforms.Affine2D().rotate_deg(90)
            #plt.plot(zneg,np.array(range(len(zneg))))
            #plt.plot(zneg[peaks_neg_true],peaks_neg_true,'*')
            #plt.gca().invert_yaxis()
            
            #plt.xlabel('Dichte')
            #plt.ylabel('Y')
            ##plt.plot([0,len(y)], [grenze,grenze])
            #plt.show()
            """
            peaks_neg_true = peaks_neg_true - 20 - 20
            
            
            #print(peaks_neg_true)
            for i in range(len(peaks_neg_true)):
                img_path[peaks_neg_true[i]-6:peaks_neg_true[i]+6,:]=0
                
                
        else:
            pass
        
        if len(peaks_pos_true)>0:
            peaks_pos_true=np.array(peaks_pos_true)
            peaks_pos_true = peaks_pos_true - 20
            
            for i in range(len(peaks_pos_true)):
                img_path[peaks_pos_true[i]-8:peaks_pos_true[i]+8,:]=1
        else:
            pass
        kernel = np.ones((5,5),np.uint8)

        #img_path = cv2.erode(img_path,kernel,iterations = 3)
        img_path = cv2.erode(img_path,kernel,iterations = 2)
        return img_path
            
                
        
    def seperate_lines_new(self, img_path, thetha,num_col):
        
        if num_col==1:
            num_patches=int(img_path.shape[1]/200.)
        else:
            num_patches=int(img_path.shape[1]/100.)
        #num_patches=int(img_path.shape[1]/200.)
        if num_patches==0:
            num_patches=1
        (h, w) = img_path.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, -thetha, 1.0)
        x_d = M[0, 2]
        y_d = M[1, 2]

        thetha = thetha / 180. * np.pi
        rotation_matrix = np.array([[np.cos(thetha), -np.sin(thetha)], [np.sin(thetha), np.cos(thetha)]])


        x_min_cont = 0
        x_max_cont = img_path.shape[1]
        y_min_cont = 0
        y_max_cont = img_path.shape[0]

        xv = np.linspace(x_min_cont, x_max_cont, 1000)

        mada_n = img_path.sum(axis=1)
            
        ##plt.plot(mada_n)
        ##plt.show()
        first_nonzero = 0  # (next((i for i, x in enumerate(mada_n) if x), None))

        y = mada_n[:]  # [first_nonzero:last_nonzero]
        y_help = np.zeros(len(y) + 40)
        y_help[20:len(y) + 20] = y
        x = np.array(range(len(y)))

        peaks_real, _ = find_peaks(gaussian_filter1d(y, 3), height=0)
        if len(peaks_real)<=2 and len(peaks_real)>1:
            sigma_gaus=10
        else:
            sigma_gaus=6
    
    
        z= gaussian_filter1d(y_help, sigma_gaus)
        zneg_rev=-y_help+np.max(y_help)
        zneg=np.zeros(len(zneg_rev)+40)
        zneg[20:len(zneg_rev)+20]=zneg_rev
        zneg= gaussian_filter1d(zneg, sigma_gaus)

        peaks, _ = find_peaks(z, height=0)
        peaks_neg, _ = find_peaks(zneg, height=0)
        
        
        for nn in range(len(peaks_neg)):
            if peaks_neg[nn]>len(z)-1:
                peaks_neg[nn]=len(z)-1
            if peaks_neg[nn]<0:
                peaks_neg[nn]=0
        
        diff_peaks=np.abs( np.diff(peaks_neg) )
        cut_off=20
        peaks_neg_true=[]
        forest=[]
        
        for i in range(len(peaks_neg)):
            if i==0:
                forest.append(peaks_neg[i])
            if i<(len(peaks_neg)-1):
                if diff_peaks[i]<=cut_off:
                    forest.append(peaks_neg[i+1])
                if diff_peaks[i]>cut_off:
                    #print(forest[np.argmin(z[forest]) ] )
                    if not self.isNaN(forest[np.argmin(z[forest]) ]):
                        #print(len(z),forest)
                        peaks_neg_true.append(forest[np.argmin(z[forest]) ])
                    forest=[]
                    forest.append(peaks_neg[i+1])
            if i==(len(peaks_neg)-1):
                #print(print(forest[np.argmin(z[forest]) ] ))
                if not self.isNaN(forest[np.argmin(z[forest]) ]):
                    
                    peaks_neg_true.append(forest[np.argmin(z[forest]) ])
        
        
        
        peaks_neg_true=np.array(peaks_neg_true)
        
        """
        #plt.figure(figsize=(40,40))
        #plt.subplot(1,2,1)
        #plt.title('Textline segmentation von Textregion')
        #plt.imshow(img_path)
        #plt.xlabel('X')
        #plt.ylabel('Y')
        #plt.subplot(1,2,2)
        #plt.title('Dichte entlang X')
        #base = pyplot.gca().transData
        #rot = transforms.Affine2D().rotate_deg(90)
        #plt.plot(zneg,np.array(range(len(zneg))))
        #plt.plot(zneg[peaks_neg_true],peaks_neg_true,'*')
        #plt.gca().invert_yaxis()
        
        #plt.xlabel('Dichte')
        #plt.ylabel('Y')
        ##plt.plot([0,len(y)], [grenze,grenze])
        #plt.show()
        """
        
        peaks_neg_true = peaks_neg_true - 20 - 20
        peaks = peaks - 20
        
        #dis_up=peaks_neg_true[14]-peaks_neg_true[0]
        #dis_down=peaks_neg_true[18]-peaks_neg_true[14]
        
        img_patch_ineterst=img_path[:,:]#[peaks_neg_true[14]-dis_up:peaks_neg_true[15]+dis_down ,:]
        
        ##plt.imshow(img_patch_ineterst)
        ##plt.show()
        
        
        
        
        length_x=int(img_path.shape[1]/float(num_patches))
        margin = int(0.04 * length_x)
        
        width_mid = length_x - 2 * margin





        nxf = img_path.shape[1] / float(width_mid)

        if nxf > int(nxf):
            nxf = int(nxf) + 1
        else:
            nxf = int(nxf)

        slopes_tile_wise=[]
        for i in range(nxf):
            if i == 0:
                index_x_d = i * width_mid
                index_x_u = index_x_d + length_x
            elif i > 0:
                index_x_d = i * width_mid
                index_x_u = index_x_d + length_x


            if index_x_u > img_path.shape[1]:
                index_x_u = img_path.shape[1]
                index_x_d = img_path.shape[1] - length_x

                    
                

            #img_patch = img[index_y_d:index_y_u, index_x_d:index_x_u, :]
            img_xline=img_patch_ineterst[:,index_x_d:index_x_u]
        
        
            sigma=2
            try:
                slope_xline=self.return_deskew_slop(img_xline,sigma)
            except:
                slope_xline=0
            slopes_tile_wise.append(slope_xline)
            #print(slope_xline,'xlineeee')
            img_line_rotated=self.rotate_image(img_xline,slope_xline)
            img_line_rotated[:,:][img_line_rotated[:,:]!=0]=1
        
        """
        
        xline=np.linspace(0,img_path.shape[1],nx)
        slopes_tile_wise=[]
        
        for ui in range( nx-1 ):
            img_xline=img_patch_ineterst[:,int(xline[ui]):int(xline[ui+1])]
            
        
            ##plt.imshow(img_xline)
            ##plt.show()
            
            sigma=3
            try:
                slope_xline=self.return_deskew_slop(img_xline,sigma)
            except:
                slope_xline=0
            slopes_tile_wise.append(slope_xline)
            print(slope_xline,'xlineeee')
            img_line_rotated=self.rotate_image(img_xline,slope_xline)
            
            ##plt.imshow(img_line_rotated)
            ##plt.show()
        """
        
        #dis_up=peaks_neg_true[14]-peaks_neg_true[0]
        #dis_down=peaks_neg_true[18]-peaks_neg_true[14]
        
        img_patch_ineterst=img_path[:,:]#[peaks_neg_true[14]-dis_up:peaks_neg_true[14]+dis_down ,:]
        
        img_patch_ineterst_revised=np.zeros(img_patch_ineterst.shape)
        

        for i in range(nxf):
            if i == 0:
                index_x_d = i * width_mid
                index_x_u = index_x_d + length_x
            elif i > 0:
                index_x_d = i * width_mid
                index_x_u = index_x_d + length_x


            if index_x_u > img_path.shape[1]:
                index_x_u = img_path.shape[1]
                index_x_d = img_path.shape[1] - length_x
                
            img_xline=img_patch_ineterst[:,index_x_d:index_x_u]
            
            img_int=np.zeros((img_xline.shape[0],img_xline.shape[1]))
            img_int[:,:]=img_xline[:,:]#img_patch_org[:,:,0]

            img_resized=np.zeros((int( img_int.shape[0]*(1.2) ) , int( img_int.shape[1]*(3) ) ))
            
            img_resized[ int( img_int.shape[0]*(.1)):int( img_int.shape[0]*(.1))+img_int.shape[0] , int( img_int.shape[1]*(1)):int( img_int.shape[1]*(1))+img_int.shape[1] ]=img_int[:,:]
            ##plt.imshow(img_xline)
            ##plt.show()
            img_line_rotated=self.rotate_image(img_resized,slopes_tile_wise[i])
            img_line_rotated[:,:][img_line_rotated[:,:]!=0]=1
            
            
            
            img_patch_seperated=self.seperate_lines_new_inside_teils(img_line_rotated,0)
            
            ##plt.imshow(img_patch_seperated)
            ##plt.show()
            img_patch_seperated_returned=self.rotate_image(img_patch_seperated,-slopes_tile_wise[i])
            img_patch_seperated_returned[:,:][img_patch_seperated_returned[:,:]!=0]=1
            
            img_patch_seperated_returned_true_size=img_patch_seperated_returned[ int( img_int.shape[0]*(.1)):int( img_int.shape[0]*(.1))+img_int.shape[0] , int( img_int.shape[1]*(1)):int( img_int.shape[1]*(1))+img_int.shape[1] ]
            
            img_patch_seperated_returned_true_size = img_patch_seperated_returned_true_size[:, margin:length_x - margin]
            img_patch_ineterst_revised[:,index_x_d + margin:index_x_u - margin]=img_patch_seperated_returned_true_size

            
        """
        for ui in range( nx-1 ):
            img_xline=img_patch_ineterst[:,int(xline[ui]):int(xline[ui+1])]
            
            
            img_int=np.zeros((img_xline.shape[0],img_xline.shape[1]))
            img_int[:,:]=img_xline[:,:]#img_patch_org[:,:,0]

            img_resized=np.zeros((int( img_int.shape[0]*(1.2) ) , int( img_int.shape[1]*(3) ) ))
            
            img_resized[ int( img_int.shape[0]*(.1)):int( img_int.shape[0]*(.1))+img_int.shape[0] , int( img_int.shape[1]*(1)):int( img_int.shape[1]*(1))+img_int.shape[1] ]=img_int[:,:]
            ##plt.imshow(img_xline)
            ##plt.show()
            img_line_rotated=self.rotate_image(img_resized,slopes_tile_wise[ui])
            
            
            #img_patch_seperated=self.seperate_lines_new_inside_teils(img_line_rotated,0)
            
            img_patch_seperated=self.seperate_lines_new_inside_teils(img_line_rotated,0)
            
            img_patch_seperated_returned=self.rotate_image(img_patch_seperated,-slopes_tile_wise[ui])
            ##plt.imshow(img_patch_seperated)
            ##plt.show()
            print(img_patch_seperated_returned.shape)
            #plt.imshow(img_patch_seperated_returned[ int( img_int.shape[0]*(.1)):int( img_int.shape[0]*(.1))+img_int.shape[0] , int( img_int.shape[1]*(1)):int( img_int.shape[1]*(1))+img_int.shape[1] ])
            #plt.show()
            
            img_patch_ineterst_revised[:,int(xline[ui]):int(xline[ui+1])]=img_patch_seperated_returned[ int( img_int.shape[0]*(.1)):int( img_int.shape[0]*(.1))+img_int.shape[0] , int( img_int.shape[1]*(1)):int( img_int.shape[1]*(1))+img_int.shape[1] ]
            
            
        """
            
        #print(img_patch_ineterst_revised.shape,np.unique(img_patch_ineterst_revised))
        ##plt.imshow(img_patch_ineterst_revised)
        ##plt.show()
        return img_patch_ineterst_revised

    def seperate_lines(self, img_patch, contour_text_interest, thetha):
        (h, w) = img_patch.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, -thetha, 1.0)
        x_d = M[0, 2]
        y_d = M[1, 2]

        thetha = thetha / 180. * np.pi
        rotation_matrix = np.array([[np.cos(thetha), -np.sin(thetha)], [np.sin(thetha), np.cos(thetha)]])
        contour_text_interest_copy = contour_text_interest.copy()

        x_cont = contour_text_interest[:, 0, 0]
        y_cont = contour_text_interest[:, 0, 1]
        x_cont = x_cont - np.min(x_cont)
        y_cont = y_cont - np.min(y_cont)

        x_min_cont = 0
        x_max_cont = img_patch.shape[1]
        y_min_cont = 0
        y_max_cont = img_patch.shape[0]

        xv = np.linspace(x_min_cont, x_max_cont, 1000)

        textline_patch_sum_along_width = img_patch.sum(axis=1)

        first_nonzero = 0  # (next((i for i, x in enumerate(mada_n) if x), None))

        y = textline_patch_sum_along_width[:]  # [first_nonzero:last_nonzero]
        y_padded = np.zeros(len(y) + 40)
        y_padded[20:len(y) + 20] = y
        x = np.array(range(len(y)))

        peaks_real, _ = find_peaks(gaussian_filter1d(y, 3), height=0)
        if 1>0:

            try:

                y_padded_smoothed_e= gaussian_filter1d(y_padded, 2)
                y_padded_up_to_down_e=-y_padded+np.max(y_padded)
                y_padded_up_to_down_padded_e=np.zeros(len(y_padded_up_to_down_e)+40)
                y_padded_up_to_down_padded_e[20:len(y_padded_up_to_down_e)+20]=y_padded_up_to_down_e
                y_padded_up_to_down_padded_e= gaussian_filter1d(y_padded_up_to_down_padded_e, 2)
                

                peaks_e, _ = find_peaks(y_padded_smoothed_e, height=0)
                peaks_neg_e, _ = find_peaks(y_padded_up_to_down_padded_e, height=0)
                neg_peaks_max=np.max(y_padded_up_to_down_padded_e[peaks_neg_e])

                arg_neg_must_be_deleted= np.array(range(len(peaks_neg_e)))[y_padded_up_to_down_padded_e[peaks_neg_e]/float(neg_peaks_max)<0.3  ] 
                diff_arg_neg_must_be_deleted=np.diff(arg_neg_must_be_deleted)
                

                
                arg_diff=np.array(range(len(diff_arg_neg_must_be_deleted)))
                arg_diff_cluster=arg_diff[diff_arg_neg_must_be_deleted>1]
                

                peaks_new=peaks_e[:]
                peaks_neg_new=peaks_neg_e[:]

                clusters_to_be_deleted=[]
                if len(arg_diff_cluster)>0:
                    
                    clusters_to_be_deleted.append(arg_neg_must_be_deleted[0:arg_diff_cluster[0]+1])
                    for i in range(len(arg_diff_cluster)-1):
                        clusters_to_be_deleted.append(arg_neg_must_be_deleted[arg_diff_cluster[i]+1:arg_diff_cluster[i+1]+1])
                    clusters_to_be_deleted.append(arg_neg_must_be_deleted[arg_diff_cluster[len(arg_diff_cluster)-1]+1:])
                    

                if len(clusters_to_be_deleted)>0:
                    peaks_new_extra=[]
                    for m in range(len(clusters_to_be_deleted)):
                        min_cluster=np.min(peaks_e[clusters_to_be_deleted[m]])
                        max_cluster=np.max(peaks_e[clusters_to_be_deleted[m]])
                        peaks_new_extra.append( int( (min_cluster+max_cluster)/2.0) )
                        for m1 in range(len(clusters_to_be_deleted[m])):
                            peaks_new=peaks_new[peaks_new!=peaks_e[clusters_to_be_deleted[m][m1]-1]]
                            peaks_new=peaks_new[peaks_new!=peaks_e[clusters_to_be_deleted[m][m1]]]
                            
                            peaks_neg_new=peaks_neg_new[peaks_neg_new!=peaks_neg_e[clusters_to_be_deleted[m][m1]]]
                    peaks_new_tot=[]
                    for i1 in peaks_new:
                        peaks_new_tot.append(i1)
                    for i1 in peaks_new_extra:
                        peaks_new_tot.append(i1)
                    peaks_new_tot=np.sort(peaks_new_tot)
                    
                    
                else:
                    peaks_new_tot=peaks_e[:]


                textline_con,hierachy=self.return_contours_of_image(img_patch)
                textline_con_fil=self.filter_contours_area_of_image(img_patch,textline_con,hierachy,max_area=1,min_area=0.0008)
                y_diff_mean=np.mean(np.diff(peaks_new_tot))#self.find_contours_mean_y_diff(textline_con_fil)

                sigma_gaus=int(  y_diff_mean * (7./40.0) )
                #print(sigma_gaus,'sigma_gaus')
            except:
                sigma_gaus=12
            if sigma_gaus<3:
                sigma_gaus=3
            #print(sigma_gaus,'sigma')
    
    
        y_padded_smoothed= gaussian_filter1d(y_padded, sigma_gaus)
        y_padded_up_to_down=-y_padded+np.max(y_padded)
        y_padded_up_to_down_padded=np.zeros(len(y_padded_up_to_down)+40)
        y_padded_up_to_down_padded[20:len(y_padded_up_to_down)+20]=y_padded_up_to_down
        y_padded_up_to_down_padded= gaussian_filter1d(y_padded_up_to_down_padded, sigma_gaus)
        

        peaks, _ = find_peaks(y_padded_smoothed, height=0)
        peaks_neg, _ = find_peaks(y_padded_up_to_down_padded, height=0)
        
        
        ##plt.plot(y_padded_up_to_down_padded)
        ##plt.plot(peaks_neg,y_padded_up_to_down_padded[peaks_neg],'*')
        ##plt.title('negs')
        ##plt.show()
        

        
        ##plt.plot(y_padded_smoothed)
        ##plt.plot(peaks,y_padded_smoothed[peaks],'*')
        ##plt.title('poss')
        ##plt.show()

            
        try:
            neg_peaks_max=np.max(y_padded_smoothed[peaks])
            

            arg_neg_must_be_deleted= np.array(range(len(peaks_neg)))[y_padded_up_to_down_padded[peaks_neg]/float(neg_peaks_max)<0.42  ] 


            diff_arg_neg_must_be_deleted=np.diff(arg_neg_must_be_deleted)
            

            
            arg_diff=np.array(range(len(diff_arg_neg_must_be_deleted)))
            arg_diff_cluster=arg_diff[diff_arg_neg_must_be_deleted>1]
        except:
            arg_neg_must_be_deleted=[]
            arg_diff_cluster=[]
            
        
        try:
            peaks_new=peaks[:]
            peaks_neg_new=peaks_neg[:]
            clusters_to_be_deleted=[]
            

            if len(arg_diff_cluster)>=2 and len(arg_diff_cluster)>0:
            
                clusters_to_be_deleted.append(arg_neg_must_be_deleted[0:arg_diff_cluster[0]+1])
                for i in range(len(arg_diff_cluster)-1):
                    clusters_to_be_deleted.append(arg_neg_must_be_deleted[arg_diff_cluster[i]+1:arg_diff_cluster[i+1]+1])
                clusters_to_be_deleted.append(arg_neg_must_be_deleted[arg_diff_cluster[len(arg_diff_cluster)-1]+1:])
            elif len(arg_neg_must_be_deleted)>=2 and len(arg_diff_cluster)==0:
                clusters_to_be_deleted.append(arg_neg_must_be_deleted[:])
                
        
        
            if  len(arg_neg_must_be_deleted)==1:
                clusters_to_be_deleted.append(arg_neg_must_be_deleted)
                

            if len(clusters_to_be_deleted)>0:
                peaks_new_extra=[]
                for m in range(len(clusters_to_be_deleted)):
                    min_cluster=np.min(peaks[clusters_to_be_deleted[m]])
                    max_cluster=np.max(peaks[clusters_to_be_deleted[m]])
                    peaks_new_extra.append( int( (min_cluster+max_cluster)/2.0) )
                    for m1 in range(len(clusters_to_be_deleted[m])):
                        peaks_new=peaks_new[peaks_new!=peaks[clusters_to_be_deleted[m][m1]-1]]
                        peaks_new=peaks_new[peaks_new!=peaks[clusters_to_be_deleted[m][m1]]]
                        
                        peaks_neg_new=peaks_neg_new[peaks_neg_new!=peaks_neg[clusters_to_be_deleted[m][m1]]]
                peaks_new_tot=[]
                for i1 in peaks_new:
                    peaks_new_tot.append(i1)
                for i1 in peaks_new_extra:
                    peaks_new_tot.append(i1)
                peaks_new_tot=np.sort(peaks_new_tot)
                
                ##plt.plot(y_padded_up_to_down_padded)
                ##plt.plot(peaks_neg,y_padded_up_to_down_padded[peaks_neg],'*')
                ##plt.show()
                
                ##plt.plot(y_padded_up_to_down_padded)
                ##plt.plot(peaks_neg_new,y_padded_up_to_down_padded[peaks_neg_new],'*')
                ##plt.show()
                
                ##plt.plot(y_padded_smoothed)
                ##plt.plot(peaks,y_padded_smoothed[peaks],'*')
                ##plt.show()
                
                ##plt.plot(y_padded_smoothed)
                ##plt.plot(peaks_new_tot,y_padded_smoothed[peaks_new_tot],'*')
                ##plt.show()
                
                peaks=peaks_new_tot[:]
                peaks_neg=peaks_neg_new[:]
                
                
            else:
                peaks_new_tot=peaks[:]
                peaks=peaks_new_tot[:]
                peaks_neg=peaks_neg_new[:]
        except:
            pass
            
        
        mean_value_of_peaks=np.mean(y_padded_smoothed[peaks])
        std_value_of_peaks=np.std(y_padded_smoothed[peaks])
        peaks_values=y_padded_smoothed[peaks]
        

        peaks_neg = peaks_neg - 20 - 20
        peaks = peaks - 20

        for jj in range(len(peaks_neg)):
            if peaks_neg[jj] > len(x) - 1:
                peaks_neg[jj] = len(x) - 1

        for jj in range(len(peaks)):
            if peaks[jj] > len(x) - 1:
                peaks[jj] = len(x) - 1
                
        

        textline_boxes = []
        textline_boxes_rot = []

        if len(peaks_neg) == len(peaks) + 1 and len(peaks) >= 3:
            #print('11')
            for jj in range(len(peaks)):
                
                if jj==(len(peaks)-1):
                    dis_to_next_up = abs(peaks[jj] - peaks_neg[jj])
                    dis_to_next_down = abs(peaks[jj] - peaks_neg[jj + 1])
                    
                    if peaks_values[jj]>mean_value_of_peaks-std_value_of_peaks/2.:
                        point_up = peaks[jj] + first_nonzero - int(1.3 * dis_to_next_up)  ##+int(dis_to_next_up*1./4.0)
                        point_down =y_max_cont-1##peaks[jj] + first_nonzero + int(1.3 * dis_to_next_down) #point_up# np.max(y_cont)#peaks[jj] + first_nonzero + int(1.4 * dis_to_next_down)  ###-int(dis_to_next_down*1./4.0)
                    else:
                        point_up = peaks[jj] + first_nonzero - int(1.4 * dis_to_next_up)  ##+int(dis_to_next_up*1./4.0)
                        point_down =y_max_cont-1##peaks[jj] + first_nonzero + int(1.6 * dis_to_next_down) #point_up# np.max(y_cont)#peaks[jj] + first_nonzero + int(1.4 * dis_to_next_down)  ###-int(dis_to_next_down*1./4.0)

                    point_down_narrow = peaks[jj] + first_nonzero + int(
                        1.4 * dis_to_next_down)  ###-int(dis_to_next_down*1./2)
                else:
                    dis_to_next_up = abs(peaks[jj] - peaks_neg[jj])
                    dis_to_next_down = abs(peaks[jj] - peaks_neg[jj + 1])
                    
                    if peaks_values[jj]>mean_value_of_peaks-std_value_of_peaks/2.:
                        point_up = peaks[jj] + first_nonzero - int(1.1 * dis_to_next_up)  ##+int(dis_to_next_up*1./4.0)
                        point_down = peaks[jj] + first_nonzero + int(1.1 * dis_to_next_down)  ###-int(dis_to_next_down*1./4.0)
                    else:
                        point_up = peaks[jj] + first_nonzero - int(1.23 * dis_to_next_up)  ##+int(dis_to_next_up*1./4.0)
                        point_down = peaks[jj] + first_nonzero + int(1.33 * dis_to_next_down)  ###-int(dis_to_next_down*1./4.0)

                    point_down_narrow = peaks[jj] + first_nonzero + int(
                        1.1 * dis_to_next_down)  ###-int(dis_to_next_down*1./2)



                if point_down_narrow >= img_patch.shape[0]:
                    point_down_narrow = img_patch.shape[0] - 2

                distances = [cv2.pointPolygonTest(contour_text_interest_copy, (xv[mj], peaks[jj] + first_nonzero), True)
                             for mj in range(len(xv))]
                distances = np.array(distances)

                xvinside = xv[distances >= 0]

                if len(xvinside) == 0:
                    x_min = x_min_cont
                    x_max = x_max_cont
                else:
                    x_min = np.min(xvinside)  # max(x_min_interest,x_min_cont)
                    x_max = np.max(xvinside)  # min(x_max_interest,x_max_cont)

                p1 = np.dot(rotation_matrix, [int(x_min), int(point_up)])
                p2 = np.dot(rotation_matrix, [int(x_max), int(point_up)])
                p3 = np.dot(rotation_matrix, [int(x_max), int(point_down)])
                p4 = np.dot(rotation_matrix, [int(x_min), int(point_down)])

                x_min_rot1, point_up_rot1 = p1[0] + x_d, p1[1] + y_d
                x_max_rot2, point_up_rot2 = p2[0] + x_d, p2[1] + y_d
                x_max_rot3, point_down_rot3 = p3[0] + x_d, p3[1] + y_d
                x_min_rot4, point_down_rot4 = p4[0] + x_d, p4[1] + y_d
                
                if x_min_rot1<0:
                    x_min_rot1=0
                if x_min_rot4<0:
                    x_min_rot4=0
                if point_up_rot1<0:
                    point_up_rot1=0
                if point_up_rot2<0:
                    point_up_rot2=0

                textline_boxes_rot.append(np.array([[int(x_min_rot1), int(point_up_rot1)],
                                                    [int(x_max_rot2), int(point_up_rot2)],
                                                    [int(x_max_rot3), int(point_down_rot3)],
                                                    [int(x_min_rot4), int(point_down_rot4)]]))

                textline_boxes.append(np.array([[int(x_min), int(point_up)],
                                                [int(x_max), int(point_up)],
                                                [int(x_max), int(point_down)],
                                                [int(x_min), int(point_down)]]))

        elif len(peaks) < 1:
            pass

        elif len(peaks) == 1:
            x_min = x_min_cont
            x_max = x_max_cont

            y_min = y_min_cont
            y_max = y_max_cont

            p1 = np.dot(rotation_matrix, [int(x_min), int(y_min)])
            p2 = np.dot(rotation_matrix, [int(x_max), int(y_min)])
            p3 = np.dot(rotation_matrix, [int(x_max), int(y_max)])
            p4 = np.dot(rotation_matrix, [int(x_min), int(y_max)])

            x_min_rot1, point_up_rot1 = p1[0] + x_d, p1[1] + y_d
            x_max_rot2, point_up_rot2 = p2[0] + x_d, p2[1] + y_d
            x_max_rot3, point_down_rot3 = p3[0] + x_d, p3[1] + y_d
            x_min_rot4, point_down_rot4 = p4[0] + x_d, p4[1] + y_d
            
            
            if x_min_rot1<0:
                x_min_rot1=0
            if x_min_rot4<0:
                x_min_rot4=0
            if point_up_rot1<0:
                point_up_rot1=0
            if point_up_rot2<0:
                point_up_rot2=0

            textline_boxes_rot.append(np.array([[int(x_min_rot1), int(point_up_rot1)],
                                                [int(x_max_rot2), int(point_up_rot2)],
                                                [int(x_max_rot3), int(point_down_rot3)],
                                                [int(x_min_rot4), int(point_down_rot4)]]))

            textline_boxes.append(np.array([[int(x_min), int(y_min)],
                                            [int(x_max), int(y_min)],
                                            [int(x_max), int(y_max)],
                                            [int(x_min), int(y_max)]]))



        elif len(peaks) == 2:
            dis_to_next = np.abs(peaks[1] - peaks[0])
            for jj in range(len(peaks)):
                if jj == 0:
                    point_up = 0#peaks[jj] + first_nonzero - int(1. / 1.7 * dis_to_next)
                    if point_up < 0:
                        point_up = 1
                    point_down = peaks[jj] + first_nonzero + int(1. / 1.8 * dis_to_next)
                elif jj == 1:
                    point_down = peaks[jj] + first_nonzero + int(1. / 1.8 * dis_to_next)
                    if point_down >= img_patch.shape[0]:
                        point_down = img_patch.shape[0] - 2
                    point_up = peaks[jj] + first_nonzero - int(1. / 1.8 * dis_to_next)

                distances = [cv2.pointPolygonTest(contour_text_interest_copy, (xv[mj], peaks[jj] + first_nonzero), True)
                             for mj in range(len(xv))]
                distances = np.array(distances)

                xvinside = xv[distances >= 0]

                if len(xvinside) == 0:
                    x_min = x_min_cont
                    x_max = x_max_cont
                else:
                    x_min = np.min(xvinside)
                    x_max = np.max(xvinside)

                p1 = np.dot(rotation_matrix, [int(x_min), int(point_up)])
                p2 = np.dot(rotation_matrix, [int(x_max), int(point_up)])
                p3 = np.dot(rotation_matrix, [int(x_max), int(point_down)])
                p4 = np.dot(rotation_matrix, [int(x_min), int(point_down)])

                x_min_rot1, point_up_rot1 = p1[0] + x_d, p1[1] + y_d
                x_max_rot2, point_up_rot2 = p2[0] + x_d, p2[1] + y_d
                x_max_rot3, point_down_rot3 = p3[0] + x_d, p3[1] + y_d
                x_min_rot4, point_down_rot4 = p4[0] + x_d, p4[1] + y_d
                
                if x_min_rot1<0:
                    x_min_rot1=0
                if x_min_rot4<0:
                    x_min_rot4=0
                if point_up_rot1<0:
                    point_up_rot1=0
                if point_up_rot2<0:
                    point_up_rot2=0

                textline_boxes_rot.append(np.array([[int(x_min_rot1), int(point_up_rot1)],
                                                    [int(x_max_rot2), int(point_up_rot2)],
                                                    [int(x_max_rot3), int(point_down_rot3)],
                                                    [int(x_min_rot4), int(point_down_rot4)]]))

                textline_boxes.append(np.array([[int(x_min), int(point_up)],
                                                [int(x_max), int(point_up)],
                                                [int(x_max), int(point_down)],
                                                [int(x_min), int(point_down)]]))
        else:
            for jj in range(len(peaks)):

                if jj == 0:
                    dis_to_next = peaks[jj + 1] - peaks[jj]
                    # point_up=peaks[jj]+first_nonzero-int(1./3*dis_to_next)
                    point_up = peaks[jj] + first_nonzero - int(1. / 1.9 * dis_to_next)
                    if point_up < 0:
                        point_up = 1
                    # point_down=peaks[jj]+first_nonzero+int(1./3*dis_to_next)
                    point_down = peaks[jj] + first_nonzero + int(1. / 1.9 * dis_to_next)
                elif jj == len(peaks) - 1:
                    dis_to_next = peaks[jj] - peaks[jj - 1]
                    # point_down=peaks[jj]+first_nonzero+int(1./3*dis_to_next)
                    point_down = peaks[jj] + first_nonzero + int(1. / 1.7 * dis_to_next)
                    if point_down >= img_patch.shape[0]:
                        point_down = img_patch.shape[0] - 2
                    # point_up=peaks[jj]+first_nonzero-int(1./3*dis_to_next)
                    point_up = peaks[jj] + first_nonzero - int(1. / 1.9 * dis_to_next)
                else:
                    dis_to_next_down = peaks[jj + 1] - peaks[jj]
                    dis_to_next_up = peaks[jj] - peaks[jj - 1]

                    point_up = peaks[jj] + first_nonzero - int(1. / 1.9 * dis_to_next_up)
                    point_down = peaks[jj] + first_nonzero + int(1. / 1.9 * dis_to_next_down)

                distances = [cv2.pointPolygonTest(contour_text_interest_copy, (xv[mj], peaks[jj] + first_nonzero), True)
                             for mj in range(len(xv))]
                distances = np.array(distances)

                xvinside = xv[distances >= 0]

                if len(xvinside) == 0:
                    x_min = x_min_cont
                    x_max = x_max_cont
                else:
                    x_min = np.min(xvinside)  # max(x_min_interest,x_min_cont)
                    x_max = np.max(xvinside)  # min(x_max_interest,x_max_cont)

                p1 = np.dot(rotation_matrix, [int(x_min), int(point_up)])
                p2 = np.dot(rotation_matrix, [int(x_max), int(point_up)])
                p3 = np.dot(rotation_matrix, [int(x_max), int(point_down)])
                p4 = np.dot(rotation_matrix, [int(x_min), int(point_down)])

                x_min_rot1, point_up_rot1 = p1[0] + x_d, p1[1] + y_d
                x_max_rot2, point_up_rot2 = p2[0] + x_d, p2[1] + y_d
                x_max_rot3, point_down_rot3 = p3[0] + x_d, p3[1] + y_d
                x_min_rot4, point_down_rot4 = p4[0] + x_d, p4[1] + y_d
                
                
                if x_min_rot1<0:
                    x_min_rot1=0
                if x_min_rot4<0:
                    x_min_rot4=0
                if point_up_rot1<0:
                    point_up_rot1=0
                if point_up_rot2<0:
                    point_up_rot2=0
                    


                textline_boxes_rot.append(np.array([[int(x_min_rot1), int(point_up_rot1)],
                                                    [int(x_max_rot2), int(point_up_rot2)],
                                                    [int(x_max_rot3), int(point_down_rot3)],
                                                    [int(x_min_rot4), int(point_down_rot4)]]))

                textline_boxes.append(np.array([[int(x_min), int(point_up)],
                                                [int(x_max), int(point_up)],
                                                [int(x_max), int(point_down)],
                                                [int(x_min), int(point_down)]]))


        return peaks, textline_boxes_rot
    
    def return_rotated_contours(self,slope,img_patch):
            dst = self.rotate_image(img_patch, slope)
            dst = dst.astype(np.uint8)
            dst = dst[:, :, 0]
            dst[dst != 0] = 1
            
            imgray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(imgray, 0, 255, 0)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            return contours
            
    def textline_contours_postprocessing(self, textline_mask, slope, contour_text_interest, box_ind):
        

        textline_mask = np.repeat(textline_mask[:, :, np.newaxis], 3, axis=2) * 255
        textline_mask = textline_mask.astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        textline_mask = cv2.morphologyEx(textline_mask, cv2.MORPH_OPEN, kernel)
        textline_mask = cv2.morphologyEx(textline_mask, cv2.MORPH_CLOSE, kernel)
        textline_mask = cv2.erode(textline_mask, kernel, iterations=2)
        
        try:

            dst = self.rotate_image(textline_mask, slope)
            dst = dst[:, :, 0]
            dst[dst != 0] = 1

            contour_text_copy = contour_text_interest.copy()

            contour_text_copy[:, 0, 0] = contour_text_copy[:, 0, 0] - box_ind[
                0]
            contour_text_copy[:, 0, 1] = contour_text_copy[:, 0, 1] - box_ind[1]

            img_contour = np.zeros((box_ind[3], box_ind[2], 3))
            img_contour = cv2.fillPoly(img_contour, pts=[contour_text_copy], color=(255, 255, 255))


 
            img_contour_rot = self.rotate_image(img_contour, slope)

            img_contour_rot = img_contour_rot.astype(np.uint8)
            imgrayrot = cv2.cvtColor(img_contour_rot, cv2.COLOR_BGR2GRAY)
            _, threshrot = cv2.threshold(imgrayrot, 0, 255, 0)
            contours_text_rot, _ = cv2.findContours(threshrot.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            len_con_text_rot = [len(contours_text_rot[ib]) for ib in range(len(contours_text_rot))]
            ind_big_con = np.argmax(len_con_text_rot)



            _, contours_rotated_clean = self.seperate_lines(dst, contours_text_rot[ind_big_con], slope)


        except:

            contours_rotated_clean = []

        return contours_rotated_clean

    def textline_contours_to_get_slope_correctly(self, textline_mask, img_patch, contour_interest):

        slope_new = 0  # deskew_images(img_patch)

        textline_mask = np.repeat(textline_mask[:, :, np.newaxis], 3, axis=2) * 255

        textline_mask = textline_mask.astype(np.uint8)
        textline_mask = cv2.morphologyEx(textline_mask, cv2.MORPH_OPEN, self.kernel)
        textline_mask = cv2.morphologyEx(textline_mask, cv2.MORPH_CLOSE, self.kernel)
        textline_mask = cv2.erode(textline_mask, self.kernel, iterations=1)
        imgray = cv2.cvtColor(textline_mask, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(imgray, 0, 255, 0)

        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, self.kernel)

        contours, hirarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        main_contours = self.filter_contours_area_of_image_tables(thresh, contours, hirarchy, max_area=1, min_area=0.003)

        textline_maskt = textline_mask[:, :, 0]
        textline_maskt[textline_maskt != 0] = 1

        peaks_point, _ = self.seperate_lines(textline_maskt, contour_interest, slope_new)

        mean_dis = np.mean(np.diff(peaks_point))

        len_x = thresh.shape[1]

        slope_lines = []
        contours_slope_new = []

        for kk in range(len(main_contours)):

            if len(main_contours[kk].shape)==2:
                xminh=np.min(main_contours[kk][:,0])
                xmaxh=np.max(main_contours[kk][:,0])

                yminh=np.min(main_contours[kk][:,1])
                ymaxh=np.max(main_contours[kk][:,1])
            elif len(main_contours[kk].shape)==3:
                xminh=np.min(main_contours[kk][:,0,0])
                xmaxh=np.max(main_contours[kk][:,0,0])

                yminh=np.min(main_contours[kk][:,0,1])
                ymaxh=np.max(main_contours[kk][:,0,1])


            if ymaxh - yminh <= mean_dis and (
                    xmaxh - xminh) >= 0.3 * len_x:  # xminh>=0.05*len_x and xminh<=0.4*len_x and xmaxh<=0.95*len_x and xmaxh>=0.6*len_x:
                contours_slope_new.append(main_contours[kk])

                rows, cols = thresh.shape[:2]
                [vx, vy, x, y] = cv2.fitLine(main_contours[kk], cv2.DIST_L2, 0, 0.01, 0.01)

                slope_lines.append((vy / vx) / np.pi * 180)

            if len(slope_lines) >= 2:

                slope = np.mean(slope_lines)  # slope_true/np.pi*180
            else:
                slope = 999

        else:
            slope = 0

        return slope
    def return_contours_of_image(self,image_box_tabels_1):
        
        image_box_tabels=np.repeat(image_box_tabels_1[:, :, np.newaxis], 3, axis=2)
        image_box_tabels=image_box_tabels.astype(np.uint8)
        imgray = cv2.cvtColor(image_box_tabels, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 0, 255, 0)
        contours,hierachy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        return contours,hierachy
    
    def find_contours_mean_y_diff(self,contours_main):
        M_main=[cv2.moments(contours_main[j]) for j in range(len(contours_main))]
        cy_main=[(M_main[j]['m01']/(M_main[j]['m00']+1e-32)) for j in range(len(M_main))]
        return np.mean( np.diff( np.sort( np.array(cy_main) ) ) )
    
    
    def isNaN(self,num):
        return num != num
    
    def find_num_col_olddd(self,regions_without_seperators,sigma_,multiplier=3.8 ):
        regions_without_seperators_0=regions_without_seperators[:,:].sum(axis=1)

        meda_n_updown=regions_without_seperators_0[len(regions_without_seperators_0)::-1]

        first_nonzero=(next((i for i, x in enumerate(regions_without_seperators_0) if x), 0))
        last_nonzero=(next((i for i, x in enumerate(meda_n_updown) if x), 0))

        last_nonzero=len(regions_without_seperators_0)-last_nonzero


        y=regions_without_seperators_0#[first_nonzero:last_nonzero]

        y_help=np.zeros(len(y)+20)

        y_help[10:len(y)+10]=y

        x=np.array( range(len(y)) )




        zneg_rev=-y_help+np.max(y_help)

        zneg=np.zeros(len(zneg_rev)+20)

        zneg[10:len(zneg_rev)+10]=zneg_rev

        z=gaussian_filter1d(y, sigma_)
        zneg= gaussian_filter1d(zneg, sigma_)


        peaks_neg, _ = find_peaks(zneg, height=0)
        peaks, _ = find_peaks(z, height=0)

        peaks_neg=peaks_neg-10-10

        

        last_nonzero=last_nonzero-0#100
        first_nonzero=first_nonzero+0#+100

        peaks_neg=peaks_neg[(peaks_neg>first_nonzero) & (peaks_neg<last_nonzero)]
        
        peaks=peaks[(peaks>.06*regions_without_seperators.shape[1]) & (peaks<0.94*regions_without_seperators.shape[1])]

        interest_pos=z[peaks]
        
        interest_pos=interest_pos[interest_pos>10]

        interest_neg=z[peaks_neg]
        
        
        if interest_neg[0]<0.1:
            interest_neg=interest_neg[1:]
        if interest_neg[len(interest_neg)-1]<0.1:
            interest_neg=interest_neg[:len(interest_neg)-1]
            
        
        
        min_peaks_pos=np.min(interest_pos)
        min_peaks_neg=0#np.min(interest_neg)
        

        dis_talaei=(min_peaks_pos-min_peaks_neg)/multiplier
        grenze=min_peaks_pos-dis_talaei#np.mean(y[peaks_neg[0]:peaks_neg[len(peaks_neg)-1]])-np.std(y[peaks_neg[0]:peaks_neg[len(peaks_neg)-1]])/2.0

        interest_neg_fin=interest_neg#[(interest_neg<grenze)]
        peaks_neg_fin=peaks_neg#[(interest_neg<grenze)]
        interest_neg_fin=interest_neg#[(interest_neg<grenze)]

        num_col=(len(interest_neg_fin))+1


        p_l=0
        p_u=len(y)-1
        p_m=int(len(y)/2.)
        p_g_l=int(len(y)/3.)
        p_g_u=len(y)-int(len(y)/3.)
        
        
        diff_peaks=np.abs( np.diff(peaks_neg_fin) )
        diff_peaks_annormal=diff_peaks[diff_peaks<30]
        

        return interest_neg_fin
    
    def find_num_col_deskew(self,regions_without_seperators,sigma_,multiplier=3.8 ):
        regions_without_seperators_0=regions_without_seperators[:,:].sum(axis=1)

        meda_n_updown=regions_without_seperators_0[len(regions_without_seperators_0)::-1]

        first_nonzero=(next((i for i, x in enumerate(regions_without_seperators_0) if x), 0))
        last_nonzero=(next((i for i, x in enumerate(meda_n_updown) if x), 0))

        last_nonzero=len(regions_without_seperators_0)-last_nonzero


        y=regions_without_seperators_0#[first_nonzero:last_nonzero]

        y_help=np.zeros(len(y)+20)

        y_help[10:len(y)+10]=y

        x=np.array( range(len(y)) )




        zneg_rev=-y_help+np.max(y_help)

        zneg=np.zeros(len(zneg_rev)+20)

        zneg[10:len(zneg_rev)+10]=zneg_rev

        z=gaussian_filter1d(y, sigma_)
        zneg= gaussian_filter1d(zneg, sigma_)


        peaks_neg, _ = find_peaks(zneg, height=0)
        peaks, _ = find_peaks(z, height=0)

        peaks_neg=peaks_neg-10-10
        
        #print(np.std(z),'np.std(z)np.std(z)np.std(z)')
        
        ##plt.plot(z)
        ##plt.show()
        
        ##plt.imshow(regions_without_seperators)
        ##plt.show()
        """
        last_nonzero=last_nonzero-0#100
        first_nonzero=first_nonzero+0#+100

        peaks_neg=peaks_neg[(peaks_neg>first_nonzero) & (peaks_neg<last_nonzero)]
        
        peaks=peaks[(peaks>.06*regions_without_seperators.shape[1]) & (peaks<0.94*regions_without_seperators.shape[1])]
        """
        interest_pos=z[peaks]
        
        interest_pos=interest_pos[interest_pos>10]
        
        interest_neg=z[peaks_neg]
        
        min_peaks_pos=np.mean(interest_pos)
        min_peaks_neg=0#np.min(interest_neg)
        
        dis_talaei=(min_peaks_pos-min_peaks_neg)/multiplier
        #print(interest_pos)
        grenze=min_peaks_pos-dis_talaei#np.mean(y[peaks_neg[0]:peaks_neg[len(peaks_neg)-1]])-np.std(y[peaks_neg[0]:peaks_neg[len(peaks_neg)-1]])/2.0

        interest_neg_fin=interest_neg[(interest_neg<grenze)]
        peaks_neg_fin=peaks_neg[(interest_neg<grenze)]
        interest_neg_fin=interest_neg[(interest_neg<grenze)]
        
        """
        if interest_neg[0]<0.1:
            interest_neg=interest_neg[1:]
        if interest_neg[len(interest_neg)-1]<0.1:
            interest_neg=interest_neg[:len(interest_neg)-1]
            
        
        
        min_peaks_pos=np.min(interest_pos)
        min_peaks_neg=0#np.min(interest_neg)
        

        dis_talaei=(min_peaks_pos-min_peaks_neg)/multiplier
        grenze=min_peaks_pos-dis_talaei#np.mean(y[peaks_neg[0]:peaks_neg[len(peaks_neg)-1]])-np.std(y[peaks_neg[0]:peaks_neg[len(peaks_neg)-1]])/2.0
        """
        #interest_neg_fin=interest_neg#[(interest_neg<grenze)]
        #peaks_neg_fin=peaks_neg#[(interest_neg<grenze)]
        #interest_neg_fin=interest_neg#[(interest_neg<grenze)]

        num_col=(len(interest_neg_fin))+1


        p_l=0
        p_u=len(y)-1
        p_m=int(len(y)/2.)
        p_g_l=int(len(y)/3.)
        p_g_u=len(y)-int(len(y)/3.)
        
        
        diff_peaks=np.abs( np.diff(peaks_neg_fin) )
        diff_peaks_annormal=diff_peaks[diff_peaks<30]
        
        #print(len(interest_neg_fin),np.mean(interest_neg_fin))
        return interest_neg_fin,np.std(z)
    def return_deskew_slop(self,img_patch_org,sigma_des):
        img_int=np.zeros((img_patch_org.shape[0],img_patch_org.shape[1]))
        img_int[:,:]=img_patch_org[:,:]#img_patch_org[:,:,0]

        img_resized=np.zeros((int( img_int.shape[0]*(1.2) ) , int( img_int.shape[1]*(2.6) ) ))
        
        img_resized[ int( img_int.shape[0]*(.1)):int( img_int.shape[0]*(.1))+img_int.shape[0] , int( img_int.shape[1]*(.8)):int( img_int.shape[1]*(.8))+img_int.shape[1] ]=img_int[:,:]
        angels=np.linspace(-12,12,40)

        res=[]
        num_of_peaks=[]
        index_cor=[]
        var_res=[]
        
        indexer=0
        for rot in angels:
            img_rot=self.rotate_image(img_resized,rot)
            ##plt.imshow(img_rot)
            ##plt.show()
            img_rot[img_rot!=0]=1
            #res_me=np.mean(self.find_num_col_deskew(img_rot,sigma_des,2.0  ))
            try:
                neg_peaks,var_spectrum=self.find_num_col_deskew(img_rot,sigma_des,20.3  )
                #print(indexer,'indexer')
                res_me=np.mean(neg_peaks)
                if res_me==0:
                    res_me=1000000000000000000000
                else:
                    pass
                    
                res_num=len(neg_peaks)
            except:
                res_me=1000000000000000000000
                res_num=0
                var_spectrum=0
            if self.isNaN(res_me):
                pass
            else:
                res.append( res_me )
                var_res.append(var_spectrum)
                num_of_peaks.append( res_num )
                index_cor.append(indexer)
            indexer=indexer+1


        try:
            var_res=np.array(var_res)
            
            ang_int=angels[np.argmax(var_res)]#angels_sorted[arg_final]#angels[arg_sort_early[arg_sort[arg_final]]]#angels[arg_fin]
        except:
            ang_int=0
        #print(ang_int,'ang_int')
        
        #img_rot=self.rotate_image(img_resized,ang_int)
        #img_rot[img_rot!=0]=1

        return ang_int

        
    def do_work_of_slopes(self,q,poly,box_sub,boxes_per_process,textline_mask_tot,contours_per_process):
        slope_biggest=0
        slopes_sub = []
        boxes_sub_new=[]
        poly_sub=[]
        for mv in range(len(boxes_per_process)):
            
            
            crop_img, _ = self.crop_image_inside_box(boxes_per_process[mv],
                                                                        np.repeat(textline_mask_tot[:, :, np.newaxis], 3, axis=2))
            crop_img=crop_img[:,:,0]
            crop_img=cv2.erode(crop_img,self.kernel,iterations = 2)
            
            try:
                textline_con,hierachy=self.return_contours_of_image(crop_img)
                textline_con_fil=self.filter_contours_area_of_image(crop_img,textline_con,hierachy,max_area=1,min_area=0.0008)
                y_diff_mean=self.find_contours_mean_y_diff(textline_con_fil)

                sigma_des=int(  y_diff_mean * (4./40.0) )

                if sigma_des<1:
                    sigma_des=1

                crop_img[crop_img>0]=1
                slope_corresponding_textregion=self.return_deskew_slop(crop_img,sigma_des)

                
            except:
                slope_corresponding_textregion=999
                
        
            if np.abs(slope_corresponding_textregion)>12.5 and slope_corresponding_textregion!=999:
                slope_corresponding_textregion=slope_biggest
            elif slope_corresponding_textregion==999:
                slope_corresponding_textregion=slope_biggest
            slopes_sub.append(slope_corresponding_textregion)
            
            cnt_clean_rot = self.textline_contours_postprocessing(crop_img
                                                                                        , slope_corresponding_textregion,
                                                                                        contours_per_process[mv], boxes_per_process[mv])
            
            poly_sub.append(cnt_clean_rot)
            boxes_sub_new.append(boxes_per_process[mv] )
            

        q.put(slopes_sub)
        poly.put(poly_sub)
        box_sub.put(boxes_sub_new )

    def get_slopes_and_deskew(self, contours,textline_mask_tot):

        slope_biggest=0#self.return_deskew_slop(img_int_p,sigma_des)
        
        num_cores = cpu_count()
        q = Queue()
        poly=Queue()
        box_sub=Queue()
        
        processes = []
        nh=np.linspace(0, len(self.boxes), num_cores+1)
        
        
        for i in range(num_cores):
            boxes_per_process=self.boxes[int(nh[i]):int(nh[i+1])]
            contours_per_process=contours[int(nh[i]):int(nh[i+1])]
            processes.append(Process(target=self.do_work_of_slopes, args=(q,poly,box_sub,  boxes_per_process, textline_mask_tot, contours_per_process)))
        
        for i in range(num_cores):
            processes[i].start()
            
        self.slopes = []
        self.all_found_texline_polygons=[]
        self.boxes=[]
        
        for i in range(num_cores):
            slopes_for_sub_process=q.get(True)
            boxes_for_sub_process=box_sub.get(True)
            polys_for_sub_process=poly.get(True)
            
            for j in range(len(slopes_for_sub_process)):
                self.slopes.append(slopes_for_sub_process[j])
                self.all_found_texline_polygons.append(polys_for_sub_process[j])
                self.boxes.append(boxes_for_sub_process[j])
                
        for i in range(num_cores):
            processes[i].join()
            
        
    def order_of_regions_old(self, textline_mask,contours_main):
        mada_n=textline_mask.sum(axis=1)
        y=mada_n[:]

        y_help=np.zeros(len(y)+40)
        y_help[20:len(y)+20]=y
        x=np.array( range(len(y)) )


        peaks_real, _ = find_peaks(gaussian_filter1d(y, 3), height=0)
        

        sigma_gaus=8

        z= gaussian_filter1d(y_help, sigma_gaus)
        zneg_rev=-y_help+np.max(y_help)

        zneg=np.zeros(len(zneg_rev)+40)
        zneg[20:len(zneg_rev)+20]=zneg_rev
        zneg= gaussian_filter1d(zneg, sigma_gaus)


        peaks, _ = find_peaks(z, height=0)
        peaks_neg, _ = find_peaks(zneg, height=0)

        peaks_neg=peaks_neg-20-20
        peaks=peaks-20
        

        
        if contours_main!=None:
            areas_main=np.array([cv2.contourArea(contours_main[j]) for j in range(len(contours_main))])
            M_main=[cv2.moments(contours_main[j]) for j in range(len(contours_main))]
            cx_main=[(M_main[j]['m10']/(M_main[j]['m00']+1e-32)) for j in range(len(M_main))]
            cy_main=[(M_main[j]['m01']/(M_main[j]['m00']+1e-32)) for j in range(len(M_main))]
            x_min_main=np.array([np.min(contours_main[j][:,0,0]) for j in range(len(contours_main))])
            x_max_main=np.array([np.max(contours_main[j][:,0,0]) for j in range(len(contours_main))])

            y_min_main=np.array([np.min(contours_main[j][:,0,1]) for j in range(len(contours_main))])
            y_max_main=np.array([np.max(contours_main[j][:,0,1]) for j in range(len(contours_main))])


        
        if contours_main!=None:
            indexer_main=np.array(range(len(contours_main)))

        
        if contours_main!=None:
            len_main=len(contours_main)
        else:
            len_main=0

        
        matrix_of_orders=np.zeros((len_main,5))
        
        matrix_of_orders[:,0]=np.array( range( len_main ) )
        
        matrix_of_orders[:len_main,1]=1
        matrix_of_orders[len_main:,1]=2
        
        matrix_of_orders[:len_main,2]=cx_main
        matrix_of_orders[:len_main,3]=cy_main

        matrix_of_orders[:len_main,4]=np.array( range( len_main ) )

        peaks_neg_new=[]
        peaks_neg_new.append(0)
        for iii in range(len(peaks_neg)):
            peaks_neg_new.append(peaks_neg[iii])
        peaks_neg_new.append(textline_mask.shape[0])
        
        final_indexers_sorted=[]
        for i in range(len(peaks_neg_new)-1):
            top=peaks_neg_new[i]
            down=peaks_neg_new[i+1]
            
            indexes_in=matrix_of_orders[:,0][(matrix_of_orders[:,3]>=top) & ((matrix_of_orders[:,3]<down))]
            cxs_in=matrix_of_orders[:,2][(matrix_of_orders[:,3]>=top) & ((matrix_of_orders[:,3]<down))]
            
            sorted_inside=np.argsort(cxs_in)
            
            ind_in_int=indexes_in[sorted_inside]
            
            for j in range(len(ind_in_int)):
                final_indexers_sorted.append(int(ind_in_int[j]) )
        

        return final_indexers_sorted, matrix_of_orders

            

    
    def order_and_id_of_texts_old(self, found_polygons_text_region ,matrix_of_orders ,indexes_sorted ):
        id_of_texts=[]
        order_of_texts=[]
        index_b=0
        for mm in range(len(found_polygons_text_region)):
            id_of_texts.append('r'+str(index_b) )
            index_matrix=matrix_of_orders[:,0][( matrix_of_orders[:,1]==1 ) & ( matrix_of_orders[:,4]==mm ) ]
            order_of_texts.append(np.where(indexes_sorted == index_matrix)[0][0])

            index_b+=1
            
        order_of_texts
        return order_of_texts, id_of_texts
    
    def write_into_page_xml_only_textlines(self,contours,page_coord ,all_found_texline_polygons,all_box_coord,dir_of_image):

        found_polygons_text_region=contours



        # create the file structure
        data = ET.Element('PcGts')

        data.set('xmlns',"http://schema.primaresearch.org/PAGE/gts/pagecontent/2017-07-15")
        data.set('xmlns:xsi',"http://www.w3.org/2001/XMLSchema-instance")
        data.set('xsi:schemaLocation',"http://schema.primaresearch.org/PAGE/gts/pagecontent/2017-07-15")



        metadata=ET.SubElement(data,'Metadata')

        author=ET.SubElement(metadata, 'Creator')
        author.text = 'SBB_QURATOR'


        created=ET.SubElement(metadata, 'Created')
        created.text = '2019-06-17T18:15:12'

        changetime=ET.SubElement(metadata, 'LastChange')
        changetime.text = '2019-06-17T18:15:12' 



        page=ET.SubElement(data,'Page')

        page.set('imageFilename', self.image_dir)
        page.set('imageHeight',str(self.height_org) ) 
        page.set('imageWidth',str(self.width_org) )
        page.set('type',"content")
        page.set('readingDirection',"left-to-right")
        page.set('textLineOrder',"top-to-bottom" )


        
        page_print_sub=ET.SubElement(page, 'PrintSpace')
        coord_page = ET.SubElement(page_print_sub, 'Coords')
        points_page_print=''

        for lmm in range(len(self.cont_page[0])):
            if len(self.cont_page[0][lmm])==2:
                points_page_print=points_page_print+str( int( (self.cont_page[0][lmm][0])/self.scale_x ) )
                points_page_print=points_page_print+','
                points_page_print=points_page_print+str( int( (self.cont_page[0][lmm][1])/self.scale_y ) )
            else:
                points_page_print=points_page_print+str( int((self.cont_page[0][lmm][0][0])/self.scale_x) )
                points_page_print=points_page_print+','
                points_page_print=points_page_print+str( int((self.cont_page[0][lmm][0][1])/self.scale_y) )

            if lmm<(len(self.cont_page[0])-1):
                points_page_print=points_page_print+' '
        coord_page.set('points',points_page_print)
        

        if len(contours)>0:

            id_indexer=0
            id_indexer_l=0
    
            for mm in range(len(found_polygons_text_region)):
                textregion=ET.SubElement(page, 'TextRegion')
    
                textregion.set('id','r'+str(id_indexer))
                id_indexer+=1
                
                textregion.set('type','paragraph')
                #if mm==0:
                #    textregion.set('type','heading')
                #else:
                #    textregion.set('type','paragraph')
                coord_text = ET.SubElement(textregion, 'Coords')
                
                points_co=''
                for lmm in range(len(found_polygons_text_region[mm])):
                    if len(found_polygons_text_region[mm][lmm])==2:
                        points_co=points_co+str( int( (found_polygons_text_region[mm][lmm][0] +page_coord[2])/self.scale_x ) )
                        points_co=points_co+','
                        points_co=points_co+str( int( (found_polygons_text_region[mm][lmm][1] +page_coord[0])/self.scale_y ) )
                    else:
                        points_co=points_co+str( int((found_polygons_text_region[mm][lmm][0][0] +page_coord[2])/self.scale_x) )
                        points_co=points_co+','
                        points_co=points_co+str( int((found_polygons_text_region[mm][lmm][0][1] +page_coord[0])/self.scale_y) )
    
                    if lmm<(len(found_polygons_text_region[mm])-1):
                        points_co=points_co+' '
                #print(points_co)
                coord_text.set('points',points_co)
                
                
                
                for j in range(len(all_found_texline_polygons[mm])):
    
                    textline=ET.SubElement(textregion, 'TextLine')
                    
                    textline.set('id','l'+str(id_indexer_l))
                    
                    id_indexer_l+=1
                    
    
                    coord = ET.SubElement(textline, 'Coords')
    
                    texteq=ET.SubElement(textline, 'TextEquiv')
    
                    uni=ET.SubElement(texteq, 'Unicode')
                    uni.text = ' ' 
    
                    #points = ET.SubElement(coord, 'Points') 
    
                    points_co=''
                    for l in range(len(all_found_texline_polygons[mm][j])):
                        #point = ET.SubElement(coord, 'Point') 
    
    
    
                        #point.set('x',str(found_polygons[j][l][0]))  
                        #point.set('y',str(found_polygons[j][l][1]))
                        if len(all_found_texline_polygons[mm][j][l])==2:
                            points_co=points_co+str( int( (all_found_texline_polygons[mm][j][l][0] +page_coord[2])/self.scale_x) )
                            points_co=points_co+','
                            points_co=points_co+str( int( (all_found_texline_polygons[mm][j][l][1] +page_coord[0])/self.scale_y) )
                        else:
                            points_co=points_co+str( int( ( all_found_texline_polygons[mm][j][l][0][0] +page_coord[2])/self.scale_x ) )
                            points_co=points_co+','
                            points_co=points_co+str( int( ( all_found_texline_polygons[mm][j][l][0][1] +page_coord[0])/self.scale_y) ) 
    
                        if l<(len(all_found_texline_polygons[mm][j])-1):
                            points_co=points_co+' '
                    #print(points_co)
                    coord.set('points',points_co)
                    
                texteqreg=ET.SubElement(textregion, 'TextEquiv')
    
                unireg=ET.SubElement(texteqreg, 'Unicode')
                unireg.text = ' ' 
                    



            
        #print(dir_of_image)
        print(self.f_name)
        #print(os.path.join(dir_of_image, self.f_name) + ".xml")
        tree = ET.ElementTree(data)
        tree.write(os.path.join(dir_of_image, self.f_name) + ".xml")
    
    def write_into_page_xml(self,contours,page_coord,dir_of_image,order_of_texts , id_of_texts,all_found_texline_polygons,all_box_coord,found_polygons_text_region_img):

        found_polygons_text_region=contours
        ##found_polygons_text_region_h=contours_h


        # create the file structure
        data = ET.Element('PcGts')

        data.set('xmlns',"http://schema.primaresearch.org/PAGE/gts/pagecontent/2017-07-15")
        data.set('xmlns:xsi',"http://www.w3.org/2001/XMLSchema-instance")
        data.set('xsi:schemaLocation',"http://schema.primaresearch.org/PAGE/gts/pagecontent/2017-07-15")



        metadata=ET.SubElement(data,'Metadata')

        author=ET.SubElement(metadata, 'Creator')
        author.text = 'SBB_QURATOR'


        created=ET.SubElement(metadata, 'Created')
        created.text = '2019-06-17T18:15:12'

        changetime=ET.SubElement(metadata, 'LastChange')
        changetime.text = '2019-06-17T18:15:12' 



        page=ET.SubElement(data,'Page')

        page.set('imageFilename', self.image_dir)
        page.set('imageHeight',str(self.height_org) ) 
        page.set('imageWidth',str(self.width_org) )
        page.set('type',"content")
        page.set('readingDirection',"left-to-right")
        page.set('textLineOrder',"top-to-bottom" )


        
        ###page_print_sub=ET.SubElement(page, 'PrintSpace')
        ###coord_page = ET.SubElement(page_print_sub, 'Coords')
        ###points_page_print=''

        ###for lmm in range(len(self.cont_page[0])):
            ###if len(self.cont_page[0][lmm])==2:
                ###points_page_print=points_page_print+str( int( (self.cont_page[0][lmm][0])/self.scale_x ) )
                ###points_page_print=points_page_print+','
                ###points_page_print=points_page_print+str( int( (self.cont_page[0][lmm][1])/self.scale_y ) )
            ###else:
                ###points_page_print=points_page_print+str( int((self.cont_page[0][lmm][0][0])/self.scale_x) )
                ###points_page_print=points_page_print+','
                ###points_page_print=points_page_print+str( int((self.cont_page[0][lmm][0][1])/self.scale_y) )

            ###if lmm<(len(self.cont_page[0])-1):
                ###points_page_print=points_page_print+' '
        ###coord_page.set('points',points_page_print)
        

        if len(contours)>0:
            region_order=ET.SubElement(page, 'ReadingOrder')
            region_order_sub = ET.SubElement(region_order, 'OrderedGroup')
            
            region_order_sub.set('id',"ro357564684568544579089")
    
            #args_sort=order_of_texts
            for vj in order_of_texts:
                name="coord_text_"+str(vj)
                name = ET.SubElement(region_order_sub, 'RegionRefIndexed')
                name.set('index',str(order_of_texts[vj]) )
                name.set('regionRef',id_of_texts[vj])
    
    
            id_indexer=0
            id_indexer_l=0
    
            for mm in range(len(found_polygons_text_region)):
                textregion=ET.SubElement(page, 'TextRegion')
    
                textregion.set('id','r'+str(id_indexer))
                id_indexer+=1
                
                textregion.set('type','paragraph')
                #if mm==0:
                #    textregion.set('type','heading')
                #else:
                #    textregion.set('type','paragraph')
                coord_text = ET.SubElement(textregion, 'Coords')
                
                points_co=''
                for lmm in range(len(found_polygons_text_region[mm])):
                    if len(found_polygons_text_region[mm][lmm])==2:
                        points_co=points_co+str( int( (found_polygons_text_region[mm][lmm][0] )/self.scale_x ) )
                        points_co=points_co+','
                        points_co=points_co+str( int( (found_polygons_text_region[mm][lmm][1] )/self.scale_y ) )
                    else:
                        points_co=points_co+str( int((found_polygons_text_region[mm][lmm][0][0] )/self.scale_x) )
                        points_co=points_co+','
                        points_co=points_co+str( int((found_polygons_text_region[mm][lmm][0][1] )/self.scale_y) )
    
                    if lmm<(len(found_polygons_text_region[mm])-1):
                        points_co=points_co+' '
                #print(points_co)
                coord_text.set('points',points_co)
                
                
                
                for j in range(len(all_found_texline_polygons[mm])):
    
                    textline=ET.SubElement(textregion, 'TextLine')
                    
                    textline.set('id','l'+str(id_indexer_l))
                    
                    id_indexer_l+=1
                    
    
                    coord = ET.SubElement(textline, 'Coords')
    
                    texteq=ET.SubElement(textline, 'TextEquiv')
    
                    uni=ET.SubElement(texteq, 'Unicode')
                    uni.text = ' ' 
    
                    #points = ET.SubElement(coord, 'Points') 
    
                    points_co=''
                    for l in range(len(all_found_texline_polygons[mm][j])):
                        #point = ET.SubElement(coord, 'Point') 
    
    
    
                        #point.set('x',str(found_polygons[j][l][0]))  
                        #point.set('y',str(found_polygons[j][l][1]))
                        if len(all_found_texline_polygons[mm][j][l])==2:
                            points_co=points_co+str( int( (all_found_texline_polygons[mm][j][l][0]
                                                    +all_box_coord[mm][2])/self.scale_x) )
                            points_co=points_co+','
                            points_co=points_co+str( int( (all_found_texline_polygons[mm][j][l][1] 
                                                    +all_box_coord[mm][0])/self.scale_y) )
                        else:
                            points_co=points_co+str( int( ( all_found_texline_polygons[mm][j][l][0][0] 
                                                    +all_box_coord[mm][2])/self.scale_x ) )
                            points_co=points_co+','
                            points_co=points_co+str( int( ( all_found_texline_polygons[mm][j][l][0][1] 
                                                    +all_box_coord[mm][0])/self.scale_y) ) 
    
                        if l<(len(all_found_texline_polygons[mm][j])-1):
                            points_co=points_co+' '
                    #print(points_co)
                    coord.set('points',points_co)
                    
                texteqreg=ET.SubElement(textregion, 'TextEquiv')
    
                unireg=ET.SubElement(texteqreg, 'Unicode')
                unireg.text = ' ' 
                
        ###print(len(contours_h))
        ###if len(contours_h)>0:
            ###for mm in range(len(found_polygons_text_region_h)):
                ###textregion=ET.SubElement(page, 'TextRegion')
                ###try:
                    ###id_indexer=id_indexer
                    ###id_indexer_l=id_indexer_l
                ###except:
                    ###id_indexer=0
                    ###id_indexer_l=0
                ###textregion.set('id','r'+str(id_indexer))
                ###id_indexer+=1

                ###textregion.set('type','heading')
                ####if mm==0:
                ####    textregion.set('type','heading')
                ####else:
                ####    textregion.set('type','paragraph')
                ###coord_text = ET.SubElement(textregion, 'Coords')

                ###points_co=''
                ###for lmm in range(len(found_polygons_text_region_h[mm])):

                    ###if len(found_polygons_text_region_h[mm][lmm])==2:
                        ###points_co=points_co+str( int( (found_polygons_text_region_h[mm][lmm][0] +page_coord[2])/self.scale_x ) )
                        ###points_co=points_co+','
                        ###points_co=points_co+str( int( (found_polygons_text_region_h[mm][lmm][1] +page_coord[0])/self.scale_y ) )
                    ###else:
                        ###points_co=points_co+str( int((found_polygons_text_region_h[mm][lmm][0][0] +page_coord[2])/self.scale_x) )
                        ###points_co=points_co+','
                        ###points_co=points_co+str( int((found_polygons_text_region_h[mm][lmm][0][1] +page_coord[0])/self.scale_y) )

                    ###if lmm<(len(found_polygons_text_region_h[mm])-1):
                        ###points_co=points_co+' '
                ####print(points_co)
                ###coord_text.set('points',points_co)

                ###for j in range(len(all_found_texline_polygons_h[mm])):

                    ###textline=ET.SubElement(textregion, 'TextLine')

                    ###textline.set('id','l'+str(id_indexer_l))

                    ###id_indexer_l+=1


                    ###coord = ET.SubElement(textline, 'Coords')

                    ###texteq=ET.SubElement(textline, 'TextEquiv')

                    ###uni=ET.SubElement(texteq, 'Unicode')
                    ###uni.text = ' ' 

                    ####points = ET.SubElement(coord, 'Points') 

                    ###points_co=''
                    ###for l in range(len(all_found_texline_polygons_h[mm][j])):
                        ####point = ET.SubElement(coord, 'Point') 



                        ####point.set('x',str(found_polygons[j][l][0]))  
                        ####point.set('y',str(found_polygons[j][l][1]))
                        ###if len(all_found_texline_polygons_h[mm][j][l])==2:
                            ###points_co=points_co+str( int( (all_found_texline_polygons_h[mm][j][l][0] +page_coord[2]
                                                    ###+all_box_coord_h[mm][2])/self.scale_x) )
                            ###points_co=points_co+','
                            ###points_co=points_co+str( int( (all_found_texline_polygons_h[mm][j][l][1] +page_coord[0]
                                                    ###+all_box_coord_h[mm][0])/self.scale_y) )
                        ###else:
                            ###points_co=points_co+str( int( ( all_found_texline_polygons_h[mm][j][l][0][0] +page_coord[2]
                                                    ###+all_box_coord_h[mm][2])/self.scale_x ) )
                            ###points_co=points_co+','
                            ###points_co=points_co+str( int( ( all_found_texline_polygons_h[mm][j][l][0][1] +page_coord[0]
                                                    ###+all_box_coord_h[mm][0])/self.scale_y) ) 

                        ###if l<(len(all_found_texline_polygons_h[mm][j])-1):
                            ###points_co=points_co+' '
                    ####print(points_co)
                    ###coord.set('points',points_co)

                ###texteqreg=ET.SubElement(textregion, 'TextEquiv')

                ###unireg=ET.SubElement(texteqreg, 'Unicode')
                ###unireg.text = ' ' 
                
        try:
            for mm in range(len(found_polygons_text_region_img)):
                textregion=ET.SubElement(page, 'ImageRegion')

                textregion.set('id','r'+str(id_indexer))
                id_indexer+=1


                coord_text = ET.SubElement(textregion, 'Coords')

                points_co=''
                for lmm in range(len(found_polygons_text_region_img[mm])):

                    if len(found_polygons_text_region_img[mm][lmm])==2:
                        points_co=points_co+str( int( (found_polygons_text_region_img[mm][lmm][0] )/self.scale_x ) )
                        points_co=points_co+','
                        points_co=points_co+str( int( (found_polygons_text_region_img[mm][lmm][1] )/self.scale_y ) )
                    else:
                        points_co=points_co+str( int((found_polygons_text_region_img[mm][lmm][0][0] )/self.scale_x) )
                        points_co=points_co+','
                        points_co=points_co+str( int((found_polygons_text_region_img[mm][lmm][0][1] )/self.scale_y) )

                    if lmm<(len(found_polygons_text_region_img[mm])-1):
                        points_co=points_co+' '
                    
                    
                coord_text.set('points',points_co)
        except:
            pass


        ####try:
            ####for mm in range(len(found_polygons_tables)):
                ####textregion=ET.SubElement(page, 'TableRegion')

                ####textregion.set('id','r'+str(id_indexer))
                ####id_indexer+=1


                ####coord_text = ET.SubElement(textregion, 'Coords')

                ####points_co=''
                ####for lmm in range(len(found_polygons_tables[mm])):

                    ####if len(found_polygons_tables[mm][lmm])==2:
                        ####points_co=points_co+str( int( (found_polygons_tables[mm][lmm][0] +page_coord[2])/self.scale_x ) )
                        ####points_co=points_co+','
                        ####points_co=points_co+str( int( (found_polygons_tables[mm][lmm][1] +page_coord[0])/self.scale_y ) )
                    ####else:
                        ####points_co=points_co+str( int((found_polygons_tables[mm][lmm][0][0] +page_coord[2])/self.scale_x) )
                        ####points_co=points_co+','
                        ####points_co=points_co+str( int((found_polygons_tables[mm][lmm][0][1] +page_coord[0])/self.scale_y) )

                    ####if lmm<(len(found_polygons_tables[mm])-1):
                        ####points_co=points_co+' '
                    
                    
                ####coord_text.set('points',points_co)
        ####except:
            ####pass
        """
        
        try:
            for mm in range(len(found_polygons_drop_capitals)):
                textregion=ET.SubElement(page, 'DropCapitals')

                textregion.set('id','r'+str(id_indexer))
                id_indexer+=1


                coord_text = ET.SubElement(textregion, 'Coords')

                points_co=''
                for lmm in range(len(found_polygons_drop_capitals[mm])):

                    if len(found_polygons_drop_capitals[mm][lmm])==2:
                        points_co=points_co+str( int( (found_polygons_drop_capitals[mm][lmm][0] +page_coord[2])/self.scale_x ) )
                        points_co=points_co+','
                        points_co=points_co+str( int( (found_polygons_drop_capitals[mm][lmm][1] +page_coord[0])/self.scale_y ) )
                    else:
                        points_co=points_co+str( int((found_polygons_drop_capitals[mm][lmm][0][0] +page_coord[2])/self.scale_x) )
                        points_co=points_co+','
                        points_co=points_co+str( int((found_polygons_drop_capitals[mm][lmm][0][1] +page_coord[0])/self.scale_y) )

                    if lmm<(len(found_polygons_drop_capitals[mm])-1):
                        points_co=points_co+' '
                    
                    
                coord_text.set('points',points_co)
        except:
            pass
        """

            
        #print(dir_of_image)
        print(self.f_name)
        #print(os.path.join(dir_of_image, self.f_name) + ".xml")
        tree = ET.ElementTree(data)
        tree.write(os.path.join(dir_of_image, self.f_name) + ".xml")
        cv2.imwrite(os.path.join(dir_of_image, self.f_name) + ".tif",self.image_org)
 
    def deskew_region_prediction(self,regions_prediction, slope):
        image_regions_deskewd=np.zeros(regions_prediction[:,:].shape)
        for ind in np.unique(regions_prediction[:,:]):
            interest_reg=(regions_prediction[:,:]==ind)*1
            interest_reg=interest_reg.astype(np.uint8)
            deskewed_new=self.rotate_image(interest_reg,slope)
            deskewed_new=deskewed_new[:,:]
            deskewed_new[deskewed_new!=0]=ind

            image_regions_deskewd=image_regions_deskewd+deskewed_new
        return image_regions_deskewd
    

    def deskew_erarly(self,textline_mask):
        textline_mask_org=np.copy(textline_mask)
        #print(textline_mask.shape,np.unique(textline_mask),'hizzzzz')
        #slope_new=0#deskew_images(img_patch)
        
        textline_mask=np.repeat(textline_mask[:, :, np.newaxis], 3, axis=2)*255

        textline_mask=textline_mask.astype(np.uint8)
        kernel = np.ones((5,5),np.uint8)


        imgray = cv2.cvtColor(textline_mask, cv2.COLOR_BGR2GRAY)


        ret, thresh = cv2.threshold(imgray, 0, 255, 0)


        contours,hirarchy=cv2.findContours(thresh.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        #print(hirarchy)


        commenst_contours=self.filter_contours_area_of_image(thresh,contours,hirarchy,max_area=0.01,min_area=0.003)
        main_contours=self.filter_contours_area_of_image(thresh,contours,hirarchy,max_area=1,min_area=0.003)
        interior_contours=self.filter_contours_area_of_image_interiors(thresh,contours,hirarchy,max_area=1,min_area=0)

        img_comm=np.zeros(thresh.shape)
        img_comm_in=cv2.fillPoly(img_comm, pts =main_contours, color=(255,255,255))
        ###img_comm_in=cv2.fillPoly(img_comm, pts =interior_contours, color=(0,0,0))


        img_comm_in=np.repeat(img_comm_in[:, :, np.newaxis], 3, axis=2)
        img_comm_in=img_comm_in.astype(np.uint8)

        imgray = cv2.cvtColor(img_comm_in, cv2.COLOR_BGR2GRAY)
        ##imgray = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


        ##mask = cv2.inRange(imgray, lower_blue, upper_blue)
        ret, thresh = cv2.threshold(imgray, 0, 255, 0)
        #print(np.unique(mask))
        ##ret, thresh = cv2.threshold(imgray, 0, 255, 0)

        ##plt.imshow(thresh)
        ##plt.show()

        contours,hirarchy=cv2.findContours(thresh.copy(), cv2.cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


        areas=[cv2.contourArea(contours[jj]) for jj in range(len(contours))]

        median_area=np.mean(areas)
        contours_slope=contours#self.find_polugons_size_filter(contours,median_area=median_area,scaler_up=100,scaler_down=0.5)

        if len(contours_slope)>0:
            for jv in range(len(contours_slope)):
                new_poly=list(contours_slope[jv])
                if jv==0:
                    merged_all=new_poly
                else:
                    merged_all=merged_all+new_poly


            merge=np.array(merged_all)


            img_in=np.zeros(textline_mask.shape)
            img_p_in=cv2.fillPoly(img_in, pts =[merge], color=(255,255,255))
            
            ##plt.imshow(img_p_in)
            ##plt.show()
            

            
            rect = cv2.minAreaRect(merge)

            box = cv2.boxPoints(rect)

            box = np.int0(box)


            indexes=[0,1,2,3]
            x_list=box[:,0]
            y_list=box[:,1]



            index_y_sort=np.argsort(y_list)

            index_upper_left=index_y_sort[np.argmin(x_list[index_y_sort[0:2]])]
            index_upper_right=index_y_sort[np.argmax(x_list[index_y_sort[0:2]])]

            index_lower_left=index_y_sort[np.argmin(x_list[index_y_sort[2:]]) +2]
            index_lower_right=index_y_sort[np.argmax(x_list[index_y_sort[2:]])+2]


            alpha1=float(box[index_upper_right][1]-
                        box[index_upper_left][1])/(float(box[index_upper_right][0]-box[index_upper_left][0]))
            alpha2=float(box[index_lower_right][1]-
                        box[index_lower_left][1])/(float(box[index_lower_right][0]-box[index_lower_left][0]))

            slope_true=(alpha1+alpha2)/2.0

            #slope=0#slope_true/np.pi*180
            
            
            #if abs(slope)>=1:
                #slope=0

            #dst=self.rotate_image(textline_mask,slope_true)
            #dst=dst[:,:,0]
            #dst[dst!=0]=1
        image_regions_deskewd=np.zeros(textline_mask_org[:,:].shape)
        for ind in np.unique(textline_mask_org[:,:]):
            interest_reg=(textline_mask_org[:,:]==ind)*1
            interest_reg=interest_reg.astype(np.uint8)
            deskewed_new=self.rotate_image(interest_reg,slope_true)
            deskewed_new=deskewed_new[:,:]
            deskewed_new[deskewed_new!=0]=ind

            image_regions_deskewd=image_regions_deskewd+deskewed_new
        return image_regions_deskewd,slope_true
        
    def return_regions_without_seperators(self,regions_pre):
        kernel = np.ones((5,5),np.uint8)
        regions_without_seperators=( (regions_pre[:,:]!=6) & (regions_pre[:,:]!=0) )*1
        #regions_without_seperators=( (image_regions_eraly_p[:,:,:]!=6) & (image_regions_eraly_p[:,:,:]!=0) & (image_regions_eraly_p[:,:,:]!=5) & (image_regions_eraly_p[:,:,:]!=8) & (image_regions_eraly_p[:,:,:]!=7))*1

        regions_without_seperators=regions_without_seperators.astype(np.uint8)

        regions_without_seperators = cv2.erode(regions_without_seperators,kernel,iterations = 6)

        return regions_without_seperators
    
    def return_regions_without_seperators_new(self,regions_pre,regions_only_text):
        kernel = np.ones((5,5),np.uint8)
        
        regions_without_seperators=( (regions_pre[:,:]!=6) & (regions_pre[:,:]!=0)  & (regions_pre[:,:]!=1) & (regions_pre[:,:]!=2))*1
        
        #plt.imshow(regions_without_seperators)
        #plt.show()
        
        regions_without_seperators_n=( (regions_without_seperators[:,:]==1) | (regions_only_text[:,:]==1) )*1
        
        #regions_without_seperators=( (image_regions_eraly_p[:,:,:]!=6) & (image_regions_eraly_p[:,:,:]!=0) & (image_regions_eraly_p[:,:,:]!=5) & (image_regions_eraly_p[:,:,:]!=8) & (image_regions_eraly_p[:,:,:]!=7))*1
        
        regions_without_seperators_n=regions_without_seperators_n.astype(np.uint8)

        regions_without_seperators_n = cv2.erode(regions_without_seperators_n,kernel,iterations = 6)

        return regions_without_seperators_n
    def image_change_background_pixels_to_zero(self,image_page):
        image_back_zero=np.zeros((image_page.shape[0],image_page.shape[1]))
        image_back_zero[:,:]=image_page[:,:,0]
        image_back_zero[:,:][image_back_zero[:,:]==0]=-255
        image_back_zero[:,:][image_back_zero[:,:]==255]=0
        image_back_zero[:,:][image_back_zero[:,:]==-255]=255
        return image_back_zero
    
    def find_num_col_only_image(self,regions_without_seperators,multiplier=3.8):
        regions_without_seperators_0=regions_without_seperators[:,:].sum(axis=0)
        
        ##plt.plot(regions_without_seperators_0)
        ##plt.show()

        sigma_=15


        meda_n_updown=regions_without_seperators_0[len(regions_without_seperators_0)::-1]

        first_nonzero=(next((i for i, x in enumerate(regions_without_seperators_0) if x), 0))
        last_nonzero=(next((i for i, x in enumerate(meda_n_updown) if x), 0))

        last_nonzero=len(regions_without_seperators_0)-last_nonzero


        y=regions_without_seperators_0#[first_nonzero:last_nonzero]

        y_help=np.zeros(len(y)+20)

        y_help[10:len(y)+10]=y

        x=np.array( range(len(y)) )




        zneg_rev=-y_help+np.max(y_help)

        zneg=np.zeros(len(zneg_rev)+20)

        zneg[10:len(zneg_rev)+10]=zneg_rev

        z=gaussian_filter1d(y, sigma_)
        zneg= gaussian_filter1d(zneg, sigma_)


        peaks_neg, _ = find_peaks(zneg, height=0)
        peaks, _ = find_peaks(z, height=0)
        
        
        peaks_neg=peaks_neg-10-10
        
        peaks_neg_org=np.copy(peaks_neg)


        peaks_neg=peaks_neg[(peaks_neg>first_nonzero) & (peaks_neg<last_nonzero)]
        
        peaks=peaks[(peaks>.09*regions_without_seperators.shape[1]) & (peaks<0.91*regions_without_seperators.shape[1])]
        
        peaks_neg=peaks_neg[ (peaks_neg>500) & (peaks_neg< (regions_without_seperators.shape[1]-500) ) ]
        #print(peaks)
        interest_pos=z[peaks]
        
        interest_pos=interest_pos[interest_pos>10]

        interest_neg=z[peaks_neg]
        min_peaks_pos=np.mean(interest_pos)#np.min(interest_pos)
        min_peaks_neg=0#np.min(interest_neg)
        
        #$print(min_peaks_pos)
        dis_talaei=(min_peaks_pos-min_peaks_neg)/multiplier
        #print(interest_pos)
        grenze=min_peaks_pos-dis_talaei#np.mean(y[peaks_neg[0]:peaks_neg[len(peaks_neg)-1]])-np.std(y[peaks_neg[0]:peaks_neg[len(peaks_neg)-1]])/2.0

        interest_neg_fin=interest_neg[(interest_neg<grenze)]
        peaks_neg_fin=peaks_neg[(interest_neg<grenze)]

        num_col=(len(interest_neg_fin))+1

        p_l=0
        p_u=len(y)-1
        p_m=int(len(y)/2.)
        p_g_l=int(len(y)/3.)
        p_g_u=len(y)-int(len(y)/3.)

        if num_col==3:
            if (peaks_neg_fin[0]>p_g_u and peaks_neg_fin[1]>p_g_u) or (peaks_neg_fin[0]<p_g_l and peaks_neg_fin[1]<p_g_l ) or (peaks_neg_fin[0]<p_m and peaks_neg_fin[1]<p_m ) or (peaks_neg_fin[0]>p_m and peaks_neg_fin[1]>p_m ):
                num_col=1
            else:
                pass

        if num_col==2:
            if (peaks_neg_fin[0]>p_g_u) or (peaks_neg_fin[0]<p_g_l):
                num_col=1
            else:
                pass

        diff_peaks=np.abs( np.diff(peaks_neg_fin) )
        

        cut_off=400
        peaks_neg_true=[]
        forest=[]
        
        for i in range(len(peaks_neg_fin)):
            if i==0:
                forest.append(peaks_neg_fin[i])
            if i<(len(peaks_neg_fin)-1):
                if diff_peaks[i]<=cut_off:
                    forest.append(peaks_neg_fin[i+1])
                if diff_peaks[i]>cut_off:
                    #print(forest[np.argmin(z[forest]) ] )
                    if not self.isNaN(forest[np.argmin(z[forest]) ]):
                        peaks_neg_true.append(forest[np.argmin(z[forest]) ])
                    forest=[]
                    forest.append(peaks_neg_fin[i+1])
            if i==(len(peaks_neg_fin)-1):
                #print(print(forest[np.argmin(z[forest]) ] ))
                if not self.isNaN(forest[np.argmin(z[forest]) ]):
                    peaks_neg_true.append(forest[np.argmin(z[forest]) ])
        
        
        
        
        num_col=(len(peaks_neg_true))+1
        p_l=0
        p_u=len(y)-1
        p_m=int(len(y)/2.)
        p_quarter=int(len(y)/4.)
        p_g_l=int(len(y)/3.)
        p_g_u=len(y)-int(len(y)/3.)
        
        p_u_quarter=len(y)-p_quarter

        if num_col==3:
            if (peaks_neg_true[0]>p_g_u and peaks_neg_true[1]>p_g_u) or (peaks_neg_true[0]<p_g_l and peaks_neg_true[1]<p_g_l ) or (peaks_neg_true[0]<p_m and peaks_neg_true[1]<p_m ) or (peaks_neg_true[0]>p_m and peaks_neg_true[1]>p_m ):
                num_col=1
                peaks_neg_true=[]
            elif (peaks_neg_true[0]<p_g_u and peaks_neg_true[0]>p_g_l) and (peaks_neg_true[1]>p_u_quarter):
                peaks_neg_true=[ peaks_neg_true[0] ]
            elif (peaks_neg_true[1]<p_g_u and peaks_neg_true[1]>p_g_l) and (peaks_neg_true[0]<p_quarter):
                peaks_neg_true=[ peaks_neg_true[1] ]
            else:
                pass

        if num_col==2:
            if (peaks_neg_true[0]>p_g_u) or (peaks_neg_true[0]<p_g_l):
                num_col=1
                peaks_neg_true=[]
        
        if num_col==4:
            if len(np.array(peaks_neg_true)[np.array(peaks_neg_true)<p_g_l])==2 or len( np.array(peaks_neg_true)[np.array(peaks_neg_true)>(len(y)-p_g_l)] )==2:
                num_col=1
                peaks_neg_true=[]
            else:
                pass
            
            
        #no deeper hill around found hills
        
        peaks_fin_true=[]
        for i in range(len(peaks_neg_true)):
            hill_main=peaks_neg_true[i]
            #deep_depth=z[peaks_neg]
            hills_around=peaks_neg_org[( (peaks_neg_org>hill_main) & (peaks_neg_org<=hill_main+400) ) | ( (peaks_neg_org<hill_main) & (peaks_neg_org>=hill_main-400) )]
            deep_depth_around=z[hills_around]
            
            #print(hill_main,z[hill_main],hills_around,deep_depth_around,'manoooo')
            try:
                if np.min(deep_depth_around)<z[hill_main]:
                    pass
                else:
                    peaks_fin_true.append(hill_main)
            except:
                pass
        
        
        diff_peaks_annormal=diff_peaks[diff_peaks<360]
        

        
        if len(diff_peaks_annormal)>0:
            arg_help=np.array(range(len(diff_peaks)))
            arg_help_ann=arg_help[diff_peaks<360]
            
            peaks_neg_fin_new=[]
            
            for ii in range(len(peaks_neg_fin)):
                if ii in arg_help_ann:
                    arg_min=np.argmin([interest_neg_fin[ii],interest_neg_fin[ii+1] ] )
                    if arg_min==0:
                        peaks_neg_fin_new.append( peaks_neg_fin[ii])
                    else:
                        peaks_neg_fin_new.append( peaks_neg_fin[ii+1])
                                    
                
                elif (ii-1) in arg_help_ann:
                    pass
                else:
                    peaks_neg_fin_new.append(peaks_neg_fin[ii] )
        else:
            peaks_neg_fin_new=peaks_neg_fin
            
        # sometime pages with one columns gives also some negative peaks. delete those peaks
        param=z[peaks_neg_true]/float(min_peaks_pos)*100
        
        if len(param[param<=41])==0:
            peaks_neg_true=[]
            
        
        

            
        
        return len(peaks_fin_true), peaks_fin_true

    def return_hor_spliter_by_index_for_without_verticals(self,peaks_neg_fin_t,x_min_hor_some,x_max_hor_some):
        #print(peaks_neg_fin_t,x_min_hor_some,x_max_hor_some)
        arg_min_hor_sort=np.argsort(x_min_hor_some)
        x_min_hor_some_sort=np.sort(x_min_hor_some)
        x_max_hor_some_sort=x_max_hor_some[arg_min_hor_sort]
        
        arg_minmax=np.array(range(len(peaks_neg_fin_t)))
        indexer_lines=[]
        indexes_to_delete=[]
        indexer_lines_deletions_len=[]
        indexr_uniq_ind=[]
        for i in range(len(x_min_hor_some_sort)):
            min_h=peaks_neg_fin_t-x_min_hor_some_sort[i]
            

            max_h=peaks_neg_fin_t-x_max_hor_some_sort[i]

            
            min_h[0]=min_h[0]#+20
            max_h[len(max_h)-1]=max_h[len(max_h)-1]-20


            min_h_neg=arg_minmax[(min_h<0)]
            min_h_neg_n=min_h[min_h<0]
            min_h_neg=[ min_h_neg[np.argmax(min_h_neg_n)] ] 
            
            max_h_neg=arg_minmax[(max_h>0)]
            max_h_neg_n=max_h[max_h>0]
            max_h_neg=[ max_h_neg[np.argmin(max_h_neg_n)] ]
            

            if len(min_h_neg)>0 and len(max_h_neg)>0:
                deletions=list(range(min_h_neg[0]+1,max_h_neg[0]))
                unique_delets_int=[]
                #print(deletions,len(deletions),'delii')
                if len(deletions)>0:
                    
                    for j in range(len(deletions)):
                        indexes_to_delete.append(deletions[j])
                        #print(deletions,indexes_to_delete,'badiii')
                        unique_delets=np.unique(indexes_to_delete)
                        #print(min_h_neg[0],unique_delets)
                        unique_delets_int=unique_delets[unique_delets<min_h_neg[0]]
                        
                    indexer_lines_deletions_len.append(len(deletions))
                    indexr_uniq_ind.append([deletions])
                        
                else:
                    indexer_lines_deletions_len.append(0)
                    indexr_uniq_ind.append(-999)
                    
                index_line_true=min_h_neg[0]-len(unique_delets_int)
                #print(index_line_true)
                if index_line_true>0 and min_h_neg[0]>=2:
                    index_line_true=index_line_true
                else:
                    index_line_true=min_h_neg[0]

                indexer_lines.append(index_line_true)
                
                if len(unique_delets_int)>0:
                    for dd in range(len(unique_delets_int)):
                        indexes_to_delete.append(unique_delets_int[dd])
            else:
                indexer_lines.append(-999)
                indexer_lines_deletions_len.append(-999)
                indexr_uniq_ind.append(-999)
        
        peaks_true=[]
        for m in range(len(peaks_neg_fin_t)):
            if m in indexes_to_delete:
                pass
            else:
                peaks_true.append(peaks_neg_fin_t[m])
        return indexer_lines,peaks_true,arg_min_hor_sort,indexer_lines_deletions_len,indexr_uniq_ind
    
    def find_num_col(self,regions_without_seperators,multiplier=3.8):
        regions_without_seperators_0=regions_without_seperators[:,:].sum(axis=0)
        
        ##plt.plot(regions_without_seperators_0)
        ##plt.show()

        sigma_=35


        meda_n_updown=regions_without_seperators_0[len(regions_without_seperators_0)::-1]

        first_nonzero=(next((i for i, x in enumerate(regions_without_seperators_0) if x), 0))
        last_nonzero=(next((i for i, x in enumerate(meda_n_updown) if x), 0))

        #print(last_nonzero)
        #print(isNaN(last_nonzero))
        #last_nonzero=0#halalikh
        last_nonzero=len(regions_without_seperators_0)-last_nonzero


        y=regions_without_seperators_0#[first_nonzero:last_nonzero]

        y_help=np.zeros(len(y)+20)

        y_help[10:len(y)+10]=y

        x=np.array( range(len(y)) )




        zneg_rev=-y_help+np.max(y_help)

        zneg=np.zeros(len(zneg_rev)+20)

        zneg[10:len(zneg_rev)+10]=zneg_rev

        z=gaussian_filter1d(y, sigma_)
        zneg= gaussian_filter1d(zneg, sigma_)


        peaks_neg, _ = find_peaks(zneg, height=0)
        peaks, _ = find_peaks(z, height=0)

        peaks_neg=peaks_neg-10-10


        last_nonzero=last_nonzero-100
        first_nonzero=first_nonzero+200

        peaks_neg=peaks_neg[(peaks_neg>first_nonzero) & (peaks_neg<last_nonzero)]
        
        peaks=peaks[(peaks>.06*regions_without_seperators.shape[1]) & (peaks<0.94*regions_without_seperators.shape[1])]
        peaks_neg=peaks_neg[ (peaks_neg>370) & (peaks_neg< (regions_without_seperators.shape[1]-370) ) ]

        #print(peaks)
        interest_pos=z[peaks]
        
        interest_pos=interest_pos[interest_pos>10]

        interest_neg=z[peaks_neg]
        min_peaks_pos=np.min(interest_pos)
        min_peaks_neg=0#np.min(interest_neg)
        
        #$print(min_peaks_pos)
        dis_talaei=(min_peaks_pos-min_peaks_neg)/multiplier
        #print(interest_pos)
        grenze=min_peaks_pos-dis_talaei#np.mean(y[peaks_neg[0]:peaks_neg[len(peaks_neg)-1]])-np.std(y[peaks_neg[0]:peaks_neg[len(peaks_neg)-1]])/2.0

        interest_neg_fin=interest_neg[(interest_neg<grenze)]
        peaks_neg_fin=peaks_neg[(interest_neg<grenze)]
        interest_neg_fin=interest_neg[(interest_neg<grenze)]

        num_col=(len(interest_neg_fin))+1


        p_l=0
        p_u=len(y)-1
        p_m=int(len(y)/2.)
        p_g_l=int(len(y)/3.)
        p_g_u=len(y)-int(len(y)/3.)

        if num_col==3:
            if (peaks_neg_fin[0]>p_g_u and peaks_neg_fin[1]>p_g_u) or (peaks_neg_fin[0]<p_g_l and peaks_neg_fin[1]<p_g_l ) or ((peaks_neg_fin[0]+200)<p_m and peaks_neg_fin[1]<p_m ) or ((peaks_neg_fin[0]-200)>p_m and peaks_neg_fin[1]>p_m ):
                num_col=1
                peaks_neg_fin=[]
            else:
                pass

        if num_col==2:
            if (peaks_neg_fin[0]>p_g_u) or (peaks_neg_fin[0]<p_g_l):
                num_col=1
                peaks_neg_fin=[]
            else:
                pass




        

        
        diff_peaks=np.abs( np.diff(peaks_neg_fin) )
        
        cut_off=400
        peaks_neg_true=[]
        forest=[]
        
        for i in range(len(peaks_neg_fin)):
            if i==0:
                forest.append(peaks_neg_fin[i])
            if i<(len(peaks_neg_fin)-1):
                if diff_peaks[i]<=cut_off:
                    forest.append(peaks_neg_fin[i+1])
                if diff_peaks[i]>cut_off:
                    #print(forest[np.argmin(z[forest]) ] )
                    if not self.isNaN(forest[np.argmin(z[forest]) ]):
                        peaks_neg_true.append(forest[np.argmin(z[forest]) ])
                    forest=[]
                    forest.append(peaks_neg_fin[i+1])
            if i==(len(peaks_neg_fin)-1):
                #print(print(forest[np.argmin(z[forest]) ] ))
                if not self.isNaN(forest[np.argmin(z[forest]) ]):
                    peaks_neg_true.append(forest[np.argmin(z[forest]) ])
        
        
        num_col=(len(peaks_neg_true))+1
        p_l=0
        p_u=len(y)-1
        p_m=int(len(y)/2.)
        p_quarter=int(len(y)/4.)
        p_g_l=int(len(y)/3.)
        p_g_u=len(y)-int(len(y)/3.)
        
        p_u_quarter=len(y)-p_quarter

        if num_col==3:
            if (peaks_neg_true[0]>p_g_u and peaks_neg_true[1]>p_g_u) or (peaks_neg_true[0]<p_g_l and peaks_neg_true[1]<p_g_l ) or (peaks_neg_true[0]<p_m and (peaks_neg_true[1]+200)<p_m ) or ( (peaks_neg_true[0]-200)>p_m and peaks_neg_true[1]>p_m ):
                num_col=1
                peaks_neg_true=[]
            elif (peaks_neg_true[0]<p_g_u and peaks_neg_true[0]>p_g_l) and (peaks_neg_true[1]>p_u_quarter):
                peaks_neg_true=[ peaks_neg_true[0] ]
            elif (peaks_neg_true[1]<p_g_u and peaks_neg_true[1]>p_g_l) and (peaks_neg_true[0]<p_quarter):
                peaks_neg_true=[ peaks_neg_true[1] ]
            else:
                pass

        if num_col==2:
            if (peaks_neg_true[0]>p_g_u) or (peaks_neg_true[0]<p_g_l):
                num_col=1
                peaks_neg_true=[]
            else:
                pass

        diff_peaks_annormal=diff_peaks[diff_peaks<360]
        

        
        if len(diff_peaks_annormal)>0:
            arg_help=np.array(range(len(diff_peaks)))
            arg_help_ann=arg_help[diff_peaks<360]
            
            peaks_neg_fin_new=[]
            
            for ii in range(len(peaks_neg_fin)):
                if ii in arg_help_ann:
                    arg_min=np.argmin([interest_neg_fin[ii],interest_neg_fin[ii+1] ] )
                    if arg_min==0:
                        peaks_neg_fin_new.append( peaks_neg_fin[ii])
                    else:
                        peaks_neg_fin_new.append( peaks_neg_fin[ii+1])
                                    
                
                elif (ii-1) in arg_help_ann:
                    pass
                else:
                    peaks_neg_fin_new.append(peaks_neg_fin[ii] )
        else:
            peaks_neg_fin_new=peaks_neg_fin
            
                    
        #plt.plot(gaussian_filter1d(y, sigma_))
        #plt.plot(peaks_neg_true,z[peaks_neg_true],'*')
        #plt.plot([0,len(y)], [grenze,grenze])
        #plt.show()
            
        #print(peaks_neg_fin_new)
        return len(peaks_neg_true), peaks_neg_true
    
    def find_new_features_of_contoures(self,contours_main):
        
        areas_main=np.array([cv2.contourArea(contours_main[j]) for j in range(len(contours_main))])
        M_main=[cv2.moments(contours_main[j]) for j in range(len(contours_main))]
        cx_main=[(M_main[j]['m10']/(M_main[j]['m00']+1e-32)) for j in range(len(M_main))]
        cy_main=[(M_main[j]['m01']/(M_main[j]['m00']+1e-32)) for j in range(len(M_main))]
        x_min_main=np.array([np.min(contours_main[j][:,0,0]) for j in range(len(contours_main))])
        x_max_main=np.array([np.max(contours_main[j][:,0,0]) for j in range(len(contours_main))])

        y_min_main=np.array([np.min(contours_main[j][:,0,1]) for j in range(len(contours_main))])
        y_max_main=np.array([np.max(contours_main[j][:,0,1]) for j in range(len(contours_main))])


        
        #dis_x=np.abs(x_max_main-x_min_main)
        
        return cx_main,cy_main ,x_min_main , x_max_main, y_min_main ,y_max_main
    def return_points_with_boundies(self,peaks_neg_fin,first_point, last_point):
        peaks_neg_tot=[]
        peaks_neg_tot.append(first_point)
        for ii in range(len(peaks_neg_fin)):
            peaks_neg_tot.append(peaks_neg_fin[ii])
        peaks_neg_tot.append(last_point)
        return peaks_neg_tot
    def contours_in_same_horizon(self,cy_main_hor):
        X1=np.zeros((len(cy_main_hor),len(cy_main_hor)))
        X2=np.zeros((len(cy_main_hor),len(cy_main_hor)))

        X1[0::1,:]=cy_main_hor[:]
        X2=X1.T

        X_dif=np.abs(X2-X1)
        args_help=np.array(range(len(cy_main_hor)))
        all_args=[]
        for i in range(len(cy_main_hor)):
            list_h=list(args_help[X_dif[i,:]<=20] )
            list_h.append(i)
            if len(list_h)>1:
                all_args.append(list( set(list_h)  ))
        return np.unique(all_args)
    
    def return_boxes_of_images_by_order_of_reading_without_seperators(self,spliter_y_new,image_p_rev,regions_without_seperators,matrix_of_lines_ch,seperators_closeup_n):
        
        boxes=[]


        # here I go through main spliters and i do check whether a vertical seperator there is. If so i am searching for \
        # holes in the text and also finding spliter which covers more than one columns.
        for i in range(len(spliter_y_new)-1):
            #print(spliter_y_new[i],spliter_y_new[i+1])
            matrix_new=matrix_of_lines_ch[:,:][ (matrix_of_lines_ch[:,6]> spliter_y_new[i] ) & (matrix_of_lines_ch[:,7]< spliter_y_new[i+1] )  ] 
            #print(len( matrix_new[:,9][matrix_new[:,9]==1] ))
            
            #print(matrix_new[:,8][matrix_new[:,9]==1],'gaddaaa')
            
            # check to see is there any vertical seperator to find holes.
            if np.abs(spliter_y_new[i+1]-spliter_y_new[i])>1./3.*regions_without_seperators.shape[0]:#len( matrix_new[:,9][matrix_new[:,9]==1] )>0 and np.max(matrix_new[:,8][matrix_new[:,9]==1])>=0.1*(np.abs(spliter_y_new[i+1]-spliter_y_new[i] )):
                
                #org_img_dichte=-gaussian_filter1d(( image_page[int(spliter_y_new[i]):int(spliter_y_new[i+1]),:,0]/255.).sum(axis=0) ,30)
                #org_img_dichte=org_img_dichte-np.min(org_img_dichte)
                ##plt.figure(figsize=(20,20))
                ##plt.plot(org_img_dichte)      
                ##plt.show()
                ###find_num_col_both_layout_and_org(regions_without_seperators,image_page[int(spliter_y_new[i]):int(spliter_y_new[i+1]),:,:],7.)
                
                num_col, peaks_neg_fin=self.find_num_col_only_image(image_p_rev[int(spliter_y_new[i]):int(spliter_y_new[i+1]),:],multiplier=2.4)

                #num_col, peaks_neg_fin=find_num_col(regions_without_seperators[int(spliter_y_new[i]):int(spliter_y_new[i+1]),:],multiplier=7.0)
                x_min_hor_some=matrix_new[:,2][ (matrix_new[:,9]==0) ]
                x_max_hor_some=matrix_new[:,3][ (matrix_new[:,9]==0) ]
                cy_hor_some=matrix_new[:,5][ (matrix_new[:,9]==0) ]
                arg_org_hor_some=matrix_new[:,0][ (matrix_new[:,9]==0) ]
                

                
                peaks_neg_tot=self.return_points_with_boundies(peaks_neg_fin,0, seperators_closeup_n[:,:,0].shape[1])
                
                start_index_of_hor,newest_peaks,arg_min_hor_sort,lines_length_dels,lines_indexes_deleted=self.return_hor_spliter_by_index_for_without_verticals(peaks_neg_tot,x_min_hor_some,x_max_hor_some)
                
                arg_org_hor_some_sort=arg_org_hor_some[arg_min_hor_sort]
                

                start_index_of_hor_with_subset=[start_index_of_hor[vij] for vij in range(len(start_index_of_hor)) if lines_length_dels[vij]>0 ]#start_index_of_hor[lines_length_dels>0]
                arg_min_hor_sort_with_subset=[arg_min_hor_sort[vij] for vij in range(len(start_index_of_hor)) if lines_length_dels[vij]>0 ]
                lines_indexes_deleted_with_subset=[lines_indexes_deleted[vij] for vij in range(len(start_index_of_hor)) if lines_length_dels[vij]>0 ]
                lines_length_dels_with_subset=[lines_length_dels[vij] for vij in range(len(start_index_of_hor)) if lines_length_dels[vij]>0 ]
                
                arg_org_hor_some_sort_subset=[arg_org_hor_some_sort[vij] for vij in range(len(start_index_of_hor)) if lines_length_dels[vij]>0 ]

                #arg_min_hor_sort_with_subset=arg_min_hor_sort[lines_length_dels>0]
                #lines_indexes_deleted_with_subset=lines_indexes_deleted[lines_length_dels>0]
                #lines_length_dels_with_subset=lines_length_dels[lines_length_dels>0]
                
                
                

                #print(len(arg_min_hor_sort),len(arg_org_hor_some_sort),'vizzzzzz')
                
                
                
                vahid_subset=np.zeros((len(start_index_of_hor_with_subset),len(start_index_of_hor_with_subset)))-1
                for kkk1 in range(len(start_index_of_hor_with_subset)):
                    
                    print(lines_indexes_deleted,'hiii')
                    index_del_sub=np.unique(lines_indexes_deleted_with_subset[kkk1])
                    
                    for kkk2 in range(len(start_index_of_hor_with_subset)):
                        
                        if set(lines_indexes_deleted_with_subset[kkk2][0]) < set(lines_indexes_deleted_with_subset[kkk1][0]):
                            vahid_subset[kkk1,kkk2]=kkk1
                        else:
                            pass
                    #print(set(lines_indexes_deleted[kkk2][0]), set(lines_indexes_deleted[kkk1][0]))
                    
                
                
                # check the len of matrix if it has no length means that there is no spliter at all
                
                if len(vahid_subset>0):
                    #print('hihoo')


                    # find parenets args
                    line_int=np.zeros(vahid_subset.shape[0])
                    
                    childs_id=[]
                    arg_child=[]
                    for li in range(vahid_subset.shape[0]):
                        if np.all(vahid_subset[:,li]==-1):
                            line_int[li]=-1
                        else:
                            line_int[li]=1
                            
                            
                            #childs_args_in=[ idd for idd in range(vahid_subset.shape[0]) if vahid_subset[idd,li]!=-1]
                            #helpi=[]
                            #for nad in range(len(childs_args_in)):
                            #    helpi.append(arg_min_hor_sort_with_subset[childs_args_in[nad]])
                            
                                
                            arg_child.append(arg_min_hor_sort_with_subset[li] )   
                    
                    
                    

                    arg_parent=[arg_min_hor_sort_with_subset[vij] for vij in range(len(arg_min_hor_sort_with_subset)) if line_int[vij]==-1]
                    start_index_of_hor_parent=[start_index_of_hor_with_subset[vij] for vij in range(len(arg_min_hor_sort_with_subset)) if line_int[vij]==-1]
                    #arg_parent=[lines_indexes_deleted_with_subset[vij] for vij in range(len(arg_min_hor_sort_with_subset)) if line_int[vij]==-1]
                    #arg_parent=[lines_length_dels_with_subset[vij] for vij in range(len(arg_min_hor_sort_with_subset)) if line_int[vij]==-1]
                    

                    #arg_child=[arg_min_hor_sort_with_subset[vij] for vij in range(len(arg_min_hor_sort_with_subset)) if line_int[vij]!=-1]
                    start_index_of_hor_child=[start_index_of_hor_with_subset[vij] for vij in range(len(arg_min_hor_sort_with_subset)) if line_int[vij]!=-1]


                    cy_hor_some_sort=cy_hor_some[arg_parent]
                    

                    newest_y_spliter_tot=[]

                    for tj in range(len(newest_peaks)-1):
                        newest_y_spliter=[]
                        newest_y_spliter.append(spliter_y_new[i])
                        if tj in np.unique(start_index_of_hor_parent):
                            cy_help=np.array(cy_hor_some_sort)[np.array(start_index_of_hor_parent)==tj]
                            cy_help_sort=np.sort(cy_help)

                            #print(tj,cy_hor_some_sort,start_index_of_hor,cy_help,'maashhaha')
                            for mj in range(len(cy_help_sort)):
                                newest_y_spliter.append(cy_help_sort[mj])
                        newest_y_spliter.append(spliter_y_new[i+1])

                        newest_y_spliter_tot.append(newest_y_spliter)
                        

                    
                        
                else:
                    line_int=[]
                    newest_y_spliter_tot=[]

                    for tj in range(len(newest_peaks)-1):
                        newest_y_spliter=[]
                        newest_y_spliter.append(spliter_y_new[i])

                        newest_y_spliter.append(spliter_y_new[i+1])

                        newest_y_spliter_tot.append(newest_y_spliter)
                    

                
                # if line_int is all -1 means that big spliters have no child and we can easily go through
                if np.all(np.array(line_int)==-1):
                    for j in range(len(newest_peaks)-1):
                        newest_y_spliter=newest_y_spliter_tot[j]


                        for n in range(len(newest_y_spliter)-1):
                            #print(j,newest_y_spliter[n],newest_y_spliter[n+1],newest_peaks[j],newest_peaks[j+1],'maaaa')
                            ##plt.imshow(regions_without_seperators[int(newest_y_spliter[n]):int(newest_y_spliter[n+1]),newest_peaks[j]:newest_peaks[j+1]])
                            ##plt.show()

                            #print(matrix_new[:,0][ (matrix_new[:,9]==1 )])
                            for jvt in    matrix_new[:,0][ (matrix_new[:,9]==1 ) & (matrix_new[:,6]> newest_y_spliter[n] ) & (matrix_new[:,7]< newest_y_spliter[n+1] ) & ( ( matrix_new[:,1]) < newest_peaks[j+1] ) & (( matrix_new[:,1])> newest_peaks[j] ) ] :
                                pass

                                ###plot_contour(regions_without_seperators.shape[0],regions_without_seperators.shape[1], contours_lines[int(jvt)])
                            #print(matrix_of_lines_ch[matrix_of_lines_ch[:,9]==1])
                            matrix_new_new=matrix_of_lines_ch[:,:][ (matrix_of_lines_ch[:,9]==1 ) & (matrix_of_lines_ch[:,6]> newest_y_spliter[n] ) & (matrix_of_lines_ch[:,7]< newest_y_spliter[n+1] ) & ( ( matrix_of_lines_ch[:,1]+500) < newest_peaks[j+1] ) & (( matrix_of_lines_ch[:,1]-500)> newest_peaks[j] )] 
                            #print(matrix_new_new,newest_y_spliter[n],newest_y_spliter[n+1],newest_peaks[j],newest_peaks[j+1],'gada')
                            if 1>0:#len( matrix_new_new[:,9][matrix_new_new[:,9]==1] )>0 and np.max(matrix_new_new[:,8][matrix_new_new[:,9]==1])>=0.2*(np.abs(newest_y_spliter[n+1]-newest_y_spliter[n] )):
                                #num_col_sub, peaks_neg_fin_sub=find_num_col(regions_without_seperators[int(newest_y_spliter[n]):int(newest_y_spliter[n+1]),newest_peaks[j]:newest_peaks[j+1]],multiplier=2.3)
                                num_col_sub, peaks_neg_fin_sub=self.find_num_col_only_image(image_p_rev[int(newest_y_spliter[n]):int(newest_y_spliter[n+1]),newest_peaks[j]:newest_peaks[j+1]],multiplier=2.4)
                            else:
                                peaks_neg_fin_sub=[]

                            peaks_sub=[]
                            peaks_sub.append(newest_peaks[j])

                            for kj in range(len(peaks_neg_fin_sub)):
                                peaks_sub.append(peaks_neg_fin_sub[kj]+newest_peaks[j])

                            peaks_sub.append(newest_peaks[j+1])

                            #peaks_sub=return_points_with_boundies(peaks_neg_fin_sub+newest_peaks[j],newest_peaks[j], newest_peaks[j+1])

                            for kh in range(len(peaks_sub)-1):
                                boxes.append([ peaks_sub[kh], peaks_sub[kh+1] ,newest_y_spliter[n],newest_y_spliter[n+1]])
                                
                                
                else:
                    for j in range(len(newest_peaks)-1):
                        newest_y_spliter=newest_y_spliter_tot[j]
                        
                        if j in start_index_of_hor_parent:
                            
                            x_min_ch=x_min_hor_some[arg_child]
                            x_max_ch=x_max_hor_some[arg_child]
                            cy_hor_some_sort_child=cy_hor_some[arg_child]
                            cy_hor_some_sort_child=np.sort(cy_hor_some_sort_child)
                            
                            for n in range(len(newest_y_spliter)-1):
                                
                                cy_child_in=cy_hor_some_sort_child[( cy_hor_some_sort_child>newest_y_spliter[n] ) & ( cy_hor_some_sort_child<newest_y_spliter[n+1] ) ]

                                if len(cy_child_in)>0:
                                    ###num_col_ch, peaks_neg_ch=find_num_col( regions_without_seperators[int(newest_y_spliter[n]):int(newest_y_spliter[n+1]),newest_peaks[j]:newest_peaks[j+1]],multiplier=2.3)
                                    
                                    num_col_ch, peaks_neg_ch=self.find_num_col_only_image(image_p_rev[int(newest_y_spliter[n]):int(newest_y_spliter[n+1]),newest_peaks[j]:newest_peaks[j+1]],multiplier=2.3)
                                    
                                    peaks_neg_ch=peaks_neg_ch[:]+newest_peaks[j]
                                    
                                    peaks_neg_ch_tot=self.return_points_with_boundies(peaks_neg_ch,newest_peaks[j], newest_peaks[j+1])
                                    

                                    ss_in_ch,nst_p_ch,arg_n_ch,lines_l_del_ch,lines_in_del_ch=self.return_hor_spliter_by_index_for_without_verticals(peaks_neg_ch_tot,x_min_ch,x_max_ch)
                                    
                                    newest_y_spliter_ch_tot=[]

                                    for tjj in range(len(nst_p_ch)-1):
                                        newest_y_spliter_new=[]
                                        newest_y_spliter_new.append(newest_y_spliter[n])
                                        if tjj in np.unique(ss_in_ch):
                                            

                                            #print(tj,cy_hor_some_sort,start_index_of_hor,cy_help,'maashhaha')
                                            for mjj in range(len(cy_child_in)):
                                                newest_y_spliter_new.append(cy_child_in[mjj])
                                        newest_y_spliter_new.append(newest_y_spliter[n+1])

                                        newest_y_spliter_ch_tot.append(newest_y_spliter_new)
                                        
                                    
                                    
                                    
                                    for jn in range(len(nst_p_ch)-1):
                                        newest_y_spliter_h=newest_y_spliter_ch_tot[jn]

                                        for nd in range(len(newest_y_spliter_h)-1):

                                            matrix_new_new2=matrix_of_lines_ch[:,:][ (matrix_of_lines_ch[:,9]==1 ) & (matrix_of_lines_ch[:,6]> newest_y_spliter_h[nd] ) & (matrix_of_lines_ch[:,7]< newest_y_spliter_h[nd+1] ) & ( ( matrix_of_lines_ch[:,1]+500) < nst_p_ch[jn+1] ) & (( matrix_of_lines_ch[:,1]-500)>nst_p_ch[jn] ) ]
                                            #print(matrix_new_new,newest_y_spliter[n],newest_y_spliter[n+1],newest_peaks[j],newest_peaks[j+1],'gada')
                                            if 1>0:#len( matrix_new_new2[:,9][matrix_new_new2[:,9]==1] )>0 and np.max(matrix_new_new2[:,8][matrix_new_new2[:,9]==1])>=0.2*(np.abs(newest_y_spliter_h[nd+1]-newest_y_spliter_h[nd] )):
                                                #num_col_sub_ch, peaks_neg_fin_sub_ch=find_num_col(regions_without_seperators[int(newest_y_spliter_h[nd]):int(newest_y_spliter_h[nd+1]),nst_p_ch[jn]:nst_p_ch[jn+1]],multiplier=2.3)
                                                
                                                num_col_sub_ch, peaks_neg_fin_sub_ch=self.find_num_col_only_image(image_p_rev[int(newest_y_spliter_h[nd]):int(newest_y_spliter_h[nd+1]),nst_p_ch[jn]:nst_p_ch[jn+1]],multiplier=2.3)
                                                print(peaks_neg_fin_sub_ch,'gada kutullllllll')
                                            else:
                                                peaks_neg_fin_sub_ch=[]

                                            peaks_sub_ch=[]
                                            peaks_sub_ch.append(nst_p_ch[jn])

                                            for kjj in range(len(peaks_neg_fin_sub_ch)):
                                                peaks_sub_ch.append(peaks_neg_fin_sub_ch[kjj]+nst_p_ch[jn])

                                            peaks_sub_ch.append(nst_p_ch[jn+1])

                                            #peaks_sub=return_points_with_boundies(peaks_neg_fin_sub+newest_peaks[j],newest_peaks[j], newest_peaks[j+1])

                                            for khh in range(len(peaks_sub_ch)-1):
                                                boxes.append([ peaks_sub_ch[khh], peaks_sub_ch[khh+1] ,newest_y_spliter_h[nd],newest_y_spliter_h[nd+1]])


                        
                                else:
                                    
                                    matrix_new_new=matrix_of_lines_ch[:,:][ (matrix_of_lines_ch[:,9]==1 ) & (matrix_of_lines_ch[:,6]> newest_y_spliter[n] ) & (matrix_of_lines_ch[:,7]< newest_y_spliter[n+1] ) & ( ( matrix_of_lines_ch[:,1]+500) < newest_peaks[j+1] ) & (( matrix_of_lines_ch[:,1]-500)> newest_peaks[j] )] 
                                    #print(matrix_new_new,newest_y_spliter[n],newest_y_spliter[n+1],newest_peaks[j],newest_peaks[j+1],'gada')
                                    if 1>0:#len( matrix_new_new[:,9][matrix_new_new[:,9]==1] )>0 and np.max(matrix_new_new[:,8][matrix_new_new[:,9]==1])>=0.2*(np.abs(newest_y_spliter[n+1]-newest_y_spliter[n] )):
                                        ###num_col_sub, peaks_neg_fin_sub=find_num_col(regions_without_seperators[int(newest_y_spliter[n]):int(newest_y_spliter[n+1]),newest_peaks[j]:newest_peaks[j+1]],multiplier=2.3)
                                        num_col_sub, peaks_neg_fin_sub=self.find_num_col_only_image(image_p_rev[int(newest_y_spliter[n]):int(newest_y_spliter[n+1]),newest_peaks[j]:newest_peaks[j+1]],multiplier=2.3)
                                    else:
                                        peaks_neg_fin_sub=[]

                                    peaks_sub=[]
                                    peaks_sub.append(newest_peaks[j])

                                    for kj in range(len(peaks_neg_fin_sub)):
                                        peaks_sub.append(peaks_neg_fin_sub[kj]+newest_peaks[j])

                                    peaks_sub.append(newest_peaks[j+1])

                                    #peaks_sub=return_points_with_boundies(peaks_neg_fin_sub+newest_peaks[j],newest_peaks[j], newest_peaks[j+1])

                                    for kh in range(len(peaks_sub)-1):
                                        boxes.append([ peaks_sub[kh], peaks_sub[kh+1] ,newest_y_spliter[n],newest_y_spliter[n+1]])

                                    
            
                                
                                
                        else:
                            for n in range(len(newest_y_spliter)-1):

                                for jvt in    matrix_new[:,0][ (matrix_new[:,9]==1 ) & (matrix_new[:,6]> newest_y_spliter[n] ) & (matrix_new[:,7]< newest_y_spliter[n+1] ) & ( ( matrix_new[:,1]) < newest_peaks[j+1] ) & (( matrix_new[:,1])> newest_peaks[j] ) ] :
                                    pass

                                    #plot_contour(regions_without_seperators.shape[0],regions_without_seperators.shape[1], contours_lines[int(jvt)])
                                #print(matrix_of_lines_ch[matrix_of_lines_ch[:,9]==1])
                                matrix_new_new=matrix_of_lines_ch[:,:][ (matrix_of_lines_ch[:,9]==1 ) & (matrix_of_lines_ch[:,6]> newest_y_spliter[n] ) & (matrix_of_lines_ch[:,7]< newest_y_spliter[n+1] ) & ( ( matrix_of_lines_ch[:,1]+500) < newest_peaks[j+1] ) & (( matrix_of_lines_ch[:,1]-500)> newest_peaks[j] )] 
                                #print(matrix_new_new,newest_y_spliter[n],newest_y_spliter[n+1],newest_peaks[j],newest_peaks[j+1],'gada')
                                if 1>0:#len( matrix_new_new[:,9][matrix_new_new[:,9]==1] )>0 and np.max(matrix_new_new[:,8][matrix_new_new[:,9]==1])>=0.2*(np.abs(newest_y_spliter[n+1]-newest_y_spliter[n] )):
                                    ###num_col_sub, peaks_neg_fin_sub=find_num_col(regions_without_seperators[int(newest_y_spliter[n]):int(newest_y_spliter[n+1]),newest_peaks[j]:newest_peaks[j+1]],multiplier=5.0)
                                    num_col_sub, peaks_neg_fin_sub=self.find_num_col_only_image(image_p_rev[int(newest_y_spliter[n]):int(newest_y_spliter[n+1]),newest_peaks[j]:newest_peaks[j+1]],multiplier=2.3)
                                else:
                                    peaks_neg_fin_sub=[]

                                peaks_sub=[]
                                peaks_sub.append(newest_peaks[j])

                                for kj in range(len(peaks_neg_fin_sub)):
                                    peaks_sub.append(peaks_neg_fin_sub[kj]+newest_peaks[j])

                                peaks_sub.append(newest_peaks[j+1])

                                #peaks_sub=return_points_with_boundies(peaks_neg_fin_sub+newest_peaks[j],newest_peaks[j], newest_peaks[j+1])

                                for kh in range(len(peaks_sub)-1):
                                    boxes.append([ peaks_sub[kh], peaks_sub[kh+1] ,newest_y_spliter[n],newest_y_spliter[n+1]])
                    
                            

                        
                        
                        
                        
                
            else:
                boxes.append([ 0, seperators_closeup_n[:,:,0].shape[1] ,spliter_y_new[i],spliter_y_new[i+1]])
        return boxes

    def return_boxes_of_images_by_order_of_reading_without_seperators_2cols(self,spliter_y_new,image_p_rev,regions_without_seperators,matrix_of_lines_ch,seperators_closeup_n):
        
        boxes=[]


        # here I go through main spliters and i do check whether a vertical seperator there is. If so i am searching for \
        # holes in the text and also finding spliter which covers more than one columns.
        for i in range(len(spliter_y_new)-1):
            #print(spliter_y_new[i],spliter_y_new[i+1])
            matrix_new=matrix_of_lines_ch[:,:][ (matrix_of_lines_ch[:,6]> spliter_y_new[i] ) & (matrix_of_lines_ch[:,7]< spliter_y_new[i+1] )  ] 
            #print(len( matrix_new[:,9][matrix_new[:,9]==1] ))
            
            #print(matrix_new[:,8][matrix_new[:,9]==1],'gaddaaa')
            
            # check to see is there any vertical seperator to find holes.
            if np.abs(spliter_y_new[i+1]-spliter_y_new[i])>1./3.*regions_without_seperators.shape[0]:#len( matrix_new[:,9][matrix_new[:,9]==1] )>0 and np.max(matrix_new[:,8][matrix_new[:,9]==1])>=0.1*(np.abs(spliter_y_new[i+1]-spliter_y_new[i] )):
                
                #org_img_dichte=-gaussian_filter1d(( image_page[int(spliter_y_new[i]):int(spliter_y_new[i+1]),:,0]/255.).sum(axis=0) ,30)
                #org_img_dichte=org_img_dichte-np.min(org_img_dichte)
                ##plt.figure(figsize=(20,20))
                ##plt.plot(org_img_dichte)      
                ##plt.show()
                ###find_num_col_both_layout_and_org(regions_without_seperators,image_page[int(spliter_y_new[i]):int(spliter_y_new[i+1]),:,:],7.)
                
                try:
                    num_col, peaks_neg_fin=self.find_num_col_only_image(image_p_rev[int(spliter_y_new[i]):int(spliter_y_new[i+1]),:],multiplier=2.4)
                except:
                    peaks_neg_fin=[]
                    num_col=0

                peaks_neg_tot=self.return_points_with_boundies(peaks_neg_fin,0, seperators_closeup_n[:,:,0].shape[1])
                
                for kh in range(len(peaks_neg_tot)-1):
                    boxes.append([ peaks_neg_tot[kh], peaks_neg_tot[kh+1] ,spliter_y_new[i],spliter_y_new[i+1]])
            else:
                boxes.append([ 0, seperators_closeup_n[:,:,0].shape[1] ,spliter_y_new[i],spliter_y_new[i+1]])
                

                
                
                

        return boxes
    def combine_hor_lines_and_delete_cross_points_and_get_lines_features_back(self, regions_pre_p):
        seperators_closeup=( (regions_pre_p[:,:]==6))*1
        
        seperators_closeup=seperators_closeup.astype(np.uint8)
        kernel = np.ones((5,5),np.uint8)


        seperators_closeup = cv2.dilate(seperators_closeup,kernel,iterations = 1)
        seperators_closeup = cv2.erode(seperators_closeup,kernel,iterations = 1)

        seperators_closeup = cv2.erode(seperators_closeup,kernel,iterations = 1)
        seperators_closeup = cv2.dilate(seperators_closeup,kernel,iterations = 1)
        
        if len(seperators_closeup.shape)==2:
            seperators_closeup_n=np.zeros((seperators_closeup.shape[0],seperators_closeup.shape[1],3))
            seperators_closeup_n[:,:,0]=seperators_closeup
            seperators_closeup_n[:,:,1]=seperators_closeup
            seperators_closeup_n[:,:,2]=seperators_closeup
        else:
            seperators_closeup_n=seperators_closeup[:,:,:]
        #seperators_closeup=seperators_closeup.astype(np.uint8)
        seperators_closeup_n=seperators_closeup_n.astype(np.uint8)
        imgray = cv2.cvtColor(seperators_closeup_n, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 0, 255, 0)
        contours_lines,hierachy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        slope_lines,dist_x, x_min_main ,x_max_main ,cy_main,slope_lines_org,y_min_main, y_max_main, cx_main=self.find_features_of_lines(contours_lines)
        
        dist_y=np.abs(y_max_main-y_min_main)

        slope_lines_org_hor=slope_lines_org[slope_lines==0]
        args=np.array( range(len(slope_lines) ))
        len_x=seperators_closeup.shape[1]*0
        len_y=seperators_closeup.shape[0]*.01

        args_hor=args[slope_lines==0]
        dist_x_hor=dist_x[slope_lines==0]
        dist_y_hor=dist_y[slope_lines==0]
        x_min_main_hor=x_min_main[slope_lines==0]
        x_max_main_hor=x_max_main[slope_lines==0]
        cy_main_hor=cy_main[slope_lines==0]
        y_min_main_hor=y_min_main[slope_lines==0]
        y_max_main_hor=y_max_main[slope_lines==0]

        args_hor=args_hor[dist_x_hor>=len_x]
        x_max_main_hor=x_max_main_hor[dist_x_hor>=len_x]
        x_min_main_hor=x_min_main_hor[dist_x_hor>=len_x]
        cy_main_hor=cy_main_hor[dist_x_hor>=len_x]
        y_min_main_hor=y_min_main_hor[dist_x_hor>=len_x]
        y_max_main_hor=y_max_main_hor[dist_x_hor>=len_x]
        slope_lines_org_hor=slope_lines_org_hor[dist_x_hor>=len_x]
        dist_y_hor=dist_y_hor[dist_x_hor>=len_x]
        dist_x_hor=dist_x_hor[dist_x_hor>=len_x]


        args_ver=args[slope_lines==1]
        dist_y_ver=dist_y[slope_lines==1]
        dist_x_ver=dist_x[slope_lines==1]
        x_min_main_ver=x_min_main[slope_lines==1]
        x_max_main_ver=x_max_main[slope_lines==1]
        y_min_main_ver=y_min_main[slope_lines==1]
        y_max_main_ver=y_max_main[slope_lines==1]
        cx_main_ver=cx_main[slope_lines==1]

        args_ver=args_ver[dist_y_ver>=len_y]
        x_max_main_ver=x_max_main_ver[dist_y_ver>=len_y]
        x_min_main_ver=x_min_main_ver[dist_y_ver>=len_y]
        cx_main_ver=cx_main_ver[dist_y_ver>=len_y]
        y_min_main_ver=y_min_main_ver[dist_y_ver>=len_y]
        y_max_main_ver=y_max_main_ver[dist_y_ver>=len_y]
        dist_x_ver=dist_x_ver[dist_y_ver>=len_y]
        dist_y_ver=dist_y_ver[dist_y_ver>=len_y]
        
        img_p_in_ver=np.zeros(seperators_closeup_n[:,:,2].shape)
        for jv in range(len(args_ver)):
            img_p_in_ver=cv2.fillPoly(img_p_in_ver, pts =[contours_lines[args_ver[jv]]], color=(1,1,1))
            
        img_in_hor=np.zeros(seperators_closeup_n[:,:,2].shape)
        for jv in range(len(args_hor)):
            img_p_in_hor=cv2.fillPoly(img_in_hor, pts =[contours_lines[args_hor[jv]]], color=(1,1,1))
            
        all_args_uniq=self.contours_in_same_horizon(cy_main_hor)
        #print(all_args_uniq,'all_args_uniq')
        if len(all_args_uniq)>0:
            if type(all_args_uniq[0]) is list:
                contours_new=[]
                for dd in range(len(all_args_uniq)):
                    merged_all=None
                    some_args=args_hor[all_args_uniq[dd]]
                    some_cy=cy_main_hor[all_args_uniq[dd]]
                    some_x_min=x_min_main_hor[all_args_uniq[dd]]
                    some_x_max=x_max_main_hor[all_args_uniq[dd]]

                    img_in=np.zeros(seperators_closeup_n[:,:,2].shape)
                    for jv in range(len(some_args)):

                        img_p_in=cv2.fillPoly(img_p_in_hor, pts =[contours_lines[some_args[jv]]], color=(1,1,1))
                        img_p_in[int(np.mean(some_cy))-5:int(np.mean(some_cy))+5, int(np.min(some_x_min)):int(np.max(some_x_max)) ]=1

            else:
                img_p_in=seperators_closeup
        else:
            img_p_in=seperators_closeup
            
        sep_ver_hor=img_p_in+img_p_in_ver
        sep_ver_hor_cross=(sep_ver_hor==2)*1

        sep_ver_hor_cross=np.repeat(sep_ver_hor_cross[:, :, np.newaxis], 3, axis=2)
        sep_ver_hor_cross=sep_ver_hor_cross.astype(np.uint8)
        imgray = cv2.cvtColor(sep_ver_hor_cross, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 0, 255, 0)
        contours_cross,_=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        cx_cross,cy_cross ,_ , _, _ ,_=self.find_new_features_of_contoures(contours_cross)
        
        for ii in range(len(cx_cross)):
            sep_ver_hor[int(cy_cross[ii])-15:int(cy_cross[ii])+15,int(cx_cross[ii])+5:int(cx_cross[ii])+40]=0
            sep_ver_hor[int(cy_cross[ii])-15:int(cy_cross[ii])+15,int(cx_cross[ii])-40:int(cx_cross[ii])-4]=0
            
        img_p_in[:,:]=sep_ver_hor[:,:]
        
        if len(img_p_in.shape)==2:
            seperators_closeup_n=np.zeros((img_p_in.shape[0],img_p_in.shape[1],3))
            seperators_closeup_n[:,:,0]=img_p_in
            seperators_closeup_n[:,:,1]=img_p_in
            seperators_closeup_n[:,:,2]=img_p_in
        else:
            seperators_closeup_n=img_p_in[:,:,:]
        #seperators_closeup=seperators_closeup.astype(np.uint8)
        seperators_closeup_n=seperators_closeup_n.astype(np.uint8)
        imgray = cv2.cvtColor(seperators_closeup_n, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 0, 255, 0)

        contours_lines,hierachy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        slope_lines,dist_x, x_min_main ,x_max_main ,cy_main,slope_lines_org,y_min_main, y_max_main, cx_main=self.find_features_of_lines(contours_lines)

        dist_y=np.abs(y_max_main-y_min_main)

        slope_lines_org_hor=slope_lines_org[slope_lines==0]
        args=np.array( range(len(slope_lines) ))
        len_x=seperators_closeup.shape[1]*.04
        len_y=seperators_closeup.shape[0]*.08

        args_hor=args[slope_lines==0]
        dist_x_hor=dist_x[slope_lines==0]
        dist_y_hor=dist_y[slope_lines==0]
        x_min_main_hor=x_min_main[slope_lines==0]
        x_max_main_hor=x_max_main[slope_lines==0]
        cy_main_hor=cy_main[slope_lines==0]
        y_min_main_hor=y_min_main[slope_lines==0]
        y_max_main_hor=y_max_main[slope_lines==0]

        args_hor=args_hor[dist_x_hor>=len_x]
        x_max_main_hor=x_max_main_hor[dist_x_hor>=len_x]
        x_min_main_hor=x_min_main_hor[dist_x_hor>=len_x]
        cy_main_hor=cy_main_hor[dist_x_hor>=len_x]
        y_min_main_hor=y_min_main_hor[dist_x_hor>=len_x]
        y_max_main_hor=y_max_main_hor[dist_x_hor>=len_x]
        slope_lines_org_hor=slope_lines_org_hor[dist_x_hor>=len_x]
        dist_y_hor=dist_y_hor[dist_x_hor>=len_x]
        dist_x_hor=dist_x_hor[dist_x_hor>=len_x]



        args_ver=args[slope_lines==1]
        dist_y_ver=dist_y[slope_lines==1]
        dist_x_ver=dist_x[slope_lines==1]
        x_min_main_ver=x_min_main[slope_lines==1]
        x_max_main_ver=x_max_main[slope_lines==1]
        y_min_main_ver=y_min_main[slope_lines==1]
        y_max_main_ver=y_max_main[slope_lines==1]
        cx_main_ver=cx_main[slope_lines==1]

        args_ver=args_ver[dist_y_ver>=len_y]
        x_max_main_ver=x_max_main_ver[dist_y_ver>=len_y]
        x_min_main_ver=x_min_main_ver[dist_y_ver>=len_y]
        cx_main_ver=cx_main_ver[dist_y_ver>=len_y]
        y_min_main_ver=y_min_main_ver[dist_y_ver>=len_y]
        y_max_main_ver=y_max_main_ver[dist_y_ver>=len_y]
        dist_x_ver=dist_x_ver[dist_y_ver>=len_y]
        dist_y_ver=dist_y_ver[dist_y_ver>=len_y]
        
        matrix_of_lines_ch=np.zeros((len(cy_main_hor)+len(cx_main_ver),10))
        
        matrix_of_lines_ch[:len(cy_main_hor),0]=args_hor
        matrix_of_lines_ch[len(cy_main_hor):,0]=args_ver


        matrix_of_lines_ch[len(cy_main_hor):,1]=cx_main_ver

        matrix_of_lines_ch[:len(cy_main_hor),2]=x_min_main_hor
        matrix_of_lines_ch[len(cy_main_hor):,2]=x_min_main_ver

        matrix_of_lines_ch[:len(cy_main_hor),3]=x_max_main_hor
        matrix_of_lines_ch[len(cy_main_hor):,3]=x_max_main_ver

        matrix_of_lines_ch[:len(cy_main_hor),4]=dist_x_hor
        matrix_of_lines_ch[len(cy_main_hor):,4]=dist_x_ver

        matrix_of_lines_ch[:len(cy_main_hor),5]=cy_main_hor


        matrix_of_lines_ch[:len(cy_main_hor),6]=y_min_main_hor
        matrix_of_lines_ch[len(cy_main_hor):,6]=y_min_main_ver

        matrix_of_lines_ch[:len(cy_main_hor),7]=y_max_main_hor
        matrix_of_lines_ch[len(cy_main_hor):,7]=y_max_main_ver

        matrix_of_lines_ch[:len(cy_main_hor),8]=dist_y_hor
        matrix_of_lines_ch[len(cy_main_hor):,8]=dist_y_ver


        matrix_of_lines_ch[len(cy_main_hor):,9]=1
        
        return matrix_of_lines_ch,seperators_closeup_n
    
    
    def combine_hor_lines_and_delete_cross_points_and_get_lines_features_back_new(self, img_p_in_ver,img_in_hor):
        
        
        ##plt.imshow(img_p_in_ver)
        ##plt.show()
        
        #img_p_in_ver = cv2.erode(img_p_in_ver, self.kernel, iterations=2)
        img_p_in_ver=img_p_in_ver.astype(np.uint8)
        img_p_in_ver=np.repeat(img_p_in_ver[:, :, np.newaxis], 3, axis=2)
        imgray = cv2.cvtColor(img_p_in_ver, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 0, 255, 0)

        contours_lines_ver,hierachy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        slope_lines_ver,dist_x_ver, x_min_main_ver ,x_max_main_ver ,cy_main_ver,slope_lines_org_ver,y_min_main_ver, y_max_main_ver, cx_main_ver=self.find_features_of_lines(contours_lines_ver)
        
        for i in range(len(x_min_main_ver)):
            img_p_in_ver[int(y_min_main_ver[i]):int(y_min_main_ver[i])+30,int(cx_main_ver[i])-25:int(cx_main_ver[i])+25,0]=0
            img_p_in_ver[int(y_max_main_ver[i])-30:int(y_max_main_ver[i]),int(cx_main_ver[i])-25:int(cx_main_ver[i])+25,0]=0
        
        
        #plt.imshow(img_p_in_ver[:,:,0])
        #plt.show()
        img_in_hor=img_in_hor.astype(np.uint8)
        img_in_hor=np.repeat(img_in_hor[:, :, np.newaxis], 3, axis=2)
        imgray = cv2.cvtColor(img_in_hor, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 0, 255, 0)

        contours_lines_hor,hierachy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        slope_lines_hor,dist_x_hor, x_min_main_hor ,x_max_main_hor ,cy_main_hor,slope_lines_org_hor,y_min_main_hor, y_max_main_hor, cx_main_hor=self.find_features_of_lines(contours_lines_hor)
        
        args_hor=np.array( range(len(slope_lines_hor) ))
        all_args_uniq=self.contours_in_same_horizon(cy_main_hor)
        #print(all_args_uniq,'all_args_uniq')
        if len(all_args_uniq)>0:
            if type(all_args_uniq[0]) is list:
                contours_new=[]
                for dd in range(len(all_args_uniq)):
                    merged_all=None
                    some_args=args_hor[all_args_uniq[dd]]
                    some_cy=cy_main_hor[all_args_uniq[dd]]
                    some_x_min=x_min_main_hor[all_args_uniq[dd]]
                    some_x_max=x_max_main_hor[all_args_uniq[dd]]

                    #img_in=np.zeros(seperators_closeup_n[:,:,2].shape)
                    for jv in range(len(some_args)):

                        img_p_in=cv2.fillPoly(img_in_hor, pts =[contours_lines_hor[some_args[jv]]], color=(1,1,1))
                        img_p_in[int(np.mean(some_cy))-5:int(np.mean(some_cy))+5, int(np.min(some_x_min)):int(np.max(some_x_max)) ]=1

            else:
                img_p_in=img_in_hor
        else:
            img_p_in=img_in_hor

        
        img_p_in_ver[:,:,0][img_p_in_ver[:,:,0]==255]=1
        #print(img_p_in_ver.shape,np.unique(img_p_in_ver[:,:,0]))
        
        #plt.imshow(img_p_in[:,:,0])
        #plt.show()
        
        #plt.imshow(img_p_in_ver[:,:,0])
        #plt.show()
        sep_ver_hor=img_p_in+img_p_in_ver
        #print(sep_ver_hor.shape,np.unique(sep_ver_hor[:,:,0]),'sep_ver_horsep_ver_horsep_ver_hor')
        #plt.imshow(sep_ver_hor[:,:,0])
        #plt.show()

        sep_ver_hor_cross=(sep_ver_hor[:,:,0]==2)*1

        sep_ver_hor_cross=np.repeat(sep_ver_hor_cross[:, :, np.newaxis], 3, axis=2)
        sep_ver_hor_cross=sep_ver_hor_cross.astype(np.uint8)
        imgray = cv2.cvtColor(sep_ver_hor_cross, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 0, 255, 0)
        contours_cross,_=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        cx_cross,cy_cross ,_ , _, _ ,_=self.find_new_features_of_contoures(contours_cross)
        
        for ii in range(len(cx_cross)):
            img_p_in[int(cy_cross[ii])-30:int(cy_cross[ii])+30,int(cx_cross[ii])+5:int(cx_cross[ii])+40,0]=0
            img_p_in[int(cy_cross[ii])-30:int(cy_cross[ii])+30,int(cx_cross[ii])-40:int(cx_cross[ii])-4,0]=0
            
        #plt.imshow(img_p_in[:,:,0])
        #plt.show()
        
        return img_p_in[:,:,0]
 
    def return_boxes_of_images_by_order_of_reading(self,spliter_y_new,regions_without_seperators,matrix_of_lines_ch,seperators_closeup_n):
        boxes=[]


        # here I go through main spliters and i do check whether a vertical seperator there is. If so i am searching for \
        # holes in the text and also finding spliter which covers more than one columns.
        for i in range(len(spliter_y_new)-1):
            #print(spliter_y_new[i],spliter_y_new[i+1])
            matrix_new=matrix_of_lines_ch[:,:][ (matrix_of_lines_ch[:,6]> spliter_y_new[i] ) & (matrix_of_lines_ch[:,7]< spliter_y_new[i+1] )  ] 
            #print(len( matrix_new[:,9][matrix_new[:,9]==1] ))
            
            #print(matrix_new[:,8][matrix_new[:,9]==1],'gaddaaa')
            
            # check to see is there any vertical seperator to find holes.
            if len( matrix_new[:,9][matrix_new[:,9]==1] )>0 and np.max(matrix_new[:,8][matrix_new[:,9]==1])>=0.1*(np.abs(spliter_y_new[i+1]-spliter_y_new[i] )):
                
                #org_img_dichte=-gaussian_filter1d(( image_page[int(spliter_y_new[i]):int(spliter_y_new[i+1]),:,0]/255.).sum(axis=0) ,30)
                #org_img_dichte=org_img_dichte-np.min(org_img_dichte)
                ##plt.figure(figsize=(20,20))
                ##plt.plot(org_img_dichte)      
                ##plt.show()
                ###find_num_col_both_layout_and_org(regions_without_seperators,image_page[int(spliter_y_new[i]):int(spliter_y_new[i+1]),:,:],7.)
                
                num_col, peaks_neg_fin=self.find_num_col(regions_without_seperators[int(spliter_y_new[i]):int(spliter_y_new[i+1]),:],multiplier=7.0)
                
                #num_col, peaks_neg_fin=find_num_col(regions_without_seperators[int(spliter_y_new[i]):int(spliter_y_new[i+1]),:],multiplier=7.0)
                x_min_hor_some=matrix_new[:,2][ (matrix_new[:,9]==0) ]
                x_max_hor_some=matrix_new[:,3][ (matrix_new[:,9]==0) ]
                cy_hor_some=matrix_new[:,5][ (matrix_new[:,9]==0) ]
                arg_org_hor_some=matrix_new[:,0][ (matrix_new[:,9]==0) ]
                

                
                peaks_neg_tot=self.return_points_with_boundies(peaks_neg_fin,0, seperators_closeup_n[:,:,0].shape[1])
                
                start_index_of_hor,newest_peaks,arg_min_hor_sort,lines_length_dels,lines_indexes_deleted=self.return_hor_spliter_by_index(peaks_neg_tot,x_min_hor_some,x_max_hor_some)
                
                arg_org_hor_some_sort=arg_org_hor_some[arg_min_hor_sort]
                
                

                start_index_of_hor_with_subset=[start_index_of_hor[vij] for vij in range(len(start_index_of_hor)) if lines_length_dels[vij]>0 ]#start_index_of_hor[lines_length_dels>0]
                arg_min_hor_sort_with_subset=[arg_min_hor_sort[vij] for vij in range(len(start_index_of_hor)) if lines_length_dels[vij]>0 ]
                lines_indexes_deleted_with_subset=[lines_indexes_deleted[vij] for vij in range(len(start_index_of_hor)) if lines_length_dels[vij]>0 ]
                lines_length_dels_with_subset=[lines_length_dels[vij] for vij in range(len(start_index_of_hor)) if lines_length_dels[vij]>0 ]
                
                arg_org_hor_some_sort_subset=[arg_org_hor_some_sort[vij] for vij in range(len(start_index_of_hor)) if lines_length_dels[vij]>0 ]

                #arg_min_hor_sort_with_subset=arg_min_hor_sort[lines_length_dels>0]
                #lines_indexes_deleted_with_subset=lines_indexes_deleted[lines_length_dels>0]
                #lines_length_dels_with_subset=lines_length_dels[lines_length_dels>0]
                
                
                

                
                
                
                vahid_subset=np.zeros((len(start_index_of_hor_with_subset),len(start_index_of_hor_with_subset)))-1
                for kkk1 in range(len(start_index_of_hor_with_subset)):
                    
                    
                    index_del_sub=np.unique(lines_indexes_deleted_with_subset[kkk1])
                    
                    for kkk2 in range(len(start_index_of_hor_with_subset)):
                        
                        if set(lines_indexes_deleted_with_subset[kkk2][0]) < set(lines_indexes_deleted_with_subset[kkk1][0]):
                            vahid_subset[kkk1,kkk2]=kkk1
                        else:
                            pass
                    #print(set(lines_indexes_deleted[kkk2][0]), set(lines_indexes_deleted[kkk1][0]))
                    
                #print(vahid_subset,'zartt222')
                
                
                # check the len of matrix if it has no length means that there is no spliter at all
                
                if len(vahid_subset>0):
                    #print('hihoo')


                    # find parenets args
                    line_int=np.zeros(vahid_subset.shape[0])
                    
                    childs_id=[]
                    arg_child=[]
                    for li in range(vahid_subset.shape[0]):
                        #print(vahid_subset[:,li])
                        if np.all(vahid_subset[:,li]==-1):
                            line_int[li]=-1
                        else:
                            line_int[li]=1
                            
                            
                            #childs_args_in=[ idd for idd in range(vahid_subset.shape[0]) if vahid_subset[idd,li]!=-1]
                            #helpi=[]
                            #for nad in range(len(childs_args_in)):
                            #    helpi.append(arg_min_hor_sort_with_subset[childs_args_in[nad]])
                            
                                
                            arg_child.append(arg_min_hor_sort_with_subset[li] )   
                    
                    
                    
                        
                      
                    #line_int=vahid_subset[0,:]
                    
                    #print(arg_child,line_int[0],'zartt33333')
                    arg_parent=[arg_min_hor_sort_with_subset[vij] for vij in range(len(arg_min_hor_sort_with_subset)) if line_int[vij]==-1]
                    start_index_of_hor_parent=[start_index_of_hor_with_subset[vij] for vij in range(len(arg_min_hor_sort_with_subset)) if line_int[vij]==-1]
                    #arg_parent=[lines_indexes_deleted_with_subset[vij] for vij in range(len(arg_min_hor_sort_with_subset)) if line_int[vij]==-1]
                    #arg_parent=[lines_length_dels_with_subset[vij] for vij in range(len(arg_min_hor_sort_with_subset)) if line_int[vij]==-1]
                    

                    #arg_child=[arg_min_hor_sort_with_subset[vij] for vij in range(len(arg_min_hor_sort_with_subset)) if line_int[vij]!=-1]
                    start_index_of_hor_child=[start_index_of_hor_with_subset[vij] for vij in range(len(arg_min_hor_sort_with_subset)) if line_int[vij]!=-1]

                    
                
                    
                    cy_hor_some_sort=cy_hor_some[arg_parent]
                    
                    


                    
                    #print(start_index_of_hor, lines_length_dels ,lines_indexes_deleted,'zartt')

                    #args_indexes=np.array(range(len(start_index_of_hor) ))

                    newest_y_spliter_tot=[]

                    for tj in range(len(newest_peaks)-1):
                        newest_y_spliter=[]
                        newest_y_spliter.append(spliter_y_new[i])
                        if tj in np.unique(start_index_of_hor_parent):
                            ##print(cy_hor_some_sort)
                            cy_help=np.array(cy_hor_some_sort)[np.array(start_index_of_hor_parent)==tj]
                            cy_help_sort=np.sort(cy_help)

                            #print(tj,cy_hor_some_sort,start_index_of_hor,cy_help,'maashhaha')
                            for mj in range(len(cy_help_sort)):
                                newest_y_spliter.append(cy_help_sort[mj])
                        newest_y_spliter.append(spliter_y_new[i+1])

                        newest_y_spliter_tot.append(newest_y_spliter)
                        

                    
                        
                else:
                    line_int=[]
                    newest_y_spliter_tot=[]

                    for tj in range(len(newest_peaks)-1):
                        newest_y_spliter=[]
                        newest_y_spliter.append(spliter_y_new[i])

                        newest_y_spliter.append(spliter_y_new[i+1])

                        newest_y_spliter_tot.append(newest_y_spliter)
                    

                
                # if line_int is all -1 means that big spliters have no child and we can easily go through
                if np.all(np.array(line_int)==-1):
                    for j in range(len(newest_peaks)-1):
                        newest_y_spliter=newest_y_spliter_tot[j]


                        for n in range(len(newest_y_spliter)-1):
                            #print(j,newest_y_spliter[n],newest_y_spliter[n+1],newest_peaks[j],newest_peaks[j+1],'maaaa')
                            ##plt.imshow(regions_without_seperators[int(newest_y_spliter[n]):int(newest_y_spliter[n+1]),newest_peaks[j]:newest_peaks[j+1]])
                            ##plt.show()

                            #print(matrix_new[:,0][ (matrix_new[:,9]==1 )])
                            for jvt in    matrix_new[:,0][ (matrix_new[:,9]==1 ) & (matrix_new[:,6]> newest_y_spliter[n] ) & (matrix_new[:,7]< newest_y_spliter[n+1] ) & ( ( matrix_new[:,1]) < newest_peaks[j+1] ) & (( matrix_new[:,1])> newest_peaks[j] ) ] :
                                pass

                                ###plot_contour(regions_without_seperators.shape[0],regions_without_seperators.shape[1], contours_lines[int(jvt)])
                            #print(matrix_of_lines_ch[matrix_of_lines_ch[:,9]==1])
                            matrix_new_new=matrix_of_lines_ch[:,:][ (matrix_of_lines_ch[:,9]==1 ) & (matrix_of_lines_ch[:,6]> newest_y_spliter[n] ) & (matrix_of_lines_ch[:,7]< newest_y_spliter[n+1] ) & ( ( matrix_of_lines_ch[:,1]+500) < newest_peaks[j+1] ) & (( matrix_of_lines_ch[:,1]-500)> newest_peaks[j] )] 
                            #print(matrix_new_new,newest_y_spliter[n],newest_y_spliter[n+1],newest_peaks[j],newest_peaks[j+1],'gada')
                            if len( matrix_new_new[:,9][matrix_new_new[:,9]==1] )>0 and np.max(matrix_new_new[:,8][matrix_new_new[:,9]==1])>=0.2*(np.abs(newest_y_spliter[n+1]-newest_y_spliter[n] )):
                                num_col_sub, peaks_neg_fin_sub=self.find_num_col(regions_without_seperators[int(newest_y_spliter[n]):int(newest_y_spliter[n+1]),newest_peaks[j]:newest_peaks[j+1]],multiplier=5.)
                            else:
                                peaks_neg_fin_sub=[]

                            peaks_sub=[]
                            peaks_sub.append(newest_peaks[j])

                            for kj in range(len(peaks_neg_fin_sub)):
                                peaks_sub.append(peaks_neg_fin_sub[kj]+newest_peaks[j])

                            peaks_sub.append(newest_peaks[j+1])

                            #peaks_sub=return_points_with_boundies(peaks_neg_fin_sub+newest_peaks[j],newest_peaks[j], newest_peaks[j+1])

                            for kh in range(len(peaks_sub)-1):
                                boxes.append([ peaks_sub[kh], peaks_sub[kh+1] ,newest_y_spliter[n],newest_y_spliter[n+1]])
                                
                                
                else:
                    for j in range(len(newest_peaks)-1):
                        newest_y_spliter=newest_y_spliter_tot[j]
                        
                        if j in start_index_of_hor_parent:
                            
                            x_min_ch=x_min_hor_some[arg_child]
                            x_max_ch=x_max_hor_some[arg_child]
                            cy_hor_some_sort_child=cy_hor_some[arg_child]
                            cy_hor_some_sort_child=np.sort(cy_hor_some_sort_child)
                            
                            #print(cy_hor_some_sort_child,'ychilds')
                            
                            for n in range(len(newest_y_spliter)-1):
                                
                                cy_child_in=cy_hor_some_sort_child[( cy_hor_some_sort_child>newest_y_spliter[n] ) & ( cy_hor_some_sort_child<newest_y_spliter[n+1] ) ]
                                
                                if len(cy_child_in)>0:
                                    num_col_ch, peaks_neg_ch=self.find_num_col( regions_without_seperators[int(newest_y_spliter[n]):int(newest_y_spliter[n+1]),newest_peaks[j]:newest_peaks[j+1]],multiplier=5.0)
                                    #print(peaks_neg_ch,'mizzzz')
                                    #peaks_neg_ch=[]
                                    #for djh in range(len(peaks_neg_ch)):
                                    #    peaks_neg_ch.append( peaks_neg_ch[djh]+newest_peaks[j] )
                                    
                                    peaks_neg_ch_tot=self.return_points_with_boundies(peaks_neg_ch,newest_peaks[j], newest_peaks[j+1])
                                    
                                    ss_in_ch,nst_p_ch,arg_n_ch,lines_l_del_ch,lines_in_del_ch=self.return_hor_spliter_by_index(peaks_neg_ch_tot,x_min_ch,x_max_ch)
                                        
                                    
                                    
                                    
                                    
                                    newest_y_spliter_ch_tot=[]

                                    for tjj in range(len(nst_p_ch)-1):
                                        newest_y_spliter_new=[]
                                        newest_y_spliter_new.append(newest_y_spliter[n])
                                        if tjj in np.unique(ss_in_ch):
                                            

                                            #print(tj,cy_hor_some_sort,start_index_of_hor,cy_help,'maashhaha')
                                            for mjj in range(len(cy_child_in)):
                                                newest_y_spliter_new.append(cy_child_in[mjj])
                                        newest_y_spliter_new.append(newest_y_spliter[n+1])

                                        newest_y_spliter_ch_tot.append(newest_y_spliter_new)
                                        
                                    
                                    
                                    
                                    
                                    for jn in range(len(nst_p_ch)-1):
                                        newest_y_spliter_h=newest_y_spliter_ch_tot[jn]

                                        for nd in range(len(newest_y_spliter_h)-1):

                                            matrix_new_new2=matrix_of_lines_ch[:,:][ (matrix_of_lines_ch[:,9]==1 ) & (matrix_of_lines_ch[:,6]> newest_y_spliter_h[nd] ) & (matrix_of_lines_ch[:,7]< newest_y_spliter_h[nd+1] ) & ( ( matrix_of_lines_ch[:,1]+500) < nst_p_ch[jn+1] ) & (( matrix_of_lines_ch[:,1]-500)>nst_p_ch[jn] ) ]
                                            #print(matrix_new_new,newest_y_spliter[n],newest_y_spliter[n+1],newest_peaks[j],newest_peaks[j+1],'gada')
                                            if len( matrix_new_new2[:,9][matrix_new_new2[:,9]==1] )>0 and np.max(matrix_new_new2[:,8][matrix_new_new2[:,9]==1])>=0.2*(np.abs(newest_y_spliter_h[nd+1]-newest_y_spliter_h[nd] )):
                                                num_col_sub_ch, peaks_neg_fin_sub_ch=self.find_num_col(regions_without_seperators[int(newest_y_spliter_h[nd]):int(newest_y_spliter_h[nd+1]),nst_p_ch[jn]:nst_p_ch[jn+1]],multiplier=5.0)
                                                
                                            else:
                                                peaks_neg_fin_sub_ch=[]

                                            peaks_sub_ch=[]
                                            peaks_sub_ch.append(nst_p_ch[jn])

                                            for kjj in range(len(peaks_neg_fin_sub_ch)):
                                                peaks_sub_ch.append(peaks_neg_fin_sub_ch[kjj]+nst_p_ch[jn])

                                            peaks_sub_ch.append(nst_p_ch[jn+1])

                                            #peaks_sub=return_points_with_boundies(peaks_neg_fin_sub+newest_peaks[j],newest_peaks[j], newest_peaks[j+1])

                                            for khh in range(len(peaks_sub_ch)-1):
                                                boxes.append([ peaks_sub_ch[khh], peaks_sub_ch[khh+1] ,newest_y_spliter_h[nd],newest_y_spliter_h[nd+1]])


                        
                                else:
                                    
                                    matrix_new_new=matrix_of_lines_ch[:,:][ (matrix_of_lines_ch[:,9]==1 ) & (matrix_of_lines_ch[:,6]> newest_y_spliter[n] ) & (matrix_of_lines_ch[:,7]< newest_y_spliter[n+1] ) & ( ( matrix_of_lines_ch[:,1]+500) < newest_peaks[j+1] ) & (( matrix_of_lines_ch[:,1]-500)> newest_peaks[j] )] 
                                    #print(matrix_new_new,newest_y_spliter[n],newest_y_spliter[n+1],newest_peaks[j],newest_peaks[j+1],'gada')
                                    if len( matrix_new_new[:,9][matrix_new_new[:,9]==1] )>0 and np.max(matrix_new_new[:,8][matrix_new_new[:,9]==1])>=0.2*(np.abs(newest_y_spliter[n+1]-newest_y_spliter[n] )):
                                        num_col_sub, peaks_neg_fin_sub=self.find_num_col(regions_without_seperators[int(newest_y_spliter[n]):int(newest_y_spliter[n+1]),newest_peaks[j]:newest_peaks[j+1]],multiplier=5.0)
                                    else:
                                        peaks_neg_fin_sub=[]

                                    peaks_sub=[]
                                    peaks_sub.append(newest_peaks[j])

                                    for kj in range(len(peaks_neg_fin_sub)):
                                        peaks_sub.append(peaks_neg_fin_sub[kj]+newest_peaks[j])

                                    peaks_sub.append(newest_peaks[j+1])

                                    #peaks_sub=return_points_with_boundies(peaks_neg_fin_sub+newest_peaks[j],newest_peaks[j], newest_peaks[j+1])

                                    for kh in range(len(peaks_sub)-1):
                                        boxes.append([ peaks_sub[kh], peaks_sub[kh+1] ,newest_y_spliter[n],newest_y_spliter[n+1]])
                                    
                                    
            
                                
                                
                        else:
                            for n in range(len(newest_y_spliter)-1):


                                #plot_contour(regions_without_seperators.shape[0],regions_without_seperators.shape[1], contours_lines[int(jvt)])
                                #print(matrix_of_lines_ch[matrix_of_lines_ch[:,9]==1])
                                matrix_new_new=matrix_of_lines_ch[:,:][ (matrix_of_lines_ch[:,9]==1 ) & (matrix_of_lines_ch[:,6]> newest_y_spliter[n] ) & (matrix_of_lines_ch[:,7]< newest_y_spliter[n+1] ) & ( ( matrix_of_lines_ch[:,1]+500) < newest_peaks[j+1] ) & (( matrix_of_lines_ch[:,1]-500)> newest_peaks[j] )] 
                                #print(matrix_new_new,newest_y_spliter[n],newest_y_spliter[n+1],newest_peaks[j],newest_peaks[j+1],'gada')
                                if len( matrix_new_new[:,9][matrix_new_new[:,9]==1] )>0 and np.max(matrix_new_new[:,8][matrix_new_new[:,9]==1])>=0.2*(np.abs(newest_y_spliter[n+1]-newest_y_spliter[n] )):
                                    num_col_sub, peaks_neg_fin_sub=self.find_num_col(regions_without_seperators[int(newest_y_spliter[n]):int(newest_y_spliter[n+1]),newest_peaks[j]:newest_peaks[j+1]],multiplier=5.0)
                                else:
                                    peaks_neg_fin_sub=[]

                                peaks_sub=[]
                                peaks_sub.append(newest_peaks[j])

                                for kj in range(len(peaks_neg_fin_sub)):
                                    peaks_sub.append(peaks_neg_fin_sub[kj]+newest_peaks[j])

                                peaks_sub.append(newest_peaks[j+1])

                                #peaks_sub=return_points_with_boundies(peaks_neg_fin_sub+newest_peaks[j],newest_peaks[j], newest_peaks[j+1])

                                for kh in range(len(peaks_sub)-1):
                                    boxes.append([ peaks_sub[kh], peaks_sub[kh+1] ,newest_y_spliter[n],newest_y_spliter[n+1]])
                    
                            

                        
                        
                        
                        
                
            else:
                boxes.append([ 0, seperators_closeup_n[:,:,0].shape[1] ,spliter_y_new[i],spliter_y_new[i+1]])

                
        return boxes



    def return_boxes_of_images_by_order_of_reading_new(self,spliter_y_new,regions_without_seperators,matrix_of_lines_ch):
        boxes=[]


        # here I go through main spliters and i do check whether a vertical seperator there is. If so i am searching for \
        # holes in the text and also finding spliter which covers more than one columns.
        for i in range(len(spliter_y_new)-1):
            #print(spliter_y_new[i],spliter_y_new[i+1])
            matrix_new=matrix_of_lines_ch[:,:][ (matrix_of_lines_ch[:,6]> spliter_y_new[i] ) & (matrix_of_lines_ch[:,7]< spliter_y_new[i+1] )  ] 
            #print(len( matrix_new[:,9][matrix_new[:,9]==1] ))
            
            #print(matrix_new[:,8][matrix_new[:,9]==1],'gaddaaa')
            
            # check to see is there any vertical seperator to find holes.
            if 1>0:#len( matrix_new[:,9][matrix_new[:,9]==1] )>0 and np.max(matrix_new[:,8][matrix_new[:,9]==1])>=0.1*(np.abs(spliter_y_new[i+1]-spliter_y_new[i] )):
                
                #org_img_dichte=-gaussian_filter1d(( image_page[int(spliter_y_new[i]):int(spliter_y_new[i+1]),:,0]/255.).sum(axis=0) ,30)
                #org_img_dichte=org_img_dichte-np.min(org_img_dichte)
                ##plt.figure(figsize=(20,20))
                ##plt.plot(org_img_dichte)      
                ##plt.show()
                ###find_num_col_both_layout_and_org(regions_without_seperators,image_page[int(spliter_y_new[i]):int(spliter_y_new[i+1]),:,:],7.)
                
                #print(int(spliter_y_new[i]),int(spliter_y_new[i+1]),'firssst')
                try:
                    num_col, peaks_neg_fin=self.find_num_col(regions_without_seperators[int(spliter_y_new[i]):int(spliter_y_new[i+1]),:],multiplier=7.0)
                except:
                    peaks_neg_fin=[]
                
                #num_col, peaks_neg_fin=find_num_col(regions_without_seperators[int(spliter_y_new[i]):int(spliter_y_new[i+1]),:],multiplier=7.0)
                x_min_hor_some=matrix_new[:,2][ (matrix_new[:,9]==0) ]
                x_max_hor_some=matrix_new[:,3][ (matrix_new[:,9]==0) ]
                cy_hor_some=matrix_new[:,5][ (matrix_new[:,9]==0) ]
                arg_org_hor_some=matrix_new[:,0][ (matrix_new[:,9]==0) ]
                

                peaks_neg_tot=self.return_points_with_boundies(peaks_neg_fin,0, regions_without_seperators[:,:].shape[1])
                
                start_index_of_hor,newest_peaks,arg_min_hor_sort,lines_length_dels,lines_indexes_deleted=self.return_hor_spliter_by_index_for_without_verticals(peaks_neg_tot,x_min_hor_some,x_max_hor_some)
                
                arg_org_hor_some_sort=arg_org_hor_some[arg_min_hor_sort]
                

                start_index_of_hor_with_subset=[start_index_of_hor[vij] for vij in range(len(start_index_of_hor)) if lines_length_dels[vij]>0 ]#start_index_of_hor[lines_length_dels>0]
                arg_min_hor_sort_with_subset=[arg_min_hor_sort[vij] for vij in range(len(start_index_of_hor)) if lines_length_dels[vij]>0 ]
                lines_indexes_deleted_with_subset=[lines_indexes_deleted[vij] for vij in range(len(start_index_of_hor)) if lines_length_dels[vij]>0 ]
                lines_length_dels_with_subset=[lines_length_dels[vij] for vij in range(len(start_index_of_hor)) if lines_length_dels[vij]>0 ]
                
                arg_org_hor_some_sort_subset=[arg_org_hor_some_sort[vij] for vij in range(len(start_index_of_hor)) if lines_length_dels[vij]>0 ]

                #arg_min_hor_sort_with_subset=arg_min_hor_sort[lines_length_dels>0]
                #lines_indexes_deleted_with_subset=lines_indexes_deleted[lines_length_dels>0]
                #lines_length_dels_with_subset=lines_length_dels[lines_length_dels>0]
                
                
                

                
                
                
                vahid_subset=np.zeros((len(start_index_of_hor_with_subset),len(start_index_of_hor_with_subset)))-1
                for kkk1 in range(len(start_index_of_hor_with_subset)):
                    
                    
                    index_del_sub=np.unique(lines_indexes_deleted_with_subset[kkk1])
                    
                    for kkk2 in range(len(start_index_of_hor_with_subset)):
                        
                        if set(lines_indexes_deleted_with_subset[kkk2][0]) < set(lines_indexes_deleted_with_subset[kkk1][0]):
                            vahid_subset[kkk1,kkk2]=kkk1
                        else:
                            pass
                    #print(set(lines_indexes_deleted[kkk2][0]), set(lines_indexes_deleted[kkk1][0]))
                    
                
                
                # check the len of matrix if it has no length means that there is no spliter at all
                
                if len(vahid_subset>0):
                    #print('hihoo')


                    # find parenets args
                    line_int=np.zeros(vahid_subset.shape[0])
                    
                    childs_id=[]
                    arg_child=[]
                    for li in range(vahid_subset.shape[0]):
                        #print(vahid_subset[:,li])
                        if np.all(vahid_subset[:,li]==-1):
                            line_int[li]=-1
                        else:
                            line_int[li]=1
                            
                            
                            #childs_args_in=[ idd for idd in range(vahid_subset.shape[0]) if vahid_subset[idd,li]!=-1]
                            #helpi=[]
                            #for nad in range(len(childs_args_in)):
                            #    helpi.append(arg_min_hor_sort_with_subset[childs_args_in[nad]])
                            
                                
                            arg_child.append(arg_min_hor_sort_with_subset[li] )   
                    
                    
                    
                        
                      
                    #line_int=vahid_subset[0,:]
                    

                    arg_parent=[arg_min_hor_sort_with_subset[vij] for vij in range(len(arg_min_hor_sort_with_subset)) if line_int[vij]==-1]
                    start_index_of_hor_parent=[start_index_of_hor_with_subset[vij] for vij in range(len(arg_min_hor_sort_with_subset)) if line_int[vij]==-1]
                    #arg_parent=[lines_indexes_deleted_with_subset[vij] for vij in range(len(arg_min_hor_sort_with_subset)) if line_int[vij]==-1]
                    #arg_parent=[lines_length_dels_with_subset[vij] for vij in range(len(arg_min_hor_sort_with_subset)) if line_int[vij]==-1]
                    

                    #arg_child=[arg_min_hor_sort_with_subset[vij] for vij in range(len(arg_min_hor_sort_with_subset)) if line_int[vij]!=-1]
                    start_index_of_hor_child=[start_index_of_hor_with_subset[vij] for vij in range(len(arg_min_hor_sort_with_subset)) if line_int[vij]!=-1]

                    
                
                    
                    cy_hor_some_sort=cy_hor_some[arg_parent]
                    
                    


                    
                    #print(start_index_of_hor, lines_length_dels ,lines_indexes_deleted,'zartt')

                    #args_indexes=np.array(range(len(start_index_of_hor) ))

                    newest_y_spliter_tot=[]

                    for tj in range(len(newest_peaks)-1):
                        newest_y_spliter=[]
                        newest_y_spliter.append(spliter_y_new[i])
                        if tj in np.unique(start_index_of_hor_parent):
                            #print(cy_hor_some_sort)
                            cy_help=np.array(cy_hor_some_sort)[np.array(start_index_of_hor_parent)==tj]
                            cy_help_sort=np.sort(cy_help)

                            #print(tj,cy_hor_some_sort,start_index_of_hor,cy_help,'maashhaha')
                            for mj in range(len(cy_help_sort)):
                                newest_y_spliter.append(cy_help_sort[mj])
                        newest_y_spliter.append(spliter_y_new[i+1])

                        newest_y_spliter_tot.append(newest_y_spliter)
                        

                    
                        
                else:
                    line_int=[]
                    newest_y_spliter_tot=[]

                    for tj in range(len(newest_peaks)-1):
                        newest_y_spliter=[]
                        newest_y_spliter.append(spliter_y_new[i])

                        newest_y_spliter.append(spliter_y_new[i+1])

                        newest_y_spliter_tot.append(newest_y_spliter)
                    

                
                # if line_int is all -1 means that big spliters have no child and we can easily go through
                if np.all(np.array(line_int)==-1):
                    for j in range(len(newest_peaks)-1):
                        newest_y_spliter=newest_y_spliter_tot[j]


                        for n in range(len(newest_y_spliter)-1):
                            #print(j,newest_y_spliter[n],newest_y_spliter[n+1],newest_peaks[j],newest_peaks[j+1],'maaaa')
                            ##plt.imshow(regions_without_seperators[int(newest_y_spliter[n]):int(newest_y_spliter[n+1]),newest_peaks[j]:newest_peaks[j+1]])
                            ##plt.show()

                            #print(matrix_new[:,0][ (matrix_new[:,9]==1 )])
                            for jvt in    matrix_new[:,0][ (matrix_new[:,9]==1 ) & (matrix_new[:,6]> newest_y_spliter[n] ) & (matrix_new[:,7]< newest_y_spliter[n+1] ) & ( ( matrix_new[:,1]) < newest_peaks[j+1] ) & (( matrix_new[:,1])> newest_peaks[j] ) ] :
                                pass

                                ###plot_contour(regions_without_seperators.shape[0],regions_without_seperators.shape[1], contours_lines[int(jvt)])
                            #print(matrix_of_lines_ch[matrix_of_lines_ch[:,9]==1])
                            matrix_new_new=matrix_of_lines_ch[:,:][ (matrix_of_lines_ch[:,9]==1 ) & (matrix_of_lines_ch[:,6]> newest_y_spliter[n] ) & (matrix_of_lines_ch[:,7]< newest_y_spliter[n+1] ) & ( ( matrix_of_lines_ch[:,1]+500) < newest_peaks[j+1] ) & (( matrix_of_lines_ch[:,1]-500)> newest_peaks[j] )] 
                            #print(matrix_new_new,newest_y_spliter[n],newest_y_spliter[n+1],newest_peaks[j],newest_peaks[j+1],'gada')
                            if 1>0:#len( matrix_new_new[:,9][matrix_new_new[:,9]==1] )>0 and np.max(matrix_new_new[:,8][matrix_new_new[:,9]==1])>=0.2*(np.abs(newest_y_spliter[n+1]-newest_y_spliter[n] )):
                                #print( int(newest_y_spliter[n]),int(newest_y_spliter[n+1]),newest_peaks[j],newest_peaks[j+1] )
                                try:
                                    num_col_sub, peaks_neg_fin_sub=self.find_num_col(regions_without_seperators[int(newest_y_spliter[n]):int(newest_y_spliter[n+1]),newest_peaks[j]:newest_peaks[j+1]],multiplier=7.)
                                except:
                                    peaks_neg_fin_sub=[]
                            else:
                                peaks_neg_fin_sub=[]

                            peaks_sub=[]
                            peaks_sub.append(newest_peaks[j])

                            for kj in range(len(peaks_neg_fin_sub)):
                                peaks_sub.append(peaks_neg_fin_sub[kj]+newest_peaks[j])

                            peaks_sub.append(newest_peaks[j+1])

                            #peaks_sub=return_points_with_boundies(peaks_neg_fin_sub+newest_peaks[j],newest_peaks[j], newest_peaks[j+1])

                            for kh in range(len(peaks_sub)-1):
                                boxes.append([ peaks_sub[kh], peaks_sub[kh+1] ,newest_y_spliter[n],newest_y_spliter[n+1]])
                                
                                
                else:
                    for j in range(len(newest_peaks)-1):
                        

                        newest_y_spliter=newest_y_spliter_tot[j]
                        
                        if j in start_index_of_hor_parent:
                            
                            x_min_ch=x_min_hor_some[arg_child]
                            x_max_ch=x_max_hor_some[arg_child]
                            cy_hor_some_sort_child=cy_hor_some[arg_child]
                            cy_hor_some_sort_child=np.sort(cy_hor_some_sort_child)
                            
                            
                            
                            for n in range(len(newest_y_spliter)-1):
                                
                                cy_child_in=cy_hor_some_sort_child[( cy_hor_some_sort_child>newest_y_spliter[n] ) & ( cy_hor_some_sort_child<newest_y_spliter[n+1] ) ]
                                
                                if len(cy_child_in)>0:
                                    try:
                                        num_col_ch, peaks_neg_ch=self.find_num_col( regions_without_seperators[int(newest_y_spliter[n]):int(newest_y_spliter[n+1]),newest_peaks[j]:newest_peaks[j+1]],multiplier=7.0)
                                    except:
                                        peaks_neg_ch=[]
                                    #print(peaks_neg_ch,'mizzzz')
                                    #peaks_neg_ch=[]
                                    #for djh in range(len(peaks_neg_ch)):
                                    #    peaks_neg_ch.append( peaks_neg_ch[djh]+newest_peaks[j] )
                                    
                                    peaks_neg_ch_tot=self.return_points_with_boundies(peaks_neg_ch,newest_peaks[j], newest_peaks[j+1])
                                    
                                    
                                    ss_in_ch,nst_p_ch,arg_n_ch,lines_l_del_ch,lines_in_del_ch=self.return_hor_spliter_by_index_for_without_verticals(peaks_neg_ch_tot,x_min_ch,x_max_ch)
                                        
                                    
                                    
                                    
                                    
                                    newest_y_spliter_ch_tot=[]

                                    for tjj in range(len(nst_p_ch)-1):
                                        newest_y_spliter_new=[]
                                        newest_y_spliter_new.append(newest_y_spliter[n])
                                        if tjj in np.unique(ss_in_ch):
                                            

                                            #print(tj,cy_hor_some_sort,start_index_of_hor,cy_help,'maashhaha')
                                            for mjj in range(len(cy_child_in)):
                                                newest_y_spliter_new.append(cy_child_in[mjj])
                                        newest_y_spliter_new.append(newest_y_spliter[n+1])

                                        newest_y_spliter_ch_tot.append(newest_y_spliter_new)
                                        
                                    
                                    
                                    
                                    
                                    for jn in range(len(nst_p_ch)-1):
                                        newest_y_spliter_h=newest_y_spliter_ch_tot[jn]

                                        for nd in range(len(newest_y_spliter_h)-1):

                                            matrix_new_new2=matrix_of_lines_ch[:,:][ (matrix_of_lines_ch[:,9]==1 ) & (matrix_of_lines_ch[:,6]> newest_y_spliter_h[nd] ) & (matrix_of_lines_ch[:,7]< newest_y_spliter_h[nd+1] ) & ( ( matrix_of_lines_ch[:,1]+500) < nst_p_ch[jn+1] ) & (( matrix_of_lines_ch[:,1]-500)>nst_p_ch[jn] ) ]
                                            #print(matrix_new_new,newest_y_spliter[n],newest_y_spliter[n+1],newest_peaks[j],newest_peaks[j+1],'gada')
                                            if 1>0:#len( matrix_new_new2[:,9][matrix_new_new2[:,9]==1] )>0 and np.max(matrix_new_new2[:,8][matrix_new_new2[:,9]==1])>=0.2*(np.abs(newest_y_spliter_h[nd+1]-newest_y_spliter_h[nd] )):
                                                try:
                                                    num_col_sub_ch, peaks_neg_fin_sub_ch=self.find_num_col(regions_without_seperators[int(newest_y_spliter_h[nd]):int(newest_y_spliter_h[nd+1]),nst_p_ch[jn]:nst_p_ch[jn+1]],multiplier=7.0)
                                                except:
                                                    peaks_neg_fin_sub_ch=[]
                                                
                                            else:
                                                peaks_neg_fin_sub_ch=[]

                                            peaks_sub_ch=[]
                                            peaks_sub_ch.append(nst_p_ch[jn])

                                            for kjj in range(len(peaks_neg_fin_sub_ch)):
                                                peaks_sub_ch.append(peaks_neg_fin_sub_ch[kjj]+nst_p_ch[jn])

                                            peaks_sub_ch.append(nst_p_ch[jn+1])

                                            #peaks_sub=return_points_with_boundies(peaks_neg_fin_sub+newest_peaks[j],newest_peaks[j], newest_peaks[j+1])

                                            for khh in range(len(peaks_sub_ch)-1):
                                                boxes.append([ peaks_sub_ch[khh], peaks_sub_ch[khh+1] ,newest_y_spliter_h[nd],newest_y_spliter_h[nd+1]])


                        
                                else:
                                    
                                    matrix_new_new=matrix_of_lines_ch[:,:][ (matrix_of_lines_ch[:,9]==1 ) & (matrix_of_lines_ch[:,6]> newest_y_spliter[n] ) & (matrix_of_lines_ch[:,7]< newest_y_spliter[n+1] ) & ( ( matrix_of_lines_ch[:,1]+500) < newest_peaks[j+1] ) & (( matrix_of_lines_ch[:,1]-500)> newest_peaks[j] )] 
                                    #print(matrix_new_new,newest_y_spliter[n],newest_y_spliter[n+1],newest_peaks[j],newest_peaks[j+1],'gada')
                                    if 1>0:#len( matrix_new_new[:,9][matrix_new_new[:,9]==1] )>0 and np.max(matrix_new_new[:,8][matrix_new_new[:,9]==1])>=0.2*(np.abs(newest_y_spliter[n+1]-newest_y_spliter[n] )):
                                        try:
                                            num_col_sub, peaks_neg_fin_sub=self.find_num_col(regions_without_seperators[int(newest_y_spliter[n]):int(newest_y_spliter[n+1]),newest_peaks[j]:newest_peaks[j+1]],multiplier=7.0)
                                        except:
                                            peaks_neg_fin_sub=[]
                                    else:
                                        peaks_neg_fin_sub=[]

                                    peaks_sub=[]
                                    peaks_sub.append(newest_peaks[j])

                                    for kj in range(len(peaks_neg_fin_sub)):
                                        peaks_sub.append(peaks_neg_fin_sub[kj]+newest_peaks[j])

                                    peaks_sub.append(newest_peaks[j+1])

                                    #peaks_sub=return_points_with_boundies(peaks_neg_fin_sub+newest_peaks[j],newest_peaks[j], newest_peaks[j+1])

                                    for kh in range(len(peaks_sub)-1):
                                        boxes.append([ peaks_sub[kh], peaks_sub[kh+1] ,newest_y_spliter[n],newest_y_spliter[n+1]])
                                    
                                    
            
                                
                                
                        else:
                            for n in range(len(newest_y_spliter)-1):


                                #plot_contour(regions_without_seperators.shape[0],regions_without_seperators.shape[1], contours_lines[int(jvt)])
                                #print(matrix_of_lines_ch[matrix_of_lines_ch[:,9]==1])
                                matrix_new_new=matrix_of_lines_ch[:,:][ (matrix_of_lines_ch[:,9]==1 ) & (matrix_of_lines_ch[:,6]> newest_y_spliter[n] ) & (matrix_of_lines_ch[:,7]< newest_y_spliter[n+1] ) & ( ( matrix_of_lines_ch[:,1]+500) < newest_peaks[j+1] ) & (( matrix_of_lines_ch[:,1]-500)> newest_peaks[j] )] 
                                #print(matrix_new_new,newest_y_spliter[n],newest_y_spliter[n+1],newest_peaks[j],newest_peaks[j+1],'gada')
                                if 1>0:#len( matrix_new_new[:,9][matrix_new_new[:,9]==1] )>0 and np.max(matrix_new_new[:,8][matrix_new_new[:,9]==1])>=0.2*(np.abs(newest_y_spliter[n+1]-newest_y_spliter[n] )):
                                    try:
                                        num_col_sub, peaks_neg_fin_sub=self.find_num_col(regions_without_seperators[int(newest_y_spliter[n]):int(newest_y_spliter[n+1]),newest_peaks[j]:newest_peaks[j+1]],multiplier=5.0)
                                    except:
                                        peaks_neg_fin_sub=[]
                                else:
                                    peaks_neg_fin_sub=[]

                                peaks_sub=[]
                                peaks_sub.append(newest_peaks[j])

                                for kj in range(len(peaks_neg_fin_sub)):
                                    peaks_sub.append(peaks_neg_fin_sub[kj]+newest_peaks[j])

                                peaks_sub.append(newest_peaks[j+1])

                                #peaks_sub=return_points_with_boundies(peaks_neg_fin_sub+newest_peaks[j],newest_peaks[j], newest_peaks[j+1])

                                for kh in range(len(peaks_sub)-1):
                                    boxes.append([ peaks_sub[kh], peaks_sub[kh+1] ,newest_y_spliter[n],newest_y_spliter[n+1]])
                    
                            

                        
                        
                        
                        
                
            else:
                boxes.append([ 0, regions_without_seperators[:,:].shape[1] ,spliter_y_new[i],spliter_y_new[i+1]])

                
        return boxes

    def return_boxes_of_images_by_order_of_reading_2cols(self,spliter_y_new,regions_without_seperators,matrix_of_lines_ch,seperators_closeup_n):
        boxes=[]


        # here I go through main spliters and i do check whether a vertical seperator there is. If so i am searching for \
        # holes in the text and also finding spliter which covers more than one columns.
        for i in range(len(spliter_y_new)-1):
            #print(spliter_y_new[i],spliter_y_new[i+1])
            matrix_new=matrix_of_lines_ch[:,:][ (matrix_of_lines_ch[:,6]> spliter_y_new[i] ) & (matrix_of_lines_ch[:,7]< spliter_y_new[i+1] )  ] 
            #print(len( matrix_new[:,9][matrix_new[:,9]==1] ))
            
            #print(matrix_new[:,8][matrix_new[:,9]==1],'gaddaaa')
            
            # check to see is there any vertical seperator to find holes.
            if 1>0:#len( matrix_new[:,9][matrix_new[:,9]==1] )>0 and np.max(matrix_new[:,8][matrix_new[:,9]==1])>=0.1*(np.abs(spliter_y_new[i+1]-spliter_y_new[i] )):
                #print(int(spliter_y_new[i]),int(spliter_y_new[i+1]),'burayaaaa galimiirrrrrrrrrrrrrrrrrrrrrrrrrrr')
                #org_img_dichte=-gaussian_filter1d(( image_page[int(spliter_y_new[i]):int(spliter_y_new[i+1]),:,0]/255.).sum(axis=0) ,30)
                #org_img_dichte=org_img_dichte-np.min(org_img_dichte)
                ##plt.figure(figsize=(20,20))
                ##plt.plot(org_img_dichte)      
                ##plt.show()
                ###find_num_col_both_layout_and_org(regions_without_seperators,image_page[int(spliter_y_new[i]):int(spliter_y_new[i+1]),:,:],7.)
                
                try:
                    num_col, peaks_neg_fin=self.find_num_col(regions_without_seperators[int(spliter_y_new[i]):int(spliter_y_new[i+1]),:],multiplier=7.0)
                    
                
                except:
                    peaks_neg_fin=[]
                    num_col=0
                    
                peaks_neg_tot=self.return_points_with_boundies(peaks_neg_fin,0, seperators_closeup_n[:,:,0].shape[1])
                
                for kh in range(len(peaks_neg_tot)-1):
                    boxes.append([ peaks_neg_tot[kh], peaks_neg_tot[kh+1] ,spliter_y_new[i],spliter_y_new[i+1]])
                    
            else:
                boxes.append([ 0, seperators_closeup_n[:,:,0].shape[1] ,spliter_y_new[i],spliter_y_new[i+1]])
                    

                
        return boxes
    def return_hor_spliter_by_index(self,peaks_neg_fin_t,x_min_hor_some,x_max_hor_some):

        arg_min_hor_sort=np.argsort(x_min_hor_some)
        x_min_hor_some_sort=np.sort(x_min_hor_some)
        x_max_hor_some_sort=x_max_hor_some[arg_min_hor_sort]
        
        arg_minmax=np.array(range(len(peaks_neg_fin_t)))
        indexer_lines=[]
        indexes_to_delete=[]
        indexer_lines_deletions_len=[]
        indexr_uniq_ind=[]
        for i in range(len(x_min_hor_some_sort)):
            min_h=peaks_neg_fin_t-x_min_hor_some_sort[i]
            max_h=peaks_neg_fin_t-x_max_hor_some_sort[i]
            
            min_h[0]=min_h[0]#+20
            max_h[len(max_h)-1]=max_h[len(max_h)-1]##-20

            min_h_neg=arg_minmax[(min_h<0) & (np.abs(min_h)<360) ]
            max_h_neg=arg_minmax[(max_h>=0) & (np.abs(max_h)<360) ]

            if len(min_h_neg)>0 and len(max_h_neg)>0:
                deletions=list(range(min_h_neg[0]+1,max_h_neg[0]))
                unique_delets_int=[]
                #print(deletions,len(deletions),'delii')
                if len(deletions)>0:
                    #print(deletions,len(deletions),'delii2')
                    
                    for j in range(len(deletions)):
                        indexes_to_delete.append(deletions[j])
                        #print(deletions,indexes_to_delete,'badiii')
                        unique_delets=np.unique(indexes_to_delete)
                        #print(min_h_neg[0],unique_delets)
                        unique_delets_int=unique_delets[unique_delets<min_h_neg[0]]
                        
                    indexer_lines_deletions_len.append(len(deletions))
                    indexr_uniq_ind.append([deletions])
                        
                else:
                    indexer_lines_deletions_len.append(0)
                    indexr_uniq_ind.append(-999)
                    
                index_line_true=min_h_neg[0]-len(unique_delets_int)
                #print(index_line_true)
                if index_line_true>0 and min_h_neg[0]>=2:
                    index_line_true=index_line_true
                else:
                    index_line_true=min_h_neg[0]

                indexer_lines.append(index_line_true)
                
                if len(unique_delets_int)>0:
                    for dd in range(len(unique_delets_int)):
                        indexes_to_delete.append(unique_delets_int[dd])
            else:
                indexer_lines.append(-999)
                indexer_lines_deletions_len.append(-999)
                indexr_uniq_ind.append(-999)
        
        peaks_true=[]
        for m in range(len(peaks_neg_fin_t)):
            if m in indexes_to_delete:
                pass
            else:
                peaks_true.append(peaks_neg_fin_t[m])
        return indexer_lines,peaks_true,arg_min_hor_sort,indexer_lines_deletions_len,indexr_uniq_ind
    def return_region_segmentation_after_implementing_not_head_maintext_parallel(self,image_regions_eraly_p,boxes):
        image_revised=np.zeros((image_regions_eraly_p.shape[0] , image_regions_eraly_p.shape[1]))
        for i in range(len(boxes)):
            
            image_box=image_regions_eraly_p[int(boxes[i][2]):int(boxes[i][3]),int(boxes[i][0]):int(boxes[i][1])]
            image_box=np.array(image_box)
            #plt.imshow(image_box)
            #plt.show()

            #print(int(boxes[i][2]),int(boxes[i][3]),int(boxes[i][0]),int(boxes[i][1]),'addaa')
            image_box=self.implent_law_head_main_not_parallel(image_box)
            image_box=self.implent_law_head_main_not_parallel(image_box)
            image_box=self.implent_law_head_main_not_parallel(image_box)
            
            image_revised[int(boxes[i][2]):int(boxes[i][3]),int(boxes[i][0]):int(boxes[i][1])]=image_box[:,:]
        return image_revised
    
    def tear_main_texts_on_the_boundaries_of_boxes(self,img_revised_tab,boxes):
        for i in range(len(boxes)):
            img_revised_tab[ int(boxes[i][2]):int(boxes[i][3]),int(boxes[i][1]-10):int(boxes[i][1]), 0][img_revised_tab[ int(boxes[i][2]):int(boxes[i][3]),int(boxes[i][1]-10):int(boxes[i][1]), 0]==1]=0
            img_revised_tab[ int(boxes[i][2]):int(boxes[i][3]),int(boxes[i][1]-10):int(boxes[i][1]), 1][img_revised_tab[ int(boxes[i][2]):int(boxes[i][3]),int(boxes[i][1]-10):int(boxes[i][1]), 1]==1]=0
            img_revised_tab[ int(boxes[i][2]):int(boxes[i][3]),int(boxes[i][1]-10):int(boxes[i][1]), 2][img_revised_tab[ int(boxes[i][2]):int(boxes[i][3]),int(boxes[i][1]-10):int(boxes[i][1]), 2]==1]=0
        return img_revised_tab
            
        
    def implent_law_head_main_not_parallel(self,text_regions):
        #print(text_regions.shape)
        text_indexes=[1 , 2 ]# 1: main text , 2: header , 3: comments
        
        
        for t_i in text_indexes:
            textline_mask=(text_regions[:,:]==t_i)
            textline_mask=textline_mask*255.0

            textline_mask=textline_mask.astype(np.uint8)
            textline_mask=np.repeat(textline_mask[:, :, np.newaxis], 3, axis=2)
            kernel = np.ones((5,5),np.uint8)

            #print(type(textline_mask),np.unique(textline_mask),textline_mask.shape)
            imgray = cv2.cvtColor(textline_mask, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(imgray, 0, 255, 0)
            
            if t_i==1:
                contours_main,hirarchy=cv2.findContours(thresh.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                #print(type(contours_main))
                areas_main=np.array([cv2.contourArea(contours_main[j]) for j in range(len(contours_main))])
                M_main=[cv2.moments(contours_main[j]) for j in range(len(contours_main))]
                cx_main=[(M_main[j]['m10']/(M_main[j]['m00']+1e-32)) for j in range(len(M_main))]
                cy_main=[(M_main[j]['m01']/(M_main[j]['m00']+1e-32)) for j in range(len(M_main))]
                x_min_main=np.array([np.min(contours_main[j][:,0,0]) for j in range(len(contours_main))])
                x_max_main=np.array([np.max(contours_main[j][:,0,0]) for j in range(len(contours_main))])
                
                y_min_main=np.array([np.min(contours_main[j][:,0,1]) for j in range(len(contours_main))])
                y_max_main=np.array([np.max(contours_main[j][:,0,1]) for j in range(len(contours_main))])
                #print(contours_main[0],np.shape(contours_main[0]),contours_main[0][:,0,0])
            elif t_i==2:
                contours_header,hirarchy=cv2.findContours(thresh.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                #print(type(contours_header))
                areas_header=np.array([cv2.contourArea(contours_header[j]) for j in range(len(contours_header))])
                M_header=[cv2.moments(contours_header[j]) for j in range(len(contours_header))]
                cx_header=[(M_header[j]['m10']/(M_header[j]['m00']+1e-32)) for j in range(len(M_header))]
                cy_header=[(M_header[j]['m01']/(M_header[j]['m00']+1e-32)) for j in range(len(M_header))]
                
                x_min_header=np.array([np.min(contours_header[j][:,0,0]) for j in range(len(contours_header))])
                x_max_header=np.array([np.max(contours_header[j][:,0,0]) for j in range(len(contours_header))])
                
                y_min_header=np.array([np.min(contours_header[j][:,0,1]) for j in range(len(contours_header))])
                y_max_header=np.array([np.max(contours_header[j][:,0,1]) for j in range(len(contours_header))])
        
        args=np.array(range(1,len(cy_header)+1))
        args_main=np.array(range(1,len(cy_main)+1))
        for jj in range(len(contours_main)):
            headers_in_main=[(cy_header>y_min_main[jj]) & ((cy_header<y_max_main[jj]))]
            mains_in_main=[(cy_main>y_min_main[jj]) & ((cy_main<y_max_main[jj]))]
            args_log=args*headers_in_main
            res=args_log[args_log>0]
            res_true=res-1
            
            args_log_main=args_main*mains_in_main
            res_main=args_log_main[args_log_main>0]
            res_true_main=res_main-1
            
            if len(res_true)>0:
                sum_header=np.sum(areas_header[res_true])
                sum_main=np.sum(areas_main[res_true_main])
                if sum_main>sum_header:
                    cnt_int=[contours_header[j] for j in res_true]
                    text_regions=cv2.fillPoly(text_regions, pts =cnt_int, color=(1,1,1))
                else:
                    cnt_int=[contours_main[j] for j in res_true_main]
                    text_regions=cv2.fillPoly(text_regions, pts =cnt_int, color=(2,2,2))
                    
        for jj in range(len(contours_header)):
            main_in_header=[(cy_main>y_min_header[jj]) & ((cy_main<y_max_header[jj]))]
            header_in_header=[(cy_header>y_min_header[jj]) & ((cy_header<y_max_header[jj]))]
            args_log=args_main*main_in_header
            res=args_log[args_log>0]
            res_true=res-1
            
            args_log_header=args*header_in_header
            res_header=args_log_header[args_log_header>0]
            res_true_header=res_header-1
            
            if len(res_true)>0:
                
                sum_header=np.sum(areas_header[res_true_header])
                sum_main=np.sum(areas_main[res_true])
                
                if sum_main>sum_header:
    
                    cnt_int=[contours_header[j] for j in res_true_header]
                    text_regions=cv2.fillPoly(text_regions, pts =cnt_int, color=(1,1,1))
                else:
                    cnt_int=[contours_main[j] for j in res_true]
                    text_regions=cv2.fillPoly(text_regions, pts =cnt_int, color=(2,2,2))
                    
            
            
            
        return text_regions
    
    def delete_seperator_around(self,spliter_y,peaks_neg,image_by_region):
        # format of subboxes box=[x1, x2 , y1, y2]
        
        if len(image_by_region.shape)==3:
            for i in range(len(spliter_y)-1):
                for j in range(1,len(peaks_neg[i])-1):
                    image_by_region[int(spliter_y[i]):int(spliter_y[i+1]),peaks_neg[i][j]-int(1./20.*peaks_neg[i][j]):peaks_neg[i][j]+int(1./20.*peaks_neg[i][j]),0][image_by_region[int(spliter_y[i]):int(spliter_y[i+1]),peaks_neg[i][j]-int(1./20.*peaks_neg[i][j]):peaks_neg[i][j]+int(1./20.*peaks_neg[i][j]),0]==6 ]=0
                    image_by_region[spliter_y[i]:spliter_y[i+1],peaks_neg[i][j]-int(1./20.*peaks_neg[i][j]):peaks_neg[i][j]+int(1./20.*peaks_neg[i][j]),0][image_by_region[int(spliter_y[i]):int(spliter_y[i+1]),peaks_neg[i][j]-int(1./20.*peaks_neg[i][j]):peaks_neg[i][j]+int(1./20.*peaks_neg[i][j]),1]==6 ]=0
                    image_by_region[spliter_y[i]:spliter_y[i+1],peaks_neg[i][j]-int(1./20.*peaks_neg[i][j]):peaks_neg[i][j]+int(1./20.*peaks_neg[i][j]),0][image_by_region[int(spliter_y[i]):int(spliter_y[i+1]),peaks_neg[i][j]-int(1./20.*peaks_neg[i][j]):peaks_neg[i][j]+int(1./20.*peaks_neg[i][j]),2]==6 ]=0
                    
                    image_by_region[int(spliter_y[i]):int(spliter_y[i+1]),peaks_neg[i][j]-int(1./20.*peaks_neg[i][j]):peaks_neg[i][j]+int(1./20.*peaks_neg[i][j]),0][image_by_region[int(spliter_y[i]):int(spliter_y[i+1]),peaks_neg[i][j]-int(1./20.*peaks_neg[i][j]):peaks_neg[i][j]+int(1./20.*peaks_neg[i][j]),0]==7 ]=0
                    image_by_region[int(spliter_y[i]):int(spliter_y[i+1]),peaks_neg[i][j]-int(1./20.*peaks_neg[i][j]):peaks_neg[i][j]+int(1./20.*peaks_neg[i][j]),0][image_by_region[int(spliter_y[i]):int(spliter_y[i+1]),peaks_neg[i][j]-int(1./20.*peaks_neg[i][j]):peaks_neg[i][j]+int(1./20.*peaks_neg[i][j]),1]==7 ]=0
                    image_by_region[int(spliter_y[i]):int(spliter_y[i+1]),peaks_neg[i][j]-int(1./20.*peaks_neg[i][j]):peaks_neg[i][j]+int(1./20.*peaks_neg[i][j]),0][image_by_region[int(spliter_y[i]):int(spliter_y[i+1]),peaks_neg[i][j]-int(1./20.*peaks_neg[i][j]):peaks_neg[i][j]+int(1./20.*peaks_neg[i][j]),2]==7 ]=0
        else:
            for i in range(len(spliter_y)-1):
                for j in range(1,len(peaks_neg[i])-1):
                    image_by_region[int(spliter_y[i]):int(spliter_y[i+1]),peaks_neg[i][j]-int(1./20.*peaks_neg[i][j]):peaks_neg[i][j]+int(1./20.*peaks_neg[i][j])][image_by_region[int(spliter_y[i]):int(spliter_y[i+1]),peaks_neg[i][j]-int(1./20.*peaks_neg[i][j]):peaks_neg[i][j]+int(1./20.*peaks_neg[i][j])]==6 ]=0
                    
                    image_by_region[int(spliter_y[i]):int(spliter_y[i+1]),peaks_neg[i][j]-int(1./20.*peaks_neg[i][j]):peaks_neg[i][j]+int(1./20.*peaks_neg[i][j])][image_by_region[int(spliter_y[i]):int(spliter_y[i+1]),peaks_neg[i][j]-int(1./20.*peaks_neg[i][j]):peaks_neg[i][j]+int(1./20.*peaks_neg[i][j])]==7 ]=0
        return image_by_region
    
    def find_features_of_contoures(self,contours_main):
        

        areas_main=np.array([cv2.contourArea(contours_main[j]) for j in range(len(contours_main))])
        M_main=[cv2.moments(contours_main[j]) for j in range(len(contours_main))]
        cx_main=[(M_main[j]['m10']/(M_main[j]['m00']+1e-32)) for j in range(len(M_main))]
        cy_main=[(M_main[j]['m01']/(M_main[j]['m00']+1e-32)) for j in range(len(M_main))]
        x_min_main=np.array([np.min(contours_main[j][:,0,0]) for j in range(len(contours_main))])
        x_max_main=np.array([np.max(contours_main[j][:,0,0]) for j in range(len(contours_main))])

        y_min_main=np.array([np.min(contours_main[j][:,0,1]) for j in range(len(contours_main))])
        y_max_main=np.array([np.max(contours_main[j][:,0,1]) for j in range(len(contours_main))])

        
        return y_min_main ,y_max_main
    

    def add_tables_heuristic_to_layout(self,image_regions_eraly_p,boxes,slope_mean_hor,spliter_y,peaks_neg_tot,image_revised):
        
        image_revised_1=self.delete_seperator_around(spliter_y,peaks_neg_tot,image_revised)
        img_comm_e=np.zeros(image_revised_1.shape)
        img_comm=np.repeat(img_comm_e[:, :, np.newaxis], 3, axis=2)

        for indiv in np.unique(image_revised_1):
            
            #print(indiv,'indd')
            image_col=(image_revised_1==indiv)*255
            img_comm_in=np.repeat(image_col[:, :, np.newaxis], 3, axis=2)
            img_comm_in=img_comm_in.astype(np.uint8)

            imgray = cv2.cvtColor(img_comm_in, cv2.COLOR_BGR2GRAY)


            ret, thresh = cv2.threshold(imgray, 0, 255, 0)

            contours,hirarchy=cv2.findContours(thresh.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


            main_contours=self.filter_contours_area_of_image_tables(thresh,contours,hirarchy,max_area=1,min_area=0.0001)


            img_comm=cv2.fillPoly(img_comm, pts =main_contours, color=(indiv,indiv,indiv))
            ###img_comm_in=cv2.fillPoly(img_comm, pts =interior_contours, color=(0,0,0))


            #img_comm=np.repeat(img_comm[:, :, np.newaxis], 3, axis=2)
            img_comm=img_comm.astype(np.uint8)
            

        
        
        
        if not self.isNaN(slope_mean_hor):
            image_revised_last=np.zeros((image_regions_eraly_p.shape[0] , image_regions_eraly_p.shape[1],3))
            for i in range(len(boxes)):


                image_box=img_comm[int(boxes[i][2]):int(boxes[i][3]),int(boxes[i][0]):int(boxes[i][1]),:]

                image_box_tabels_1=(image_box[:,:,0]==7)*1





                contours_tab,_=self.return_contours_of_image(image_box_tabels_1)
                
                contours_tab=self.filter_contours_area_of_image_tables(image_box_tabels_1,contours_tab,_,1,0.001)

                image_box_tabels_1=(image_box[:,:,0]==6)*1


                image_box_tabels_and_m_text=( (image_box[:,:,0]==7) | (image_box[:,:,0]==1) )*1
                image_box_tabels_and_m_text=image_box_tabels_and_m_text.astype(np.uint8)


                image_box_tabels_1=image_box_tabels_1.astype(np.uint8)
                image_box_tabels_1 = cv2.dilate(image_box_tabels_1,self.kernel,iterations = 5)




                contours_table_m_text,_=self.return_contours_of_image(image_box_tabels_and_m_text)


                image_box_tabels=np.repeat(image_box_tabels_1[:, :, np.newaxis], 3, axis=2)

                image_box_tabels=image_box_tabels.astype(np.uint8)
                imgray = cv2.cvtColor(image_box_tabels, cv2.COLOR_BGR2GRAY)
                ret, thresh = cv2.threshold(imgray, 0, 255, 0)


                contours_line,hierachy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


                y_min_main_line ,y_max_main_line=self.find_features_of_contoures(contours_line)
                #_,_,y_min_main_line ,y_max_main_line,x_min_main_line,x_max_main_line=find_new_features_of_contoures(contours_line)
                y_min_main_tab ,y_max_main_tab=self.find_features_of_contoures(contours_tab)

                cx_tab_m_text,cy_tab_m_text ,x_min_tab_m_text , x_max_tab_m_text, y_min_tab_m_text ,y_max_tab_m_text=self.find_new_features_of_contoures(contours_table_m_text)
                cx_tabl,cy_tabl ,x_min_tabl , x_max_tabl, y_min_tabl ,y_max_tabl=self.find_new_features_of_contoures(contours_tab)


                if len(y_min_main_tab )>0:
                    y_down_tabs=[]
                    y_up_tabs=[]

                    for i_t in range(len(y_min_main_tab )):
                        y_down_tab=[]
                        y_up_tab=[]
                        for i_l in range(len(y_min_main_line)):
                            if y_min_main_tab[i_t]>y_min_main_line[i_l] and  y_max_main_tab[i_t]>y_min_main_line[i_l] and y_min_main_tab[i_t]>y_max_main_line[i_l] and y_max_main_tab[i_t]>y_min_main_line[i_l]:
                                pass
                            elif y_min_main_tab[i_t]<y_max_main_line[i_l] and y_max_main_tab[i_t]<y_max_main_line[i_l] and y_max_main_tab[i_t]<y_min_main_line[i_l] and y_min_main_tab[i_t]<y_min_main_line[i_l]:
                                pass
                            elif np.abs(y_max_main_line[i_l]-y_min_main_line[i_l])<100:
                                pass

                            else:
                                y_up_tab.append(np.min([y_min_main_line[i_l], y_min_main_tab[i_t] ])  )
                                y_down_tab.append( np.max([ y_max_main_line[i_l],y_max_main_tab[i_t] ]) )

                        if len(y_up_tab)==0:
                            for v_n in range(len(cx_tab_m_text)):
                                if cx_tabl[i_t]<= x_max_tab_m_text[v_n] and cx_tabl[i_t]>= x_min_tab_m_text[v_n] and cy_tabl[i_t]<= y_max_tab_m_text[v_n] and cy_tabl[i_t]>= y_min_tab_m_text[v_n] and cx_tabl[i_t]!=cx_tab_m_text[v_n] and cy_tabl[i_t]!=cy_tab_m_text[v_n]:
                                    y_up_tabs.append(y_min_tab_m_text[v_n])
                                    y_down_tabs.append(y_max_tab_m_text[v_n])
                            #y_up_tabs.append(y_min_main_tab[i_t])
                            #y_down_tabs.append(y_max_main_tab[i_t])
                        else:
                            y_up_tabs.append(np.min(y_up_tab))
                            y_down_tabs.append(np.max(y_down_tab))


                else:
                    y_down_tabs=[]
                    y_up_tabs=[]
                    pass

                for ii in range(len(y_up_tabs)):
                    image_box[y_up_tabs[ii]:y_down_tabs[ii],:,0]=7






                image_revised_last[int(boxes[i][2]):int(boxes[i][3]),int(boxes[i][0]):int(boxes[i][1]),:]=image_box[:,:,:]


        else:
            for i in range(len(boxes)):

                image_box=img_comm[int(boxes[i][2]):int(boxes[i][3]),int(boxes[i][0]):int(boxes[i][1]),:]
                image_revised_last[int(boxes[i][2]):int(boxes[i][3]),int(boxes[i][0]):int(boxes[i][1]),:]=image_box[:,:,:]

                ##plt.figure(figsize=(20,20))
                ##plt.imshow(image_box[:,:,0])
                ##plt.show()
        return image_revised_last
    def find_features_of_contours(self,contours_main):
        

        areas_main=np.array([cv2.contourArea(contours_main[j]) for j in range(len(contours_main))])
        M_main=[cv2.moments(contours_main[j]) for j in range(len(contours_main))]
        cx_main=[(M_main[j]['m10']/(M_main[j]['m00']+1e-32)) for j in range(len(M_main))]
        cy_main=[(M_main[j]['m01']/(M_main[j]['m00']+1e-32)) for j in range(len(M_main))]
        x_min_main=np.array([np.min(contours_main[j][:,0,0]) for j in range(len(contours_main))])
        x_max_main=np.array([np.max(contours_main[j][:,0,0]) for j in range(len(contours_main))])

        y_min_main=np.array([np.min(contours_main[j][:,0,1]) for j in range(len(contours_main))])
        y_max_main=np.array([np.max(contours_main[j][:,0,1]) for j in range(len(contours_main))])


        
        return y_min_main ,y_max_main,areas_main
    def remove_headers_and_mains_intersection(self,seperators_closeup_n,img_revised_tab,boxes):    
        for ind in range(len(boxes)):
            asp=np.zeros((img_revised_tab[:,:,0].shape[0],seperators_closeup_n[:,:,0].shape[1]))
            asp[ int(boxes[ind][2]):int(boxes[ind][3]),int(boxes[ind][0]):int(boxes[ind][1])] = img_revised_tab[ int(boxes[ind][2]):int(boxes[ind][3]),int(boxes[ind][0]):int(boxes[ind][1]),0] 

            head_patch_con=( asp[:,:]==2)*1
            main_patch_con=( asp[:,:]==1)*1
            #print(head_patch_con)
            head_patch_con=head_patch_con.astype(np.uint8)
            main_patch_con=main_patch_con.astype(np.uint8)

            head_patch_con=np.repeat(head_patch_con[:, :, np.newaxis], 3, axis=2)
            main_patch_con=np.repeat(main_patch_con[:, :, np.newaxis], 3, axis=2)

            imgray = cv2.cvtColor(head_patch_con, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(imgray, 0, 255, 0)


            contours_head_patch_con,hiearchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            contours_head_patch_con=self.return_parent_contours(contours_head_patch_con,hiearchy)

            imgray = cv2.cvtColor(main_patch_con, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(imgray, 0, 255, 0)


            contours_main_patch_con,hiearchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            contours_main_patch_con=self.return_parent_contours(contours_main_patch_con,hiearchy)
            
            
            y_patch_head_min, y_patch_head_max,_= self.find_features_of_contours(contours_head_patch_con)
            y_patch_main_min, y_patch_main_max,_= self.find_features_of_contours(contours_main_patch_con)
            
            
            for i in range(len(y_patch_head_min)):
                for j in range(len(y_patch_main_min)):
                    if y_patch_head_max[i]>y_patch_main_min[j] and y_patch_head_min[i]<y_patch_main_min[j]:
                        y_down=y_patch_head_max[i]
                        y_up=y_patch_main_min[j]

                        patch_intersection=np.zeros(asp.shape)
                        patch_intersection[y_up:y_down,:]=asp[y_up:y_down,:]


                        head_patch_con=( patch_intersection[:,:]==2)*1
                        main_patch_con=( patch_intersection[:,:]==1)*1
                        head_patch_con=head_patch_con.astype(np.uint8)
                        main_patch_con=main_patch_con.astype(np.uint8)

                        head_patch_con=np.repeat(head_patch_con[:, :, np.newaxis], 3, axis=2)
                        main_patch_con=np.repeat(main_patch_con[:, :, np.newaxis], 3, axis=2)

                        imgray = cv2.cvtColor(head_patch_con, cv2.COLOR_BGR2GRAY)
                        ret, thresh = cv2.threshold(imgray, 0, 255, 0)


                        contours_head_patch_con,hiearchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                        contours_head_patch_con=self.return_parent_contours(contours_head_patch_con,hiearchy)

                        imgray = cv2.cvtColor(main_patch_con, cv2.COLOR_BGR2GRAY)
                        ret, thresh = cv2.threshold(imgray, 0, 255, 0)


                        contours_main_patch_con,hiearchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                        contours_main_patch_con=self.return_parent_contours(contours_main_patch_con,hiearchy)

                        _,_,areas_head= self.find_features_of_contours(contours_head_patch_con)
                        _,_,areas_main= self.find_features_of_contours(contours_main_patch_con)

                        if np.sum(areas_head)>np.sum(areas_main):
                            img_revised_tab[y_up:y_down,int(boxes[ind][0]):int(boxes[ind][1]),0][img_revised_tab[y_up:y_down,int(boxes[ind][0]):int(boxes[ind][1]),0]==1 ]=2 
                        else:
                            img_revised_tab[y_up:y_down,int(boxes[ind][0]):int(boxes[ind][1]),0][img_revised_tab[y_up:y_down,int(boxes[ind][0]):int(boxes[ind][1]),0]==2 ]=1 



                    elif y_patch_head_min[i]<y_patch_main_max[j] and y_patch_head_max[i]>y_patch_main_max[j]:
                        y_down=y_patch_main_max[j]
                        y_up=y_patch_head_min[i]

                        patch_intersection=np.zeros(asp.shape)
                        patch_intersection[y_up:y_down,:]=asp[y_up:y_down,:]
                        
                        
                        head_patch_con=( patch_intersection[:,:]==2)*1
                        main_patch_con=( patch_intersection[:,:]==1)*1
                        head_patch_con=head_patch_con.astype(np.uint8)
                        main_patch_con=main_patch_con.astype(np.uint8)

                        head_patch_con=np.repeat(head_patch_con[:, :, np.newaxis], 3, axis=2)
                        main_patch_con=np.repeat(main_patch_con[:, :, np.newaxis], 3, axis=2)

                        imgray = cv2.cvtColor(head_patch_con, cv2.COLOR_BGR2GRAY)
                        ret, thresh = cv2.threshold(imgray, 0, 255, 0)


                        contours_head_patch_con,hiearchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                        contours_head_patch_con=self.return_parent_contours(contours_head_patch_con,hiearchy)

                        imgray = cv2.cvtColor(main_patch_con, cv2.COLOR_BGR2GRAY)
                        ret, thresh = cv2.threshold(imgray, 0, 255, 0)


                        contours_main_patch_con,hiearchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                        contours_main_patch_con=self.return_parent_contours(contours_main_patch_con,hiearchy)

                        _,_,areas_head= self.find_features_of_contours(contours_head_patch_con)
                        _,_,areas_main= self.find_features_of_contours(contours_main_patch_con)

                        if np.sum(areas_head)>np.sum(areas_main):
                            img_revised_tab[y_up:y_down,int(boxes[ind][0]):int(boxes[ind][1]),0][img_revised_tab[y_up:y_down,int(boxes[ind][0]):int(boxes[ind][1]),0]==1 ]=2 
                        else:
                            img_revised_tab[y_up:y_down,int(boxes[ind][0]):int(boxes[ind][1]),0][img_revised_tab[y_up:y_down,int(boxes[ind][0]):int(boxes[ind][1]),0]==2 ]=1 



                        #print(np.unique(patch_intersection) )
                        ##plt.figure(figsize=(20,20))
                        ##plt.imshow(patch_intersection)
                        ##plt.show()
                    else:
                        pass

        return img_revised_tab
    
    def order_of_regions(self,textline_mask,contours_main,contours_header, y_ref):
        mada_n=textline_mask.sum(axis=1)
        y=mada_n[:]

        y_help=np.zeros(len(y)+40)
        y_help[20:len(y)+20]=y
        x=np.array( range(len(y)) )


        peaks_real, _ = find_peaks(gaussian_filter1d(y, 3), height=0)
        
        ##plt.imshow(textline_mask[:,:])
        ##plt.show()
        

        sigma_gaus=8

        z= gaussian_filter1d(y_help, sigma_gaus)
        zneg_rev=-y_help+np.max(y_help)

        zneg=np.zeros(len(zneg_rev)+40)
        zneg[20:len(zneg_rev)+20]=zneg_rev
        zneg= gaussian_filter1d(zneg, sigma_gaus)


        peaks, _ = find_peaks(z, height=0)
        peaks_neg, _ = find_peaks(zneg, height=0)

        peaks_neg=peaks_neg-20-20
        peaks=peaks-20

        ##plt.plot(z)
        ##plt.show()

        if contours_main!=None:
            areas_main=np.array([cv2.contourArea(contours_main[j]) for j in range(len(contours_main))])
            M_main=[cv2.moments(contours_main[j]) for j in range(len(contours_main))]
            cx_main=[(M_main[j]['m10']/(M_main[j]['m00']+1e-32)) for j in range(len(M_main))]
            cy_main=[(M_main[j]['m01']/(M_main[j]['m00']+1e-32)) for j in range(len(M_main))]
            x_min_main=np.array([np.min(contours_main[j][:,0,0]) for j in range(len(contours_main))])
            x_max_main=np.array([np.max(contours_main[j][:,0,0]) for j in range(len(contours_main))])

            y_min_main=np.array([np.min(contours_main[j][:,0,1]) for j in range(len(contours_main))])
            y_max_main=np.array([np.max(contours_main[j][:,0,1]) for j in range(len(contours_main))])

        if len(contours_header)!=None:
            areas_header=np.array([cv2.contourArea(contours_header[j]) for j in range(len(contours_header))])
            M_header=[cv2.moments(contours_header[j]) for j in range(len(contours_header))]
            cx_header=[(M_header[j]['m10']/(M_header[j]['m00']+1e-32)) for j in range(len(M_header))]
            cy_header=[(M_header[j]['m01']/(M_header[j]['m00']+1e-32)) for j in range(len(M_header))]

            x_min_header=np.array([np.min(contours_header[j][:,0,0]) for j in range(len(contours_header))])
            x_max_header=np.array([np.max(contours_header[j][:,0,0]) for j in range(len(contours_header))])

            y_min_header=np.array([np.min(contours_header[j][:,0,1]) for j in range(len(contours_header))])
            y_max_header=np.array([np.max(contours_header[j][:,0,1]) for j in range(len(contours_header))])
            #print(cy_main,'mainy')

        #print(cy_main,'cyyyy')
        #print(cx_main,'cxxxx')


        if contours_main!=None:
            indexer_main=np.array(range(len(contours_main)))


        if contours_main!=None:
            len_main=len(contours_main)
        else:
            len_main=0


        matrix_of_orders=np.zeros((len(contours_main)+len(contours_header),5))

        matrix_of_orders[:,0]=np.array( range( len(contours_main)+len(contours_header) ) )

        matrix_of_orders[:len(contours_main),1]=1
        matrix_of_orders[len(contours_main):,1]=2

        matrix_of_orders[:len(contours_main),2]=cx_main
        matrix_of_orders[len(contours_main):,2]=cx_header

        matrix_of_orders[:len(contours_main),3]=cy_main
        matrix_of_orders[len(contours_main):,3]=cy_header


        matrix_of_orders[:len(contours_main),4]=np.array( range( len(contours_main) ) )
        matrix_of_orders[len(contours_main):,4]=np.array( range( len(contours_header) ) )


        peaks_neg_new=[]

        peaks_neg_new.append(0+y_ref)
        for iii in range(len(peaks_neg)):
            peaks_neg_new.append(peaks_neg[iii]+y_ref)

        peaks_neg_new.append(textline_mask.shape[0]+y_ref)
        


        
        #print(peaks_neg_new,np.max(peaks_neg_new))
        final_indexers_sorted=[]
        final_types=[]
        final_index_type=[]
        for i in range(len(peaks_neg_new)-1):
            top=peaks_neg_new[i]
            down=peaks_neg_new[i+1]

            indexes_in=matrix_of_orders[:,0][(matrix_of_orders[:,3]>=top) & ((matrix_of_orders[:,3]<down))]
            cxs_in=matrix_of_orders[:,2][(matrix_of_orders[:,3]>=top) & ((matrix_of_orders[:,3]<down))]
            cys_in=matrix_of_orders[:,3][(matrix_of_orders[:,3]>=top) & ((matrix_of_orders[:,3]<down))]
            types_of_text=matrix_of_orders[:,1][(matrix_of_orders[:,3]>=top) & ((matrix_of_orders[:,3]<down))]
            index_types_of_text=matrix_of_orders[:,4][(matrix_of_orders[:,3]>=top) & ((matrix_of_orders[:,3]<down))]
            
            #print(top,down)
            #print(cys_in,'cyyyins')
            #print(indexes_in,'indexes')
            sorted_inside=np.argsort(cxs_in)

            ind_in_int=indexes_in[sorted_inside]
            ind_in_type=types_of_text[sorted_inside]
            ind_ind_type=index_types_of_text[sorted_inside]

            for j in range(len(ind_in_int)):
                final_indexers_sorted.append(int(ind_in_int[j]) )
                final_types.append(int(ind_in_type[j]))
                final_index_type.append(int(ind_ind_type[j]))
                
        ##matrix_of_orders[:len_main,4]=final_indexers_sorted[:]
        
        #print(peaks_neg_new,'peaks')
        #print(final_indexers_sorted,'indexsorted')
        #print(final_types,'types')

        return final_indexers_sorted, matrix_of_orders,final_types,final_index_type




    def order_and_id_of_texts(self,found_polygons_text_region ,found_polygons_text_region_h,matrix_of_orders ,indexes_sorted,index_of_types, kind_of_texts, ref_point ):
        indexes_sorted=np.array(indexes_sorted)
        index_of_types=np.array(index_of_types)
        kind_of_texts=np.array(kind_of_texts)
        
        id_of_texts=[]
        order_of_texts=[]

        index_of_types_1=index_of_types[kind_of_texts==1]
        indexes_sorted_1=indexes_sorted[kind_of_texts==1]
        

        
        index_of_types_2=index_of_types[kind_of_texts==2]
        indexes_sorted_2=indexes_sorted[kind_of_texts==2]
        

        index_b=0+ref_point
        for mm in range(len(found_polygons_text_region)):
        
            id_of_texts.append('r'+str(index_b) )
            interest=indexes_sorted_1[indexes_sorted_1==index_of_types_1[mm] ]
            order_of_texts.append(interest[0])
            index_b+=1
            
        for mm in range(len(found_polygons_text_region_h)):
            id_of_texts.append('r'+str(index_b) )
            interest=indexes_sorted_2[index_of_types_2[mm]]
            order_of_texts.append(interest)
            index_b+=1
                
        return order_of_texts, id_of_texts
    def get_text_region_boxes_by_given_contours(self,image_textline,contours,img_org):

        
        kernel = np.ones((5,5),np.uint8)
        boxes=[]
        contours_new=[]
        for jj in range(len(contours)):
            x,y,w,h = cv2.boundingRect(contours[jj])

            boxes.append([x,y,w,h])
            contours_new.append(contours[jj])

        del contours
        return boxes,contours_new


    def return_teilwiese_deskewed_lines(self,text_regions_p,textline_rotated):
        
        kernel = np.ones((5, 5), np.uint8)
        textline_rotated = cv2.erode(textline_rotated, kernel, iterations=1)
        
        textline_rotated_new=np.zeros(textline_rotated.shape)
        rgb_m=1
        rgb_h=2
        
        cnt_m,boxes_m=self.return_contours_of_interested_region_and_bounding_box(text_regions_p,rgb_m)
        cnt_h,boxes_h=self.return_contours_of_interested_region_and_bounding_box(text_regions_p,rgb_h)
        
        areas_cnt_m=np.array([cv2.contourArea(cnt_m[j]) for j in range(len(cnt_m))])
        
        argmax=np.argmax(areas_cnt_m)
        
        #plt.imshow(textline_rotated[ boxes_m[argmax][1]:boxes_m[argmax][1]+boxes_m[argmax][3] ,boxes_m[argmax][0]:boxes_m[argmax][0]+boxes_m[argmax][2]])
        #plt.show()
        
        
        
        for argmax in range(len(boxes_m)):
            
            textline_text_region=textline_rotated[ boxes_m[argmax][1]:boxes_m[argmax][1]+boxes_m[argmax][3] ,boxes_m[argmax][0]:boxes_m[argmax][0]+boxes_m[argmax][2]  ]
            
            textline_text_region_revised=self.seperate_lines_new(textline_text_region,0)
            #except:
            #    textline_text_region_revised=textline_rotated[ boxes_m[argmax][1]:boxes_m[argmax][1]+boxes_m[argmax][3] ,boxes_m[argmax][0]:boxes_m[argmax][0]+boxes_m[argmax][2]  ]
            textline_rotated_new[boxes_m[argmax][1]:boxes_m[argmax][1]+boxes_m[argmax][3] ,boxes_m[argmax][0]:boxes_m[argmax][0]+boxes_m[argmax][2]]=textline_text_region_revised[:,:]
        
        #textline_rotated_new[textline_rotated_new>0]=1
        #textline_rotated_new[textline_rotated_new<0]=0
        #plt.imshow(textline_rotated_new)
        #plt.show()
    def return_contours_of_interested_region_and_bounding_box(self,region_pre_p,pixel):
        
        # pixels of images are identified by 5
        cnts_images=(region_pre_p[:,:,0]==pixel)*1
        cnts_images=cnts_images.astype(np.uint8)
        cnts_images=np.repeat(cnts_images[:, :, np.newaxis], 3, axis=2)
        imgray = cv2.cvtColor(cnts_images, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 0, 255, 0)
        contours_imgs,hiearchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        contours_imgs=self.return_parent_contours(contours_imgs,hiearchy)
        contours_imgs=self.filter_contours_area_of_image_tables(thresh,contours_imgs,hiearchy,max_area=1,min_area=0.0003)
        
        boxes = []
        
        for jj in range(len(contours_imgs)):
            x, y, w, h = cv2.boundingRect(contours_imgs[jj])
            boxes.append([int(x), int(y), int(w), int(h)])
        return contours_imgs,boxes
    def find_number_of_columns_in_document(self,region_pre_p):
        
        seperators_closeup=( (region_pre_p[:,:,:]==3))*1
        
        seperators_closeup[0:110,:,:]=0
        seperators_closeup[seperators_closeup.shape[0]-150:,:,:]=0
        
        kernel = np.ones((5,5),np.uint8)

        seperators_closeup=seperators_closeup.astype(np.uint8)
        seperators_closeup = cv2.dilate(seperators_closeup,kernel,iterations = 1)
        seperators_closeup = cv2.erode(seperators_closeup,kernel,iterations = 1)
        
        ##plt.imshow(seperators_closeup[:,:,0])
        ##plt.show()
        seperators_closeup_new=np.zeros((seperators_closeup.shape[0] ,seperators_closeup.shape[1] ))
        
        
        
        ##_,seperators_closeup_n=self.combine_hor_lines_and_delete_cross_points_and_get_lines_features_back(region_pre_p[:,:,0])
        seperators_closeup_n=np.copy(seperators_closeup)
        
        seperators_closeup_n=seperators_closeup_n.astype(np.uint8)
        ##plt.imshow(seperators_closeup_n[:,:,0])
        ##plt.show()
        
        seperators_closeup_n_binary=np.zeros(( seperators_closeup_n.shape[0],seperators_closeup_n.shape[1]) )
        seperators_closeup_n_binary[:,:]=seperators_closeup_n[:,:,0]
        
        seperators_closeup_n_binary[:,:][seperators_closeup_n_binary[:,:]!=0]=1
        #seperators_closeup_n_binary[:,:][seperators_closeup_n_binary[:,:]==0]=255
        #seperators_closeup_n_binary[:,:][seperators_closeup_n_binary[:,:]==-255]=0
        
        
        #seperators_closeup_n_binary=(seperators_closeup_n_binary[:,:]==2)*1
        
        #gray = cv2.cvtColor(seperators_closeup_n, cv2.COLOR_BGR2GRAY)
        
        #print(np.unique(seperators_closeup_n_binary))
        
        ##plt.imshow(seperators_closeup_n_binary)
        ##plt.show()

        
        #print( np.unique(gray),np.unique(seperators_closeup_n[:,:,1]) )
        
        gray = cv2.bitwise_not(seperators_closeup_n_binary)
        gray=gray.astype(np.uint8)
        
    
        ##plt.imshow(gray)
        ##plt.show()
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                    cv2.THRESH_BINARY, 15, -2)
        ##plt.imshow(bw[:,:])
        ##plt.show()
        
        horizontal = np.copy(bw)
        vertical = np.copy(bw)
        
        cols = horizontal.shape[1]
        horizontal_size = cols // 30
        # Create structure element for extracting horizontal lines through morphology operations
        horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
        # Apply morphology operations
        horizontal = cv2.erode(horizontal, horizontalStructure)
        horizontal = cv2.dilate(horizontal, horizontalStructure)

        kernel = np.ones((5,5),np.uint8)


        horizontal = cv2.dilate(horizontal,kernel,iterations = 2)
        horizontal = cv2.erode(horizontal,kernel,iterations = 2)
        #plt.imshow(horizontal)
        #plt.show()
        
        rows = vertical.shape[0]
        verticalsize = rows // 30
        # Create structure element for extracting vertical lines through morphology operations
        verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
        # Apply morphology operations
        vertical = cv2.erode(vertical, verticalStructure)
        vertical = cv2.dilate(vertical, verticalStructure)
        
        vertical = cv2.dilate(vertical,kernel,iterations = 1)
        # Show extracted vertical lines

        horizontal=self.combine_hor_lines_and_delete_cross_points_and_get_lines_features_back_new(vertical,horizontal)
        
        
        ##plt.imshow(vertical)
        ##plt.show()
        #print(vertical.shape,np.unique(vertical),'verticalvertical')
        seperators_closeup_new[:,:][vertical[:,:]!=0]=1
        seperators_closeup_new[:,:][horizontal[:,:]!=0]=1
        
        ##plt.imshow(seperators_closeup_new)
        ##plt.show()
        seperators_closeup_n
        vertical=np.repeat(vertical[:, :, np.newaxis], 3, axis=2)
        vertical=vertical.astype(np.uint8)
        imgray = cv2.cvtColor(vertical, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 0, 255, 0)
        
        contours_line_vers,hierachy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        slope_lines,dist_x, x_min_main ,x_max_main ,cy_main,slope_lines_org,y_min_main, y_max_main, cx_main=self.find_features_of_lines(contours_line_vers)
        #print(slope_lines,'vertical')
        args=np.array( range(len(slope_lines) ))
        args_ver=args[slope_lines==1]
        dist_x_ver=dist_x[slope_lines==1]
        y_min_main_ver=y_min_main[slope_lines==1]
        y_max_main_ver=y_max_main[slope_lines==1]
        x_min_main_ver=x_min_main[slope_lines==1]
        x_max_main_ver=x_max_main[slope_lines==1]
        cx_main_ver=cx_main[slope_lines==1]
        dist_y_ver=y_max_main_ver-y_min_main_ver
        len_y=seperators_closeup.shape[0]/3.0
        
        
        horizontal=np.repeat(horizontal[:, :, np.newaxis], 3, axis=2)
        horizontal=horizontal.astype(np.uint8)
        imgray = cv2.cvtColor(horizontal, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 0, 255, 0)
        
        contours_line_hors,hierachy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        slope_lines,dist_x, x_min_main ,x_max_main ,cy_main,slope_lines_org,y_min_main, y_max_main, cx_main=self.find_features_of_lines(contours_line_hors)
        
        slope_lines_org_hor=slope_lines_org[slope_lines==0]
        args=np.array( range(len(slope_lines) ))
        len_x=seperators_closeup.shape[1]/5.0

        dist_y=np.abs(y_max_main-y_min_main)
        
        args_hor=args[slope_lines==0]
        dist_x_hor=dist_x[slope_lines==0]
        y_min_main_hor=y_min_main[slope_lines==0]
        y_max_main_hor=y_max_main[slope_lines==0]
        x_min_main_hor=x_min_main[slope_lines==0]
        x_max_main_hor=x_max_main[slope_lines==0]
        dist_y_hor=dist_y[slope_lines==0]
        cy_main_hor=cy_main[slope_lines==0]

        args_hor=args_hor[dist_x_hor>=len_x/2.0]
        x_max_main_hor=x_max_main_hor[dist_x_hor>=len_x/2.0]
        x_min_main_hor=x_min_main_hor[dist_x_hor>=len_x/2.0]
        cy_main_hor=cy_main_hor[dist_x_hor>=len_x/2.0]
        y_min_main_hor=y_min_main_hor[dist_x_hor>=len_x/2.0]
        y_max_main_hor=y_max_main_hor[dist_x_hor>=len_x/2.0]
        dist_y_hor=dist_y_hor[dist_x_hor>=len_x/2.0]
        
        slope_lines_org_hor=slope_lines_org_hor[dist_x_hor>=len_x/2.0]
        dist_x_hor=dist_x_hor[dist_x_hor>=len_x/2.0]
        
        
        matrix_of_lines_ch=np.zeros((len(cy_main_hor)+len(cx_main_ver),10))
        
        matrix_of_lines_ch[:len(cy_main_hor),0]=args_hor
        matrix_of_lines_ch[len(cy_main_hor):,0]=args_ver


        matrix_of_lines_ch[len(cy_main_hor):,1]=cx_main_ver

        matrix_of_lines_ch[:len(cy_main_hor),2]=x_min_main_hor+50#x_min_main_hor+150
        matrix_of_lines_ch[len(cy_main_hor):,2]=x_min_main_ver

        matrix_of_lines_ch[:len(cy_main_hor),3]=x_max_main_hor-50#x_max_main_hor-150
        matrix_of_lines_ch[len(cy_main_hor):,3]=x_max_main_ver

        matrix_of_lines_ch[:len(cy_main_hor),4]=dist_x_hor
        matrix_of_lines_ch[len(cy_main_hor):,4]=dist_x_ver

        matrix_of_lines_ch[:len(cy_main_hor),5]=cy_main_hor


        matrix_of_lines_ch[:len(cy_main_hor),6]=y_min_main_hor
        matrix_of_lines_ch[len(cy_main_hor):,6]=y_min_main_ver

        matrix_of_lines_ch[:len(cy_main_hor),7]=y_max_main_hor
        matrix_of_lines_ch[len(cy_main_hor):,7]=y_max_main_ver

        matrix_of_lines_ch[:len(cy_main_hor),8]=dist_y_hor
        matrix_of_lines_ch[len(cy_main_hor):,8]=dist_y_ver


        matrix_of_lines_ch[len(cy_main_hor):,9]=1
        
        """
        
        
        
        seperators_closeup=seperators_closeup.astype(np.uint8)
        imgray = cv2.cvtColor(seperators_closeup, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 0, 255, 0)

        contours_lines,hierachy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        slope_lines,dist_x, x_min_main ,x_max_main ,cy_main,slope_lines_org,y_min_main, y_max_main, cx_main=self.find_features_of_lines(contours_lines)

        slope_lines_org_hor=slope_lines_org[slope_lines==0]
        args=np.array( range(len(slope_lines) ))
        len_x=seperators_closeup.shape[1]/4.0

        args_hor=args[slope_lines==0]
        dist_x_hor=dist_x[slope_lines==0]
        x_min_main_hor=x_min_main[slope_lines==0]
        x_max_main_hor=x_max_main[slope_lines==0]
        cy_main_hor=cy_main[slope_lines==0]

        args_hor=args_hor[dist_x_hor>=len_x/2.0]
        x_max_main_hor=x_max_main_hor[dist_x_hor>=len_x/2.0]
        x_min_main_hor=x_min_main_hor[dist_x_hor>=len_x/2.0]
        cy_main_hor=cy_main_hor[dist_x_hor>=len_x/2.0]
        slope_lines_org_hor=slope_lines_org_hor[dist_x_hor>=len_x/2.0]


        slope_lines_org_hor=slope_lines_org_hor[np.abs(slope_lines_org_hor)<1.2]
        slope_mean_hor=np.mean(slope_lines_org_hor)



        args_ver=args[slope_lines==1]
        y_min_main_ver=y_min_main[slope_lines==1]
        y_max_main_ver=y_max_main[slope_lines==1]
        x_min_main_ver=x_min_main[slope_lines==1]
        x_max_main_ver=x_max_main[slope_lines==1]
        cx_main_ver=cx_main[slope_lines==1]
        dist_y_ver=y_max_main_ver-y_min_main_ver
        len_y=seperators_closeup.shape[0]/3.0
        

        
        print(matrix_of_lines_ch[:,8][matrix_of_lines_ch[:,9]==0],'khatlarrrr')
        args_main_spliters=matrix_of_lines_ch[:,0][ (matrix_of_lines_ch[:,9]==0) & ((matrix_of_lines_ch[:,8]<=290)) & ((matrix_of_lines_ch[:,2]<=.16*region_pre_p.shape[1])) & ((matrix_of_lines_ch[:,3]>=.84*region_pre_p.shape[1]))]

        cy_main_spliters=matrix_of_lines_ch[:,5][ (matrix_of_lines_ch[:,9]==0) & ((matrix_of_lines_ch[:,8]<=290)) & ((matrix_of_lines_ch[:,2]<=.16*region_pre_p.shape[1])) & ((matrix_of_lines_ch[:,3]>=.84*region_pre_p.shape[1]))]
        """
        
        cy_main_spliters=cy_main_hor[ (x_min_main_hor<=.16*region_pre_p.shape[1]) & (x_max_main_hor>=.84*region_pre_p.shape[1] )]
        
        args_cy_spliter=np.argsort(cy_main_spliters)
        
        cy_main_spliters_sort=cy_main_spliters[args_cy_spliter]
        
        spliter_y_new=[]
        spliter_y_new.append(0)
        for i in range(len(cy_main_spliters_sort)):
            spliter_y_new.append(  cy_main_spliters_sort[i] ) 
            
        spliter_y_new.append(region_pre_p.shape[0])
        
        spliter_y_new_diff=np.diff(spliter_y_new)/float(region_pre_p.shape[0])*100
        
        args_big_parts=np.array(range(len(spliter_y_new_diff))) [ spliter_y_new_diff>22 ]
        
        
                
        regions_without_seperators=self.return_regions_without_seperators(region_pre_p)
            

        #image_page_otsu=self.otsu_copy(image_page_deskewd)
        #print(np.unique(image_page_otsu[:,:,0]))
        #image_page_background_zero=self.image_change_background_pixels_to_zero(image_page_otsu)
        
        
        length_y_threshold=regions_without_seperators.shape[0]/4.0
        
        num_col_fin=0
        peaks_neg_fin_fin=[]
        
        for iteils in args_big_parts:
            
            
            regions_without_seperators_teil=regions_without_seperators[int(spliter_y_new[iteils]):int(spliter_y_new[iteils+1]),:,0]
            #image_page_background_zero_teil=image_page_background_zero[int(spliter_y_new[iteils]):int(spliter_y_new[iteils+1]),:]
            
            #print(regions_without_seperators_teil.shape)
            ##plt.imshow(regions_without_seperators_teil)
            ##plt.show()
            
            #num_col, peaks_neg_fin=self.find_num_col(regions_without_seperators_teil,multiplier=6.0)
            num_col, peaks_neg_fin=self.find_num_col(regions_without_seperators_teil,multiplier=7.0)
            
            if num_col>num_col_fin:
                num_col_fin=num_col
                peaks_neg_fin_fin=peaks_neg_fin
            """
            #print(length_y_vertical_lines,length_y_threshold,'x_center_of_ver_linesx_center_of_ver_linesx_center_of_ver_lines')
            if len(cx_main_ver)>0 and len( dist_y_ver[dist_y_ver>=length_y_threshold] ) >=1:
                num_col, peaks_neg_fin=self.find_num_col(regions_without_seperators_teil,multiplier=6.0)
            else:
                #plt.imshow(image_page_background_zero_teil)
                #plt.show()
                #num_col, peaks_neg_fin=self.find_num_col_only_image(image_page_background_zero,multiplier=2.4)#2.3)
                num_col, peaks_neg_fin=self.find_num_col_only_image(image_page_background_zero_teil,multiplier=3.4)#2.3)
                
                print(num_col,'birda')
                if num_col>0:
                    pass
                elif num_col==0:
                    print(num_col,'birda2222')
                    num_col_regions, peaks_neg_fin_regions=self.find_num_col(regions_without_seperators_teil,multiplier=10.0)
                    if num_col_regions==0:
                        pass
                    else:
                        
                        num_col=num_col_regions
                        peaks_neg_fin=peaks_neg_fin_regions[:]
            """
            
            #print(num_col+1,'num colmsssssssss')
        
        return num_col_fin, peaks_neg_fin_fin,matrix_of_lines_ch,spliter_y_new,seperators_closeup_n
        

        
    def return_contours_of_interested_region_by_size(self,region_pre_p,pixel,min_area,max_area):
        
        # pixels of images are identified by 5
        if len(region_pre_p.shape)==3:
            cnts_images=(region_pre_p[:,:,0]==pixel)*1
        else:
            cnts_images=(region_pre_p[:,:]==pixel)*1
        cnts_images=cnts_images.astype(np.uint8)
        cnts_images=np.repeat(cnts_images[:, :, np.newaxis], 3, axis=2)
        imgray = cv2.cvtColor(cnts_images, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 0, 255, 0)
        contours_imgs,hiearchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        contours_imgs=self.return_parent_contours(contours_imgs,hiearchy)
        contours_imgs=self.filter_contours_area_of_image_tables(thresh,contours_imgs,hiearchy,max_area=max_area,min_area=min_area)
        
        img_ret=np.zeros((region_pre_p.shape[0],region_pre_p.shape[1],3))
        img_ret=cv2.fillPoly(img_ret,pts=contours_imgs, color=(1,1,1))
        return img_ret[:,:,0]
    
    def get_regions_from_xy(self,img):
        img_org=np.copy(img)
        
        img_height_h=img_org.shape[0]
        img_width_h=img_org.shape[1]
        
        model_region, session_region = self.start_new_session_and_model(self.model_region_dir_p)
        
        gaussian_filter=False
        patches=True
        binary=True
        
        
        

        ratio_x=1
        ratio_y=1
        median_blur=False
        
        if binary:
            img = self.otsu_copy_binary(img)#self.otsu_copy(img)
            img = img.astype(np.uint16)
            
        if median_blur:
            img=cv2.medianBlur(img,5)
            
        if gaussian_filter:
            img= cv2.GaussianBlur(img,(5,5),0)
            img = img.astype(np.uint16)
        prediction_regions_org=self.do_prediction(patches,img,model_region)
        
        ###plt.imshow(prediction_regions_org[:,:,0])
        ###plt.show()
        ##sys.exit()
        prediction_regions_org=prediction_regions_org[:,:,0]
        
        
        
        gaussian_filter=False
        patches=True
        binary=False
        
        
        

        ratio_x=1.1
        ratio_y=1
        median_blur=False
        
        #img= self.resize_image(img_org, int(img_org.shape[0]*0.8), int(img_org.shape[1]*1.6))
        img= self.resize_image(img_org, int(img_org.shape[0]*ratio_y), int(img_org.shape[1]*ratio_x))

        if binary:
            img = self.otsu_copy_binary(img)#self.otsu_copy(img)
            img = img.astype(np.uint16)
            
        if median_blur:
            img=cv2.medianBlur(img,5)
        if gaussian_filter:
            img= cv2.GaussianBlur(img,(5,5),0)
            img = img.astype(np.uint16)
        
        prediction_regions=self.do_prediction(patches,img,model_region)
        text_region1=self.resize_image(prediction_regions, img_height_h, img_width_h )
    
        
        ratio_x=1
        ratio_y=1.1
        binary=False
        median_blur=False
        
        
        img= self.resize_image(img_org, int(img_org.shape[0]*ratio_y), int(img_org.shape[1]*ratio_x))

        if binary:
            img = self.otsu_copy_binary(img)#self.otsu_copy(img)
            img = img.astype(np.uint16)
            
        if median_blur:
            img=cv2.medianBlur(img,5)
        if gaussian_filter:
            img= cv2.GaussianBlur(img,(5,5),0)
            img = img.astype(np.uint16)

        prediction_regions=self.do_prediction(patches,img,model_region)
        text_region2=self.resize_image(prediction_regions, img_height_h, img_width_h )
        
        session_region.close()
        del model_region
        del session_region
        gc.collect()
        
        
        mask_zeros_from_1=(text_region1[:,:,0]==0)*1
        #mask_text_from_1=(text_region1[:,:,0]==1)*1
        


        mask_img_text_region1=(text_region1[:,:,0]==2)*1
        text_region2_1st_channel=text_region2[:,:,0]
        
        text_region2_1st_channel[mask_zeros_from_1==1]=0
        
        text_region2_1st_channel[mask_img_text_region1[:,:]==1]=2
        #text_region2_1st_channel[(mask_text_from_1==1) & (text_region2_1st_channel==2)]=1
        
        mask_lines1=(text_region1[:,:,0]==3)*1
        mask_lines2=(text_region2[:,:,0]==3)*1
        
        mask_lines2[mask_lines1[:,:]==1]=1
        
        ##plt.imshow(text_region2_1st_channel)
        ##plt.show()
        
        text_region2_1st_channel = cv2.erode(text_region2_1st_channel[:,:], self.kernel, iterations=5)
        
        ##plt.imshow(text_region2_1st_channel)
        ##plt.show()
        
        text_region2_1st_channel = cv2.dilate(text_region2_1st_channel[:,:], self.kernel, iterations=5)
        
        
        text_region2_1st_channel[mask_lines2[:,:]==1]=3
        
        text_region2_1st_channel[ (prediction_regions_org[:,:]==1) & (text_region2_1st_channel[:,:]==2)]=1
        text_region2_1st_channel[prediction_regions_org[:,:]==3]=3
        
        ##plt.imshow(text_region2_1st_channel)
        ##plt.show()
        return text_region2_1st_channel
    
    
    def rotation_not_90_func(self,img,textline,thetha):
        rotated=imutils.rotate(img,thetha)
        rotated_textline=imutils.rotate(textline,thetha)
        return self.rotate_max_area(img, rotated,rotated_textline,thetha)
    
    def rotate_max_area(self,image,rotated,rotated_textline,angle):
        wr, hr =self.rotatedRectWithMaxArea(image.shape[1], image.shape[0],math.radians(angle))
        h, w, _ = rotated.shape
        y1 = h//2 - int(hr/2)
        y2 = y1 + int(hr)
        x1 = w//2 - int(wr/2)
        x2 = x1 + int(wr)
        return rotated[y1:y2, x1:x2],rotated_textline[y1:y2, x1:x2]
    
    def run(self):
        
        #get image and sclaes, then extract the page of scanned image

        self.get_image_and_scales()
        
        ###self.produce_groundtruth_for_textline()
        image_page,page_coord=self.extract_page()

        ##########  
        K.clear_session()
        gc.collect()
        
        
        
        
        
        patches=True
        scaler_h_textline=1.2#1.2
        scaler_w_textline=0.9#1
        textline_mask_tot,textline_mask_tot_long_shot=self.textline_contours(image_page,patches,scaler_h_textline,scaler_w_textline)
        

        sigma=2
        slope_first=self.return_deskew_slop(cv2.erode(textline_mask_tot, self.kernel, iterations=2),sigma)
        
        
        image_page_rotated,textline_mask_tot=self.rotation_not_90_func(image_page,textline_mask_tot,slope_first)
        
        self.get_image_and_scales_deskewd(image_page_rotated)
        
        ##text_regions_p=self.deskew_region_prediction(text_regions_p,slope_first)
        
        text_regions_p_1=self.get_regions_from_xy(image_page_rotated)
        
        #text_regions_p_1=self.deskew_region_prediction(text_regions_p_1,textline_mask_tot,slope_first)
        
        ##textline_mask_tot=self.deskew_region_prediction(textline_mask_tot,slope_first)

        mask_images=(text_regions_p_1[:,:]==2)*1
        mask_lines=(text_regions_p_1[:,:]==3)*1
        
        mask_images=mask_images.astype(np.uint16)
        mask_lines=mask_lines.astype(np.uint16)
        
        mask_images=cv2.erode(mask_images[:,:], self.kernel, iterations=10)
        
        
        
        img_only_regions_with_sep=( (text_regions_p_1[:,:]!=3) & (text_regions_p_1[:,:]!=0) )*1
        img_only_regions_with_sep=img_only_regions_with_sep.astype(np.uint8)
        img_only_regions = cv2.erode(img_only_regions_with_sep[:,:], self.kernel, iterations=6)
        

        num_col, peaks_neg_fin=self.find_num_col(img_only_regions,multiplier=6.0)
        

        
        textline_mask_tot[mask_images[:,:]==1]=0




        

        
        pixel_img=1
        min_area=0.00001
        max_area=0.0006
        textline_mask_tot_small_size=self.return_contours_of_interested_region_by_size(textline_mask_tot,pixel_img,min_area,max_area)
        
        
        text_regions_p_1[(textline_mask_tot[:,:]==1) & (text_regions_p_1[:,:]==2)]=1
        
        
        text_regions_p_1[mask_lines[:,:]==1]=3
        
        
        text_regions_p_1[textline_mask_tot_small_size[:,:]==1]=1
        
        
        text_regions_p=text_regions_p_1[:,:]#long_short_region[:,:]#self.get_regions_from_2_models(image_page)
        
        text_regions_p=np.array(text_regions_p)
        
        t3=time.time()
        
        regions_without_seperators=( (text_regions_p[:,:]==1) | (text_regions_p[:,:]==2) )*1 #self.return_regions_without_seperators_new(text_regions_p[:,:,0],img_only_regions)

        

        num_col,peaks_neg_fin,matrix_of_lines_ch,spliter_y_new,seperators_closeup_n=self.find_number_of_columns_in_document(np.repeat(text_regions_p[:, :, np.newaxis], 3, axis=2))
        
        
        boxes=self.return_boxes_of_images_by_order_of_reading_new(spliter_y_new,regions_without_seperators,matrix_of_lines_ch)
        
        
        img_revised_tab=text_regions_p[:,:]
        

        K.clear_session()
        
        pixel_img=2
        polygons_of_images=self.return_contours_of_interested_region(img_revised_tab,pixel_img)
        

        text_only=( (img_revised_tab[:,:]==1) )*1
        ##text_only_h=( (img_revised_tab[:,:,0]==2) )*1
        
        contours_only_text,hir_on_text=self.return_contours_of_image(text_only)
        contours_only_text_parent=self.return_parent_contours( contours_only_text,hir_on_text)
        
        ###contours_only_text_h,hir_on_text_h=self.return_contours_of_image(text_only_h)
        ###contours_only_text_parent_h=self.return_parent_contours( contours_only_text_h,hir_on_text_h)
        
        areas_cnt_text=np.array([cv2.contourArea(contours_only_text_parent[j]) for j in range(len(contours_only_text_parent))])
        areas_cnt_text=areas_cnt_text/float(text_only.shape[0]*text_only.shape[1])
        
        ###areas_cnt_text_h=np.array([cv2.contourArea(contours_only_text_parent_h[j]) for j in range(len(contours_only_text_parent_h))])
        ###areas_cnt_text_h=areas_cnt_text_h/float(text_only_h.shape[0]*text_only_h.shape[1])

        contours_only_text_parent=[contours_only_text_parent[jz] for jz in range(len(contours_only_text_parent)) if areas_cnt_text[jz]>0.0002]

        boxes_text,_=self.get_text_region_boxes_by_given_contours(text_only,contours_only_text_parent,image_page_rotated)
        ####boxes_text_h,_=self.get_text_region_boxes_by_given_contours(text_only_h,contours_only_text_parent_h,image_page)
        

        all_box_coord=[]
        for jk in range(len(boxes_text)):
            crop_img,crop_coor=self.crop_image_inside_box(boxes_text[jk],image_page_rotated)

            all_box_coord.append(crop_coor)
            

            
        slopes=[]
        slope_new=0
        textregion_kind='main_text'
        for mv in range(len(boxes_text)):
            denoised=None
            
            all_text_region_raw=textline_mask_tot[boxes_text[mv][1]:boxes_text[mv][1]+boxes_text[mv][3] , boxes_text[mv][0]:boxes_text[mv][0]+boxes_text[mv][2] ]
            all_text_region_raw=all_text_region_raw.astype(np.uint8)
            slope_for_all=self.textline_contours_to_get_slope_correctly(all_text_region_raw,denoised,contours_only_text_parent[mv])
            #text_patch_processed=textline_contours_postprocessing(gada)

            if np.abs(slope_for_all)>60.5 and slope_for_all!=999:
                slope_for_all=0
            elif slope_for_all==999:
                slope_for_all=slope_new
            else:
                slope_for_all=slope_for_all+slope_new
            #slopes.append(slope_for_all+slope_tot)
            slopes.append(slope_for_all)
            

        all_found_texline_polygons=[]
        textregion_kind='main_text'
        for jj in range(len(boxes_text)):
            all_text_region_raw=textline_mask_tot[boxes_text[jj][1]:boxes_text[jj][1]+boxes_text[jj][3] , boxes_text[jj][0]:boxes_text[jj][0]+boxes_text[jj][2] ]
            cnt_clean_rot=self.textline_contours_postprocessing(all_text_region_raw,slopes[jj],contours_only_text_parent[jj],boxes_text[jj])
            all_found_texline_polygons.append(cnt_clean_rot)

        cx_text_only,cy_text_only ,x_min_text_only, _, _ ,_=self.find_new_features_of_contoures(contours_only_text_parent)
        boxes_arr=np.array(boxes)

        arg_text_con=[]
        for ii in range(len(cx_text_only)):
            for jj in range(len(boxes)):
                #print(x_min_text_only[ii]+30,boxes[jj][1],boxes[jj][2])
                #if cx_text_only[ii] >=boxes[jj][0] and cx_text_only[ii] < boxes[jj][1] and cy_text_only[ii] >=boxes[jj][2] and cy_text_only[ii] < boxes[jj][3]:#this is valid if the center of region identify in which box it is located
                if (x_min_text_only[ii]+80) >=boxes[jj][0] and (x_min_text_only[ii]+80) < boxes[jj][1] and cy_text_only[ii] >=boxes[jj][2] and cy_text_only[ii] < boxes[jj][3]:#this is valid if the min of region(in x direction) identify in which box it is located
                    arg_text_con.append(jj)
                    break
        arg_arg_text_con=np.argsort(arg_text_con)
        args_contours=np.array(range( len(arg_text_con)) )


        
        order_by_con_main=np.zeros(len(arg_text_con))
        ###order_by_con_head=np.zeros(len(arg_text_con_h))

        ref_point=0
        order_of_texts_tot=[]
        id_of_texts_tot=[]
        for iij in range(len(boxes)):

            args_contours_box=args_contours[np.array(arg_text_con)==iij]
            
            ###print(args_contours_box,'args_contours_box')
            ##args_contours_box_h=args_contours_h[np.array(arg_text_con_h)==iij]
            con_inter_box=[]
            con_inter_box_h=[]

            for i in range(len(args_contours_box)):

                con_inter_box.append( contours_only_text_parent[args_contours_box[i] ]  )
            #for i in range(len(args_contours_box_h)):

                #con_inter_box_h.append( contours_only_text_parent_h[args_contours_box_h[i] ]  )
                
                
            indexes_sorted, matrix_of_orders,kind_of_texts_sorted,index_by_kind_sorted=self.order_of_regions(textline_mask_tot[int(boxes[iij][2]):int(boxes[iij][3]), int(boxes[iij][0]):int(boxes[iij][1])],con_inter_box,con_inter_box_h,boxes[iij][2])
            

                
            
            order_of_texts, id_of_texts=self.order_and_id_of_texts(con_inter_box ,con_inter_box_h,matrix_of_orders ,indexes_sorted ,index_by_kind_sorted, kind_of_texts_sorted, ref_point)
            
            
            indexes_sorted_main=np.array(indexes_sorted)[np.array(kind_of_texts_sorted)==1]
            indexes_by_type_main=np.array(index_by_kind_sorted)[np.array(kind_of_texts_sorted)==1]
            indexes_sorted_head=np.array(indexes_sorted)[np.array(kind_of_texts_sorted)==2]
            indexes_by_type_head=np.array(index_by_kind_sorted)[np.array(kind_of_texts_sorted)==2]
            

            
            
            zahler=0
            for mtv in args_contours_box:
                arg_order_v=indexes_sorted_main[zahler]
                tartib=np.where(indexes_sorted==arg_order_v )[0][0]
                order_by_con_main[ args_contours_box[indexes_by_type_main[zahler] ]]=tartib+ref_point
                zahler=zahler+1

                
            
            for jji in range(len(id_of_texts)):
                
                
                #order_of_texts_tot.append(args_contours_box_ordered[jji])
                order_of_texts_tot.append(order_of_texts[jji]+ref_point)
                id_of_texts_tot.append(id_of_texts[jji])
            ref_point=ref_point+len(id_of_texts)


        for i in range(0):#(len(order_of_texts_tot)):
            img_con=np.zeros(seperators_closeup_n[:,:,0].shape)

            img_con=cv2.fillPoly(img_con, pts =[contours_only_text_parent[order_of_texts_tot[i] ] ] , color=(255,255,255))

        
        
        order_of_texts_tot=[]
        #id_of_texts_tot=[]
        for tj1 in range(len(contours_only_text_parent)):
            order_of_texts_tot.append(int(order_by_con_main[tj1] ))
            #id_of_texts_tot.append('r'+str(int(order_by_con_main[tj1] )))
        ###for tj1 in range(len(contours_only_text_parent_h)):
            ###order_of_texts_tot.append(int(order_by_con_head[tj1]) )
            ####id_of_texts_tot.append('r'+str(int(order_by_con_head[tj1] )))
            
        order_text_new=[]
        for iii in range(len(order_of_texts_tot)):
            tartib_new=np.where(np.array(order_of_texts_tot)==iii)[0][0]
            order_text_new.append(tartib_new)

        self.write_into_page_xml(contours_only_text_parent,page_coord,self.dir_out , order_text_new , id_of_texts_tot,all_found_texline_polygons,
                                 all_box_coord,polygons_of_images )
        


        

@click.command()
@click.option('--image', '-i', help='image filename', type=click.Path(exists=True, dir_okay=False))
@click.option('--out', '-o', help='directory to write output xml data', type=click.Path(exists=True, file_okay=False))
@click.option('--model', '-m', help='directory of models', type=click.Path(exists=True, file_okay=False))
def main(image,out, model):
    possibles = globals()  # XXX unused?
    possibles.update(locals())
    x = sbb_newspapers(image, None, out, model)
    x.run()


if __name__ == "__main__":
    main()
 
