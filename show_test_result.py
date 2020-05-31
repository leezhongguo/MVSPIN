#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 17:27:57 2020

@author: li
"""

import numpy as np
from opendr.renderer import ColoredRenderer 
from opendr.lighting import LambertianPointLight
from opendr.camera import ProjectPoints
import cPickle as pkl
from models.smpl import Smpl, copy_smpl, joints_coco
import h5py
from util import im
from render_model import render_model
from util.imutils import crop
import cv2
import matplotlib.pyplot as plt
from os.path import join
import scipy.io as sio

index = 800

#test_path = '../SPIN_MV/data/h36m_train_S1s_3d.npz'
test_path = '../SPIN/data/dataset_extras/h36m_valid_protocol1.npz'
#spin = '../SPIN_MV/S1_single_smplify.npz'
spin = '../SPIN_MV/temp/logs_b16_e20_full_3d_mix/eval_h36m_spin.npz'
#our = '../SPIN_MV/S1_multi_smplify.npz'
our = '../SPIN_MV/temp/logs_b16_e20_full_3d_mix/eval_h36m_our.npz'

mpi_inf_valid = np.load(test_path)
ROOT = '../SPIN_MV/data/'
mpi_inf_spin = np.load(spin)
mpi_inf_pred = np.load(our)
IMG_RES = 224
focal_length = 5000
model_file = 'SMPL_python_v.1.0.0/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl'
with open(model_file, 'rb') as fp:
    model_data = pkl.load(fp)
fig = plt.figure()
#plt.ion()
#gt_keypoints = np.zeros((400,24,4))
for i in range(70563,70564):
    imgname = mpi_inf_valid['imgname'][i]
    #print(join(ROOT,imgname))
    rgb_img = cv2.imread(join(ROOT,imgname))[:,:,::-1].copy().astype(np.float32)
    center = mpi_inf_valid['center'][i]
    scale = mpi_inf_valid['scale'][i]
    rgb_img = crop(rgb_img, center, scale, [IMG_RES, IMG_RES])
    pose = mpi_inf_pred['pose'][i]
    betas = mpi_inf_pred['betas'][i]
    camera = mpi_inf_pred['camera'][i]
    #gt_keypoints[i*batch_size*4+k*4+j] = mpi_inf_valid['S'][i*batch_size+j][k]
    camera_t = np.array([camera[1],camera[2], 2*focal_length/(IMG_RES*camera[0] +1e-9)])
    w, h = (IMG_RES, IMG_RES)
    rn = ColoredRenderer()
    pred_base_smpl = Smpl(model_data)
    pred_base_smpl.pose[:] = pose
    pred_base_smpl.betas[:] = betas
    pred_rot = np.eye(3)
    rn.camera = ProjectPoints(t=camera_t, rt=cv2.Rodrigues(pred_rot)[0].reshape(3), c=np.array([112, 112]),
                                  f=np.array([5000,5000]), k=np.zeros(5), v=pred_base_smpl)
    dist = np.abs(rn.camera.t.r[2] - np.mean(pred_base_smpl.r, axis=0)[2])
    verts = pred_base_smpl.r
    im = (render_model(verts, pred_base_smpl.f, w, h, rn.camera, far=20+dist) * 255.).astype('uint8')
    
    pose_spin = mpi_inf_spin['pose'][i]
    betas_spin = mpi_inf_spin['betas'][i]
    camera = mpi_inf_spin['camera'][i]
    camera_t_spin = np.array([camera[1],camera[2], 2*focal_length/(IMG_RES*camera[0] +1e-9)])
    rn = ColoredRenderer()
    pred_base_smpl.pose[:] = pose_spin
    pred_base_smpl.betas[:] = betas_spin
    rn.camera = ProjectPoints(t=camera_t_spin, rt=cv2.Rodrigues(pred_rot)[0].reshape(3), c=np.array([112, 112]),
                                  f=np.array([5000,5000]), k=np.zeros(5), v=pred_base_smpl)
    dist = np.abs(rn.camera.t.r[2] - np.mean(pred_base_smpl.r, axis=0)[2])
    verts = pred_base_smpl.r
    im_spin = (render_model(verts, pred_base_smpl.f, w, h, rn.camera, far=20+dist) * 255.).astype('uint8')
    
    ort = np.reshape(pose_spin[:3],(3,1))
    #print(ort)
    ort_mat = cv2.Rodrigues(ort)[0]
    #print(ort_mat)
    trans_mat = np.array([[-1,0,0],
                          [0,-1,0],
                          [0,0,1]])
    new_ort = ort_mat.dot(trans_mat)
    pred_base_smpl.pose[:3] = cv2.Rodrigues(new_ort)[0].reshape(3)
    rn.camera = ProjectPoints(t=camera_t, rt=cv2.Rodrigues(pred_rot)[0].reshape(3), c=np.array([112, 112]),
                                  f=np.array([5000,5000]), k=np.zeros(5), v=pred_base_smpl)
    dist = np.abs(rn.camera.t.r[2] - np.mean(pred_base_smpl.r, axis=0)[2])
    verts = pred_base_smpl.r
    im_1 = (render_model(verts, pred_base_smpl.f, w, h, rn.camera, far=20+dist) * 255.).astype('uint8')
    
    fig = plt.figure()
    #plt.subplot(1,3,1)
    plt.imshow(rgb_img)
    
    height, width, channels = rgb_img.shape 
    # 如果dpi=300，那么图像大小=height*width 
    fig.set_size_inches(width/100.0/3.0, height/100.0/3.0) 
    plt.gca().xaxis.set_major_locator(plt.NullLocator()) 
    plt.gca().yaxis.set_major_locator(plt.NullLocator()) 
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0) 
    plt.margins(0,0)
    plt.axis('off')
    plt.savefig("../SPIN_MV/save_h36m/h36m_test_original_%06d.png" % (i), dpi=300)
    
    fig = plt.figure()
    #plt.subplot(1,3,2)
    plt.imshow(rgb_img)
    plt.imshow(im)
    
    # 如果dpi=300，那么图像大小=height*width 
    fig.set_size_inches(width/100.0/3.0, height/100.0/3.0) 
    plt.gca().xaxis.set_major_locator(plt.NullLocator()) 
    plt.gca().yaxis.set_major_locator(plt.NullLocator()) 
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0) 
    plt.margins(0,0)
    plt.axis('off')
    plt.savefig("../SPIN_MV/save_h36m/h36m_test_our_%06d.png" % (i), dpi=300)
    
    #fig = plt.figure()
    #plt.subplot()
    #plt.imshow(im_1)
    # 如果dpi=300，那么图像大小=height*width 
   # fig.set_size_inches(width/100.0/3.0, height/100.0/3.0) 
   # plt.gca().xaxis.set_major_locator(plt.NullLocator()) 
   # plt.gca().yaxis.set_major_locator(plt.NullLocator()) 
    #plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0) 
    #plt.margins(0,0)
    #plt.axis('off')
    #plt.savefig("../SPIN_MV/save_mpi_smpl/mpi_test_our_view_%04d.png" % (i), dpi=300)
    
    fig = plt.figure()
    #plt.subplot(1,3,3)
    plt.imshow(rgb_img)
    plt.imshow(im_spin)
    
    fig.set_size_inches(width/100.0/3.0, height/100.0/3.0) 
    plt.gca().xaxis.set_major_locator(plt.NullLocator()) 
    plt.gca().yaxis.set_major_locator(plt.NullLocator()) 
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0) 
    plt.margins(0,0)
    plt.axis('off')
    plt.savefig("../SPIN_MV/save_h36m/h36m_test_spin_%06d.png" % (i), dpi=300)
    
    #plt.savefig('../SPIN_MV/save_h36m/h36m_test_%06d.png' % i)
    #plt.pause(1e-3)
   


"""
iter_num = 25
batch_size = 4
for i in range(24,25): # iteration number
    for j in range(2,3):   #batch_size
        for k in range(1,2):
            imgname = mpi_inf_valid['imgname'][i*batch_size+j][k]
            #print(join(ROOT,imgname))
            rgb_img = cv2.imread(join(ROOT,imgname))[:,:,::-1].copy().astype(np.float32)
            center = mpi_inf_valid['center'][i*batch_size+j][k]
            scale = mpi_inf_valid['scale'][i*batch_size+j][k]
            rgb_img = crop(rgb_img, center, scale, [IMG_RES, IMG_RES])
            pose = mpi_inf_pred['pose'][i*batch_size*4+k*4+j]
            betas = mpi_inf_pred['betas'][i*batch_size*4+k*4+j]
            camera_t = mpi_inf_pred['camera'][i*batch_size*4+k*4+j]
            #gt_keypoints[i*batch_size*4+k*4+j] = mpi_inf_valid['S'][i*batch_size+j][k]
            #camera_t = np.array([camera[1],camera[2], 2*focal_length/(IMG_RES*camera[0] +1e-9)])
            w, h = (IMG_RES, IMG_RES)
            rn = ColoredRenderer()
            pred_base_smpl = Smpl(model_data)
            pred_base_smpl.pose[:] = pose
            pred_base_smpl.betas[:] = betas
            pred_rot = np.eye(3)
            rn.camera = ProjectPoints(t=camera_t, rt=cv2.Rodrigues(pred_rot)[0].reshape(3), c=np.array([112, 112]),
                                          f=np.array([5000,5000]), k=np.zeros(5), v=pred_base_smpl)
            dist = np.abs(rn.camera.t.r[2] - np.mean(pred_base_smpl.r, axis=0)[2])
            verts = pred_base_smpl.r
            im = (render_model(verts, pred_base_smpl.f, w, h, rn.camera, far=20+dist) * 255.).astype('uint8')
            
            pose_spin = mpi_inf_spin['pose'][i*batch_size*4+k*4+j]
            betas_spin = mpi_inf_spin['betas'][i*batch_size*4+k*4+j]
            camera_t_spin = mpi_inf_spin['camera'][i*batch_size*4+k*4+j]
            #camera_t_spin = np.array([camera[1],camera[2], 2*focal_length/(IMG_RES*camera[0] +1e-9)])
            rn = ColoredRenderer()
            pred_base_smpl.pose[:] = pose_spin
            pred_base_smpl.betas[:] = betas_spin
            rn.camera = ProjectPoints(t=camera_t_spin, rt=cv2.Rodrigues(pred_rot)[0].reshape(3), c=np.array([112, 112]),
                                          f=np.array([5000,5000]), k=np.zeros(5), v=pred_base_smpl)
            dist = np.abs(rn.camera.t.r[2] - np.mean(pred_base_smpl.r, axis=0)[2])
            verts = pred_base_smpl.r
            im_spin = (render_model(verts, pred_base_smpl.f, w, h, rn.camera, far=20+dist) * 255.).astype('uint8')
            
            # orignal image
            fig = plt.figure()
            #plt.imshow(im+)
            #plt.subplot(1,3,1)
            plt.imshow(rgb_img)
            plt.axis('off')
            #plt.subplot(1,3,2)
            height, width, channels = rgb_img.shape 
            # 如果dpi=300，那么图像大小=height*width 
            fig.set_size_inches(width/100.0/3.0, height/100.0/3.0) 
            plt.gca().xaxis.set_major_locator(plt.NullLocator()) 
            plt.gca().yaxis.set_major_locator(plt.NullLocator()) 
            plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0) 
            plt.margins(0,0)
            plt.savefig("../SPIN_MV/save_smpl/S1_%d_view_%d_orig.png" % (i*batch_size+j, k), dpi=300)
            
            # multi
            fig = plt.figure()
            plt.imshow(rgb_img)
            plt.imshow(im)
            plt.axis('off')
            height, width, channels = rgb_img.shape 
            # 如果dpi=300，那么图像大小=height*width 
            fig.set_size_inches(width/100.0/3.0, height/100.0/3.0) 
            plt.gca().xaxis.set_major_locator(plt.NullLocator()) 
            plt.gca().yaxis.set_major_locator(plt.NullLocator()) 
            plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0) 
            plt.margins(0,0)
            plt.savefig("../SPIN_MV/save_smpl/S1_%d_view_%d_multi.png" % (i*batch_size+j, k), dpi=300)
            #plt.imshow(img[sample_idx].transpose((1,2,0)))
            #plt.subplot(1,2,1)
            
            # single
            fig = plt.figure()
            #plt.subplot(1,3,3)
            plt.imshow(rgb_img)
            #plt.imshow(img[sample_idx].transpose((1,2,0)))
            #plt.subplot(1,2,1)
            plt.imshow(im_spin)
            plt.axis('off')
            #plt.ioff()
            fig.set_size_inches(width/100.0/3.0, height/100.0/3.0) 
            plt.gca().xaxis.set_major_locator(plt.NullLocator()) 
            plt.gca().yaxis.set_major_locator(plt.NullLocator()) 
            plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0) 
            plt.margins(0,0)
            plt.savefig("../SPIN_MV/save_smpl/S1_%d_view_%d_single.png" % (i*batch_size+j, k), dpi=300)
            #plt.pause(1e-3)
            #plt.show()
"""
"""
H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
H36M_TO_J14 = H36M_TO_J17[:14]
J24_TO_J17 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 14, 16, 17]
J24_TO_J14 = J24_TO_J17[:14]
joint_mapper_gt = J24_TO_J14
joint_mapper_h36m = H36M_TO_J14
gt_keypoints = gt_keypoints[:, joint_mapper_gt, :-1]
sio.savemat('../SPIN_MV/evaluation/S1_gt.mat',{'gt_joints17':gt_keypoints})
sio.savemat('../SPIN_MV/evaluation/S1_single_gt.mat',{'pred_joints':mpi_inf_spin['pred_joints']})
sio.savemat('../SPIN_MV/evaluation/S1_multi_gt.mat',{'pred_joints':mpi_inf_pred['pred_joints']})
"""