import os
import sys
import cv2
import glob
import h5py
import numpy as np
import argparse
from spacepy import pycdf
#from .read_openpose import read_openpose

# Illustrative script for training data extraction
# No SMPL parameters will be included in the .npz file.
def h36m_train_extract(dataset_path, openpose_path, out_path, extract_img=False):

    # convert joints to global order
    h36m_idx = [11, 6, 7, 8, 1, 2, 3, 12, 24, 14, 15, 17, 18, 19, 25, 26, 27]
    global_idx = [14, 3, 4, 5, 2, 1, 0, 16, 12, 17, 18, 9, 10, 11, 8, 7, 6]

    # structs we use
    imgnames_, scales_, centers_, parts_, Ss_, openposes_  = [], [], [], [], [], []

    # users in validation set
    user_list = [8]

    # go over each user
    for user_i in user_list:
        user_name = 'S%d' % user_i
        # path with GT bounding boxes
        bbox_path = os.path.join(dataset_path, user_name, 'MySegmentsMat', 'ground_truth_bb')
        # path with GT 3D pose
        pose_path = os.path.join(dataset_path, user_name, 'MyPoseFeatures', 'D3_Positions_mono')
        # path with GT 2D pose
        pose2d_path = os.path.join(dataset_path, user_name, 'MyPoseFeatures', 'D2_Positions')
        # path with videos
        vid_path = os.path.join(dataset_path, user_name, 'Videos')

        # go over all the sequences of each user
        seq_list = glob.glob(os.path.join(pose_path, '*.cdf'))
        seq_list.sort()
        #for seq_i in seq_list:
        for seq_idx in range(len(seq_list)/4):
            print(seq_idx)
            # four views for one action
            camera_mv, poses_3d_mv, poses_2d_mv, bbox_h5py_mv, vidcap_mv = [], [], [], [], []
            for i in range(4):
                seq_i = seq_list[seq_idx*4+i]
                # sequence info
                seq_name = seq_i.split('/')[-1]
                action, camera, _ = seq_name.split('.')
                action = action.replace(' ', '_')
                camera_mv.append(camera)
                # irrelevant sequences
                if action == '_ALL':
                    continue

                # 3D pose file
                poses_3d = pycdf.CDF(seq_i)['Pose'][0]
                poses_3d_mv.append(poses_3d)
                # 2D pose file
                pose2d_file = os.path.join(pose2d_path, seq_name)
                poses_2d = pycdf.CDF(pose2d_file)['Pose'][0]
                poses_2d_mv.append(poses_2d)

                # bbox file
                bbox_file = os.path.join(bbox_path, seq_name.replace('cdf', 'mat'))
                bbox_h5py = h5py.File(bbox_file)
                bbox_h5py_mv.append(bbox_h5py)

                # video file
                if extract_img:
                    vid_file = os.path.join(vid_path, seq_name.replace('cdf', 'mp4'))
                    imgs_path = os.path.join(dataset_path, 'images')
                    vidcap = cv2.VideoCapture(vid_file)
                    vidcap_mv.append(vidcap)

            # go over each frame of the four views sequences
            for frame_i in range(poses_3d_mv[0].shape[0]):
                image_mv = []
                for i in range(4):
                    # read video frame
                    if extract_img:
                        success, image = vidcap_mv[i].read()
                        if not success:
                            break
                        image_mv.append(image)

                # check if you can keep this frame
                protocol = 1
                if frame_i %10 == 0 and (protocol == 1 or camera == '60457274'):
                    # image names
                    imgname_mv, center_mv, scale_mv, part_mv, Ss_mv = [], [], [], [], []
                    for i in range(4):
                        imgname = '%s_%s.%s_%06d.jpg' % (user_name, action, camera_mv[i], frame_i+1)
                        imgname_mv.append(imgname)
                        # save image
                        if extract_img:
                            img_out = os.path.join(imgs_path, imgname)
                            cv2.imwrite(img_out, image_mv[i])
                        # read GT bounding box
                        mask = bbox_h5py_mv[i][bbox_h5py_mv[i]['Masks'][frame_i,0]].value.T
                        ys, xs = np.where(mask==1)
                        bbox = np.array([np.min(xs), np.min(ys), np.max(xs)+1, np.max(ys)+1])
                        center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
                        scale = 0.9*max(bbox[2]-bbox[0], bbox[3]-bbox[1])/200.
                        center_mv.append(center)
                        scale_mv.append(scale)

                        # read GT 3D pose
                        partall = np.reshape(poses_2d_mv[i][frame_i,:], [-1,2])
                        part17 = partall[h36m_idx]
                        part = np.zeros([24,3])
                        part[global_idx, :2] = part17
                        part[global_idx, 2] = 1
                        part_mv.append(part)
                        
                        # read GT 3D pose
                        Sall = np.reshape(poses_3d_mv[i][frame_i,:], [-1,3])/1000.
                        S17 = Sall[h36m_idx]
                        S17 -= S17[0] # root-centered
                        S24 = np.zeros([24,4])
                        S24[global_idx, :3] = S17
                        S24[global_idx, 3] = 1
                        Ss_mv.append(S24)
                        
                        # read openpose detections
                        #json_file = os.path.join(openpose_path, 'coco',
                        #    imgname.replace('.jpg', '_keypoints.json'))
                        #openpose = read_openpose(json_file, part, 'h36m')

                    # store data of the four views
                    imgnames_.append(imgname_mv)
                    centers_.append(center_mv)
                    scales_.append(scale_mv)
                    parts_.append(part_mv)
                    Ss_.append(Ss_mv)
                    #openposes_.append(openpose)

    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, 'h36m_train_8.npz')
    np.savez(out_file, imgname=imgnames_,
                       center=centers_,
                       scale=scales_,
                       part=parts_,
                       S=Ss_)
#   np.savez(out_file, imgname=imgnames_,
#                       center=centers_,
#                       scale=scales_,
#                       part=parts_,
#                       S=Ss_,
#                       openpose=openposes_)
dataset_path = "/media/li/Seagate Expansion Drive/dataset/human36m"
#dataset_path = "/usr/vision/data/Human36m"
out_path = "/media/li/Seagate Expansion Drive/dataset/human36m"
openpose_path = "/path/openpose"
h36m_train_extract(dataset_path, openpose_path, out_path, extract_img=True)