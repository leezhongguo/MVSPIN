import torch
import torch.nn as nn
import numpy as np
import os
import pickle

from utils.geometry import perspective_projection,batch_rodrigues
import constants


def gmof(x, sigma):
    """
    Geman-McClure error function
    """
    x_squared =  x ** 2
    sigma_squared = sigma ** 2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)

def angle_prior(pose):
    """
    Angle prior that penalizes unnatural bending of the knees and elbows
    """
    # We subtract 3 because pose does not include the global rotation of the model
    return torch.exp(pose[:, [55-3, 58-3, 12-3, 15-3]] * torch.tensor([1., -1., -1, -1.], device=pose.device)) ** 2

def body_fitting_loss(body_pose, betas, model_joints, camera_t,camera_center,
                      joints_2d, joints_conf, pose_prior,
                      focal_length=5000, sigma=100, pose_prior_weight=4.78,
                      shape_prior_weight=5, angle_prior_weight=15.2,
                      output='sum'):
    """
    Loss function for body fitting
    """
    
    batch_size = model_joints.shape[0]
    rotation = torch.eye(3, device=body_pose.device).unsqueeze(0).expand(batch_size, -1, -1)
    projected_joints = perspective_projection(model_joints, rotation, camera_t,
                                              focal_length, camera_center)
    reprojection_error_1 = gmof(projected_joints - joints_2d, sigma)
    reprojection_loss = (joints_conf[:int(batch_size/4)] ** 2) * reprojection_error_1[:int(batch_size/4)].sum(dim=-1) +\
                        (joints_conf[int(batch_size/4):int(batch_size/2)] ** 2) * reprojection_error_1[int(batch_size/4):int(batch_size/2)].sum(dim=-1) +\
                        (joints_conf[int(batch_size/2):int(batch_size*3/4)] ** 2) * reprojection_error_1[int(batch_size/2):int(batch_size*3/4)].sum(dim=-1) +\
                        (joints_conf[int(batch_size*3/4):] ** 2) * reprojection_error_1[int(batch_size*3/4):].sum(dim=-1)
    #rotation = camera_rot[0]
    #print(camera_rot[1])
    #input()
    #camera_rot_mat0 = batch_rodrigues(camera_rot[0])
    #print(model_joints)
    #projected_joints_0 = perspective_projection(model_joints[:int(batch_size/4)], rotation, camera_t[:int(batch_size/4)],
    #                                          focal_length, camera_center[:int(batch_size/4)])
    #rotation = camera_rot[1]
    #camera_rot_mat1 = batch_rodrigues(camera_rot[1])
    #projected_joints_1 = perspective_projection(model_joints[int(batch_size/4):int(batch_size/2)], rotation, camera_t[int(batch_size/4):int(batch_size/2)],
    #                                          focal_length, camera_center[int(batch_size/4):int(batch_size/2)])

    #rotation = camera_rot[2]
    #camera_rot_mat2 = batch_rodrigues(camera_rot[2])
    #projected_joints_2 = perspective_projection(model_joints[int(batch_size/2):int(batch_size*3/4)], rotation, camera_t[int(batch_size/2):int(batch_size*3/4)],
    #                                          focal_length, camera_center[int(batch_size/2):int(batch_size*3/4)])
    #rotation = camera_rot[3]
    #camera_rot_mat3 = batch_rodrigues(camera_rot[3])
    #projected_joints_3 = perspective_projection(model_joints[int(batch_size*3/4):], rotation, camera_t[int(batch_size*3/4):],
    #                                          focal_length, camera_center[int(batch_size*3/4):])
    # Weighted robust reprojection error
    #reprojection_error_1 = gmof(projected_joints_0 - joints_2d[:int(batch_size/4)], sigma)
    #reprojection_error_2 = gmof(projected_joints_1 - joints_2d[int(batch_size/4):int(batch_size/2)], sigma)
    #reprojection_error_3 = gmof(projected_joints_2 - joints_2d[int(batch_size/2):int(batch_size*3/4)], sigma)
    #reprojection_error_4 = gmof(projected_joints_3 - joints_2d[int(batch_size*3/4):], sigma)
    
    #reprojection_loss = (joints_conf[:int(batch_size/4)] ** 2) * reprojection_error_1.sum(dim=-1) +\
    #                    (joints_conf[int(batch_size/4):int(batch_size/2)] ** 2) * reprojection_error_2.sum(dim=-1) +\
    #                    (joints_conf[int(batch_size/2):int(batch_size*3/4)] ** 2) * reprojection_error_3.sum(dim=-1) +\
    #                    (joints_conf[int(batch_size*3/4):] ** 2) * reprojection_error_4.sum(dim=-1)
    # Pose prior loss
    pose_prior_loss = (pose_prior_weight ** 2) * pose_prior(body_pose, betas)

    # Angle prior for knees and elbows
    angle_prior_loss = (angle_prior_weight ** 2) * angle_prior(body_pose).sum(dim=-1)

    # Regularizer to prevent betas from taking large values
    shape_prior_loss = (shape_prior_weight ** 2) * (betas ** 2).sum(dim=-1)
    #print(reprojection_loss.size(),pose_prior_loss.size(),angle_prior_loss.size(),shape_prior_loss.size())
    total_loss = reprojection_loss.sum(dim=-1) + pose_prior_loss + angle_prior_loss + shape_prior_loss

    if output == 'sum':
        return total_loss.sum()
    elif output == 'reprojection':
        return reprojection_loss

def camera_fitting_loss(model_joints, camera_t, camera_t_est, camera_center, joints_2d, joints_conf,
                        focal_length=5000, depth_loss_weight=100):
    """
    Loss function for camera optimization.
    """

    # Project model joints
    batch_size = model_joints.shape[0]
    rotation = torch.eye(3, device=model_joints.device).unsqueeze(0).expand(batch_size, -1, -1)
    #print(model_joints.size(),rotation.size(),camera_t.size(),camera_center.size())
    projected_joints = perspective_projection(model_joints, rotation, camera_t,
                                              focal_length, camera_center)
    
    op_joints = ['OP RHip', 'OP LHip', 'OP RShoulder', 'OP LShoulder']
    op_joints_ind = [constants.JOINT_IDS[joint] for joint in op_joints]
    gt_joints = ['Right Hip', 'Left Hip', 'Right Shoulder', 'Left Shoulder']
    gt_joints_ind = [constants.JOINT_IDS[joint] for joint in gt_joints]
    #print(op_joints_ind,gt_joints_ind)
    #print(joints_2d[:, gt_joints_ind])
    #input()
    reprojection_error_op = (joints_2d[:, op_joints_ind] -
                             projected_joints[:, op_joints_ind]) ** 2
    reprojection_error_gt = (joints_2d[:, gt_joints_ind] -
                             projected_joints[:, gt_joints_ind]) ** 2
    #print('joint_2d',joints_2d[:, gt_joints_ind])
    #print('projected_2d',projected_joints[:, gt_joints_ind])
    # Check if for each example in the batch all 4 OpenPose detections are valid, otherwise use the GT detections
    # OpenPose joints are more reliable for this task, so we prefer to use them if possible
    is_valid = (joints_conf[:, op_joints_ind].min(dim=-1)[0][:,None,None] > 0).float()
    reprojection_loss = (is_valid * reprojection_error_op + (1-is_valid) * reprojection_error_gt).sum(dim=(1,2))

    # Loss that penalizes deviation from depth estimate
    depth_loss = (depth_loss_weight ** 2) * (camera_t[:, 2] - camera_t_est[:, 2]) ** 2

    total_loss = reprojection_loss + depth_loss
    return total_loss.sum()
