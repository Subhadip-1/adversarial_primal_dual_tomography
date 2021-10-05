#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 10:12:17 2021

@author: subhadip
"""

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import odl

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = ['cuda:4', 'cuda:5', 'cuda:6', 'cuda:7']
print('device = %s'%device)

import mayo_utils, networks, torch_wrapper
from skimage.measure import compare_ssim, compare_psnr




##############specify geometry parameters#################
from simulate_projections_for_train_and_test import img_size, space_range, num_angles, det_shape, noise_std_dev, geom

#img_size, space_range = 512, 10.0 #space discretization
#num_angles, det_shape = 128, 512 #projection parameters
#noise_std_dev = 0.1
#geom = 'parallel_beam' # 'cone_beam' or 'parallel_beam'

space = odl.uniform_discr([-space_range, -space_range], [space_range, space_range],\
                              (img_size, img_size), dtype='float32', weighting=1.0)
if(geom=='parallel_beam'):
    geometry = odl.tomo.geometry.parallel.parallel_beam_geometry(space, num_angles=num_angles, det_shape=det_shape)
else:
    geometry = odl.tomo.geometry.conebeam.cone_beam_geometry(space, src_radius=1.5*space_range, \
                                                             det_radius=5.0, num_angles=num_angles, det_shape=det_shape)
    
fwd_op_odl = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')
op_norm = 1.1 * odl.power_method_opnorm(fwd_op_odl)
print('operator norm = {:.4f}'.format(op_norm))

fbp_op_odl = odl.tomo.analytic.filtered_back_projection.fbp_op(ray_trafo=fwd_op_odl)
adjoint_op_odl = fwd_op_odl.adjoint

fwd_op = torch_wrapper.OperatorModule(fwd_op_odl).to(device)
fbp_op = torch_wrapper.OperatorModule(fbp_op_odl).to(device)
adjoint_op = torch_wrapper.OperatorModule(adjoint_op_odl).to(device)

##############create the network models#################
model_path = './trained_models/'
multi_gpu = False
device_ids = [i for i in range(torch.cuda.device_count())]
print(device_ids)
#lpd_generator = nn.DataParallel(networks.LPD(fwd_op, adjoint_op, niter=20, sigma=0.01, tau=0.01), device_ids=[4, 5, 6, 7]).to(device)
lpd_generator = networks.LPD(fwd_op, adjoint_op, niter=15, sigma=0.01, tau=0.01).to(device)
lpd_generator.load_state_dict(torch.load(model_path + "lpd_generator_epoch_25.pt")) #may need to modify the filename based on the name of the saved model
num_lpd_params = sum(p.numel() for p in lpd_generator.parameters())


load_discriminator = False
if load_discriminator:
    discriminator = nn.DataParallel(networks.Discriminator(in_channels=1, n_filters=64), device_ids=[4, 5, 6, 7]).to(device)
    discriminator = networks.Discriminator(in_channels=1, n_filters=64).to(device)
    discriminator.load_state_dict(torch.load(model_path + "discriminator_epoch_20.pt")) #may need to change this name
    num_discriminator_params = sum(p.numel() for p in discriminator.parameters())
    print('# params: LPD-net = {:d}, discriminator = {:d}'.format(num_lpd_params, num_discriminator_params))  
      
############ dataloaders #######################
print('creating dataloaders...')
output_datapath = './mayo_data_arranged_patientwise/'
transform_to_tensor = [transforms.ToTensor()]
#testing dataloader
eval_dataloader = DataLoader(mayo_utils.mayo_dataset(output_datapath, transforms_=transform_to_tensor, aligned = True, mode = 'test'),\
                              batch_size = 1, shuffle = False)
print('number of minibatches during eval = %d'%len(eval_dataloader))      

psnr_fbp_avg, ssim_fbp_avg, psnr_alpd_avg, ssim_alpd_avg = 0.0, 0.0, 0.0, 0.0

for index, batch in enumerate(eval_dataloader):
    #load the images and compute the loss
    phantom = batch["phantom"].to(device)
    phantom_image = phantom.cpu().detach().numpy().squeeze()
    data_range=np.max(phantom_image) - np.min(phantom_image)
    fbp = batch["fbp"].to(device)
    sinogram = batch["sinogram"].to(device)
    
    
#    sinogram, fbp = simulate_projections_for_train_and_test.compute_projection\
#    (phantom, num_angles=num_angles, det_shape=det_shape, space_range=space_range, geom=geom, noise_std_dev=noise_std_dev)
    
#    #fbp = batch["fbp"].to(device)
#    phantom_image = phantom.cpu().detach().numpy().squeeze()
#    data_range=np.max(phantom_image) - np.min(phantom_image)
#    
#    sinogram = batch["sinogram"].to(device)
#    fbp = fbp_op(sinogram) #corresponding FBP for performance tracking
    fbp_image = fbp.cpu().detach().numpy().squeeze()
    psnr_fbp = compare_psnr(phantom_image, fbp_image, data_range=data_range)
    ssim_fbp = compare_ssim(phantom_image, fbp_image, data_range=data_range)
    
    psnr_fbp_avg += psnr_fbp
    ssim_fbp_avg += ssim_fbp
    
    #clamp FBP and use an initialization for the unrolled gradient network
    x_init = torch.from_numpy(fbp_image).view(fbp.size()).to(device).requires_grad_(True)
    
    # Generate a batch of images
    recon = lpd_generator(sinogram, x_init)      
    recon_image = recon.cpu().detach().numpy().squeeze()
    
    psnr_alpd = compare_psnr(phantom_image, recon_image, data_range=data_range)
    ssim_alpd = compare_ssim(phantom_image, recon_image, data_range=data_range)
    
    psnr_alpd_avg += psnr_alpd
    ssim_alpd_avg += ssim_alpd
    
    eval_log = 'image: [{}/{}]\t FBP: PSNR {:.4f}, SSIM {:.4f} \t adv_LPD: PSNR {:.4f}, SSIM {:.4f}\n'.\
                  format(index+1, len(eval_dataloader), psnr_fbp, ssim_fbp, psnr_alpd, ssim_alpd)
    print(eval_log)

print('########## average performance ##########')
num_images = len(eval_dataloader)
print('FBP: PSNR {:.4f}, SSIM {:.4f} \t adv_LPD: PSNR {:.4f}, SSIM {:.4f}\n'.\
                  format(psnr_fbp_avg/num_images, ssim_fbp_avg/num_images, psnr_alpd_avg/num_images, ssim_alpd_avg/num_images))





















