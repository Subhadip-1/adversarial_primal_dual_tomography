#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 18:56:04 2021

@author: subhadip
"""

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import odl
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = ['cuda:4', 'cuda:5', 'cuda:6', 'cuda:7']
print('device = %s'%device)

import mayo_utils, networks, torch_wrapper
from skimage.measure import compare_ssim, compare_psnr


##############specify geometry parameters#################
img_size, space_range = 512, 10.0 #space discretization
num_angles, det_shape = 128, 512 #projection parameters
noise_std_dev = 0.1
geom = 'parallel_beam' # 'cone_beam' or 'parallel_beam'

space = odl.uniform_discr([-space_range, -space_range], [space_range, space_range],\
                              (img_size, img_size), dtype='float32', weighting=1.0)
if(geom=='parallel_beam'):
    geometry = odl.tomo.geometry.parallel.parallel_beam_geometry(space, num_angles=num_angles, det_shape=det_shape)
else:
    geometry = odl.tomo.geometry.conebeam.cone_beam_geometry(space, src_radius=1.5*space_range, \
                                                             det_radius=5.0, num_angles=num_angles, det_shape=det_shape)
    
fwd_op_odl = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')
fbp_op_odl = odl.tomo.analytic.filtered_back_projection.fbp_op(ray_trafo=fwd_op_odl)
adjoint_op_odl = fwd_op_odl.adjoint

fwd_op = torch_wrapper.OperatorModule(fwd_op_odl).to(device)
fbp_op = torch_wrapper.OperatorModule(fbp_op_odl).to(device)
adjoint_op = torch_wrapper.OperatorModule(adjoint_op_odl).to(device)


##############create the network models#################
multi_gpu = False
device_ids = [i for i in range(torch.cuda.device_count())]
print(device_ids)
#lpd_generator = nn.DataParallel(networks.LPD(fwd_op, adjoint_op, niter=20, sigma=0.01, tau=0.01), device_ids=[4, 5, 6, 7]).to(device)
lpd_generator = networks.LPD(fwd_op, adjoint_op, niter=15, sigma=0.01, tau=0.01).to(device)
num_lpd_params = sum(p.numel() for p in lpd_generator.parameters())

#discriminator = nn.DataParallel(networks.Discriminator(in_channels=1, n_filters=64), device_ids=[4, 5, 6, 7]).to(device)
discriminator = networks.Discriminator(in_channels=1, n_filters=64).to(device)
discriminator.apply(networks.weights_init)
num_discriminator_params = sum(p.numel() for p in discriminator.parameters())

print('# params: LPD-net = {:d}, discriminator = {:d}'.format(num_lpd_params, num_discriminator_params))  
      
############ dataloaders #######################
print('creating dataloaders...')
output_datapath = './mayo_data_arranged_patientwise/'
transform_to_tensor = [transforms.ToTensor()]
train_dataloader = DataLoader(mayo_utils.mayo_dataset(output_datapath, transforms_=transform_to_tensor, aligned = False, mode = 'train'),\
                              batch_size = 2, shuffle = True)
print('number of minibatches during training = %d'%len(train_dataloader))
#testing dataloader
eval_dataloader = DataLoader(mayo_utils.mayo_dataset(output_datapath, transforms_=transform_to_tensor, aligned = True, mode = 'test'),\
                              batch_size = 1, shuffle = True)
print('number of minibatches during eval = %d'%len(eval_dataloader))      

################# gradient penalty loss #######################
import torch.autograd as autograd
def compute_gradient_penalty(network, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.cuda.FloatTensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    #validity = net(interpolates)
    validity = network(interpolates)
    fake = torch.cuda.FloatTensor(np.ones(validity.shape)).requires_grad_(False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(outputs=validity, inputs=interpolates,
                              grad_outputs=fake, create_graph=True, retain_graph=True,
                              only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


########################### training process #####################################
print('############ training the model ############')
mse_loss = torch.nn.MSELoss()
optimizer_G = torch.optim.Adam(lpd_generator.parameters(), lr = 5e-5, betas=(0.9, 0.99))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr = 5e-5, betas=(0.9, 0.99))

############## main training loop ##############
lambda_cycle_image, lambda_cycle_data, lambda_gp = 10.0, 10.0, 10.0 

num_epochs = 20 #see the artifacts in the reconstruction and then modify the penalties. 
num_minibatches_for_avg = 10
n_critic_steps = 1

lpd_generator.train()
discriminator.train()
log_file = open("adv_LPD_training_log.txt","w+")
log_file.write("################ training log\n ################")
print("################ training log ################")
for epoch in range(num_epochs):
    G_loss_total, D_loss_total, recon_loss_clean_data_total = 0.0, 0.0, 0.0
    psnr_on_clean_data_total = 0.0
    num_pairs, num_batches = 0, 0
    for index, batch in enumerate(train_dataloader):
        #load the images and compute the loss
        phantom = batch["phantom"].to(device)
        sinogram = batch["sinogram"].to(device)
        fbp = fbp_op(sinogram) #corresponding FBP for performance tracking
        #clamp FBP and use an initialization for the unrolled gradient network
        fbp_image = fbp.cpu().detach().numpy().squeeze()
        x_init = torch.from_numpy(fbp_image).view(fbp.size()).to(device).requires_grad_(True)
        
        # Generate a batch of images
        recon = lpd_generator(sinogram, x_init)
        # ----------------------
        #Train Discriminator
        # ----------------------
        for nD_updates in range(n_critic_steps):
            optimizer_D.zero_grad()
            # Compute gradient penalty for improved WGAN
            grad_penalty = compute_gradient_penalty(discriminator, phantom.data, recon.data)
            # Adversarial loss
            D_loss = -torch.mean(discriminator(phantom)) + torch.mean(discriminator(recon)) + lambda_gp * grad_penalty
            D_loss.backward(retain_graph=True)
            optimizer_D.step()        
        # ------------------
        #  Train Generators
        # ------------------
        optimizer_G.zero_grad()
        # Translate images to opposite domain
        fake_clean_data = fwd_op(phantom)
        fake_fbp = fbp_op(fake_clean_data)
        fake_fbp_image = fake_fbp.cpu().detach().numpy().squeeze()
        x_init_dummy = torch.from_numpy(fake_fbp_image).view(fbp.size()).to(device).requires_grad_(True)
        
        recovered_phantom = lpd_generator(fake_clean_data, x_init_dummy)
        recovered_sinogram = fwd_op(recon)

        # Adversarial loss
        G_adv = -torch.mean(discriminator(recon))
        # Cycle loss
        recon_loss_clean_data = mse_loss(recovered_phantom, phantom)
        G_cycle = lambda_cycle_data * mse_loss(recovered_sinogram, sinogram) + lambda_cycle_image * mse_loss(recovered_phantom, phantom)
        # Total loss
        G_loss = G_adv + G_cycle
        G_loss.backward()
        optimizer_G.step()     
            
            
        G_loss_total += G_loss.item()
        D_loss_total += D_loss.item()
        recon_loss_clean_data_total += recon_loss_clean_data.item()
        num_pairs += phantom.size()[0]
        num_batches += 1
        
        #### PSNR on reconstruction from clean data ###############
        recovered_phantom_image = recovered_phantom.cpu().detach().numpy().squeeze()
        phantom_image = phantom.cpu().detach().numpy().squeeze()
        data_range=np.max(phantom_image) - np.min(phantom_image)
        psnr_on_clean_data_total += compare_psnr(phantom_image, recovered_phantom_image, data_range=data_range)
        
        #scheduler steps
        #scheduler_D.step()
        #scheduler_G.step()
        if(index % num_minibatches_for_avg == num_minibatches_for_avg-1):
            #compute average loss and reset total loss
            G_loss_avg = G_loss_total/num_batches
            D_loss_avg = D_loss_total/num_batches
            recon_loss_clean_data_avg = recon_loss_clean_data_total/num_batches
            psnr_on_clean_data_avg = psnr_on_clean_data_total/num_batches
            
            G_loss_total, D_loss_total, recon_loss_clean_data_total = 0.0, 0.0, 0.0
            psnr_on_clean_data_total = 0.0
            num_pairs, num_batches = 0, 0
            training_log = 'batch: [{}/{}]\t epoch: [{}/{}] \t G_loss: {:.6f}\t D_loss: {:.6f}\t recon_loss {:.8f} recon_psnr_clean {:.4f}\n'.\
                  format(index+1, len(train_dataloader), epoch+1, num_epochs, G_loss_avg, D_loss_avg, recon_loss_clean_data_avg, psnr_on_clean_data_avg)
            print(training_log)
            log_file.write(training_log)
            
########save the final model################
model_path = './trained_models/'
os.makedirs(model_path, exist_ok=True)
torch.save(lpd_generator.state_dict(), model_path + "lpd_generator_epoch_{:d}.pt".format(num_epochs))  
torch.save(discriminator.state_dict(), model_path + "discriminator_epoch_{:d}.pt".format(num_epochs))            

log_file.write("################ eval log\n ################")
print("################ eval log ################")    
      
for index, batch in enumerate(eval_dataloader):
    #load the images and compute the loss
    phantom = batch["phantom"].to(device)
    phantom_image = phantom.cpu().detach().numpy().squeeze()
    data_range=np.max(phantom_image) - np.min(phantom_image)
    
    sinogram = batch["sinogram"].to(device)
    fbp = fbp_op(sinogram) #corresponding FBP for performance tracking
    fbp_image = fbp.cpu().detach().numpy().squeeze()
    
    psnr_fbp = compare_psnr(phantom_image, fbp_image, data_range=data_range)
    ssim_fbp = compare_ssim(phantom_image, fbp_image, data_range=data_range)
    
    #clamp FBP and use an initialization for the unrolled gradient network
    x_init = torch.from_numpy(fbp_image).view(fbp.size()).to(device).requires_grad_(True)
    
    # Generate a batch of images
    recon = lpd_generator(sinogram, x_init)      
    recon_image = recon.cpu().detach().numpy().squeeze()
    
    psnr_alpd = compare_psnr(phantom_image, recon_image, data_range=data_range)
    ssim_alpd = compare_ssim(phantom_image, recon_image, data_range=data_range)
    
    eval_log = 'image: [{}/{}]\t FBP: PSNR {:.4f}, SSIM {:.4f} \t adv_LPD: PSNR {:.4f}, SSIM {:.4f}\n'.\
                  format(index+1, len(train_dataloader), psnr_fbp, ssim_fbp, psnr_alpd, ssim_alpd)
    print(eval_log)
    log_file.write(eval_log)
      
log_file.close()      
      
      
      
      
      
      
      
      
      
      
      
      
