import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torchvision.transforms as transforms
from torchvision import datasets
import torch
import sklearn
import wandb

desktop_path = '/Users/jocelynornelasmunoz/Desktop/Research/coded-aperture/jornelasmunoz/'
laptop_path = '/Users/jocelynornelas/iCloud Drive (Archive)/Desktop/UC Merced/Research/coded-aperture/jornelasmunoz/'
if desktop_path in sys.path[0]: sys.path.insert(0, desktop_path + 'lib/'); path = desktop_path
elif laptop_path in sys.path[0]: sys.path.insert(0, laptop_path + 'lib/'); path = laptop_path
print('Using path = ', path)

import MURA as mura
from Reconstruct import CNN
import confusion_matrix as cm
import wandb_functions as wf

# Update plotting parameters
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Times", #"Computer Modern Serif"
    "figure.figsize" : [15,7],#[15,10],
})

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Log into WandB
wandb.login()

# ------------------ Define Parameters ---------------
params = {
  "dataset": "encoded_MNIST",
  "learning_rate": 1e-3,
  "batch_size": 100,
  "epochs": 50,
  "kernel_size":23,
  "p": 23, # size of array (has to be prime)
  "image_size": 23,
  "kind": "recon",
  "suffix": "l_1_pen_lambda_",
  "lambda": 5e-1,
}

# Define model name and save path
params['model'] = params['kind']+'_' + params['suffix'] + f"{params['lambda']:.0e}"
params['model_save_path'] = f'../models/{params["kind"]}/{params["model"]}.pth'
# Compute MURA encoder and decoder
params['A'] = mura.create_binary_aperture_arr(params['p'])
params['G'] = mura.create_decoding_arr(params['A'])

# Save params in WandB runs
wandb.config = wf.wandb_config(params)

# ------------------ Load Data ----------------------
mura_train_data, mura_eval_data, mura_test_data, loaders = CNN.load_data(params)

# ------------------ Model and Training ---------------
# Instantiate model 
model = CNN(params)#.to(device)
model.optimizer = torch.optim.Adam(model.parameters(), lr = model.params['learning_rate']) 

# Initialize project in Weights and Biases
wandb.init(config=wandb.config, project="coded-aperture-MNIST", group=model.params["kind"], name=f"{model.params['model']}_exp1")

# Store values for later 
train_loss = []
val_loss = []
frob_per_epoch = []

#Dictionary that will store different images and outputs for various epochs (not sure if needed)
outputs = {}

# Training loop starts
for epoch in range(params['epochs']):
    
    # Initialize variable to store loss
    running_loss = 0
    model.train()
    # Iterate over training set
    for i, data in enumerate(loaders['train']):
        # get the inputs; data is a list of [images, labels, digit]
        inputs, targets, digits = data
        
        # Generate output
        out = model(inputs)
        
        # Calculate loss
        loss = model.criterion(out, targets)
        # Modified (04/06/23) to include l_1 penalty
        l_1_pen = sum((w.abs()-1).abs().sum() for w in model.parameters())
        loss = loss + model.params['lambda'] * l_1_pen
        # zero the parameter gradients
        model.optimizer.zero_grad()
        # Backprop and update weights
        loss.backward()
        model.optimizer.step()
        
        # Increment loss
        running_loss += loss.item()
    
    # Average loss over entire dataset
    running_loss/= len(loaders['train'].dataset)#params['batch_size']
    train_loss.append(running_loss)  
    
    
    # --------------------- Validation ------------------
    model.eval()
    running_vloss = 0.0
    with torch.no_grad():
        for i, vdata in enumerate(loaders['eval']):
            vinputs, vtargets, vdigits = vdata
            voutputs = model(vinputs)
            # validation loss modified to include penalty
            vloss = (model.criterion(voutputs, vtargets) + model.params['lambda'] * l_1_pen).item()
            #vloss = model.criterion(voutputs, vtargets)
            running_vloss += vloss
        running_vloss/= len(loaders['eval'].dataset)
        val_loss.append(running_vloss) 

    
    
    print(f"Epoch {epoch +1} | Loss: {running_loss:.7f} | Val_loss: {running_vloss:.7f}")
    wandb.log({"epoch": epoch, "loss": running_loss, "val_loss": running_vloss})
    
    # Storing images, reconstructed outputs, and labels
    outputs[epoch+1] = {'img': inputs, 'out': out, 'targets': targets, 'digits': digits}
    
    # Save weights every 10 epochs
    if epoch % 10 == 9:
        if abs(train_loss[epoch] < train_loss[epoch-1]):
            torch.save(model.state_dict(), model.params['model_save_path'])
        # epoch_save_model = path+f'models/CNN/{params["model"]}_model_epoch_'+str(epoch)+'.pth'
        # torch.save(model.state_dict(),epoch_save_model)
        weight_epoch_list = list(model.parameters())
        weight_epoch = np.squeeze(weight_epoch_list[0].detach().cpu().numpy())
        plt.figure(figsize=(15,15))
        heat = sns.heatmap(weight_epoch, cmap='gray')
        figure = heat.get_figure()
        figure.savefig(path+f'/metrics/CNN/{params["model"]}_model_heat_epoch_'+str(epoch)+'.png',dpi=400)
        plt.close()
    
    # Calculate Frobenius norm between weights and decoder every epoch
    weights = list(model.parameters())
    weight_map = np.squeeze(weights[0].detach().cpu().numpy())
    diff_block = params['G'] - weight_map
    frob_per_epoch.append(np.linalg.norm(np.abs(diff_block),ord='fro'))

# End WandB run
wandb.finish()

# Save Model
torch.save(model.state_dict(), model.params['model_save_path'])



#----------------------------- Save Other Stuff -----------------------------
import pickle
with open(path+f'metrics/CNN/train_loss_{params["model"]}.pkl','wb') as fp:
    pickle.dump(train_loss, fp)
with open(path+f'metrics/CNN/validation_train_loss_{params["model"]}.pkl','wb') as fp:
    pickle.dump(val_loss, fp)
with open(path+f'metrics/CNN/weight_norm_{params["model"]}.pkl','wb') as fp:
    pickle.dump(frob_per_epoch, fp)
with open(path+f'metrics/CNN/outputs_per_epoch_{params["model"]}.pkl','wb') as fp:
    pickle.dump(outputs, fp)


    
# ------------------ Evaluate Validation Set -----------------
# Initialize variables
images_all = []
predicted_all = []
labels_all = []
digit_labels_all = []
mse_all = []
test_outputs = {}

with torch.no_grad():
    for data in loaders['eval']:
        # get images and labels from test set
        img_batch, label_batch, digit_batch = data
        
        # calculate outputs by running images through the network (done in batches)
        pred_batch = model(img_batch)
        
        # Calculate MSE for each pair of label,prediction images
        for image, label, prediction, digit in zip(img_batch, label_batch, pred_batch, digit_batch):
            one_mse = sklearn.metrics.mean_squared_error(torch.squeeze(label), torch.squeeze(prediction))
            # Store values in lists
            mse_all.append(one_mse)
            images_all.append(image)#.reshape(-1, model.img_size, model.img_size))
            labels_all.append(label)#.reshape(-1, model.img_size, model.img_size))
            predicted_all.append(prediction)#.reshape(-1, model.img_size, model.img_size))
            digit_labels_all.append(digit.numpy())
            
            
    # Storing information in dictionary
    test_outputs['img']       = images_all
    test_outputs['label']     = labels_all
    test_outputs['pred']      = predicted_all
    test_outputs['digit']     = digit_labels_all
    test_outputs['mse_score'] = mse_all
    
# ------------------ Save Figures ----------------------------
fontsize = 30

# ---- Training Loss
fig, axs = plt.subplots(1,1)
axs.plot(range(1,params['epochs']+1), train_loss, label="Training Loss")
axs.plot(range(1,params['epochs']+1), val_loss, label="Validation Loss")

axs.set_xlabel("Number of epochs", fontsize = fontsize-2)
axs.set_ylabel("MSE", fontsize = fontsize-2)
axs.set_title("MSE Training Loss", fontsize=fontsize)
plt.xticks(fontsize=fontsize-6)
plt.yticks(fontsize=fontsize-6)
plt.legend(fontsize=fontsize-6)
axs.legend(fontsize=fontsize)
plt.savefig(f'../figs/training_loss_{params["model"]}_{params["epochs"]}epochs.png')
plt.show(); plt.close()

# ---- Frobenius Norm
plt.plot(np.arange(params['epochs']), frob_per_epoch)
plt.title("Frobenius norm, $\|G-G_L\|_F$", fontsize=fontsize)
plt.xlabel("Epoch", fontsize=fontsize-4)
plt.ylabel("Norm", fontsize=fontsize-4)
plt.xticks(fontsize=fontsize-14)
plt.yticks(fontsize=fontsize-14)
plt.savefig(path+f'figs/frob_per_epoch_{params["model"]}.png')
plt.show(); plt.close()

# ---- Reconstructed images
# Initializing subplot counter
counter = 1

# Plotting original images

# Plotting first 10 images
for idx in range(10):
    val = test_outputs['img']
    plt.subplot(3, 10, counter)
    plt.imshow(val[idx].reshape(model.img_size, model.img_size), cmap='gray')
    plt.title(f"Input Image: {test_outputs['digit'][idx]}")
    plt.axis('off')
  
    # Incrementing subplot counter
    counter += 1

# Plotting reconstructions
val = test_outputs['pred']#.numpy()
  
# Plotting first 10 images of the batch
for idx in range(10):
    plt.subplot(3, 10, counter)
    plt.title("Reconstructed")
    plt.imshow(val[idx].reshape(model.img_size, model.img_size), cmap='gray')
    plt.axis('off')
  
    # Incrementing subplot counter
    counter += 1
    
# Plotting label images

# Plotting first 10 images
for idx in range(10):
    val = test_outputs['label']
    plt.subplot(3, 10, counter)
    plt.imshow(val[idx].reshape(model.img_size, model.img_size), cmap='gray')
    plt.title("Original Image")
    plt.axis('off')
  
    # Incrementing subplot counter
    counter += 1
    
plt.suptitle(f"Example reconstructions {params['model']}", fontsize=30)
plt.savefig(path+f'figs/example_recons_{params["model"]}.png')
plt.tight_layout()
plt.show(); plt.close()

#------------------------- Save the final weights --------------------------
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Times", #"Computer Modern Serif"
    "figure.figsize" : [20,16],
})

final_weights = list(model.parameters())
final_weight_map = final_weights[0].detach().cpu().numpy()
sns.heatmap(np.squeeze(final_weight_map), cmap='gray')
plt.savefig(f'../figs/final_weights_{params["model"]}.png')
plt.show(); plt.close()