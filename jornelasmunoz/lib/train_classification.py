import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.transforms as transforms
import torch
import sklearn
import wandb

# Choose path depending in which device I am working with
desktop_path = '/Users/jocelynornelasmunoz/Desktop/Research/coded-aperture/jornelasmunoz/'
laptop_path = '/Users/jocelynornelas/iCloud Drive (Archive)/Desktop/UC Merced/Research/coded-aperture/jornelasmunoz/'
if desktop_path in sys.path[0]: sys.path.insert(0, desktop_path + 'lib/'); path = desktop_path
elif laptop_path in sys.path[0]: sys.path.insert(0, laptop_path + 'lib/'); path = laptop_path
print('Using path = ', path)

import MURA as mura
from Classify import classification_cnn
import confusion_matrix as cm

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Log into WandB
wandb.login()

# ---------------- LOAD DATA ---------------------------
# define list of parameters
SNR = 10 #noise level in dB
method = 'direct'
params = {
  "dataset": f"MNIST_mura_{SNR}dB",#_reconstructed_{method}_method",
  "learning_rate": 1e-3,
  "batch_size": 100,
  "epochs": 50,
  "p": 23, # size of array (has to be prime)
  "image_size": 23,
  "kernel_size": 3,
  "SNR": SNR,
  "method": method,
  "kind": "classification",
  "suffix": f"{SNR}dB"#_reconstructed_{method}_method",
}
params['model'] = params['kind']+'_' + params['suffix'] 
params['model_save_path'] = f'../models/{params["kind"]}/{params["model"]}.pth'

# Compute MURA encoder and decoder
params['A'] = mura.create_binary_aperture_arr(params['p'])
params['G'] = mura.create_decoding_arr(params['A'])
wandb.config = params

# ---------------- define CNN and get data ----------------
model = classification_cnn(params)
train_data, eval_data, test_data, loaders = classification_cnn.load_encoded_data(model.params)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = model.params['learning_rate']) 

# -------------------------------- TRAIN --------------------------------
# Initialize project in Weights and Biases
wandb.init(config=wandb.config, 
           project="coded-aperture-MNIST", 
           group=model.params["kind"], 
           name=f"{model.params['model']}")

# Store values for later 
train_loss = []
val_loss = []

print("Starting training...")
# Start training
for epoch in range(params['epochs']):
    # --------------------- Training ------------------
    model.train()
    running_loss = 0.0
    for i, data in enumerate(loaders['train']):
        # get the inputs; data is a list of [encoded image, original image, digit label, noise level]
        img_batch, _, digit_batch, _ = data

        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        outputs = model(img_batch)
        loss = criterion(outputs, digit_batch)
        loss.backward()
        optimizer.step()
    

        running_loss += loss.item()
        
    # Average loss over entire dataset
    running_loss /= len(loaders['train'].dataset)
    train_loss.append(running_loss)
    
    # --------------------- Validation ------------------
    model.eval()
    running_vloss = 0.0
    with torch.no_grad():
        for i, vdata in enumerate(loaders['eval']):
            vinputs, _, vdigits, _ = vdata
            voutputs = model(vinputs)
            vloss = criterion(voutputs, vdigits)
            running_vloss += vloss
        running_vloss/= len(loaders['eval'].dataset)
        val_loss.append(running_vloss) 
        
    print(f"Epoch {epoch +1} | Loss: {running_loss:.7f} | Val_loss: {running_vloss:.7f}")
    wandb.log({"epoch": epoch, "loss": running_loss, "val_loss": running_vloss})
print('Finished Training')

# Save model
wandb.finish()
print(f"Model will be saved in {model.params['model_save_path']}")
torch.save(model.state_dict(), model.params['model_save_path'])
print("Model saved")

#----------------------------- Save The Losses -----------------------------
print("Saving training and validation loss")
import pickle
with open(path+f'metrics/{params["kind"]}/train_loss_{params["model"]}.pkl','wb') as fp:
    pickle.dump(train_loss, fp)
with open(path+f'metrics/{params["kind"]}/validation_train_loss_{params["model"]}.pkl','wb') as fp:
    pickle.dump(val_loss, fp)

    
#----------------------------- Evaluate Test set -----------------------------

# Initialize variables
correct = 0
total = 0
incorrect_examples = []
predicted_all = []
labels_all = []

# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in loaders['test']:
        # get images and labels from test set
        images, _, labels, _ = data
    
        # calculate outputs by running images through the network (done in batches)
        outputs = model(images)
        
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Save misclassifications
        idxs_mask = torch.nonzero(predicted != labels) #((predicted == labels) == False).nonzero()
        for single_sample in idxs_mask:
            incorrect_examples.append([np.squeeze(images[single_sample].numpy()), 
                                       labels[single_sample].numpy()[0], 
                                       predicted[single_sample].numpy()[0]])
        predicted_all.append(predicted.tolist())
        labels_all.append(labels.tolist())

print(f'Accuracy of the model {params["model"]} on the {total} test images: {100 * correct / total} %')

predicted_all = list(np.concatenate(predicted_all).flat) 
labels_all = list(np.concatenate(labels_all).flat) 



#----------------------------- PLOTTING -----------------------------
# Change plotting parameters
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Times", #"Computer Modern Serif"
    "figure.figsize" : [15,12],
})

FIG_FOLDER = path + f'figs/{params["kind"]}/{params["method"]}_method/'
title_size = 30
label_size = 28
tick_size = 24

# --- Plot training and val loss
fig, axs = plt.subplots(1,1)
axs.plot(range(1,params['epochs']+1), train_loss, label="Training Loss")
axs.plot(range(1,params['epochs']+1), val_loss, label="Validation Loss")


axs.set_xlabel("Number of epochs", fontsize = label_size)
axs.set_ylabel("Loss", fontsize = label_size)
axs.set_title("Cross Entropy Training Loss", fontsize=title_size)
plt.xticks(fontsize=tick_size)
plt.yticks(fontsize=tick_size)
axs.legend(fontsize=title_size)
plt.savefig(FIG_FOLDER + f'training_loss_{params["model"]}.png')
plt.show();plt.close()

# --- Confusion Matrix
# Converting list of tensors to list of integers
preds = [pred for pred in predicted_all]
truths = [tru for tru in labels_all]

con_mat = sklearn.metrics.confusion_matrix(truths,preds,normalize='true')
axlabels= [ str(num) for num in np.arange(10)]

fig, ax = plt.subplots(figsize=(10,7))
im, cbar = cm.heatmap(con_mat, axlabels, axlabels, ax=ax,cmap = 'Blues', cbarlabel="Probability")
plt.ylabel("Truth")
plt.xlabel("Prediction")
texts = cm.annotate_heatmap(im, valfmt="{x:.3f}")
fig1 = plt.gcf()
fig1.savefig(FIG_FOLDER + f'conf_mat_{params["model"]}.png')

# Save dataframe with misclassifications
df = pd.DataFrame(incorrect_examples, columns=['image_array', 'label', 'prediction'])
print("Misclassifications stats")
print(df.label.value_counts(normalize=True))

plt.rcParams.update({"figure.figsize" : [25,15]})
crosstab_misclass = pd.crosstab(df.label, df.prediction, margins=False)
crosstab_misclass.plot(kind="bar", stacked=True, rot=0, cmap="PuBu")
plt.ylabel("Count", fontsize=label_size)
plt.xlabel("Label", fontsize=label_size)
plt.title(f"Misclassifications for {params['SNR']}dB data reconstructed using $G$", fontsize=title_size)
plt.xticks(fontsize=tick_size)
plt.yticks(fontsize=tick_size)
plt.legend(fontsize=label_size,loc=(1.04, 0.15))
plt.savefig(FIG_FOLDER + f'misclass_{params["model"]}.png')
plt.show(); plt.close()
print(crosstab_misclass)