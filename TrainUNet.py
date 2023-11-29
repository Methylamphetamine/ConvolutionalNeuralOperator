import copy
import json
import os
import sys

import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


# from Problems.CNOBenchmarks import Darcy, Airfoil, DiscContTranslation, ContTranslation, AllenCahn, SinFrequency, WaveEquation, ShearLayer
from Problems.UNetBenchmarks import SinFrequency



if len(sys.argv) >= 2:
    
    training_properties = {
        "learning_rate": 0.001, 
        "weight_decay": 1e-6,
        "scheduler_step": 10,
        "scheduler_gamma": 0.98,
        "epochs": 1000,
        "batch_size": 16,
        "exp": 1,                # Do we use L1 or L2 errors? Default: L1
        "training_samples": 1024  # How many training samples?
    }
    model_architecture_ = {
        
        #Other parameters:
        "in_size": 64,            # Resolution of the computational grid
        "retrain": 4,             # Random seed
        "FourierF": 0,            # Number of Fourier Features in the input channels. Default is 0.
        "initial_channels" : 1,
        "start_channels" : 64,


    }


    which_example = str(sys.argv[1])
    unet_type = str(sys.argv[2])
    #which_example = "shear_layer"

    # Save the models here:
    folder = "TrainedModels/"+"UNet_" + unet_type + "_" +which_example+"_1"
        
else:
    raise NotImplementedError
    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter(log_dir=folder) #usage of TensorBoard

learning_rate = training_properties["learning_rate"]
epochs = training_properties["epochs"]
batch_size = training_properties["batch_size"]
weight_decay = training_properties["weight_decay"]
scheduler_step = training_properties["scheduler_step"]
scheduler_gamma = training_properties["scheduler_gamma"]
training_samples = training_properties["training_samples"]
p = training_properties["exp"]

if not os.path.isdir(folder):
    print("Generated new folder")
    os.mkdir(folder)

df = pd.DataFrame.from_dict([training_properties]).T
df.to_csv(folder + '/training_properties.txt', header=False, index=True, mode='w')
df = pd.DataFrame.from_dict([model_architecture_]).T
df.to_csv(folder + '/net_architecture.txt', header=False, index=True, mode='w')

model_architecture_["mode"] = unet_type
if which_example == "poisson":
    model_architecture_["out_dim"] = 1
    example = SinFrequency(model_architecture_, device, batch_size, training_samples)
    ood_example = SinFrequency(model_architecture_, device, batch_size, training_samples, in_dist=False)

else:
    raise NotImplementedError
    
#-----------------------------------Train--------------------------------------
model = example.model
n_params = model.print_size()
train_loader = example.train_loader #TRAIN LOADER
val_loader = example.val_loader #VALIDATION LOADER
test_loader = example.test_loader
ood_test_loader = ood_example.test_loader

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
freq_print = 1

if p == 1:
    loss = torch.nn.L1Loss()
elif p == 2:
    loss = torch.nn.MSELoss()
    
best_model_testing_error = 1000 #Save the model once it has less than 1000% relative L1 error
patience = int(0.2 * epochs)    # Early stopping parameter
counter = 0

if str(device) == 'cpu':
    print("------------------------------------------")
    print("YOU ARE RUNNING THE CODE ON A CPU.")
    print("WE SUGGEST YOU TO RUN THE CODE ON A GPU!")
    print("------------------------------------------")
    print(" ")





for epoch in range(epochs):
    with tqdm(unit="batch", disable=False) as tepoch:
        
        model.train()
        tepoch.set_description(f"Epoch {epoch}")
        train_mse = 0.0
        running_relative_train_mse = 0.0
        for step, (input_batch, output_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            input_batch = input_batch.to(device)
            output_batch = output_batch.to(device)

            output_pred_batch = model(input_batch)

            if which_example == "airfoil": #Mask the airfoil shape
                output_pred_batch[input_batch==1] = 1
                output_batch[input_batch==1] = 1

            loss_f = loss(output_pred_batch, output_batch) / loss(torch.zeros_like(output_batch).to(device), output_batch)

            loss_f.backward()
            optimizer.step()
            train_mse = train_mse * step / (step + 1) + loss_f.item() / (step + 1)
            tepoch.set_postfix({'Batch': step + 1, 'Train loss (in progress)': train_mse})

        writer.add_scalar("train_loss/train_loss", train_mse, epoch)
        
        with torch.no_grad():
            model.eval()
            test_relative_l2 = 0.0
            train_relative_l2 = 0.0
            
            for step, (input_batch, output_batch) in enumerate(val_loader):
                
                input_batch = input_batch.to(device)
                output_batch = output_batch.to(device)
                output_pred_batch = model(input_batch)
                
                if which_example == "airfoil": #Mask the airfoil shape
                    output_pred_batch[input_batch==1] = 1
                    output_batch[input_batch==1] = 1
                
                loss_f = torch.mean(abs(output_pred_batch - output_batch)) / torch.mean(abs(output_batch)) * 100
                test_relative_l2 += loss_f.item()
            test_relative_l2 /= len(val_loader)

            for step, (input_batch, output_batch) in enumerate(train_loader):
                input_batch = input_batch.to(device)
                output_batch = output_batch.to(device)
                output_pred_batch = model(input_batch)
                    
                if which_example == "airfoil": #Mask the airfoil shape
                    output_pred_batch[input_batch==1] = 1
                    output_batch[input_batch==1] = 1

                    loss_f = torch.mean(abs(output_pred_batch - output_batch)) / torch.mean(abs(output_batch)) * 100
                    train_relative_l2 += loss_f.item()
            train_relative_l2 /= len(train_loader)
            
            writer.add_scalar("train_loss/train_loss_rel", train_relative_l2, epoch)
            writer.add_scalar("val_loss/val_loss", test_relative_l2, epoch)

            if test_relative_l2 < best_model_testing_error:
                best_model_testing_error = test_relative_l2
                best_model = copy.deepcopy(model)
                torch.save(best_model, folder + "/model.pkl")
                writer.add_scalar("val_loss/Best Relative Validation Error", best_model_testing_error, epoch)
                counter = 0
            else:
                counter+=1

        tepoch.set_postfix({'Train loss': train_mse, "Relative Train": train_relative_l2, "Relative Val loss": test_relative_l2})
        tepoch.close()

        with open(folder + '/errors.txt', 'w') as file:
            file.write("Training Error: " + str(train_mse) + "\n")
            file.write("Best Validation Error: " + str(best_model_testing_error) + "\n")
            file.write("Current Epoch: " + str(epoch) + "\n")
            file.write("Params: " + str(n_params) + "\n")
        scheduler.step()

    if counter>patience:
        print("Early Stopping")
        break


with torch.no_grad():
    model.eval()
    final_test_relative_l2 = 0.0
    
    for step, (input_batch, output_batch) in enumerate(test_loader):
        
        input_batch = input_batch.to(device)
        output_batch = output_batch.to(device)
        output_pred_batch = model(input_batch)
        
        if which_example == "airfoil": #Mask the airfoil shape
            output_pred_batch[input_batch==1] = 1
            output_batch[input_batch==1] = 1
        
        loss_f = torch.mean(abs(output_pred_batch - output_batch)) / torch.mean(abs(output_batch)) * 100
        final_test_relative_l2 += loss_f.item()
    final_test_relative_l2 /= len(test_loader)

    ood_test_relative_l2 = 0.0
    
    for step, (input_batch, output_batch) in enumerate(ood_test_loader):
        
        input_batch = input_batch.to(device)
        output_batch = output_batch.to(device)
        output_pred_batch = model(input_batch)
        
        if which_example == "airfoil": #Mask the airfoil shape
            output_pred_batch[input_batch==1] = 1
            output_batch[input_batch==1] = 1
        
        loss_f = torch.mean(abs(output_pred_batch - output_batch)) / torch.mean(abs(output_batch)) * 100
        ood_test_relative_l2 += loss_f.item()
    ood_test_relative_l2 /= len(ood_test_loader)

print(final_test_relative_l2)
print(ood_test_relative_l2)

with open(folder + '/errors.txt', 'a') as file:
    file.write("Final Testing Error: " + str(final_test_relative_l2) + "\n")
    file.write("OOD Testing Error: " + str(ood_test_relative_l2) + "\n")