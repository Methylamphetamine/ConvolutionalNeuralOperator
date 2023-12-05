import copy
import json
import os
import sys

import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
# NOTE changed normalization
from Problems.CNOBenchmarks_new_normalization import Darcy, Airfoil, DiscContTranslation, ContTranslation, AllenCahn, SinFrequency, WaveEquation, ShearLayer



if len(sys.argv) == 2:
    
    # training_properties = {
    #     "learning_rate": 0.001, 
    #     "weight_decay": 1e-6,
    #     "scheduler_step": 10,
    #     "scheduler_gamma": 0.98,
    #     "epochs": 1000,
    #     "batch_size": 16,
    #     "exp": 1,                # Do we use L1 or L2 errors? Default: L1
    #     "training_samples": 256  # How many training samples?
    # }
    # model_architecture_ = {
        
    #     #Parameters to be chosen with model selection:
    #     "N_layers": 3,            # Number of (D) & (U) blocks 
    #     "channel_multiplier": 32, # Parameter d_e (how the number of channels changes)
    #     "N_res": 4,               # Number of (R) blocks in the middle networs.
    #     "N_res_neck" : 6,         # Number of (R) blocks in the BN
        
    #     #Other parameters:
    #     "in_size": 64,            # Resolution of the computational grid
    #     "retrain": 4,             # Random seed
    #     "kernel_size": 3,         # Kernel size.
    #     "FourierF": 0,            # Number of Fourier Features in the input channels. Default is 0.
    #     "activation": 'cno_lrelu',# cno_lrelu or lrelu
        
    #     #Filter properties:
    #     "cutoff_den": 2.0001,     # Cutoff parameter.
    #     "lrelu_upsampling": 2,    # Coefficient N_{\sigma}. Default is 2.
    #     "half_width_mult": 0.8,   # Coefficient c_h. Default is 1
    #     "filter_size": 6,         # 2xfilter_size is the number of taps N_{tap}. Default is 6.
    #     "radial_filter": 0,       # Is the filter radially symmetric? Default is 0 - NO.
    # }
# NOTE: change the hyperparameters to the reported

    training_properties = {
        "learning_rate": 0.001, # CHECKED
        "scheduler_step": 10,
        "scheduler_gamma": 0.98, # CHECKED
        "epochs": 1000,
        "batch_size": 32,
        "exp": 1,                # Do we use L1 or L2 errors? Default: L1
        "training_samples": 256  # How many training samples?
    }
    model_architecture_ = {
        
        #Parameters to be chosen with model selection:
        # "N_layers": 3,            # Number of (D) & (U) blocks CHANGE M
        # "channel_multiplier": 32, # Parameter d_e (how the number of channels changes) CHANGE
        # "N_res": 4,               # Number of (R) blocks in the middle networs. CHANGE
        # "N_res_neck" : 6,         # Number of (R) blocks in the BN CHANGE, r
        
        #Other parameters:
        "in_size": 64,            # Resolution of the computational grid
        "retrain": 4,             # Random seed
        "kernel_size": 3,         # Kernel size. CHECKED
        "FourierF": 0,            # Number of Fourier Features in the input channels. Default is 0.
        "activation": 'cno_lrelu',# cno_lrelu or lrelu
        
        #Filter properties:
        "cutoff_den": 2.0001,     # Cutoff parameter.
        "lrelu_upsampling": 2,    # Coefficient N_{\sigma}. Default is 2.
        "half_width_mult": 0.8,   # Coefficient c_h. Default is 1 CHECKED
        "filter_size": 6,         # 2xfilter_size is the number of taps N_{tap}. Default is 6. CHECKED
        "radial_filter": 0,       # Is the filter radially symmetric? Default is 0 - NO.
    }
    
    #   "which_example" can be 
    
    #   poisson             : Poisson equation 
    #   wave_0_5            : Wave equation
    #   cont_tran           : Smooth Transport
    #   disc_tran           : Discontinuous Transport
    #   allen               : Allen-Cahn equation
    #   shear_layer         : Navier-Stokes equations
    #   airfoil             : Compressible Euler equations
    #   darcy               : Darcy Flow

    which_example = sys.argv[1]
    #which_example = "shear_layer"

    # Save the models here: NOTE change saving directory
    folder = "TrainedReportedModels_NoEarlyStopping_new_normalization/"+"CNO_"+which_example+"_1"
        
else:
    
    # Do we use a script to run the code (for cluster):
    folder = sys.argv[1]
    training_properties = json.loads(sys.argv[2].replace("\'", "\""))
    model_architecture_ = json.loads(sys.argv[3].replace("\'", "\""))
    which_example = sys.argv[4]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter(log_dir=folder) #usage of TensorBoard



if not os.path.isdir(folder):
    print("Generated new folder")
    os.mkdir(folder)

# NOTE moved most of the hyperparameter variables to after the individual example setup
batch_size = training_properties["batch_size"]


# NOTE add ood example, set correct training sample size
if which_example == "shear_layer":
    training_properties.update({
        "weight_decay": 1e-10, # CHANGE
    })
    model_architecture_.update({
        #Parameters to be chosen with model selection:
        "N_layers": 3,            # Number of (D) & (U) blocks CHANGE M
        "channel_multiplier": 32, # Parameter d_e (how the number of channels changes) CHANGE
        "N_res": 1,               # Number of (R) blocks in the middle networs. CHANGE
        "N_res_neck" : 8,         # Number of (R) blocks in the BN CHANGE, r
    })
    training_samples = training_properties["training_samples"] = 750
    example = ShearLayer(model_architecture_, device, batch_size, training_samples, size = 64)
    ood_example = ShearLayer(model_architecture_, device, batch_size, training_samples, size = 64, in_dist=False)
elif which_example == "poisson":
    training_properties.update({
        "weight_decay": 1e-6, # CHANGE
    })
    model_architecture_.update({
        #Parameters to be chosen with model selection:
        "N_layers": 3,            # Number of (D) & (U) blocks CHANGE M
        "channel_multiplier": 16, # Parameter d_e (how the number of channels changes) CHANGE
        "N_res": 4,               # Number of (R) blocks in the middle networs. CHANGE
        "N_res_neck" : 6,         # Number of (R) blocks in the BN CHANGE, r
    })
    training_samples = training_properties["training_samples"] = 1024
    example = SinFrequency(model_architecture_, device, batch_size, training_samples)
    ood_example = SinFrequency(model_architecture_, device, batch_size, training_samples, in_dist=False)
elif which_example == "wave_0_5":
    training_properties.update({
        "weight_decay": 1e-10, # CHANGE
    })
    model_architecture_.update({
        #Parameters to be chosen with model selection:
        "N_layers": 3,            # Number of (D) & (U) blocks CHANGE M
        "channel_multiplier": 48, # Parameter d_e (how the number of channels changes) CHANGE
        "N_res": 4,               # Number of (R) blocks in the middle networs. CHANGE
        "N_res_neck" : 6,         # Number of (R) blocks in the BN CHANGE, r
    })
    training_samples = training_properties["training_samples"] = 512
    example = WaveEquation(model_architecture_, device, batch_size, training_samples)
    ood_example = WaveEquation(model_architecture_, device, batch_size, training_samples, in_dist=False)
elif which_example == "allen":
    training_properties.update({
        "weight_decay": 1e-6, # CHANGE
    })
    model_architecture_.update({
        #Parameters to be chosen with model selection:
        "N_layers": 3,            # Number of (D) & (U) blocks CHANGE M
        "channel_multiplier": 48, # Parameter d_e (how the number of channels changes) CHANGE
        "N_res": 4,               # Number of (R) blocks in the middle networs. CHANGE
        "N_res_neck" : 8,         # Number of (R) blocks in the BN CHANGE, r
    })
    training_samples = training_properties["training_samples"] = 256
    example = AllenCahn(model_architecture_, device, batch_size, training_samples)
    ood_example = AllenCahn(model_architecture_, device, batch_size, training_samples, in_dist=False)
elif which_example == "cont_tran":
    training_properties.update({
        "weight_decay": 1e-6, # CHANGE
    })
    model_architecture_.update({
        #Parameters to be chosen with model selection:
        "N_layers": 3,            # Number of (D) & (U) blocks CHANGE M
        "channel_multiplier": 32, # Parameter d_e (how the number of channels changes) CHANGE
        "N_res": 2,               # Number of (R) blocks in the middle networs. CHANGE
        "N_res_neck" : 6,         # Number of (R) blocks in the BN CHANGE, r
    })
    training_samples = training_properties["training_samples"] = 512
    example = ContTranslation(model_architecture_, device, batch_size, training_samples)
    ood_example = ContTranslation(model_architecture_, device, batch_size, training_samples, in_dist=False)
elif which_example == "disc_tran":
    training_properties.update({
        "weight_decay": 1e-6, # CHANGE
    })
    model_architecture_.update({
        #Parameters to be chosen with model selection:
        "N_layers": 3,            # Number of (D) & (U) blocks CHANGE M
        "channel_multiplier": 32, # Parameter d_e (how the number of channels changes) CHANGE
        "N_res": 5,               # Number of (R) blocks in the middle networs. CHANGE
        "N_res_neck" : 4,         # Number of (R) blocks in the BN CHANGE, r
    })
    training_samples = training_properties["training_samples"] = 512
    example = DiscContTranslation(model_architecture_, device, batch_size, training_samples)
    ood_example = DiscContTranslation(model_architecture_, device, batch_size, training_samples, in_dist=False)
elif which_example == "airfoil":
    training_properties.update({
        "weight_decay": 1e-10, # CHANGE
    })
    model_architecture_.update({
        #Parameters to be chosen with model selection:
        "N_layers": 4,            # Number of (D) & (U) blocks CHANGE M
        "channel_multiplier": 48, # Parameter d_e (how the number of channels changes) CHANGE
        "N_res": 1,               # Number of (R) blocks in the middle networs. CHANGE
        "N_res_neck" : 8,         # Number of (R) blocks in the BN CHANGE, r
    })
    training_samples = training_properties["training_samples"] = 750
    model_architecture_["in_size"] = 128
    example = Airfoil(model_architecture_, device, batch_size, training_samples)
    ood_example = Airfoil(model_architecture_, device, batch_size, training_samples, in_dist=False)
elif which_example == "darcy":
    training_properties.update({
        "weight_decay": 1e-6, # CHANGE
    })
    model_architecture_.update({
        #Parameters to be chosen with model selection:
        "N_layers": 3,            # Number of (D) & (U) blocks CHANGE M
        "channel_multiplier": 48, # Parameter d_e (how the number of channels changes) CHANGE
        "N_res": 4,               # Number of (R) blocks in the middle networs. CHANGE
        "N_res_neck" : 4,         # Number of (R) blocks in the BN CHANGE, r
    })
    training_samples = training_properties["training_samples"] = 256
    example = Darcy(model_architecture_, device, batch_size, training_samples)
    ood_example = Darcy(model_architecture_, device, batch_size, training_samples, in_dist=False)
else:
    raise ValueError()
# NOTE load some training hyperparameters after setting for individual examples
learning_rate = training_properties["learning_rate"]
epochs = training_properties["epochs"]

weight_decay = training_properties["weight_decay"]
scheduler_step = training_properties["scheduler_step"]
scheduler_gamma = training_properties["scheduler_gamma"]
# training_samples = training_properties["training_samples"]
p = training_properties["exp"]
# NOTE save train and architecture config after setting.
df = pd.DataFrame.from_dict([training_properties]).T
df.to_csv(folder + '/training_properties.txt', header=False, index=True, mode='w')
df = pd.DataFrame.from_dict([model_architecture_]).T
df.to_csv(folder + '/net_architecture.txt', header=False, index=True, mode='w')
    
#-----------------------------------Train--------------------------------------
model = example.model
n_params = model.print_size()
train_loader = example.train_loader #TRAIN LOADER
val_loader = example.val_loader #VALIDATION LOADER
# NOTE add ood loader
test_loader = example.test_loader
ood_test_loader = ood_example.test_loader

#####################
# NOTE in dist and ood test trunk



####################

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
freq_print = 1

if p == 1:
    loss = torch.nn.L1Loss()
elif p == 2:
    loss = torch.nn.MSELoss()
    
best_model_testing_error = 1000 #Save the model once it has less than 1000% relative L1 error
# patience = int(0.2 * epochs)    # Early stopping parameter
# NOTE : set patience to inf to stop early stopping
patience = np.inf

counter = 0

if str(device) == 'cpu':
    print("------------------------------------------")
    print("YOU ARE RUNNING THE CODE ON A CPU.")
    print("WE SUGGEST YOU TO RUN THE CODE ON A GPU!")
    print("------------------------------------------")
    print(" ")

print(which_example)
train_error= []
test_error = []
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

            train_error.append(train_relative_l2)
            test_error.append(test_relative_l2)

            np.save(folder + '/train_error.npy', np.array(train_error))
            np.save(folder + '/test_error.npy', np.array(test_error))
            
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

################

# NOTE add test and ood error, median

with torch.no_grad():
    model.eval()
    final_test_relative_l2 = []
    
    for step, (input_batch, output_batch) in enumerate(test_loader):
        
        input_batch = input_batch.to(device)
        output_batch = output_batch.to(device)
        output_pred_batch = model(input_batch)
        
        if which_example == "airfoil": #Mask the airfoil shape
            output_pred_batch[input_batch==1] = 1
            output_batch[input_batch==1] = 1
        
        loss_f = torch.mean(abs(output_pred_batch - output_batch), axis = [1,2,3]) / torch.mean(abs(output_batch), axis = [1,2,3]) * 100
        final_test_relative_l2.append(loss_f.detach().cpu().numpy())
    final_test_relative_l2 = np.concatenate(final_test_relative_l2, 0)

    ood_test_relative_l2 = []
    
    for step, (input_batch, output_batch) in enumerate(ood_test_loader):
        
        input_batch = input_batch.to(device)
        output_batch = output_batch.to(device)
        output_pred_batch = model(input_batch)
        
        if which_example == "airfoil": #Mask the airfoil shape
            output_pred_batch[input_batch==1] = 1
            output_batch[input_batch==1] = 1
        
        loss_f = torch.mean(abs(output_pred_batch - output_batch), axis = [1,2,3]) / torch.mean(abs(output_batch), axis = [1,2,3]) * 100
        ood_test_relative_l2.append(loss_f.detach().cpu().numpy())
    ood_test_relative_l2 = np.concatenate(ood_test_relative_l2, 0)

print(np.median(final_test_relative_l2).item(), final_test_relative_l2.shape)
print(np.median(ood_test_relative_l2).item(), ood_test_relative_l2.shape)
# NOTE save files.
np.save(folder + '/final_test_relative_l2.npy', final_test_relative_l2)
np.save(folder + '/ood_test_relative_l2.npy', ood_test_relative_l2)

with open(folder + '/errors.txt', 'a') as file:
    file.write("Final Median Testing Error: " + str(np.median(final_test_relative_l2).item()) + "\n")
    file.write("OOD Median Testing Error: " + str(np.median(ood_test_relative_l2).item()) + "\n")