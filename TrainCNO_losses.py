import copy
import json
import os
import sys

import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


from Problems.CNOBenchmarks import Darcy, Airfoil, DiscContTranslation, ContTranslation, AllenCahn, SinFrequency, WaveEquation, ShearLayer
from Problems.PDEArenaBenchmarks import StandardNavierStokes

import functools


def relative_error(pred, y, p, test, axis = (-2,-1)):
    # NOTE new relative l2 error defined here
    # NCHW, summing the over the HW dimension

    errors = (torch.abs(y - pred)**p).sum(axis)
    norms = (torch.abs(y) ** p).sum(axis)

    relative_errors = errors / norms
    if test:
        return torch.pow(relative_errors, 1/p).mean()
    else:
        return relative_errors.mean()

test_rel_p = 2

model_architecture_ = {
    
    #Parameters to be chosen with model selection:
    "N_layers": 3,            # Number of (D) & (U) blocks 
    "channel_multiplier": 32, # Parameter d_e (how the number of channels changes)
    "N_res": 4,               # Number of (R) blocks in the middle networs.
    "N_res_neck" : 6,         # Number of (R) blocks in the BN
    
    #Other parameters:
    "in_size": 64,            # Resolution of the computational grid
    "retrain": 4,             # Random seed
    "kernel_size": 3,         # Kernel size.
    "FourierF": 0,            # Number of Fourier Features in the input channels. Default is 0.
    "activation": 'cno_lrelu',# cno_lrelu or lrelu
    
    #Filter properties:
    "cutoff_den": 2.0001,     # Cutoff parameter.
    "lrelu_upsampling": 2,    # Coefficient N_{\sigma}. Default is 2.
    "half_width_mult": 0.8,   # Coefficient c_h. Default is 1
    "filter_size": 6,         # 2xfilter_size is the number of taps N_{tap}. Default is 6.
    "radial_filter": 0,       # Is the filter radially symmetric? Default is 0 - NO.
}


which_example = sys.argv[1]
loss_type = str(sys.argv[2])

if loss_type == "rel_l1":
    loss_function = functools.partial(relative_error, p = 1, test = False, axis = (-2, -1))
elif loss_type == "rel_l2":
    loss_function = functools.partial(relative_error, p = 2, test = False, axis = (-2, -1))
elif loss_type == "mse":
    loss_function = torch.nn.MSELoss()
elif loss_type == "cno_rel_l1":
    aux_fn = torch.nn.L1Loss()
    loss_function = lambda output_pred_batch, output_batch : aux_fn(output_pred_batch, output_batch) / aux_fn(torch.zeros_like(output_batch), output_batch)
else:
    raise NotImplementedError
print(loss_function)
# Save the models here:
folder = "results/" + loss_type + "_TrainedModels/"+"CNO_"+which_example+"_1"

training_properties = {
    "learning_rate": 0.001, 
    "weight_decay": 1e-6,
    "scheduler_step": 10,
    "scheduler_gamma": 0.98,
    "epochs": 1000,
    "batch_size": 16,
    # "exp": 1,                # Do we use L1 or L2 errors? Default: L1
    "loss_function": loss_type,
    "test_rel_p": 2
#    "training_samples": 256  # How many training samples?
}
        


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter(log_dir=folder) #usage of TensorBoard

learning_rate = training_properties["learning_rate"]
epochs = training_properties["epochs"]
batch_size = training_properties["batch_size"]
weight_decay = training_properties["weight_decay"]
scheduler_step = training_properties["scheduler_step"]
scheduler_gamma = training_properties["scheduler_gamma"]
# training_samples = training_properties["training_samples"]
# p = training_properties["exp"]





if which_example == "shear_layer":
    training_samples = training_properties["training_samples"] = 750
    example = ShearLayer(model_architecture_, device, batch_size, training_samples, size = 64)
elif which_example == "poisson":
    training_samples = training_properties["training_samples"] = 1024
    example = SinFrequency(model_architecture_, device, batch_size, training_samples)
elif which_example == "wave_0_5":
    training_samples = training_properties["training_samples"] = 512
    example = WaveEquation(model_architecture_, device, batch_size, training_samples)
elif which_example == "allen":
    training_samples = training_properties["training_samples"] = 256
    example = AllenCahn(model_architecture_, device, batch_size, training_samples)
elif which_example == "cont_tran":
    training_samples = training_properties["training_samples"] = 512
    example = ContTranslation(model_architecture_, device, batch_size, training_samples)
elif which_example == "disc_tran":
    training_samples = training_properties["training_samples"] = 512
    example = DiscContTranslation(model_architecture_, device, batch_size, training_samples)
elif which_example == "airfoil":
    training_samples = training_properties["training_samples"] = 750
    model_architecture_["in_size"] = 128
    example = Airfoil(model_architecture_, device, batch_size, training_samples)
elif which_example == "darcy":
    training_samples = training_properties["training_samples"] = 256
    example = Darcy(model_architecture_, device, batch_size, training_samples)
elif which_example == "ns":
    training_samples = training_properties["training_samples"] = 4096
    example = StandardNavierStokes(model_architecture_, device, batch_size, training_samples, size = None)
else:
    raise ValueError()

if not os.path.isdir(folder):
    print("Generated new folder")
    os.mkdir(folder)

df = pd.DataFrame.from_dict([training_properties]).T
df.to_csv(folder + '/training_properties.txt', header=False, index=True, mode='w')
df = pd.DataFrame.from_dict([model_architecture_]).T
df.to_csv(folder + '/net_architecture.txt', header=False, index=True, mode='w')
#-----------------------------------Train--------------------------------------
model = example.model
n_params = model.print_size()
train_loader = example.train_loader #TRAIN LOADER
val_loader = example.val_loader #VALIDATION LOADER

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
freq_print = 1

# if p == 1:
#     loss = torch.nn.L1Loss()
# elif p == 2:
#     loss = torch.nn.MSELoss()
    
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
            # print(output_pred_batch.shape)
            # raise EOFError
            # NOTE loss changed here
            # loss_f = loss(output_pred_batch, output_batch) / loss(torch.zeros_like(output_batch).to(device), output_batch)
            # loss_f = relative_error(output_pred_batch, output_batch, p = train_rel_p, test = False)
            loss_f = loss_function(output_pred_batch, output_batch)

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
                
                # loss_f = torch.mean(abs(output_pred_batch - output_batch)) / torch.mean(abs(output_batch)) * 100
                # NOTE error changed here
                loss_f = relative_error(output_pred_batch, output_batch, p = test_rel_p, test = True) * 100
                test_relative_l2 += loss_f.item()
            test_relative_l2 /= len(val_loader)

            for step, (input_batch, output_batch) in enumerate(train_loader):
                input_batch = input_batch.to(device)
                output_batch = output_batch.to(device)
                output_pred_batch = model(input_batch)
                    
                if which_example == "airfoil": #Mask the airfoil shape
                    output_pred_batch[input_batch==1] = 1
                    output_batch[input_batch==1] = 1

                    # loss_f = torch.mean(abs(output_pred_batch - output_batch)) / torch.mean(abs(output_batch)) * 100
                    # NOTE error changed here
                    loss_f = relative_error(output_pred_batch, output_batch, p = test_rel_p, test = True) * 100
                    train_relative_l2 += loss_f.item()
            train_relative_l2 /= len(train_loader)
            
            writer.add_scalar("train_loss/train_loss_rel", train_relative_l2, epoch)
            writer.add_scalar("val_loss/val_loss", test_relative_l2, epoch)

            if test_relative_l2 < best_model_testing_error:
                best_model_testing_error = test_relative_l2
                best_model = copy.deepcopy(model)
                torch.save(best_model, folder + "/model.pkl")
                writer.add_scalar("val_loss/Best Relative Testing Error", best_model_testing_error, epoch)
                counter = 0
            else:
                counter+=1

        tepoch.set_postfix({'Train loss': train_mse, "Relative Train": train_relative_l2, "Relative Val loss": test_relative_l2})
        tepoch.close()

        with open(folder + '/errors.txt', 'w') as file:
            file.write("Training Error: " + str(train_mse) + "\n")
            file.write("Best Testing Error: " + str(best_model_testing_error) + "\n")
            file.write("Current Epoch: " + str(epoch) + "\n")
            file.write("Params: " + str(n_params) + "\n")
            file.write("Loss : " + loss_type + "\n")
            file.write("Test : " + f'L_{test_rel_p}' + "\n")
        scheduler.step()

    if counter>patience:
        print("Early Stopping")
        break