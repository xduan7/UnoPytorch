# UnoPytorch
Multi-tasking (drug response, cell line classification, etc.) Uno Implemented in PyTorch.

## Todos
* More labels for the network like drug labels;
* Dataloader hanging problem when num_workers set to more than 0;
* Better pre-processing for drug descriptor integer features;
* Network regularization with weight decay and/or dropout;
* Hyper-parameter searching; 

## Prerequisites
```
Python 3.6.4
PyTorch 0.4.1
SciPy 1.1.0
pandas 0.23.4
Scikit-Learn 0.19.1
urllib3 1.23
```


The default network structure is shown below: 
<img src="https://github.com/xduan7/UnoPytorch/blob/master/images/default_network.jpg" width="100%">

An example of the program output for training on NCI60 and valdiation on all other data sources is shown below:
```
python3.6 ~/Projects/UnoPytorch/launcher.py
Training Arguments:
{
    "trn_src": "NCI60",
    "val_srcs": [
        "NCI60",
        "CTRP",
        "GDSC",
        "CCLE",
        "gCSI"
    ],
    "growth_scaling": "none",
    "descriptor_scaling": "std",
    "rnaseq_scaling": "std",
    "nan_threshold": 0.0,
    "rnaseq_feature_usage": "source_scale",
    "drug_feature_usage": "both",
    "validation_size": 0.15,
    "disjoint_drugs": false,
    "disjoint_cells": true,
    "gene_layer_dim": 1024,
    "gene_latent_dim": 256,
    "gene_num_layers": 2,
    "drug_layer_dim": 4096,
    "drug_latent_dim": 1024,
    "drug_num_layers": 2,
    "ae_init": true,
    "resp_layer_dim": 1024,
    "resp_num_layers": 2,
    "resp_dropout": 0.0,
    "resp_num_blocks": 4,
    "resp_activation": "none",
    "clf_layer_dim": 32,
    "clf_num_layers": 1,
    "resp_loss_func": "mse",
    "resp_opt": "SGD",
    "resp_lr": 1e-05,
    "resp_val_start_epoch": 0,
    "early_stop_patience": 10,
    "clf_opt": "SGD",
    "clf_lr": 0.0,
    "decay_factor": 0.95,
    "trn_batch_size": 32,
    "val_batch_size": 256,
    "max_num_batches": 1000,
    "max_num_epochs": 100,
    "precision": "full",
    "multi_gpu": false,
    "no_cuda": false,
    "rand_state": 0
}
================================================================================
Training Epoch   1:
	Drug Response Regression Loss:  1759.77

Validation Results:
	Cell Line Classification: 
		Category Accuracy: 		18.82%; 
		Site Accuracy: 			10.00%; 
		Type Accuracy: 			 1.27%
	Drug Response Regression:
		NCI60  	 MSE:  1298.50 	 MAE:    25.75 	 R2: +0.60
		CTRP   	 MSE:  2588.82 	 MAE:    37.37 	 R2: +0.28
		GDSC   	 MSE:  2719.53 	 MAE:    37.82 	 R2: +0.18
		CCLE   	 MSE:  2628.39 	 MAE:    38.04 	 R2: +0.37
		gCSI   	 MSE:  2485.53 	 MAE:    36.61 	 R2: +0.41
Epoch Running Time: 104.8 Seconds.
================================================================================
Training Epoch   2:
    ...
```

For default hyper parameters, the transfer learning matrix results are shown below:
<img src="https://github.com/xduan7/UnoPytorch/blob/master/images/default_results.jpg" width="100%">

Note that the green cells represents R2 score of higher than 0.1, red cells are R2 scores lower than -0.1 and yellows are for all the values in between. 
