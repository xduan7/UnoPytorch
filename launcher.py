""" 
    File Name:          UnoPytorch/launcher.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               8/17/18
    Python Version:     3.6.6
    File Description:   

"""

import sys
import runpy
import datetime

from utils.miscellaneous.tee import Tee

if __name__ == '__main__':

    param_dict_list = [

        # {'trn_src': ['GDSC'],
        #  'val_src': ['GDSC'], },

        {'trn_src': ['NCI60'],
         'val_src': ['NCI60', 'CTRP', 'GDSC', 'CCLE', 'gCSI'], },

        {'trn_src': ['CTRP'],
         'val_src': ['NCI60', 'CTRP', 'GDSC', 'CCLE', 'gCSI'], },

        {'trn_src': ['GDSC'],
         'val_src': ['NCI60', 'CTRP', 'GDSC', 'CCLE', 'gCSI'], },

        {'trn_src': ['CCLE'],
         'val_src': ['NCI60', 'CTRP', 'GDSC', 'CCLE', 'gCSI'], },

        {'trn_src': ['gCSI'],
         'val_src': ['NCI60', 'CTRP', 'GDSC', 'CCLE', 'gCSI'], },
    ]

    for param_dict in param_dict_list:

        # Try until the models are successfully trained

        # module_finished = False
        # while not module_finished:

        now = datetime.datetime.now()

        # Save log with name = (training data source + time)
        tee = Tee('./results/logs/%s_(%02d%02d_%02d%02d).txt'
                  % (param_dict['trn_src'],
                     now.month, now.day, now.hour, now.minute))
        sys.stdout = tee

        sys.argv = [
            'uno_pytorch',

            '--training_src', *param_dict['trn_src'],
            '--validation_src', *param_dict['val_src'],
            '--precision', 'full',

            '--growth_scaling', 'none',
            '--descriptor_scaling', 'std',
            '--rnaseq_scaling', 'std',
            '--nan_threshold', '0.0',

            '--rnaseq_feature_usage', 'combat',
            '--drug_feature_usage', 'both',
            '--validation_size', '0.15',
            # '--disjoint_drugs',
            '--disjoint_cells',
            
            '--autoencoder_init',

            '--gene_layer_dim', '1024',
            '--gene_latent_dim', '256',
            '--gene_num_layers', '2',

            '--drug_layer_dim', '4096',
            '--drug_latent_dim', '1024',
            '--drug_num_layers', '2',

            '--resp_layer_dim', '1024',
            '--resp_num_layers', '2',
            '--resp_dropout', '0.0',
            '--resp_num_blocks', '3',
            '--resp_activation', 'none',

            '--loss_func', 'mse',
            '--optimizer', 'Adam',
            '--resp_lr', '2e-5',
            '--decay_factor', '0.95',
            '--early_stop_patience', '10',
            '--training_batch_size', '32',
            '--validation_batch_size', '256',
            '--max_num_batches', '1000',
            '--max_num_epochs', '100',
            '--log_interval', '10',
        ]

        runpy.run_module('uno_pytorch')

            # module_finished = True
        # except Exception as e:
        #     print('Encountering Exception %s' % e)
        #     print('Re-initiate a new run ...')

        sys.stdout = tee.default_stdout()
