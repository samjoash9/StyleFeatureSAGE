import os
from argparse import Namespace

from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
import sys
import json

sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from utils.common import tensor2im, log_input_image
from options.test_options import TestOptions

def run():
    test_opts = TestOptions().parse()
    
    # Determine which model to use based on provided checkpoints
    if test_opts.fse_checkpoint_path is not None:
        # Use FSE-based model
        from models.age_with_fse import AGEWithFSE
        
        if test_opts.checkpoint_path is not None:
            # Load AGE model that uses FSE as backbone
            ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
            opts = ckpt['opts']
            opts.update(vars(test_opts))
            # Set default values for missing attributes
            if 'learn_in_w' not in opts:
                opts['learn_in_w'] = False
            if 'output_size' not in opts:
                opts['output_size'] = 1024
            if 'start_from_latent_avg' not in opts:
                opts['start_from_latent_avg'] = True
            if 'label_nc' not in opts:
                opts['label_nc'] = 0  # Add missing attribute
            if 'input_nc' not in opts:
                opts['input_nc'] = 3  # Add missing attribute
            opts = Namespace(**opts)
            opts.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            net = AGEWithFSE(opts)
            # Load the AGE weights (Ax module)
            net.load_weights()
        else:
            # Create FSE model directly
            opts = Namespace(**vars(test_opts))
            # Set default values for missing attributes
            if not hasattr(opts, 'learn_in_w'):
                opts.learn_in_w = False
            if not hasattr(opts, 'output_size'):
                opts.output_size = 1024
            if not hasattr(opts, 'start_from_latent_avg'):
                opts.start_from_latent_avg = True
            if not hasattr(opts, 'label_nc'):
                opts.label_nc = 0  # Add missing attribute
            if not hasattr(opts, 'input_nc'):
                opts.input_nc = 3  # Add missing attribute
            opts.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
            net = AGEWithFSE(opts)
            
    else:
        # Use original AGE model (backward compatibility)
        from models.age import AGE
        
        if test_opts.checkpoint_path is None and test_opts.psp_checkpoint_path is None:
            raise ValueError("Either --checkpoint_path or --psp_checkpoint_path must be provided")
            
        checkpoint_to_load = test_opts.checkpoint_path if test_opts.checkpoint_path is not None else test_opts.psp_checkpoint_path
        ckpt = torch.load(checkpoint_to_load, map_location='cpu')
        opts = ckpt['opts']
        opts.update(vars(test_opts))
        # Set default values for missing attributes
        if 'learn_in_w' not in opts:
            opts['learn_in_w'] = False
        if 'output_size' not in opts:
            opts['output_size'] = 1024
        if 'label_nc' not in opts:
            opts['label_nc'] = 0  # Add missing attribute
        if 'input_nc' not in opts:
            opts['input_nc'] = 3  # Add missing attribute
        opts = Namespace(**opts)

        net = AGE(opts)

    net.eval()
    net.cuda()

    print('Loading dataset for {}'.format(opts.dataset_type))
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    
    # Ensure opts has all required attributes for transforms
    if not hasattr(opts, 'label_nc'):
        opts.label_nc = 0
    if not hasattr(opts, 'input_nc'):
        opts.input_nc = 3
        
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    data_path = test_opts.train_data_path
    class_embedding_path = test_opts.class_embedding_path
    
    if data_path is None:
        raise ValueError("--train_data_path must be provided for computing class embeddings")
    if class_embedding_path is None:
        raise ValueError("--class_embedding_path must be provided for saving class embeddings")
        
    os.makedirs(class_embedding_path, exist_ok=True)
    
    # Ensure the dataset gets the opts with all required attributes
    dataset = InferenceDataset(root=data_path,
                            transform=transforms_dict['transform_inference'],
                            opts=opts)
    dataloader = DataLoader(dataset,
                            batch_size=opts.test_batch_size,
                            shuffle=False,
                            num_workers=int(opts.test_workers),
                            drop_last=True)

    codes = {}
    counts = {}
    
    print("Computing class embeddings...")
    for input_batch, cate_batch in tqdm(dataloader):
        with torch.no_grad():
            input_batch = input_batch.cuda()
            for image_idx, input in enumerate(input_batch):
                input_image = input
                cate = cate_batch[image_idx]
                
                # Use get_test_code which works for both original AGE and AGEWithFSE
                outputs = net.get_test_code(input_image.unsqueeze(0).float())
                
                # save codes
                if cate not in codes.keys():
                    codes[cate] = outputs['ocodes'][0].detach().clone()
                    counts[cate] = 1
                else:
                    codes[cate] += outputs['ocodes'][0].detach().clone()
                    counts[cate] += 1
    
    # Compute means
    means = {}
    for cate in codes.keys():
        means[cate] = codes[cate] / counts[cate]
    
    # Save embeddings
    output_path = os.path.join(class_embedding_path, 'class_embeddings.pt')
    torch.save(means, output_path)
    print(f"Class embeddings saved to {output_path}")
    
    # Also save as JSON for readability (optional)
    try:
        means_json = {}
        for k, v in means.items():
            # Convert tensor to list, handling both CPU and GPU tensors
            if isinstance(v, torch.Tensor):
                v_cpu = v.cpu()
                means_json[k] = v_cpu.numpy().tolist() if v_cpu.numel() > 1 else float(v_cpu.item())
            else:
                means_json[k] = v
                
        json_path = os.path.join(class_embedding_path, 'class_embeddings.json')
        with open(json_path, 'w') as f:
            json.dump(means_json, f, indent=2)
        print(f"Class embeddings (JSON) saved to {json_path}")
    except Exception as e:
        print(f"Could not save JSON file: {e}")

if __name__ == '__main__':
    run()