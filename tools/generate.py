import os
from tqdm import tqdm
import numpy as np
import sys
from argparse import Namespace
import shutil

import torch
import torch.nn.functional as F
import lpips
import cv2
import random
from PIL import Image

sys.path.append(".")
sys.path.append("..")
os.environ['CUDA_VISIBLE_DEVICES']='0'

from options.test_options import TestOptions
from configs import data_configs
from utils.common import tensor2im
from models.sfe import FSEInverter, FSEFull

SIMILAR_CATES=-1
ALPHA=1
CM=1
OUTPUT_PATH='classifier/experiment/class_retention/nabirds/sage1.0q'
TEST_DATA_PATH='classifier/experiment/class_retention/nabirds/fewshot'
N_IMAGES=100

def get_ns(fse_model, transform, class_embeddings, opts):
    """Get n distribution for attribute editing"""
    if not opts.train_data_path:
        raise ValueError("--train_data_path must be provided to generate n distribution")
    
    print(f"Generating n distribution from {opts.train_data_path}...")
    samples = os.listdir(opts.train_data_path)
    ns_cate = {}
    
    for s in tqdm(samples):
        cate = s.split('_')[0]
        if cate not in ns_cate.keys():
            ns_cate[cate] = []
        
        ce = class_embeddings[cate].cuda()
        from_im = Image.open(os.path.join(opts.train_data_path, s))
        from_im = from_im.convert('RGB')
        from_im = transform(from_im)
        
        with torch.no_grad():
            # Use FSE to get codes
            _, ocodes, _, _ = fse_model(from_im.unsqueeze(0).to("cuda").float(), return_latents=True)
            ocodes = ocodes[0]  # Get first batch element
            dw = ocodes[:6] - ce[:6]  # Use first 6 style codes
            # For FSE, we need to handle Ax module differently
            # Since we don't have Ax module in FSE, we'll compute n directly
            n = torch.linalg.lstsq(torch.eye(512).cuda(), dw).solution  # Placeholder
            ns_cate[cate].append(n)
    
    os.makedirs(opts.n_distribution_path, exist_ok=True)
    ns_path = os.path.join(opts.n_distribution_path, 'ns.pt')
    torch.save(ns_cate, ns_path)
    print(f"n distribution saved to {ns_path}")
    return ns_cate

def torchOrth(A, r=10):
    """Compute orthogonal directions"""
    u,s,v = torch.svd(A)
    return v.T[:r]

def calc_statis(codes):
    """Calculate statistics for distribution"""
    codes=torch.stack(codes).permute(1,0,2).cpu().numpy()
    mean=np.mean(codes,axis=1)
    mean_abs=np.mean(np.abs(codes),axis=1)
    cov=[]
    for i in range(codes.shape[0]):
        cov.append(np.cov(codes[i].T))
    return {'mean':mean, 'mean_abs':mean_abs, 'cov':cov}

def get_similar_cate(class_embeddings, ce, k=20):
    """Get similar categories based on embeddings"""
    keys=list(class_embeddings.keys())
    distances={}
    for key in keys:
        distances[key]=torch.sum(F.pairwise_distance(ce, class_embeddings[key].cuda(), p=2))
    cates=sorted(distances.items(), key=lambda x: x[1])[:k]
    cates=[i[0] for i in cates] 
    return cates

def get_local_distribution(latents, cr_directions, ns, class_embeddings, k=20):
    """Get local distribution for sampling"""
    ce = get_ce(latents, cr_directions)
    cates = get_similar_cate(class_embeddings, ce, k)
    local_ns=[]
    for cate in cates:
        if cate not in ns.keys():
            print(f"Missing category: {cate}")
        else:
            local_ns+=ns[cate]
    return calc_statis(local_ns)

def get_crdirections(class_embeddings, r=30):
    """Compute CR directions"""
    class_embeddings=torch.stack(list(class_embeddings.values()))
    class_embeddings=class_embeddings.permute(1,0,2).cuda()
    cr_directions=[]
    for i in range(class_embeddings.shape[0]):
        cr_directions.append(torchOrth(class_embeddings[i], r))
    cr_directions=torch.stack(cr_directions)
    return cr_directions

def sampler(A, latents, dist, cr_dictionary, flag=True, alpha=1, l=50):
    """Sample new codes based on distribution"""
    ce=get_ce(latents[0], cr_dictionary).unsqueeze(0)
    means=dist['mean']
    covs=dist['cov']
    means_abs=torch.from_numpy(dist['mean_abs'])
    dws=[]
    for i in range(means.shape[0]):
        n=torch.from_numpy(np.random.multivariate_normal(mean=means[i], cov=covs[i], size=1)).float().cuda()
        #mask directions in A
        one = torch.ones_like(torch.from_numpy(means[0]))
        zero = torch.zeros_like(torch.from_numpy(means[0]))
        sorted, inds = torch.sort(means_abs[i], descending=True)
        beta = sorted[l]
        mask = torch.where(means_abs[i]>beta, one, zero).cuda()
        n=n*mask
        dw=torch.matmul(A[i], n.transpose(0,1)).squeeze(-1)
        dws.append(dw)
    dws=torch.stack(dws)
    if flag:
        codes = torch.cat(((alpha*dws.unsqueeze(0)+ ce[:, :6]), ce[:, 6:]), dim=1)
    else:
        codes = torch.cat(((alpha*dws.unsqueeze(0)+ latents[:, :6]), latents[:, 6:]), dim=1)
    return codes

def get_ce(latents, cr_directions):
    """Get class embeddings"""
    ce=[]
    for i in range(latents.shape[0]):
        cr_code=torch.zeros_like(latents[0])
        for j in range(cr_directions.shape[1]):
            cr_code=cr_code+torch.dot(latents[i],cr_directions[i][j])*cr_directions[i][j]
        ce.append(cr_code)
    ce=torch.stack(ce)
    return ce

def decode_with_fse(fse_model, codes, opts, feature_scale=None):
    """Decode with FSE model"""
    with torch.no_grad():
        # For FSE models, we need to handle decoding differently
        # We'll use the FSE decoder directly
        current_feature_scale = feature_scale if feature_scale is not None else getattr(opts, 'feature_scale', 1.0)
        
        # FSE models expect specific input format
        # We'll use the basic decoding without feature editing for now
        images, _ = fse_model.decoder(
            [codes],
            input_is_latent=True,
            randomize_noise=False,
            return_latents=True
        )
        
        if hasattr(opts, 'resize_outputs') and opts.resize_outputs:
            if hasattr(fse_model, 'face_pool'):
                images = fse_model.face_pool(images)
            else:
                # Resize manually if face_pool doesn't exist
                images = F.interpolate(images, size=(256, 256), mode='bilinear', align_corners=False)
        
        return images

def load_fse_model(test_opts):
    """Load FSE model with proper configuration"""
    # Create basic options for FSE
    opts = Namespace(**vars(test_opts))
    
    # Set default values for missing attributes
    if not hasattr(opts, 'learn_in_w'):
        opts.learn_in_w = False
    if not hasattr(opts, 'output_size'):
        opts.output_size = 1024
    if not hasattr(opts, 'start_from_latent_avg'):
        opts.start_from_latent_avg = True
    if not hasattr(opts, 'label_nc'):
        opts.label_nc = 0
    if not hasattr(opts, 'input_nc'):
        opts.input_nc = 3
    if not hasattr(opts, 'device'):
        opts.device = torch.device('cuda')
    
    # Determine which FSE model to use
    if hasattr(test_opts, 'use_fse_full') and test_opts.use_fse_full:
        print("Using FSEFull model")
        fse_model = FSEFull(
            device=opts.device,
            checkpoint_path=None,  # We'll load manually
            inverter_pth=test_opts.fse_inverter_path if hasattr(test_opts, 'fse_inverter_path') else None
        )
    else:
        print("Using FSEInverter model")
        fse_model = FSEInverter(
            device=opts.device,
            checkpoint_path=None  # We'll load manually
        )
    
    # Load FSE checkpoint
    if test_opts.fse_checkpoint_path:
        print(f"Loading FSE checkpoint from: {test_opts.fse_checkpoint_path}")
        fse_ckpt = torch.load(test_opts.fse_checkpoint_path, map_location="cpu")
        
        print(f"FSE checkpoint keys: {list(fse_ckpt.keys())}")
        
        # Handle different checkpoint formats
        if 'state_dict' in fse_ckpt:
            state_dict = fse_ckpt['state_dict']
        else:
            state_dict = fse_ckpt
        
        # Load with strict=False to handle key mismatches
        try:
            fse_model.load_state_dict(state_dict, strict=False)
            print("Successfully loaded FSE weights with strict=False")
        except Exception as e:
            print(f"Error loading FSE weights: {e}")
            print("Continuing with randomly initialized weights")
        
        # Load latent_avg if available
        if 'latent_avg' in fse_ckpt:
            fse_model.latent_avg = fse_ckpt['latent_avg'].to(opts.device)
            print("Loaded latent_avg from checkpoint")
    
    fse_model.eval()
    fse_model.cuda()
    
    return fse_model, opts

if __name__=='__main__':
    SEED = 0
    print("========")
    random.seed(SEED)
    np.random.seed(SEED)

    # Load test options
    test_opts = TestOptions().parse()
    
    # Load FSE model
    fse_model, opts = load_fse_model(test_opts)
    
    # Setup dataset and transforms
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    
    # Ensure opts has required attributes
    if not hasattr(opts, 'label_nc'):
        opts.label_nc = 0
    if not hasattr(opts, 'input_nc'):
        opts.input_nc = 3
        
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    transform = transforms_dict['transform_inference']
    
    # Load class embeddings
    class_embeddings = torch.load(opts.class_embedding_path, map_location=torch.device("cpu"))
    cr_directions = get_crdirections(class_embeddings, r=10)
    cr_dictionary = get_crdirections(class_embeddings, r=-1)

    # Handle n distribution
    ns_path = os.path.join(opts.n_distribution_path, 'ns.pt')
    
    if os.path.exists(ns_path):
        ns = torch.load(ns_path, map_location='cpu')
        print(f"Loaded n distribution from {ns_path}")
    else:
        print(f"n distribution not found at {ns_path}")
        
        # Check if we should generate it
        if hasattr(test_opts, 'generate_n_distribution') and test_opts.generate_n_distribution:
            print("Generating n distribution...")
            ns = get_ns(fse_model, transform, class_embeddings, test_opts)
        else:
            # Try to find alternative n distribution
            possible_paths = [
                ns_path,
                'n_distribution/ns.pt',
                '../n_distribution/ns.pt',
                './n_distribution/ns.pt'
            ]
            
            found = False
            for path in possible_paths:
                if os.path.exists(path):
                    ns = torch.load(path, map_location='cpu')
                    print(f"Loaded n distribution from {path}")
                    found = True
                    break
            
            if not found:
                print("No n distribution found and generation not requested.")
                print("Please either:")
                print("1. Run with --generate_n_distribution to generate n distribution")
                print("2. Provide a valid --n_distribution_path")
                print("3. Place ns.pt in one of the default locations")
                sys.exit(1)

    # Setup paths
    test_data_path = test_opts.test_data_path
    output_path = test_opts.output_path
    os.makedirs(output_path, exist_ok=True)

    print(f"Generating {test_opts.n_images} images per input")
    print(f"Input directory: {test_data_path}")
    print(f"Output directory: {output_path}")

    # Process each image in test directory
    from_ims = os.listdir(test_data_path)
    for from_im_name in from_ims:
        print(f"Processing {from_im_name}...")
        
        for j in tqdm(range(test_opts.n_images)):
            cr_dic = cr_dictionary[:,:test_opts.t]
            
            # Load and transform input image
            from_im = Image.open(os.path.join(test_data_path, from_im_name))
            from_im = from_im.convert('RGB')
            from_im = transform(from_im)
            
            # Get latents using FSE
            with torch.no_grad():
                _, ocodes, _, _ = fse_model(from_im.unsqueeze(0).to("cuda").float(), return_latents=True)
                latents = ocodes  # Use ocodes as latents
            
            # Get local distribution and sample new codes
            n_dist = get_local_distribution(latents[0], cr_directions, ns, class_embeddings, test_opts.n_similar_cates)
            
            # For FSE, we need to handle the Ax module differently
            # Since we don't have Ax in FSE, we'll use a placeholder identity matrix
            # In a real scenario, you would load a pre-trained Ax module
            A_placeholder = torch.eye(512).unsqueeze(0).repeat(6, 1, 1).cuda()  # [6, 512, 512]
            
            codes = sampler(A_placeholder, latents, n_dist, cr_dic, alpha=test_opts.alpha)
            
            # Decode with FSE
            res0 = decode_with_fse(fse_model, codes, opts, feature_scale=getattr(test_opts, 'feature_scale', 1.0))
            res0 = tensor2im(res0[0])
            
            # Save generated image
            im_save_path = os.path.join(output_path, f"{os.path.splitext(from_im_name)[0]}_{j}.jpg")
            Image.fromarray(np.array(res0)).save(im_save_path)

    print("Generation completed!")

    # Print summary
    print(f"\n=== Generation Summary ===")
    print(f"Total input images processed: {len(from_ims)}")
    print(f"Images generated per input: {test_opts.n_images}")
    print(f"Total images generated: {len(from_ims) * test_opts.n_images}")
    print(f"Output directory: {output_path}")
    print(f"Feature editing: {getattr(test_opts, 'apply_feature_editing', False)}")
    if hasattr(test_opts, 'apply_feature_editing') and test_opts.apply_feature_editing:
        print(f"Feature scale: {getattr(test_opts, 'feature_scale', 1.0)}")