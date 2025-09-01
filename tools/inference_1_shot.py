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
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from options.test_options import TestOptions
from configs import data_configs
from utils.common import tensor2im
from models.age import AGE

SIMILAR_CATES = -1
ALPHA = 1
CM = 1
OUTPUT_PATH = 'classifier/experiment/class_retention/nabirds/sage1.0q'
TEST_DATA_PATH = 'classifier/experiment/class_retention/nabirds/fewshot'
N_IMAGES = 100

def get_ns(net, transform, class_embeddings, opts):
    """
    Build ns.pt by iterating through training images, computing per-class n (coefficients)
    using least-squares with net.ax.A and saving dict {class: [n1, n2, ...]}.
    """
    samples = os.listdir(opts.train_data_path)
    ns_cate = {}
    device = next(net.parameters()).device
    for s in tqdm(samples):
        cate = s.split('_')[0]
        if cate not in ns_cate:
            ns_cate[cate] = []
        ce = class_embeddings[cate].to(device)
        from_im = Image.open(os.path.join(opts.train_data_path, s))
        from_im = from_im.convert('RGB')
        from_im = transform(from_im)
        with torch.no_grad():
            latents = net.encoder(from_im.unsqueeze(0).to(device).float())

            # keep the same indexing you used before (first 6 elements)
            dw = latents[0][:6] - ce[:6]
            # solve for n using least squares against A
            n = torch.linalg.lstsq(net.ax.A, dw).solution
            ns_cate[cate].append(n.cpu())  # store on cpu to make saved file lighter
    os.makedirs(opts.n_distribution_path, exist_ok=True)
    torch.save(ns_cate, os.path.join(opts.n_distribution_path, 'ns.pt'))
    print(f"Saved ns to {os.path.join(opts.n_distribution_path, 'ns.pt')} (classes: {len(ns_cate)})")

def torchOrth(A, r=10):
    # fallback for svd (works for CPU/GPU). Returns top-r orthonormal vectors (transposed)
    u, s, v = torch.svd(A)
    return v.T[:r]

def calc_statis(codes):
    codes = torch.stack(codes).permute(1, 0, 2).cpu().numpy()
    mean = np.mean(codes, axis=1)
    mean_abs = np.mean(np.abs(codes), axis=1)
    cov = []
    for i in range(codes.shape[0]):
        cov.append(np.cov(codes[i].T))
    return {'mean': mean, 'mean_abs': mean_abs, 'cov': cov}

def get_similar_cate(class_embeddings, ce, k=20):
    keys = list(class_embeddings.keys())
    distances = {}
    for key in keys:
        distances[key] = torch.sum(F.pairwise_distance(ce, class_embeddings[key].cuda(), p=2))
    cates = sorted(distances.items(), key=lambda x: x[1])[:k]
    cates = [i[0] for i in cates]
    return cates

def get_local_distribution(latents, cr_directions, ns, class_embeddings, k=20):
    ce = get_ce(latents, cr_directions)
    cates = get_similar_cate(class_embeddings, ce, k)
    local_ns = []
    for cate in cates:
        if cate not in ns:
            print(f"{cate} missed")
        else:
            # ns[cate] is a list of tensors on CPU (we saved them that way)
            local_ns += ns[cate]
    return calc_statis(local_ns)

def get_crdirections(class_embeddings, r=30):
    class_embeddings_stack = torch.stack(list(class_embeddings.values()))
    class_embeddings_stack = class_embeddings_stack.permute(1, 0, 2).cuda()
    cr_directions = []
    for i in range(class_embeddings_stack.shape[0]):
        cr_directions.append(torchOrth(class_embeddings_stack[i], r))
    cr_directions = torch.stack(cr_directions)
    return cr_directions

def sampler(A, latents, dist, cr_dictionary, flag=True, alpha=1, l=50):
    # latents is expected to be shape (1, n_layers, feat_dim) or (batch, n_layers, feat_dim)
    ce = get_ce(latents[0], cr_dictionary).unsqueeze(0)
    means = dist['mean']
    covs = dist['cov']
    means_abs = torch.from_numpy(dist['mean_abs'])
    device = next(A.parameters()).device if hasattr(A, 'parameters') else A.device if isinstance(A, torch.Tensor) else latents.device
    dws = []
    for i in range(means.shape[0]):
        n = torch.from_numpy(np.random.multivariate_normal(mean=means[i], cov=covs[i], size=1)).float().to(device)
        # mask directions in A based on mean magnitudes
        one = torch.ones_like(torch.from_numpy(means[0])).to(device)
        zero = torch.zeros_like(torch.from_numpy(means[0])).to(device)
        sorted_vals, inds = torch.sort(means_abs[i], descending=True)
        # protect index l
        if l >= sorted_vals.numel():
            beta = sorted_vals[-1]
        else:
            beta = sorted_vals[l]
        mask = torch.where(torch.from_numpy(dist['mean_abs'][i]).to(device) > beta, one, zero).to(device)
        n = n * mask
        # A may be a tensor of shape (num_dirs, feat_dim) or similar; ensure device
        A_i = A[i].to(device) if isinstance(A, torch.Tensor) else A[i]
        dw = torch.matmul(A_i, n.transpose(0, 1)).squeeze(-1)
        dws.append(dw)
    dws = torch.stack(dws).to(device)
    if flag:
        codes = torch.cat(((alpha * dws.unsqueeze(0) + ce[:, :6]), ce[:, 6:]), dim=1)
    else:
        codes = torch.cat(((alpha * dws.unsqueeze(0) + latents[:, :6]), latents[:, 6:]), dim=1)
    return codes

def get_ce(latents, cr_directions):
    ce = []
    for i in range(latents.shape[0]):
        cr_code = torch.zeros_like(latents[0])
        for j in range(cr_directions.shape[1]):
            cr_code = cr_code + torch.dot(latents[i], cr_directions[i][j]) * cr_directions[i][j]
        ce.append(cr_code)
    ce = torch.stack(ce)
    return ce

if __name__ == '__main__':
    SEED = 0
    print("========")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # load test options
    test_opts = TestOptions().parse()

    # load checkpoint and merge opts
    ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
    opts_dict = ckpt.get('opts', {})
    opts_dict.update(vars(test_opts))
    if 'learn_in_w' not in opts_dict:
        opts_dict['learn_in_w'] = False
    if 'output_size' not in opts_dict:
        opts_dict['output_size'] = 1024
    opts = Namespace(**opts_dict)

    # build net
    net = AGE(opts)
    net.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    transform = transforms_dict['transform_inference']

    class_embeddings = torch.load(opts.class_embedding_path, map_location=torch.device("cpu"))
    cr_directions = get_crdirections(class_embeddings, r=10)
    cr_dictionary = get_crdirections(class_embeddings, r=-1)

    # ensure n_distribution path exists and build ns.pt if not already present
    os.makedirs(opts.n_distribution_path, exist_ok=True)
    ns_path = os.path.join(opts.n_distribution_path, 'ns.pt')
    if not os.path.exists(ns_path):
        print("ns.pt not found — building distribution (this may take a while)...")
        # pass the merged opts so get_ns can access train_data_path etc.
        get_ns(net, transform, class_embeddings, opts)
    else:
        print(f"Found existing ns.pt at {ns_path} — skipping build.")

    # load ns and proceed to generate
    ns = torch.load(ns_path, map_location='cpu')

    test_data_path = test_opts.test_data_path
    output_path = test_opts.output_path
    os.makedirs(output_path, exist_ok=True)

    from_ims = os.listdir(test_data_path)
    for from_im_name in from_ims:
        for j in tqdm(range(test_opts.n_images)):
            cr_dic = cr_dictionary[:, :test_opts.t]
            from_im = Image.open(os.path.join(test_data_path, from_im_name))
            from_im = from_im.convert('RGB')
            from_im = transform(from_im)
            latents = net.encoder(from_im.unsqueeze(0).to(device).float())
            n_dist = get_local_distribution(latents[0], cr_directions, ns, class_embeddings, test_opts.n_similar_cates)
            codes = sampler(net.ax.A, latents, n_dist, cr_dic, alpha=test_opts.alpha)
            with torch.no_grad():
                res0 = net.decode(codes, randomize_noise=False, resize=opts.resize_outputs)
            res0 = tensor2im(res0[0])
            im_save_path = os.path.join(output_path, from_im_name + '_' + str(j) + '.jpg')
            Image.fromarray(np.array(res0)).save(im_save_path)

    print("Done.")
