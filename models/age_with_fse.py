"""
AGE Model with FSE Inverter and Feature Editing
"""
import math
import torch
from torch import nn
import torch.nn.functional as F
from models.stylegan2.model import Generator
from models.sfe import FSEInverter, FSEFull

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(EqualLinear, self).__init__()
        self.out_dim = out_dim
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(in_dim*3, in_dim*2)
        self.fc2 = nn.Linear(in_dim*2, in_dim*2)
        self.fc3 = nn.Linear(in_dim*2, in_dim)
        self.fc4 = nn.Linear(in_dim, out_dim)
        self.fc5 = nn.Linear(out_dim, out_dim)
        self.nonlinearity = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, input):
        out = self.flat(input)
        out = self.fc1(out)
        out = self.nonlinearity(out)
        out = self.fc2(out)
        out = self.nonlinearity(out)
        out = self.fc3(out)
        out = self.nonlinearity(out)
        out = self.fc4(out)
        out = self.nonlinearity(out)
        out = self.fc5(out)
        return out

class Ax(nn.Module):
    def __init__(self, dim):
        super(Ax, self).__init__()
        self.A = nn.Parameter(torch.randn(6, 512, dim), requires_grad=True)
        self.encoder0 = EqualLinear(512, dim)
        self.encoder1 = EqualLinear(512, dim)
    
    def forward(self, dw):
        x0 = self.encoder0(dw[:, :3])
        x0 = x0.unsqueeze(-1).unsqueeze(1)
        x1 = self.encoder1(dw[:, 3:6])
        x1 = x1.unsqueeze(-1).unsqueeze(1)
        x = [x0.squeeze(-1), x1.squeeze(-1)]
        
        output_dw0 = torch.matmul(self.A[:3], x0).squeeze(-1)
        output_dw1 = torch.matmul(self.A[3:6], x1).squeeze(-1)
        output_dw = torch.cat((output_dw0, output_dw1), dim=1)
        return output_dw, self.A, x

def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt

class AGEWithFSE(nn.Module):
    def __init__(self, opts):
        super(AGEWithFSE, self).__init__()
        self.set_opts(opts)
        self.opts.n_styles = int(math.log(self.opts.output_size, 2)) * 2 - 2
        
        # Initialize FSE Inverter
        if hasattr(opts, 'use_fse_full') and opts.use_fse_full:
            self.encoder = FSEFull(
                device=opts.device,
                checkpoint_path=opts.fse_checkpoint_path,
                inverter_pth=opts.fse_inverter_path if hasattr(opts, 'fse_inverter_path') else None
            )
        else:
            self.encoder = FSEInverter(
                device=opts.device,
                checkpoint_path=opts.fse_checkpoint_path
            )
        
        # Attribute factorization module
        self.ax = Ax(self.opts.A_length)
        
        # Use decoder from FSE
        self.decoder = self.encoder.decoder
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        
        # Freeze or unfreeze components based on training mode
        self._setup_training_mode()
        
        self.load_weights()

    def _setup_training_mode(self):
        """Setup which components are trainable"""
        if hasattr(self.opts, 'train_feature_editor') and self.opts.train_feature_editor:
            self._unfreeze_feature_editor()
        else:
            self._freeze_fse_encoder()
            
        # Always train Ax module
        for param in self.ax.parameters():
            param.requires_grad = True

    def _freeze_fse_encoder(self):
        """Freeze entire FSE encoder"""
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

    def _unfreeze_feature_editor(self):
        """Unfreeze only the feature editor part of FSE"""
        if hasattr(self.encoder, 'encoder'):  # FSEFull case
            # In FSEFull, the feature editor is the trainable part
            for param in self.encoder.encoder.parameters():
                param.requires_grad = True
            # Freeze the rest
            for name, param in self.encoder.named_parameters():
                if 'encoder' not in name:
                    param.requires_grad = False
        else:  # FSEInverter case
            # In FSEInverter, we might want to train the fuser
            for name, param in self.encoder.named_parameters():
                if 'fuser' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    def load_weights(self):
        if self.opts.checkpoint_path is not None:
            print('Loading AGE from checkpoint: {}'.format(self.opts.checkpoint_path))
            ckpt = torch.load(self.opts.checkpoint_path, map_location=torch.device('cpu'))
            
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in ckpt['state_dict'].items():
                name = k.replace('module.', '') 
                new_state_dict[name] = v
            ckpt['state_dict'] = new_state_dict
            
            # Load Ax module weights
            self.ax.load_state_dict(get_keys(ckpt, 'ax'), strict=True)
            self.__load_latent_avg(ckpt)
        else:
            print('Using FSEInverter with checkpoint: {}'.format(self.opts.fse_checkpoint_path))
            # Get latent average from FSE
            self.latent_avg = self.encoder.latent_avg

    def forward(self, x, av_codes, resize=True, latent_mask=None, input_code=False, 
                randomize_noise=True, inject_latent=None, return_latents=False, 
                alpha=None, feature_scale=None):
        
        if input_code:
            codes = x
            # For feature editing with input codes, we need to handle this differently
            if hasattr(self.opts, 'apply_feature_editing') and self.opts.apply_feature_editing:
                raise NotImplementedError("Feature editing with input codes not implemented")
        else:
            # Use FSE to get initial codes and features
            if self.encoder.training:
                # If feature editor is trainable, don't use torch.no_grad()
                if hasattr(self.encoder, 'encoder'):  # FSEFull
                    images, ocodes, fused_feat, predicted_feat = self.encoder(x, return_latents=True)
                else:  # FSEInverter
                    images, ocodes, fused_feat, w_feat = self.encoder(x, return_latents=True)
            else:
                # Frozen encoder - use no_grad for efficiency
                with torch.no_grad():
                    if hasattr(self.encoder, 'encoder'):  # FSEFull
                        images, ocodes, fused_feat, predicted_feat = self.encoder(x, return_latents=True)
                    else:  # FSEInverter
                        images, ocodes, fused_feat, w_feat = self.encoder(x, return_latents=True)
            
            # Apply attribute factorization
            odw = ocodes[:, :6] - av_codes[:, :6]
            dw, A, x_factors = self.ax(odw)
            
            # Reconstruct codes with edited attributes
            codes = torch.cat((dw + av_codes[:, :6], ocodes[:, 6:]), dim=1)

            # Normalize with latent average
            if self.opts.start_from_latent_avg:
                if self.opts.learn_in_w:
                    # Make sure latent_avg is at least 2D: [num_styles, 512]
                    if self.latent_avg.dim() == 1:
                        latent_avg_expanded = self.latent_avg.unsqueeze(0).unsqueeze(0).repeat(codes.shape[0], codes.shape[1], 1)
                    else:  # latent_avg is [num_styles, 512]
                        # Slice if needed
                        latent_avg_expanded = self.latent_avg[:codes.shape[1], :].unsqueeze(0).repeat(codes.shape[0], 1, 1)
                    codes = codes + latent_avg_expanded
                else:
                    latent_avg_expanded = self.latent_avg.unsqueeze(0).repeat(codes.shape[0], 1, 1)
                    codes = codes + latent_avg_expanded

        # Apply latent masking if needed
        if latent_mask is not None:
            for i in latent_mask:
                if inject_latent is not None:
                    if alpha is not None:
                        codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
                    else:
                        codes[:, i] = inject_latent[:, i]
                else:
                    codes[:, i] = 0

        # Apply feature editing if enabled
        if hasattr(self.opts, 'apply_feature_editing') and self.opts.apply_feature_editing:
            # Get current feature scale
            current_feature_scale = feature_scale if feature_scale is not None else getattr(self.opts, 'feature_scale', 1.0)
            
            # Prepare features for editing
            feats = [None] * 9 + [fused_feat] + [None] * (17 - 9)
            
            # Generate with feature editing
            images, result_latent = self.decoder(
                [codes],
                input_is_latent=True,
                return_features=True,
                new_features=feats,
                feature_scale=current_feature_scale,
                randomize_noise=randomize_noise,
                return_latents=return_latents
            )
        else:
            # Standard generation without feature editing
            input_is_latent = not input_code
            images, result_latent = self.decoder(
                [codes],
                input_is_latent=input_is_latent,
                randomize_noise=randomize_noise,
                return_latents=return_latents
            )

        if resize:
            images = self.face_pool(images)

        if return_latents:
            return {
                'y_hat': images, 
                'latent': result_latent, 
                'dw': [odw, dw], 
                'A': A, 
                'x': x_factors,
                'ocodes': ocodes,
                'fused_feat': fused_feat if 'fused_feat' in locals() else None
            }
        else:
            return {
                'y_hat': images, 
                'dw': [odw, dw], 
                'A': A, 
                'x': x_factors,
                'ocodes': ocodes
            }

    def get_code(self, x, av_codes, resize=True, latent_mask=None, return_latents=False):
        """Get style codes with attribute editing applied"""
        with torch.no_grad():
            if hasattr(self.encoder, 'encoder'):  # FSEFull
                _, ocodes, _, _ = self.encoder(x, return_latents=True)
            else:  # FSEInverter
                _, ocodes, _, _ = self.encoder(x, return_latents=True)
        
        odw = ocodes - av_codes
        dw, A, x_factors = self.ax(odw)
        codes = torch.cat((dw + av_codes[:, :6], ocodes[:, 6:]), dim=1)
        
        return {
            'odw': odw, 
            'dw': dw, 
            'A': A, 
            'x': x_factors, 
            'codes': codes, 
            'ocodes': ocodes
        }

    def get_test_code(self, x, resize=True, latent_mask=None, return_latents=False):
        """Get test codes without average codes (for inference)"""
        with torch.no_grad():
            if hasattr(self.encoder, 'encoder'):  # FSEFull
                _, ocodes, _, _ = self.encoder(x, return_latents=True)
            else:  # FSEInverter
                _, ocodes, _, _ = self.encoder(x, return_latents=True)
        
        odw = ocodes[:, :6]  # Use raw codes without average subtraction
        dw, A, x_factors = self.ax(odw)
        
        return { 
            'A': A,  
            'ocodes': ocodes,
            'dw': dw,
            'x': x_factors
        }

    def set_opts(self, opts):
        self.opts = opts

    def __load_latent_avg(self, ckpt, repeat=None):
        if 'latent_avg' in ckpt:
            self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
            if repeat is not None:
                self.latent_avg = self.latent_avg.repeat(repeat, 1)
        else:
            self.latent_avg = self.encoder.latent_avg

    def toggle_feature_editor_training(self, enable=True):
        """Dynamically enable/disable feature editor training"""
        if hasattr(self.encoder, 'encoder'):  # FSEFull
            for param in self.encoder.encoder.parameters():
                param.requires_grad = enable
            if enable:
                self.encoder.encoder.train()
            else:
                self.encoder.encoder.eval()
        else:  # FSEInverter
            for name, param in self.encoder.named_parameters():
                if 'fuser' in name:
                    param.requires_grad = enable