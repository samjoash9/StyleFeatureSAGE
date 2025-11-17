from argparse import ArgumentParser


class TrainOptions:

    def __init__(self): 
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        self.parser.add_argument('--exp_dir', type=str, help='Path to experiment output directory')
        self.parser.add_argument('--dataset_type', default='af_encode', type=str, help='Type of dataset/experiment to run')
        self.parser.add_argument('--encoder_type', default='GradualStyleEncoder', type=str, help='Which encoder to use')
        self.parser.add_argument('--input_nc', default=3, type=int, help='Number of input image channels to the psp encoder')
        self.parser.add_argument('--label_nc', default=0, type=int, help='Number of input label channels to the psp encoder')

        self.parser.add_argument('--batch_size', default=2, type=int, help='Batch size for training')
        self.parser.add_argument('--valid_batch_size', default=2, type=int, help='Batch size for testing and inference')
        self.parser.add_argument('--workers', default=2, type=int, help='Number of train dataloader workers')
        self.parser.add_argument('--valid_workers', default=2, type=int, help='Number of test/inference dataloader workers')
        self.parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')

        self.parser.add_argument('--output_size', default=1024, type=int, help='Output size of generator')
        self.parser.add_argument('--A_length', default=100, type=int, help='length of A')
        self.parser.add_argument('--learning_rate', default=0.0001, type=float, help='Optimizer learning rate')
        self.parser.add_argument('--optim_name', default='ranger', type=str, help='Which optimizer to use')
        self.parser.add_argument('--train_decoder', default=False, type=bool, help='Whether to train the decoder model')
        self.parser.add_argument('--start_from_latent_avg', action='store_true',
                                 help='Whether to add average latent vector to generate codes from encoder.')
        self.parser.add_argument('--learn_in_w', action='store_true', help='Whether to learn in w space insteaf of w+')

        self.parser.add_argument('--class_embedding_path', default=None, type=str, help='path to class embedding')
        self.parser.add_argument('--psp_checkpoint_path', default=None, type=str, help='Path to pretrained pSp model checkpoint')
        self.parser.add_argument('--generator_checkpoint_path', default=None, type=str, help='Path to pretrained StyleGAN2/pSp generator checkpoint (decoder)')
        self.parser.add_argument('--arcface_model_path', default=None, type=str, help='Path to ArcFace iresnet50 weights for SFE inverter')
        self.parser.add_argument('--checkpoint_path', default=None, type=str, help='Path to AGE model checkpoint')

        # FSE-specific options
        self.parser.add_argument('--fse_checkpoint_path', default=None, type=str, 
                                help='Path to FSE model checkpoint (replaces psp_checkpoint_path when using FSE)')  
        self.parser.add_argument('--fse_inverter_path', default=None, type=str,
                                help='Path to FSE inverter checkpoint (for FSEFull model)')
        self.parser.add_argument('--use_fse_full', action='store_true',
                                help='Use FSEFull model instead of FSEInverter')
        self.parser.add_argument('--train_feature_editor', action='store_true',
                                help='Train the feature editor along with Ax module')
        self.parser.add_argument('--apply_feature_editing', action='store_true', default=True,
                                help='Apply feature editing during training')
        self.parser.add_argument('--feature_scale', type=float, default=1.0,
                                help='Strength of feature editing (0.0 to 1.0)')
        self.parser.add_argument('--feature_scale_schedule', action='store_true',
                                help='Gradually increase feature scale during training')
        self.parser.add_argument('--feature_scale_warmup', type=int, default=10000,
                                help='Number of steps to warm up feature scale from 0 to 1')
        self.parser.add_argument('--freeze_fse_encoder', action='store_true', default=True,
                                help='Freeze FSE encoder and only train Ax module (recommended)')
        self.parser.add_argument(
            '--use_fse_encoder', 
            action='store_true', 
            help='Enable FSE mode without requiring pSp/E4E checkpoint'
)


        # Loss parameters
        self.parser.add_argument('--l2_lambda', default=1.0, type=float, help='L2 loss multiplier factor')
        self.parser.add_argument('--sparse_lambda', default=0.005, type=float, help='sparse loss for n')
        self.parser.add_argument('--orthogonal_lambda', default=0.0005, type=float, help='orthogonal loss multiplier factor for A')
        self.parser.add_argument('--lpips_lambda', default=1.0, type=float, help='LPIPS loss multiplier factor')

        # Training schedule parameters
        self.parser.add_argument('--max_steps', default=500000, type=int, help='Maximum number of training steps')
        self.parser.add_argument('--image_interval', default=100, type=int, help='Interval for logging train images during training')
        self.parser.add_argument('--board_interval', default=50, type=int, help='Interval for logging metrics to tensorboard')
        self.parser.add_argument('--val_interval', default=1000, type=int, help='Validation interval')
        self.parser.add_argument('--save_interval', default=3000, type=int, help='Model checkpoint interval')


    def parse(self):
        opts = self.parser.parse_args()
        
        # Set default behavior for FSE training
        if opts.fse_checkpoint_path is not None:
            # If using FSE, automatically enable feature editing unless explicitly disabled
            if not hasattr(opts, 'apply_feature_editing_set'):
                opts.apply_feature_editing = True
            
            # Freeze FSE encoder by default unless training feature editor
            if not opts.train_feature_editor and not hasattr(opts, 'freeze_fse_encoder_set'):
                opts.freeze_fse_encoder = True
        
        return opts