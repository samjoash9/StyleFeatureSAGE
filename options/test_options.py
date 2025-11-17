from argparse import ArgumentParser

class TestOptions:

    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        # arguments for inference script
        self.parser.add_argument('--dataset_type', default='fl_encode', type=str, help='Type of dataset/experiment to run')
        self.parser.add_argument('--psp_checkpoint_path', default=None, type=str, help='Path to pSp model checkpoint')
        self.parser.add_argument('--checkpoint_path', default=None, type=str, help='Path to AGE model checkpoint')
        self.parser.add_argument('--n_distribution_path', type=str, default=None, help='Path to distribution of n')
        self.parser.add_argument('--train_data_path', type=str, default=None, help='Path to directory of training set')
        self.parser.add_argument('--test_data_path', type=str, default=None, help='Path to to directory of inference inputs')
        self.parser.add_argument('--output_path', type=str, default=None, help='Path to save outputs')
        self.parser.add_argument('--class_embedding_path', type=str, default=None, help='Path to save class embeddings')
        
        # FSE-specific options
        self.parser.add_argument('--fse_checkpoint_path', default=None, type=str, 
                                help='Path to FSE model checkpoint (alternative to psp_checkpoint_path)')
        self.parser.add_argument('--fse_inverter_path', default=None, type=str,
                                help='Path to FSE inverter checkpoint (for FSEFull model)')
        self.parser.add_argument('--use_fse_full', action='store_true',
                                help='Use FSEFull model instead of FSEInverter')
        self.parser.add_argument('--apply_feature_editing', action='store_true', default=True,
                                help='Apply feature editing during inference')
        self.parser.add_argument('--feature_scale', type=float, default=1.0,
                                help='Strength of feature editing (0.0 to 1.0)')
        
        self.parser.add_argument('--test_batch_size', default=2, type=int, help='Batch size for testing and inference')
        self.parser.add_argument('--test_workers', default=2, type=int, help='Number of test/inference dataloader workers')
        
        self.parser.add_argument('--n_images', type=int, default=128, help='Number of images to generate on per input')
        self.parser.add_argument('--n_similar_cates', type=int, default=20, help='Number of similar cates for ')
        self.parser.add_argument('--t', type=int, default=10, help='Value of t_B')
        self.parser.add_argument('--A_length', default=100, type=int, help='Length of A')
        self.parser.add_argument('--alpha', default=1, type=float, help='Editing intensity alpha')
        self.parser.add_argument('--beta', default=0.000, type=float, help='Direction selection threshold in A')
        self.parser.add_argument('--resize_outputs', action='store_true', help='Whether to resize outputs to 256x256 or keep at 1024x1024')
        
        # Model configuration (for backward compatibility and dataset requirements)
        self.parser.add_argument('--encoder_type', default='GradualStyleEncoder', type=str, help='Which encoder to use')
        self.parser.add_argument('--input_nc', default=3, type=int, help='Number of input image channels')
        self.parser.add_argument('--label_nc', default=0, type=int, help='Number of input label channels')  # ADDED THIS
        self.parser.add_argument('--output_size', default=1024, type=int, help='Output size of generator')
        self.parser.add_argument('--start_from_latent_avg', action='store_true', default=True,
                                 help='Whether to add average latent vector to generate codes from encoder.')
        self.parser.add_argument('--learn_in_w', action='store_true', help='Whether to learn in w space instead of w+')

    def parse(self):
        opts = self.parser.parse_args()
        
        # Handle backward compatibility for model selection
        if opts.fse_checkpoint_path is not None:
            print(f"Using FSE model with checkpoint: {opts.fse_checkpoint_path}")
            if opts.use_fse_full:
                print("Using FSEFull model")
            else:
                print("Using FSEInverter model")
        elif opts.psp_checkpoint_path is not None:
            print(f"Using original PSP model with checkpoint: {opts.psp_checkpoint_path}")
        else:
            print("Warning: No model checkpoint specified. Please provide either --fse_checkpoint_path or --psp_checkpoint_path")
        
        return opts