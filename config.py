import numpy as np

class GlobalConfig:
  def __init__(self):
    # Whether the model in and outputs will be visualized and saved into SAVE_PATH
    self.debug = False
    # -----------------------------------------------------------------------------
    # Dataloader
    # -----------------------------------------------------------------------------
    # self.carla_fps = 20  # Simulator Frames per second
    # self.seq_len = 1  # input timesteps #!
    # self.img_seq_len = 1 #!
    # self.lidar_seq_len = 1 #!
    
    # self.lidar_resolution_width = 256 #!
    # self.lidar_resolution_height = 256 #!
    # # 1 / pixels_per_meter = size of pixel in meters
    # self.pixels_per_meter = 4.0 #!
    # Max number of LiDAR points per pixel in voxelized LiDAR
    # self.hist_max_per_pixel = 5
    # Height at which the LiDAR points are split into the 2 channels.
    # Is relative to lidar_pos[2]
    # self.lidar_split_height = 0.2
    # Max and minimum LiDAR ranges used for voxelization
    # self.min_x = -32 #!
    # self.max_x = 32 #!
    # self.min_y = -32 #!
    # self.max_y = 32 #!
    # self.min_z = -4
    # self.max_z = 4
    # self.min_z_projection = -10
    # self.max_z_projection = 14
    
    # self.target_speed_slow = 10.0  # Speed at junctions, km/s
    # self.target_speed_fast = 30.0  # Speed outside junctions, km/s
    # self.target_speed_normal = 15.0  # Normal Speed, km/s
    # Bin in for the target speed one hot vector.
    # self.target_speed_bins = [
    #     self.target_speed_object + 0.1, self.target_speed_slow + 0.1, self.target_speed_fast + 0.1
    # ]
    # self.target_speed_bins = [
    #     self.target_speed_slow + 0.1, self.target_speed_fast + 0.1
    # ]
    # Index 0 is the brake action
    # self.target_speeds = [20.0] #! Let's do with 1 first
    # self.target_speeds = [0.0, self.target_speed_normal] #! 0, 15
    # self.target_speeds = [0.0, self.target_speed_slow, self.target_speed_fast] #! new trial
    # self.target_speed_weights = [0.866605263873406, 7.4527377240841775, 1.2281629310898465, 0.5269622904065803] #!
    # self.target_speed_weights = [76.55, 1.0]
    # self.target_speed_weights = [50.0, 1.0]
    # self.target_speed_weights = [3.5, 1.0]
    # self.semantic_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] #! 19

    # -----------------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------------
    self.num_workers = 8
    self.id = 'transfuser'  # Unique experiment identifier. #!
    self.epochs = 31  # Number of epochs to train #!
    self.lr = 1e-4  # Learning rate used for training #!
    self.batch_size = 64  # Batch size used during training #!
    self.logdir = ''  # Directory to log data to.
    self.load_file = None  # File to continue training from
    self.root_dir = ''  # Dataset root dir 
    # When to reduce the learning rate for the first and second  time
    self.schedule_reduce_epoch_01 = 30 #!
    self.schedule_reduce_epoch_02 = 40 #!
    # self.val_every = 1  # Validation frequency in epochs #!
    # Whether zero_redundancy_optimizer was used during training
    # self.detect_boxes = 0  # Whether to use the bounding box auxiliary task #!
    # Number of route points we use for prediction in TF or input in planT
    # self.learn_origin = 1  # Whether to learn the origin of the waypoints or use 0 / 0 #!
    # self.augment = 1  # Whether to use rotation and translation augmentation
    # If this is true we convert the batch norms, to synced bach norms.
    # At which interval to save debug files to disk during training
    # self.train_debug_save_freq = 1
    # self.backbone = 'transFuser'  # Vision backbone architecture used #!
    # self.use_velocity = 1  # Whether to use the velocity as input to the network #!
    # self.image_architecture = 'regnety_032'  # Image architecture used in the backbone #!
    # self.lidar_architecture = 'regnety_032'  # LiDAR architecture used in the backbone #!
    # Whether to classify target speeds and regress a path as output representation.
    # self.use_controller_input_prediction = True #! checkpoint & Target speed separated
    # Label smoothing applied to the cross entropy losses
    # self.label_smoothing_alpha = 0.1
    # Optimization
    # self.lr = 0.0003  # learning rate
    # Whether to use focal loss instead of cross entropy for classification
    # self.use_focal_loss = False #!
    # Gamma hyperparameter of focal loss
    # self.focal_loss_gamma = 2.0 #!
    # Learning rate decay, applied when using multi-step scheduler
    self.multi_step_lr_decay = 0.1
    # Whether to use a cosine schedule instead of the linear one.
    self.use_cosine_schedule = False
    # Epoch of the first restart
    self.cosine_t0 = 1
    # Multiplier applied to t0 after every restart
    self.cosine_t_mult = 2
    # Weights applied to each of these losses, when combining them
    self.detailed_loss_weights = { #! Multi task learning weights
        # 'loss_wp': 1.0,
        # 'loss_target_speed': 1.0,
        'loss_ctrl': 1.0,
        'loss_checkpoint': 1.0,
        # 'loss_collision': 1.0,
        'loss_tl': 1.0,
        # 'loss_steer': 1.0,
        # 'loss_semantic': 1.0,
        # 'loss_depth': 1.0,
        # 'loss_wh': 1.0,
        # 'loss_offset': 1.0,
        # 'loss_yaw_class': 1.0,
        # 'loss_yaw_res': 1.0,
        # 'loss_velocity': 1.0,
        # 'loss_brake': 1.0,
        # 'loss_forcast': 0.2,
        # 'loss_selection': 0.0,
    }
    self.root_dir = ''
    # NOTE currently leads to inf gradients do not use! Whether to use automatic mixed precision during training.
    self.use_amp = 1
    self.use_grad_clip = 0  # Whether to clip the gradients
    # self.grad_clip_max_norm = 1.0  # Max value for the gradients if gradient clipping is used.
    # self.color_aug_prob = 0.5  # With which probability to apply the different image color augmentations.
    # self.lidar_aug_prob = 1.0  # Probability with which data augmentation is applied to the LiDAR image.
    # self.freeze_backbone = False  # Whether to freeze the image backbone during training. Useful for 2 stage training. #!
    # self.auxiliary_task_only = False #!
    self.learn_multi_task_weights = False  # Whether to learn the multi-task weights #!
    # self.use_bev_semantic = False  # Whether to use bev semantic segmentation as auxiliary loss for training. #!
    # self.use_depth = False  # Whether to use depth prediction as auxiliary loss for training. #!
    # self.continue_epoch = True  # Whether to continue the training from the loaded epoch or from 0.

    # self.smooth_route = True  # Whether to smooth the route points with a spline.
    # self.ignore_index = -999  # Index to ignore for future bounding box prediction task.
    # self.use_speed_weights = True  # Whether to weight target speed classes #!
    # self.weight_decay = 0.01  # Weight decay coefficient used during training #!
    # self.use_label_smoothing = False  # Whether to use label smoothing in the classification losses #!

    # -----------------------------------------------------------------------------
    # TransFuser Model
    # -----------------------------------------------------------------------------
    # Waypoint GRU
    self.gru_hidden_size = 128 #!
    self.gru_input_size = 256 #!

    # self.camera_width = 1024  # Camera width in pixel during data collection
    # self.camera_height = 256  # Camera height in pixel during data collection
    # self.camera_fov = 110
    # Conv Encoder
    # self.img_vert_anchors = self.camera_height // 32 #!
    # self.img_horz_anchors = self.camera_width // 32 #!

    # self.lidar_vert_anchors = self.lidar_resolution_height // 32 #!
    # self.lidar_horz_anchors = self.lidar_resolution_width // 32 #!

    # self.img_anchors = self.img_vert_anchors * self.img_horz_anchors
    # self.lidar_anchors = self.lidar_vert_anchors * self.lidar_horz_anchors

    # Resolution at which the perspective auxiliary tasks are predicted
    # self.perspective_downsample_factor = 1 #!

    # self.bev_features_chanels = 64  # Number of channels for the BEV feature pyramid #!
    # Resolution at which the BEV auxiliary tasks are predicted
    # self.bev_down_sample_factor = 4 #!
    # self.bev_upsample_factor = 2 #!

    # GPT Encoder
    # self.block_exp = 4 #!
    # self.n_layer = 2  # Number of transformer layers used in the vision backbone #!
    # self.n_head = 4 #!
    # self.n_scale = 4
    # self.embd_pdrop = 0.1 #!
    # self.resid_pdrop = 0.1 #!
    # self.attn_pdrop = 0.1 #!
    # Mean of the normal distribution initialization for linear layers in the GPT
    # self.gpt_linear_layer_init_mean = 0.0 #!
    # Std of the normal distribution initialization for linear layers in the GPT
    # self.gpt_linear_layer_init_std = 0.02 #!
    # Initial weight of the layer norms in the gpt.
    # self.gpt_layer_norm_init_weight = 1.0 #!

    # Number of route checkpoints to predict. Needs to be smaller than num_route_points!
    self.predict_checkpoint_len = 100 #!
 
    # Whether to normalize the camera image by the imagenet distribution
    self.normalize_imagenet = False #!
    # self.use_wp_gru = False  # Whether to use the WP output GRU. #! as in Transfuser

    # Semantic Segmentation
    # self.use_semantic = True  # Whether to use semantic segmentation as auxiliary loss #!
    # self.num_semantic_classes = 19 #! #TODO change to # of pseudolabel class
    # self.class_mapping = { #! newdata
    #     (0, 0, 0): 0,          # road
    #     (0, 128, 0): 1,        # sky
    #     (0, 0, 255): 2,        # building
    #     (255, 0, 255): 3,      # pole
    #     (128, 0, 0): 4,        # vegetation
    #     (0, 255, 0): 5,        # sidewalk
    #     (128, 128, 128): 6,    # terrain
    #     (0, 255, 255): 7,      # traffic light
    #     (255, 0, 0): 8,        # fence or wall
    #     (255, 255, 0): 9,      # fence or wall
    #     (255, 255, 255): 10,   # traffic sign
    # }
    # self.class_mapping = { #! Officialdataset
    #     (128, 0, 128): 0,    # road
    #     (128, 128, 0): 1,    # sidewalk
    #     (128, 128, 128): 2,  # building
    #     (128, 0, 0): 3,      # wall
    #     (128, 64, 0): 4,     # fence
    #     (192, 192, 192): 5,  # pole
    #     (255, 255, 0): 6,    # traffic light
    #     (0, 255, 255): 7,    # traffic sign
    #     (0, 128, 0): 8,      # vegetation
    #     (128, 255, 128): 9,  # terrain
    #     (0, 0, 255): 10,     # sky

    #     (255, 0, 0): 11,     # person
    #     (255, 128, 0): 12,   # rider
    #     (0, 0, 128): 13,     # car
    #     (0, 0, 64): 14,      # truck
    #     (0, 128, 128): 15,   # bus
    #     (0, 64, 64): 16,     # train
    #     (0, 0, 255): 17,     # motorcycle
    #     (128, 0, 64): 18,    # bicycle
    # }
    
    # self.deconv_channel_num_0 = 128  # Number of channels at the first deconvolution layer #!
    # self.deconv_channel_num_1 = 64  # Number of channels at the second deconvolution layer #!
    # self.deconv_channel_num_2 = 32  # Number of channels at the third deconvolution layer #!

    # Fraction of the down-sampling factor that will be up-sampled in the first Up-sample
    # self.deconv_scale_factor_0 = 4 #!
    # # Fraction of the down-sampling factor that will be up-sampled in the second Up-sample
    # self.deconv_scale_factor_1 = 8 #!

    # self.use_discrete_command = True  # Whether to input the discrete target point as input to the network. #!
    # self.add_features = True  # Whether to add (true) or concatenate (false) the features at the end of the backbone. #!

    # self.image_u_net_output_features = 512  # Channel dimension of the up-sampled encoded image in bev_encoder
    # self.bev_latent_dim = 32  # Channel dimensions of the image projected to BEV in the bev_encoder

    # Whether to use a transformer decoder instead of global average pool + MLP for planning
    # self.transformer_decoder_join = True #! same as Interfuser transformer decoder
    self.num_transformer_decoder_layers = 6  # Number of layers in the TransFormer decoder
    self.num_decoder_heads = 8

    # Ratio by which the height size of the voxel grid in BEV decoder are larger than width and depth
    # self.bev_grid_height_downsample_factor = 1.0

    self.extra_sensor_channels = 128  # Number of channels the extra sensors are embedded to

    # self.use_tp = True  # Whether to use the target point as input to TransFuser #!

    # self.tp_attention = False  # Adds a TP at the TF decoder and computes it with attention visualization. #!
    # self.multi_wp_output = False  # Predicts 2 WP outputs and uses the min loss of both. #!

    
    # self.intrinsics = [
    #     np.array([[320., 0., 320.], [0., 320., 180.], [0., 0., 1.]]),
    #     np.array([[320., 0., 320.], [0., 320., 180.], [0., 0., 1.]]),
    #     np.array([[320., 0., 320.], [0., 320., 180.], [0., 0., 1.]]),
    #     np.array([[184.75208614, 0., 320.], [0., 184.75208614, 180.], [0., 0., 1.]]),
    #     np.array([[184.75208614, 0., 320.], [0., 184.75208614, 180.], [0., 0., 1.]])
    # ]

    # self.extrinsics = [
    #     np.array([[2.22044605e-16, -1.00000000e+00, 4.93038066e-32, -3.99680289e-16],
    #               [2.22044605e-16, 4.93038066e-32, -1.00000000e+00, 1.30000000e+00],
    #               [1.00000000e+00, 2.22044605e-16, 2.22044605e-16, -1.80000000e+00],
    #               [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
    #     np.array([[7.07106781e-01, -7.07106781e-01, 0.00000000e+00, -7.07106781e-01],
    #               [1.57009246e-16, 1.57009246e-16, -1.00000000e+00, 1.30000000e+00],
    #               [7.07106781e-01, 7.07106781e-01, 2.22044605e-16, -1.41421356e+00],
    #               [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
    #     np.array([[-7.07106781e-01, -7.07106781e-01, -0.00000000e+00, 7.07106781e-01],
    #               [1.57009246e-16, -1.57009246e-16, -1.00000000e+00, 1.30000000e+00],
    #               [7.07106781e-01, -7.07106781e-01, 2.22044605e-16, -1.41421356e+00],
    #               [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
    #     np.array([[7.07106781e-01, 7.07106781e-01, 0.00000000e+00, -9.89949494e-01],
    #               [-1.57009246e-16, 1.57009246e-16, -1.00000000e+00, 1.30000000e+00],
    #               [-7.07106781e-01, 7.07106781e-01, 2.22044605e-16, 1.41421356e-01],
    #               [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
    #     np.array([[-7.07106781e-01, 7.07106781e-01, -0.00000000e+00, 9.89949494e-01],
    #               [-1.57009246e-16, -1.57009246e-16, -1.00000000e+00, 1.30000000e+00],
    #               [-7.07106781e-01, -7.07106781e-01, 2.22044605e-16, 1.41421356e-01],
    #               [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    # ]

    # # World range
    # self.world_x_min, self.world_x_max = -20, 20
    # self.world_y_min, self.world_y_max = -10, 10

    # # World points
    # self.world_pts = np.array([
    #     [self.world_x_min, self.world_y_min, 0],
    #     [self.world_x_max, self.world_y_min, 0],
    #     [self.world_x_max, self.world_y_max, 0],
    #     [self.world_x_min, self.world_y_max, 0],
    # ])

    # self.left_bd = 5.8
    # self.right_bd = 5.1
    # self.height_bd = 1.2


  def initialize(self, **kwargs):
    for k, v in kwargs.items():
      # print(f"Key: {k}, Value: {v}") # Debugging
      setattr(self, k, v)
