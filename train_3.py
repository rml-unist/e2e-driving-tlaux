import argparse
import json
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.distributed.elastic.multiprocessing.errors import record
import torch.multiprocessing as mp

from config import GlobalConfig
# from model_AR import LidarCenterNet
# from model_aux import LidarCenterNet
# from model_noaux import LidarCenterNet
from model_aux_2 import LidarCenterNet

import pathlib
import random
import pickle

from collections import defaultdict

import dataset_single_official
import draw


@record  # Records error and tracebacks in case of failure
def main():
    torch.cuda.empty_cache()

    config = GlobalConfig()

    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, default='w_aux_official', help='Unique experiment identifier.')
    parser.add_argument('--epochs', type=int, default=51, help='Number of train epochs.')
    parser.add_argument('--lr', type=float, default=config.lr, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for one GPU. When training with multiple GPUs the effective batch size will be batch_size*num_gpus.')
    parser.add_argument('--logdir', type=str, default='./exp',
                        help='Directory to log data and models to.')
    # parser.add_argument('--cache', type=str, default='./dataset/cache.pkl', help='Directory to log data and models to.')fstep1
    # parser.add_argument('--load_file', type=str, default=config.load_file, help='Model to load for initialization. Expects the full path with ending /path/to/model.pth. Optimizer files are expected to exist in the same directory.')
    parser.add_argument('--root_dir', type=str, default="./dataset",
                        help='Root directory of your training data.')
    parser.add_argument('--schedule_reduce_epoch_01', type=int, default=config.schedule_reduce_epoch_01,
                        help='Epoch at which to reduce the lr by a factor of 10 the first time. Only used with --schedule 1.')
    parser.add_argument('--schedule_reduce_epoch_02', type=int, default=config.schedule_reduce_epoch_02,
                        help='Epoch at which to reduce the lr by a factor of 10 the second time. Only used with --schedule 1.')
    # parser.add_argument('--backbone', type=str, default=config.backbone, help='Which fusion backbone to use. Options: transFuser, aim, bev_encoder.')
    # parser.add_argument('--image_architecture', type=str, default=config.image_architecture, help='Which architecture to use for the image branch. resnet34, regnety_032, etc. All options of the TIMM lib can be used but some might need adjustments to the backbone.')
    # parser.add_argument('--lidar_architecture', type=str, default=config.lidar_architecture, help='Which architecture to use for the lidar branch. Tested: resnet34, regnety_032. Has the special video option video_resnet18 and video_swin_tiny.')
    # parser.add_argument('--use_velocity', type=int, default=config.use_velocity, help='Whether to use the velocity input. Expected values are 0:False, 1:True.')
    # parser.add_argument('--n_layer', type=int, default=config.n_layer, help='Number of transformer layers used in the transfuser.')
    # parser.add_argument('--val_every', type=int, default=config.val_every, help='At which epoch frequency to validate.')
    # parser.add_argument('--lidar_seq_len', type=int, default=config.lidar_seq_len, help='How many temporal frames in the LiDAR to use. 1 equals single timestep.')
    # parser.add_argument('--use_controller_input_prediction', type=int, default=int(config.use_controller_input_prediction), help='Whether to classify target speeds and regress a path as output representation.')
    # parser.add_argument('--use_wp_gru', type=int, default=int(config.use_wp_gru), help='Whether to predict the waypoint output representation.')
    # parser.add_argument('--use_focal_loss', type=int, default=int(config.use_focal_loss), help='Whether to use focal loss instead of cross entropy for target speed classification.')
    parser.add_argument('--use_cosine_schedule', type=int, default=int(config.use_cosine_schedule),
                        help='Whether to use a cyclic cosine learning rate schedule instead of the linear one.')
    # parser.add_argument('--augment', type=int, default=int(config.augment), help='Whether to use rotation and translation augmentation.')
    # parser.add_argument('--learn_origin', type=int, default=int(config.learn_origin), help='Whether to learn the origin of the waypoints or use 0/0.')
    parser.add_argument('--use_amp', type=int, default=int(config.use_amp),
                        help='Whether to use automatic mixed precision with fp16 during training. Currently produces inf gradients.')
    parser.add_argument('--use_grad_clip', type=int, default=int(config.use_grad_clip),
                        help='Whether to clip the gradients during training.')
    # parser.add_argument('--use_semantic', type=int, default=int(config.use_semantic), help='Whether to use semantic segmentation as auxiliary loss.')
    # parser.add_argument('--use_depth', type=int, default=int(config.use_depth), help='Whether to use depth prediction as auxiliary loss for training.')
    # parser.add_argument('--detect_boxes', type=int, default=int(config.detect_boxes), help='Whether to use the bounding box auxiliary task.')
    # parser.add_argument('--use_bev_semantic', type=int, default=int(config.use_bev_semantic), help='Whether to use bev semantic segmentation as auxiliary loss for training.')
    # parser.add_argument('--use_discrete_command', type=int, default=int(config.use_discrete_command), help='Whether the discrete command is an input for the model.')
    parser.add_argument('--gru_hidden_size', type=int, default=int(config.gru_hidden_size),
                        help='Number of features used in the hidden size of the GRUs.')
    # parser.add_argument('--add_features', type=int, default=int(config.add_features), help='Whether to add (or concatenate) the features at the end of the backbone.')
    # parser.add_argument('--auxiliary_task_only', type=int, default=int(config.auxiliary_task_only), help='Freezes decoders. Should be used when only training backbone and auxiliary tasks.')
    parser.add_argument('--learn_multi_task_weights', type=int, default=int(config.learn_multi_task_weights),
                        help='Whether to learn the multi-task weights according to https://arxiv.org/abs/1705.07115.')
    # parser.add_argument('--transformer_decoder_join', type=int, default=int(config.transformer_decoder_join), help='Whether to use a transformer decoder instead of global average pool + MLP for planning.')
    # parser.add_argument('--bev_down_sample_factor', type=int, default=int(config.bev_down_sample_factor), help='Factor (int) by which the bev auxiliary tasks are down-sampled.')
    # parser.add_argument('--perspective_downsample_factor', type=int, default=int(config.perspective_downsample_factor), help='Factor (int) by which the perspective auxiliary tasks are down-sampled.')
    parser.add_argument('--gru_input_size', type=int, default=int(config.gru_input_size),
                        help='Number of channels in the InterFuser GRU input and Transformer decoder. Must be divisible by number of heads (8).')
    # parser.add_argument('--bev_grid_height_downsample_factor', type=int, default=int(config.bev_grid_height_downsample_factor), help='Ratio by which the height size of the voxel grid in BEV decoder are larger than width and depth. Value should be >= 1. Larger values uses less gpu memory. Only relevant for the bev_encoder backbone.')
    # parser.add_argument('--use_tp', type=int, default=int(config.use_tp), help='Whether to use the target point as input to the network.')
    # parser.add_argument('--continue_epoch', type=int, default=int(config.continue_epoch), help='Whether to continue the training from the loaded epoch or from 0.')
    # parser.add_argument('--smooth_route', type=int, default=int(config.smooth_route), help='Whether to smooth the route points with linear interpolation.')
    # parser.add_argument('--use_speed_weights', type=int, default=int(config.use_speed_weights), help='Whether to weight target speed classes.')
    # parser.add_argument('--weight_decay', type=float, default=float(config.weight_decay), help='Weight decay coefficient used during training.')
    # parser.add_argument('--use_label_smoothing', type=int, default=int(config.use_label_smoothing), help='Whether to use label smoothing in the classification losses. Not working as intended when combined with use_speed_weights.')
    # parser.add_argument('--tp_attention', type=int, default=int(config.tp_attention), help='Adds a TP at the TF decoder and computes it with attention visualization. Only compatible with transformer decoder.')
    # parser.add_argument('--multi_wp_output', type=int, default=int(config.multi_wp_output), help='Predict 2 WP outputs and select between them. Only compatible with use_wp=1, transformer_decoder_join=1.')

    args = parser.parse_args()
    args.logdir = os.path.join(args.logdir, args.id)

    device = torch.device('cuda:0')

    config.initialize(**vars(args))

    config.debug = int(os.environ.get('DEBUG_CHALLENGE', 0))  # 0

    if args.learn_multi_task_weights:  # False
        for k in config.detailed_loss_weights:
            if config.detailed_loss_weights[k] > 0.0:
                config.detailed_loss_weights[k] = torch.nn.Parameter(
                    torch.zeros(1, dtype=torch.float32, requires_grad=True))
            else:
                # These losses we don't train
                config.detailed_loss_weights[k] = None
        # Convert to pytorch dictionary for proper parameter handling
        config.detailed_loss_weights = torch.nn.ParameterDict(config.detailed_loss_weights)
    else:
        # Normalize loss weights.
        factor = 1.0 / sum(config.detailed_loss_weights.values())
        for k in config.detailed_loss_weights:
            config.detailed_loss_weights[k] = config.detailed_loss_weights[k] * factor

    train_dataset_official = dataset_single_official.MORAIDataset(
        "./dataset", train=True,
        cache_file='P_official_path_inc_0424_0.pkl', ratio=0.8, config=config)
    val_dataset_official = dataset_single_official.MORAIDataset(
        "./dataset", train=False,
        cache_file='P_official_path_inc_0424_0.pkl', ratio=0.8, config=config)

    train_dataset = ConcatDataset([train_dataset_official])
    val_dataset = ConcatDataset([val_dataset_official])

    model = LidarCenterNet(config)

    start_epoch = 101
    args.epochs = 151
    model.load_state_dict(torch.load('./exp/w_aux_official/model_0100.pth'), strict=True)

    model.cuda(device=device)

    with open(os.path.join(args.logdir, 'param_require_grad.txt'), 'w') as f:
        for name, param in model.named_parameters():
            f.write(f"{name}: requires_grad={param.requires_grad}\n")

    # params = model.parameters()
    params = filter(lambda p: p.requires_grad, model.parameters())  # ! Filtering only params whose requires_grad=True
    optimizer = optim.AdamW(params, lr=args.lr, amsgrad=True)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = sum(np.prod(p.size()) for p in model_parameters)
    print('Total trainable parameters: ', num_params)

    g_cuda = torch.Generator(device='cpu')
    g_cuda.manual_seed(torch.initial_seed())

    dataloader_train = DataLoader(train_dataset, batch_size=config.batch_size, generator=g_cuda, shuffle=False,
                                  worker_init_fn=seed_worker, num_workers=config.num_workers, drop_last=True,
                                  pin_memory=True)
    dataloader_val = DataLoader(val_dataset, batch_size=config.batch_size, generator=g_cuda, shuffle=False,
                                worker_init_fn=seed_worker, num_workers=config.num_workers, drop_last=True,
                                pin_memory=True)

    # Create logdir
    if not os.path.isdir(args.logdir):
        print('Created dir:', args.logdir)
        os.makedirs(args.logdir, exist_ok=True)

    # Log args
    with open(os.path.join(args.logdir, 'args.txt'), 'w', encoding='utf-8') as f:
        json.dump(args.__dict__, f, indent=2)

    with open(os.path.join(args.logdir, 'config.pickle'), 'wb') as f2:
        pickle.dump(config, f2, protocol=4)

    if config.use_cosine_schedule:  # False
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config.cosine_t0,
                                                                         T_mult=config.cosine_t_mult, verbose=False)
    else:
        milestones = [args.schedule_reduce_epoch_01, args.schedule_reduce_epoch_02]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=config.multi_step_lr_decay,
                                                         verbose=True)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(config.use_amp))

    trainer = Engine(model=model,
                     optimizer=optimizer,
                     dataloader_train=dataloader_train,
                     dataloader_val=dataloader_val,
                     args=args,
                     config=config,
                     device=device,
                     cur_epoch=start_epoch,
                     scheduler=scheduler,
                     scaler=scaler,
                     id=args.id)

    logfile = open(os.path.join(args.logdir, "training_log.txt"), "w+")
    for epoch in range(trainer.cur_epoch, args.epochs):
        print(f'Epoch: {epoch}/{args.epochs}')
        train_loss_epoch, train_detailed_losses_epoch, tl_accuracy = trainer.train()  # ! train
        logfile.write(
            f'Epoch: {epoch}, Total Loss: {train_loss_epoch}, Detailed Loss: {train_detailed_losses_epoch}, TL Acc: {tl_accuracy}\n')

        torch.cuda.empty_cache()

        val_loss_epoch, val_detailed_losses_epoch, tl_accuracy = trainer.validate()  # ! val
        logfile.write(
            f'Epoch: {epoch}, Total Loss: {val_loss_epoch}, Detailed Loss: {val_detailed_losses_epoch}, TL Acc: {tl_accuracy}\n\n')
        torch.cuda.empty_cache()

        if not config.use_cosine_schedule:
            scheduler.step()

        # if epoch % 10 == 0:
        # if val_loss_epoch < best_val:
        trainer.save()
        # best_val = val_loss_epoch

        if epoch == args.epochs - 1:
            trainer.plot_loss(args.epochs, save_path=os.path.join(args.logdir, 'loss.png'))

        trainer.cur_epoch += 1


class Engine(object):
    def __init__(self,
                 model,
                 optimizer,
                 dataloader_train,
                 dataloader_val,
                 args,
                 config,
                 device,
                 scheduler,
                 scaler,
                 id,
                 world_size=1,
                 cur_epoch=0):
        self.cur_epoch = cur_epoch
        self.start_epoch = cur_epoch
        self.bestval_epoch = cur_epoch
        self.bestval = 1e10
        self.model = model
        self.optimizer = optimizer
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.args = args
        self.config = config
        self.device = device
        self.world_size = world_size
        self.vis_save_path = self.args.logdir + r'/visualizations'
        self.scheduler = scheduler
        self.iters_per_epoch = len(self.dataloader_train)
        self.scaler = scaler
        self.id = id

        # loss logging
        self.train_loss_total = []
        self.train_loss_ctrl = []
        self.train_loss_checkpoint = []
        self.train_tl_loss = []
        self.train_tl_acc = []  ### 추가: 신호등 분류 정확도 저장 리스트
        self.val_loss_total = []
        self.val_loss_ctrl = []
        self.val_loss_checkpoint = []
        self.val_tl_loss = []
        self.val_tl_acc = []  ### 추가: 검증 신호등 분류 정확도 저장 리스트

        if self.config.debug:
            pathlib.Path(self.vis_save_path).mkdir(parents=True, exist_ok=True)

        self.detailed_loss_weights = config.detailed_loss_weights

    def plot_loss(self, epochs, save_path="loss_plot.png"):
        plt.figure(figsize=(15, 10))
        epoch_plot = list(range(self.start_epoch, self.cur_epoch + 1))

        # Total Loss
        plt.subplot(2, 3, 1)
        plt.plot(epoch_plot, self.train_loss_total, label='Train Total Loss', color='blue')
        plt.plot(epoch_plot, self.val_loss_total, label='Validation Total Loss', color='orange')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Total Loss per Epoch')
        plt.legend()
        plt.grid(True)

        # Control Loss
        plt.subplot(2, 3, 2)
        plt.plot(epoch_plot, self.train_loss_ctrl, label='Train Ctrl Loss', color='blue')
        plt.plot(epoch_plot, self.val_loss_ctrl, label='Validation Ctrl Loss', color='orange')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Ctrl Loss per Epoch')
        plt.legend()
        plt.grid(True)

        # Checkpoint Loss
        plt.subplot(2, 3, 3)
        plt.plot(epoch_plot, self.train_loss_checkpoint, label='Train Checkpoint Loss', color='blue')
        plt.plot(epoch_plot, self.val_loss_checkpoint, label='Validation Checkpoint Loss', color='orange')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Checkpoint Loss per Epoch')
        plt.legend()
        plt.grid(True)

        # tl Acc
        plt.subplot(2, 3, 4)
        plt.plot(epoch_plot, self.train_tl_acc, label='Train TL Acc', color='blue')
        plt.plot(epoch_plot, self.val_tl_acc, label='Validation TL Acc', color='orange')
        plt.xlabel('Epochs')
        plt.ylabel('Acc')
        plt.title('TL Acc per Epoch')
        plt.legend()
        plt.grid(True)

        # tl Loss
        plt.subplot(2, 3, 5)
        plt.plot(epoch_plot, self.train_tl_loss, label='Train TL Loss', color='blue')
        plt.plot(epoch_plot, self.val_tl_loss, label='Validation TL Loss', color='orange')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('TL Loss per Epoch')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
        # plt.show()

    def load_data_compute_loss(self, data, idx, validation=False):
        ego_vel = data['speed'].to(self.device, dtype=torch.float32).unsqueeze(1)
        target_ctrl = data['target_ctrl'].to(self.device, dtype=torch.float32)

        # Load model specific data and execute model
        checkpoint = data['local_checkpoint'][:, :self.config.predict_checkpoint_len].to(self.device,
                                                                                         dtype=torch.float32)
        target_point = data['local_targetpoint'].to(self.device, dtype=torch.float32)  # .squeeze(1)
        local_path = data['local_path'].to(self.device, dtype=torch.float32)

        rgb = data['rgb'].to(self.device, dtype=torch.float32)

        tl_label = data['tl_label'].to(self.device, dtype=torch.long)

        pred_ctrl, \
            pred_checkpoint, \
            pred_tl = self.model(rgb=rgb,
                                 ego_vel=ego_vel,
                                 target_point=target_point,
                                 path=local_path
                                 )

        # if validation and self.cur_epoch % 10 == 0:
        # if validation and idx % 10 == 0:
        #   visualize(data, self.config,  pred_checkpoint, pred_ctrl, pred_tl, self.cur_epoch, idx)
        # else:
        #   visualize(data, self.config,  pred_checkpoint, pred_ctrl, pred_tl, self.cur_epoch, idx, 1)

        compute_loss = self.model.compute_loss
        losses, loss_ctrl = compute_loss(pred_ctrl=pred_ctrl,
                                         pred_checkpoint=pred_checkpoint,
                                         pred_tl=pred_tl,
                                         ctrl_label=target_ctrl,
                                         checkpoint_label=checkpoint,
                                         tl_label=tl_label
                                         # obstacles=obstacles
                                         )

        metrics = {}

        return losses, loss_ctrl, metrics, pred_tl, tl_label  ### 추가: 신호등 예측값과 정답 라벨 반환

    def train(self):
        self.model.train()

        num_batches = 0
        loss_epoch = 0.0
        detailed_losses_epoch = {key: 0.0 for key in self.detailed_loss_weights}
        total_tl_correct = 0  ### 추가: 신호등 분류 맞춘 개수
        total_tl_samples = 0  ### 추가: 신호등 분류 전체 개수
        self.optimizer.zero_grad(set_to_none=False)

        # Train loop
        for i, data in enumerate(tqdm(self.dataloader_train)):
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=bool(self.config.use_amp)):
                losses, loss_ctrl, metrics, pred_tl, tl_label = self.load_data_compute_loss(data, i, validation=False)
                loss = torch.zeros(1, dtype=torch.float32, device=self.device)
                loss2 = torch.zeros(1, dtype=torch.float32, device=self.device)

                for key, value in losses.items():
                    loss += self.detailed_loss_weights[key] * value
                    detailed_losses_epoch[key] += float(self.detailed_loss_weights[key] * float(value.item()))

                for key, value in loss_ctrl.items():
                    loss2 += self.detailed_loss_weights[key] * value
                    detailed_losses_epoch[key] += float(self.detailed_loss_weights[key] * float(value.item()))
                # 신호등 분류 정확도 계산
                pred_tl_class = torch.argmax(pred_tl, dim=1)  ### 추가
                correct_tl = (pred_tl_class == tl_label).sum().item()  ### 추가
                total_tl_correct += correct_tl  ### 추가
                total_tl_samples += tl_label.shape[0]  ### 추가

            # ! step1 freeze control_network
            self.scaler.scale(loss + loss2).backward()

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            num_batches += 1
            loss_epoch += float(loss.item() + loss2.item()) / len(self.dataloader_train)  # !!!!!

            if self.config.use_cosine_schedule:
                self.scheduler.step(self.cur_epoch + i / self.iters_per_epoch)

        self.optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

        for key in detailed_losses_epoch.keys():
            detailed_losses_epoch[key] /= len(self.dataloader_train)  # 배치 개수로 나눠 평균 계산

        tl_accuracy = (total_tl_correct / total_tl_samples) * 100  ### 추가: 신호등 분류 정확도 계산
        print(f'Loss: {loss_epoch}, {detailed_losses_epoch}, Traffic Light Accuracy: {tl_accuracy:.4f}')  ### 추가: 출력

        self.train_loss_total.append(loss_epoch)
        self.train_loss_checkpoint.append(detailed_losses_epoch['loss_checkpoint'])
        self.train_loss_ctrl.append(detailed_losses_epoch['loss_ctrl'])
        self.train_tl_loss.append(detailed_losses_epoch['loss_tl'])
        self.train_tl_acc.append(tl_accuracy)  ### 추가: 신호등 분류 정확도 저장

        return loss_epoch, detailed_losses_epoch, tl_accuracy

    @torch.inference_mode()
    def validate(self):
        self.model.eval()

        num_batches = 0
        loss_epoch = 0.0
        detailed_val_losses_epoch = defaultdict(float)
        total_tl_correct = 0  ### 추가
        total_tl_samples = 0  ### 추가

        # Evaluation loop loop
        for i, data in enumerate(tqdm(self.dataloader_val)):
            losses, loss_ctrl, metrics, pred_tl, tl_label = self.load_data_compute_loss(data, i, validation=True)
            loss = torch.zeros(1, dtype=torch.float32, device=self.device)
            loss2 = torch.zeros(1, dtype=torch.float32, device=self.device)

            for key, value in losses.items():
                loss += self.detailed_loss_weights[key] * value
                detailed_val_losses_epoch[key] += float(self.detailed_loss_weights[key] * float(value.item()))

            for key, value in loss_ctrl.items():
                loss2 += self.detailed_loss_weights[key] * value
                detailed_val_losses_epoch[key] += float(self.detailed_loss_weights[key] * float(value.item()))
            pred_tl_class = torch.argmax(pred_tl, dim=1)  ### 추가
            correct_tl = (pred_tl_class == tl_label).sum().item()  ### 추가
            total_tl_correct += correct_tl  ### 추가
            total_tl_samples += tl_label.shape[0]  ### 추가
            for key, value in metrics.items():
                detailed_val_losses_epoch[key] += float(value)

            num_batches += 1
            loss_epoch += float(loss.item() + loss2.item()) / len(self.dataloader_train)  # !!!!!

            del losses
            del loss_ctrl
            del metrics

        for key in detailed_val_losses_epoch.keys():
            detailed_val_losses_epoch[key] /= len(self.dataloader_val)  # 배치 개수로 나눠 평균 계산

        tl_accuracy = (total_tl_correct / total_tl_samples) * 100  ### 추가: 신호등 분류 정확도 계산
        print(f'Loss: {loss_epoch}, {detailed_val_losses_epoch}, Traffic Light Accuracy: {tl_accuracy:.4f}')  ### 추가

        self.val_loss_total.append(loss_epoch)
        self.val_loss_checkpoint.append(detailed_val_losses_epoch['loss_checkpoint'])
        self.val_loss_ctrl.append(detailed_val_losses_epoch['loss_ctrl'])
        self.val_tl_loss.append(detailed_val_losses_epoch['loss_tl'])
        self.val_tl_acc.append(tl_accuracy)

        return loss_epoch, detailed_val_losses_epoch, tl_accuracy

    def save(self):

        model_file = os.path.join(self.args.logdir, f'model_{self.cur_epoch:04d}.pth')
        # optimizer_file = os.path.join(self.args.logdir, f'optimizer_{self.cur_epoch:04d}.pth')
        # scaler_file = os.path.join(self.args.logdir, f'scaler_{self.cur_epoch:04d}.pth')
        # scheduler_file = os.path.join(self.args.logdir, f'scheduler_{self.cur_epoch:04d}.pth')

        # The parallel weights are named differently with the module.
        # We remove that, so that we can load the model with the same code.
        torch.save(self.model.state_dict(), model_file)
        # torch.save(self.optimizer.state_dict(), optimizer_file)
        # torch.save(self.scaler.state_dict(), scaler_file)
        # torch.save(self.scheduler.state_dict(), scheduler_file)

        # Remove last epochs files to avoid accumulating storage
        # if self.cur_epoch > 0:
        #   last_model_file = os.path.join(self.args.logdir, f'model_{self.cur_epoch - 1:04d}.pth')
        #   last_optimizer_file = os.path.join(self.args.logdir, f'optimizer_{self.cur_epoch - 1:04d}.pth')
        #   last_scaler_file = os.path.join(self.args.logdir, f'scaler_{self.cur_epoch - 1:04d}.pth')
        #   last_scheduler_file = os.path.join(self.args.logdir, f'scheduler_{self.cur_epoch - 1:04d}.pth')
        #   if os.path.isfile(last_model_file):
        #     os.remove(last_model_file)
        #   if os.path.isfile(last_optimizer_file):
        #     os.remove(last_optimizer_file)
        #   if os.path.isfile(last_scaler_file):
        #     os.remove(last_scaler_file)
        #   if os.path.isfile(last_scheduler_file):
        #     os.remove(last_scheduler_file)


# We need to seed the workers individually otherwise random processes in the
# dataloader return the same values across workers!
def seed_worker(worker_id):  # pylint: disable=locally-disabled, unused-argument
    # Torch initial seed is properly set across the different workers, we need to pass it to numpy and random.
    worker_seed = (torch.initial_seed()) % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


if __name__ == '__main__':
    # Select how the threads in the data loader are spawned
    available_start_methods = mp.get_all_start_methods()
    if 'fork' in available_start_methods:
        mp.set_start_method('fork')
    # Available on all OS.
    elif 'spawn' in available_start_methods:
        mp.set_start_method('spawn')
    elif 'forkserver' in available_start_methods:
        mp.set_start_method('forkserver')
    print('Start method of multiprocessing:', mp.get_start_method())

    main()
