import torchvision.models as models
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math
      
class LidarCenterNet(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config

    # ================== (A) RGB ResNet + Conv => d_model ==================
    self.rgb_resnet = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-2])
    self.rgb_change = nn.Conv2d(512, self.config.gru_input_size, kernel_size=1)
    self.rgb_pos_embed = PositionEmbeddingSine(self.config.gru_input_size // 2, normalize=True) #!!!
    
    # ================== (A-1) "이미지 피처"에 Self-Attention  ==================
    self.img_self_attn = nn.MultiheadAttention(self.config.gru_input_size, #256
                                               self.config.num_decoder_heads, #8
                                               batch_first=True)

    # ================== (B) Global path enc ==================
    self.path_enc = nn.Linear(2, 256)
    self.path_pos = nn.Parameter(torch.randn(1, 100, 256) * 0.02) #!!!
    
    # (C) Speed token encoding (single token)
    self.speed_norm = nn.LayerNorm(1)  # scalar normalization
    self.speed_encoder = nn.Sequential(nn.Linear(1, 128), nn.ReLU(inplace=True),
                                        nn.Linear(128, config.gru_input_size), nn.ReLU(inplace=True))
    
    # (C‑1) Positional embedding for KV sequence (101 tokens: 1 speed + 100 path)
    self.kv_pos = nn.Parameter(torch.randn(1, 101, config.gru_input_size) * 0.02) #!!!
    
    # ================== () img, path Attention ==================
    self.cross_attn = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)
    self.cross_norm = nn.LayerNorm(256)
    
    # ================== (D) Transformer Decoder ==================
    decoder_norm = nn.LayerNorm(self.config.gru_input_size)
    decoder_layer = nn.TransformerDecoderLayer(self.config.gru_input_size,
                                                self.config.num_decoder_heads,
                                                activation=nn.GELU(),
                                                batch_first=True)
    self.decoder = torch.nn.TransformerDecoder(decoder_layer,
                                            num_layers=self.config.num_transformer_decoder_layers,
                                            norm=decoder_norm)
    
    # (E‑1) Query content & positional embedding (traj_len = 100)
    self.query_content = nn.Parameter(torch.randn(1, config.predict_checkpoint_len, config.gru_input_size) * 0.02) #!!!
    self.query_pos = nn.Parameter(torch.randn(1, config.predict_checkpoint_len, config.gru_input_size) * 0.02) #!!!

    # ================== (E) GRU ==================
    self.checkpoint_net = GRUWaypointsPredictorAR(input_dim=self.config.gru_input_size,
                                                  hidden_size=self.config.gru_hidden_size,
                                                  steps=self.config.predict_checkpoint_len)
    
    # ================== (F) Control Net ==================
    self.control_net = control_net()
    
    # ================== () TL Classifier ==================
    self.pool = nn.AdaptiveAvgPool2d((1, 1))
    self.traffic_feature_extractor = nn.Sequential(
        nn.Linear(self.config.gru_input_size, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.1),
    )
    # 최종 TL classifier: traffic_feature_extractor의 출력을 받아 3 클래스 분류
    self.traffic_classifier = nn.Linear(128, 10)
    
    
    #* Loss definition
    self.loss_ctrl = nn.MSELoss()
    self.loss_tl = nn.CrossEntropyLoss()


  def forward(self, rgb, ego_vel, target_point, path):
    bs = rgb.shape[0]

    # (A) Image features + 2D PE
    img_feat = self.rgb_resnet(rgb) # [B, 512, 12, 20]
    img_feat_ = self.rgb_change(img_feat) # => [B, 256, 12, 20]
    img_feat = img_feat_ + self.rgb_pos_embed(img_feat_)

    # (A-1) Self-attention over flattened patches
    img_feat_flat = img_feat.flatten(2).transpose(1, 2)      # [B,240,256]
    img_feat_attn, _ = self.img_self_attn(img_feat_flat, img_feat_flat, img_feat_flat)

    # (B) Encode path + own PE
    path_feat = self.path_enc(path) + self.path_pos          # [B,100,256]

    # (C) Speed token
    speed = self.speed_norm(ego_vel)                         # [B,1]
    speed_tok = self.speed_encoder(speed)                    # [B,256]
    speed_tok = speed_tok.unsqueeze(1)                       # [B,1,256]

    # (D) Cross-attention: path Q, image K/V
    cross_out, _ = self.cross_attn(path_feat, img_feat_attn, img_feat_attn)  # [B,100,256]
    path_feat = self.cross_norm(path_feat + cross_out)       # residual + norm

    # (E) Build KV sequence (speed + path) and add single PE (once)
    kv = torch.cat([speed_tok, path_feat], dim=1) + self.kv_pos  # [B,101,256]
    # (F) Build Q (content + pos)
    q = (self.query_content + self.query_pos).repeat(bs, 1, 1)    # [B,100,256]
    
    # (G) Transformer decoder
    dec_out = self.decoder(tgt=q, memory=kv)               # [B,100,256]

    # (H) GRU waypoint predictor
    pred_checkpoint = None
    pred_checkpoint, hidden_seq = self.checkpoint_net(dec_out, target_point)
    
    # (I) Control prediction
    pred_ctrl = self.control_net(hidden_seq)
    
    # (J) Traffic light classification (global image context)
    pooled = self.pool(img_feat).view(bs, -1)               # [B,256]
    tl_feat = self.traffic_feature_extractor(pooled)
    traffic_logits = self.traffic_classifier(tl_feat)

    return pred_ctrl, pred_checkpoint, traffic_logits


  def compute_loss(self, pred_ctrl, pred_checkpoint, pred_tl, ctrl_label, checkpoint_label, tl_label, obstacles=None):
    loss = {}
    loss_ctrl = {}

    loss_control = self.loss_ctrl(pred_ctrl, ctrl_label) * 10
    loss_ctrl.update({'loss_ctrl': loss_control})
    
    # diff = pred_checkpoint - checkpoint_label               # shape: [B, T, 2]
    # dist = torch.norm(diff, dim=2)                         # 각 waypoint마다의 L2 distance, shape: [B, T]
    # ade = torch.mean(dist)                                 # 배치+전체 timestep 평균
    # loss.update({'loss_checkpoint': ade})
    
    # ---- 기존 L2 dist 대신 Weighted L2 dist ----
    diff = pred_checkpoint - checkpoint_label       # shape: [B, T, 2]
    alpha = 2.0  # y축 가중치 (원하는 값으로 설정)
    dx = diff[..., 0]   # (B, T)
    dy = diff[..., 1]
    # sqrt( dx^2 + (alpha * dy)^2 )
    dist = torch.sqrt(dx**2 + (alpha * dy)**2)      # shape: [B, T]
    ade = torch.mean(dist)                          # (batch+timesteps 평균)
    loss.update({'loss_checkpoint': ade})
    
    loss_tl = self.loss_tl(pred_tl, tl_label)
    loss.update({'loss_tl': loss_tl})
      
    # # 추가: Collision Loss 계산 (예측 웨이포인트와 장애물 간의 충돌 패널티)
    # if obstacles is not None:
    #   # pred_checkpoint: [B, T, 2], obstacles: [B, N, 3]
    #   margin = 1.0  # 안전 margin (하이퍼파라미터, config로 설정 가능)
    #   B, T, _ = pred_checkpoint.shape
    #   collision_loss = 0.0
    #   for b in range(B):
    #     # 만약 해당 배치에 장애물이 하나도 없으면 패널티 0
    #     if obstacles[b].shape[0] == 0:
    #       continue
    #     # predicted waypoints [T,2]와 obstacles centers [N,2]
    #     wp = pred_checkpoint[b]  # [T,2]
    #     obs_centers = obstacles[b][:, :2]  # [N,2]
    #     obs_radii = obstacles[b][:, 2].unsqueeze(0)  # [1, N]
    #     # wp: [T, 2] → expand to [T, N, 2]
    #     wp_exp = wp.unsqueeze(1)  # [T,1,2]
    #     # obs_centers: [N,2] → expand to [1, N,2]
    #     obs_exp = obs_centers.unsqueeze(0)  # [1,N,2]
    #     d = torch.norm(wp_exp - obs_exp, dim=2)  # [T, N]
    #     safe_distance = obs_radii + margin  # [1, N]
    #     diff_collision = F.relu(safe_distance - d)  # [T, N]
    #     collision_loss += torch.mean(diff_collision**2)
    #   collision_loss = collision_loss / B
    #   loss.update({'loss_collision': collision_loss})

    return loss, loss_ctrl

class GRUWaypointsPredictorAR(nn.Module):
    """
    Autoregressive GRU:
      - input_dim = feature dimension per step (예: 256)
      - hidden_size = GRU hidden
      - steps = 예측할 waypoint 수 (T)
    """
    def __init__(self, input_dim, hidden_size, steps, learnable_init=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.steps = steps

        # GRU는 (x_in + y_prev + maybe target_point)
        in_dim = input_dim + 2 # 3= (x,y) 이전 waypoint + velocity   &   32= target point

        # self.tp_enc = nn.Linear(2, hidden_size)
        self.gru = nn.GRU(input_size=in_dim, hidden_size=hidden_size, batch_first=True)
        self.delta_fc = nn.Linear(hidden_size, 2)  # 다음 waypoint로의 delta

        # h₀ 초기화 방식
        if learnable_init:
            self.h0 = nn.Parameter(torch.zeros(1, 1, hidden_size))
            nn.init.normal_(self.h0, std=0.02)
        else:
            self.register_buffer("h0", torch.zeros(1, 1, hidden_size),
                                 persistent=False)
            
        
    def forward(self, features, target_point):
        """
        Args:
          features: shape (B, steps, input_dim) - 각 timestep별 feature
        Returns:
          pred_wp: (B, steps, 2) - 누적 (auto-regressive) waypoint
          hidden_seq: (B, steps, hidden_size) - 각 timestep의 hidden (Control Net용)
        """
        B, T, _ = features.shape
        assert T == self.steps, f"features time dim {T} != steps {self.steps}"

        # hidden = self.tp_enc(target_point).unsqueeze(0)

        # initial hidden state (learnable or zeros)
        hidden = self.h0.expand(-1, B, -1).contiguous()  # (1,B,H)

        # auto-reg loop
        y_prev = torch.zeros(B, 2, device=features.device)  # (x0,y0)= (0,0)
        traj = []
        all_hidden = []


        for t in range(T):
            x_t = features[:, t, :]  # (B, input_dim)
            x_in = torch.cat([x_t, y_prev], dim=1)

            x_in = x_in.unsqueeze(1)  # GRU(batch_first=True) expects (B, seq=1, in_dim)
            out, hidden = self.gru(x_in, hidden)  # out: (B,1,hid), hidden: (1,B,hid)
            out = F.dropout(out, p=0.2, training=self.training)  # 추가

            # Δwp
            delta_xy = self.delta_fc(out.squeeze(1))  # shape (B,2)
            y_curr = y_prev + delta_xy
            traj.append(y_curr)
            y_prev = y_curr

            all_hidden.append(hidden.squeeze(0))  # shape (B,hid)

        pred_wp = torch.stack(traj, dim=1)         # (B,T,2)
        hidden_seq = torch.stack(all_hidden, dim=1)  # (B,T,hid)
        return pred_wp, hidden_seq
      
      
## 단순 mlp -> 1D CNN + mlp

class control_net(nn.Module):
    def __init__(self):
        super(control_net, self).__init__()
        
        # ----- (1) 1D Conv for the GRU hidden_seq -----
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # => (B,256,1)
        )
        
        # ----- (3) 최종 MLP -----
        #  최종 concat 시 => 256( from CNN ) + 64( from sensor_embed ) = 320
        combined_dim = 256 # 256 + 128
        self.fc = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # steer, accel
        )

    def forward(self, hidden_seq):
        """
        hidden_seq: shape (B, T, 128)
        """
        
        # 1) 1D Conv for GRU output
        #    Permute => (B, 128, T)
        x = hidden_seq.permute(0, 2, 1)
        x = self.cnn(x)             # => (B, 256, 1)
        x = x.squeeze(-1)          # => (B, 256)

        # Concatenate control net features with traffic feature [B, 128]
        # combined = torch.cat([x, traffic_feature], dim=1)  # [B, 256+128]
        pred_ctrl = self.fc(x)       # => (B,2)
        return pred_ctrl
      
class PositionEmbeddingSine(nn.Module):
  """
  Taken from InterFuser
  This is a more standard version of the position embedding, very similar to the one
  used by the Attention is all you need paper, generalized to work on images.
  """

  def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
    super().__init__()
    self.num_pos_feats = num_pos_feats
    self.temperature = temperature
    self.normalize = normalize
    if scale is not None and normalize is False:
      raise ValueError('normalize should be True if scale is passed')
    if scale is None:
      scale = 2 * math.pi
    self.scale = scale

  def forward(self, tensor):
    x = tensor
    bs, _, h, w = x.shape
    not_mask = torch.ones((bs, h, w), device=x.device)
    y_embed = not_mask.cumsum(1, dtype=torch.float32)
    x_embed = not_mask.cumsum(2, dtype=torch.float32)
    if self.normalize:
      eps = 1e-6
      y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
      x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

    dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
    dim_t = self.temperature**(2 * (torch.div(dim_t, 2, rounding_mode='floor')) / self.num_pos_feats)

    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
    return pos
