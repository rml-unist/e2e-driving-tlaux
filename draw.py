import numpy as np
import matplotlib.pyplot as plt
import torch


def transform_to_bev_coordinates(checkpoint, pixels_per_meter=300/45, image_center=(150,300)):
  bev_coords = []
  for x, y in checkpoint:
    pixel_x = float(x * pixels_per_meter + image_center[0])
    pixel_y = float(-y * pixels_per_meter + image_center[1])
    bev_coords.append((pixel_x, pixel_y))

  return bev_coords

def transform_to_bev_coordinates_vertical(xy_local_list, pixels_per_meter=6.67, image_center=(150,300)): #!!
  # 이미지 위에서 아래 방향으로 waypoint 나오도록 계산
  bev_coords = []
  for x, y in xy_local_list:
    pixel_x = float(-y * pixels_per_meter + image_center[0]) #-y
    pixel_y = float(-x * pixels_per_meter + image_center[1]) #-x
    bev_coords.append((pixel_x, pixel_y)) 
    
  return bev_coords

class SegmentationVisualizer:
  def __init__(self, class_mapping):
    self.class_mapping = class_mapping
    self.reverse_mapping = {k: v for k, v in class_mapping.items()}  # BGR to Class ID

  def on_click(self, event, seg):
    if event.inaxes is not None:
      x, y = int(event.xdata), int(event.ydata)
      pixel_value = tuple(seg[y, x])  # (B, G, R)
      class_id = self.reverse_mapping.get(pixel_value, "Unknown")
      print(f"Clicked Pixel: ({x}, {y}), BGR: {pixel_value}, Class ID: {class_id}")

  def visualize_with_click(self, seg):
    seg = np.array(seg)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(seg)
    ax.axis("off")
    ax.set_title("Click on the Image to Get Class ID")

    # Connect the click event
    fig.canvas.mpl_connect('button_press_event', lambda event: self.on_click(event, seg[..., ::-1]))

    plt.show()
    
    
import torch
import numpy as np
import matplotlib
from PIL import Image

def depth_to_colormapped_image(depth_tensor, colormap_name="magma_r"):
    """
    depth_tensor: shape (1,H,W) or (H,W), float
    colormap_name: e.g. "magma_r", "viridis", "plasma", ...
    return: PIL Image (RGB)
    """
    # 1) (1,H,W) => (H,W)
    if len(depth_tensor.shape) == 3 and depth_tensor.shape[0] == 1:
        depth_tensor = depth_tensor.squeeze(0)  # => (H,W)

    # 2) CPU로 복사 & numpy로 변환
    depth_array = depth_tensor.detach().cpu().numpy()  # shape (H,W), float

    # 3) min-max 정규화
    d_min, d_max = depth_array.min(), depth_array.max()
    # 만약 특정 범위로 클리핑/정규화하고 싶다면:
    # d_min, d_max = 0.0, 80.0  # 예시
    # depth_array = np.clip(depth_array, d_min, d_max)

    denom = max(d_max - d_min, 1e-8)
    normalized = (depth_array - d_min) / denom

    # 4) matplotlib 컬러맵 적용 => RGBA 배열 (H,W,4)
    colormap = matplotlib.colormaps[colormap_name]
    color_rgba = colormap(normalized, bytes=True)  # => uint8
    # shape: (H,W,4), each pixel = (R,G,B,A)

    # 5) RGB만 추출 => PIL Image 변환
    color_rgb = color_rgba[..., :3]  # (H,W,3)
    pil_img = Image.fromarray(color_rgb)  # => PIL Image (RGB)
    return pil_img

# 사용 예시
# depth_tensor: (1,H,W)
# depth_viz_img = depth_to_colormapped_image(depth_tensor, "magma_r")
# depth_viz_img.show()  # or depth_viz_img.save("depth_viz.png")
