import yaml
from pathlib import Path

config_path = Path(__file__).parent / 'config_616.yaml'
cfg = yaml.safe_load(open(config_path))
prof = cfg['profiles'][cfg['active_profile']]

print(f"Active profile: {cfg['active_profile'].upper()}")
print(f"Screen capture: {prof['screen']['capture_resolution']} @ {prof['screen']['capture_fps']} FPS")
print(f"YOLO input: {prof['yolo']['input_resolution']}")
print(f"YOLO skip_frames: {prof['yolo']['skip_frames']}")
print(f"YOLO enabled: {prof['yolo']['enabled']}")
