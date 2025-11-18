import glob, os, torch
import tyro
from depth_anything_3.api import DepthAnything3

from webpolicy.deploy.server.policy_base import PolicyBase
from webpolicy.deploy.server.server import Server

from dataclasses import dataclass

@dataclass
class Config:
    host: str 
    port: int = 8080

class DA3Policy(PolicyBase):
    def __init__(self):
        self.device = torch.device("cuda")
        self.model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
        self.model = self.model.to(device=self.device)

    def infer(self, example_path):
        images = sorted(glob.glob(os.path.join(example_path, "*.png")))
        prediction = self.model.inference(
            images,
        )
        return prediction

def main(cfg:Config):

    policy = DA3Policy()
    server = Server(policy, cfg.host, cfg.port)

    example_path = "assets/examples/SOH"
    images = sorted(glob.glob(os.path.join(example_path, "*.png")))
    prediction = model.inference(
        images,
    )
# prediction.processed_images : [N, H, W, 3] uint8   array
    print(prediction.processed_images.shape)
# prediction.depth            : [N, H, W]    float32 array
    print(prediction.depth.shape)  
# prediction.conf             : [N, H, W]    float32 array
    print(prediction.conf.shape)  
# prediction.extrinsics       : [N, 3, 4]    float32 array # opencv w2c or colmap format
    print(prediction.extrinsics.shape)
# prediction.intrinsics       : [N, 3, 3]    float32 array
    print(prediction.intrinsics.shape)

if __name__ == "__main__":
    main(tyro.cli(Config))
