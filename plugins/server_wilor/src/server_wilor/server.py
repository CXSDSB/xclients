import os
import sys
import torch
import cv2
import numpy as np
from pathlib import Path

from dataclasses import dataclass

@dataclass
class WilorConfig:
    image: str = None


PLUGIN_DIR = Path(__file__).parent
ROOT_DIR   = (PLUGIN_DIR / "../../../../").resolve()         # xclients/
WILOR_ROOT = (ROOT_DIR / "external/wilor").resolve()         # xclients/external/wilor
sys.path.append(str(WILOR_ROOT))

WEBPOLICY_SRC = (PLUGIN_DIR / "../../external/webpolicy/src").resolve()
sys.path.append(str(WEBPOLICY_SRC))

from webpolicy.base_policy import BasePolicy   

from wilor.models import WiLoR, load_wilor
from wilor.utils import recursive_to
from wilor.datasets.vitdet_dataset import ViTDetDataset
from wilor.utils.renderer import Renderer, cam_crop_to_full
from ultralytics import YOLO
from .dummy import RealWiLoR, RealYOLO, real_cfg
LIGHT_PURPLE=(0.25098039,  0.274117647,  0.65882353)

class WilorModel(BasePolicy):
    def __init__(self):
        print(" Initializing WiLoR server...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ckpt = WILOR_ROOT / "pretrained_models" / "wilor_final.ckpt"
        cfg  = WILOR_ROOT / "pretrained_models" / "model_config.yaml"
        det  = WILOR_ROOT / "pretrained_models" / "detector.pt"

        self.model = RealWiLoR(ckpt, cfg, self.device)
        self.cfg = real_cfg(cfg)
        self.detector = RealYOLO(det, self.device)


    # def load_wilor_model(self, ckpt_path, cfg_path, device):
    #     model, cfg = load_wilor(ckpt_path, cfg_path)
    #     return model.to(device), cfg


    # # need path of modle
    # def load_detector(self, detector_path, device):
    #     det = YOLO(detector_path)
    #     return det.to(device)
    

    # def load_renderer(self, model_cfg, model.mano.faces)


    def detect_hands(self, image):
        detections = self.detector(image)[0]
        boxes = []
        is_right = []   
        for det in detections:
            box = det.boxes.data.cpu().numpy().squeeze()[:4]
            hand_type = int(det.boxes.cls.cpu().item())  # 0=left, 1=right
            boxes.append(box)
            is_right.append(hand_type)
        return np.array(boxes), np.array(is_right)

    def preprocess(self, image, boxes, is_right, rescale_factor):
        dataset = ViTDetDataset(
            self.cfg,
            image,
            boxes,
            is_right,
            rescale_factor=rescale_factor
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=16,
            shuffle=False,
            num_workers=0
        )
        return dataloader


    def infer(self, batch):
        # move to device
        batch = recursive_to(batch, self.device)
        # forward pass
        with torch.no_grad():
            out = self.model(batch)
            print("infer output keys:", out.keys())
        return out   

    def postprocess(self, out, batch):

        # make multiplier follow model device (GPU/CPU)
        multiplier = (2 * batch['right'] - 1).to(out['pred_cam'].device)

        pred_cam = out['pred_cam']
        pred_cam[:, 1] = multiplier * pred_cam[:, 1]

        box_center = batch["box_center"].float().to(pred_cam.device)
        box_size   = batch["box_size"].float().to(pred_cam.device)
        img_size   = batch["img_size"].float().to(pred_cam.device)

        scaled_focal_length = (
            self.cfg.EXTRA.FOCAL_LENGTH
            / self.cfg.MODEL.IMAGE_SIZE
            * img_size.max()
        )

        pred_cam_t_full = cam_crop_to_full(
            pred_cam,
            box_center,
            box_size,
            img_size,
            scaled_focal_length
        ).detach().cpu().numpy()

        all_verts = []
        all_cam_t = []
        all_right = []

        batch_size = batch["img"].shape[0]

        for n in range(batch_size):
            verts = out['pred_vertices'][n].detach().cpu().numpy()

            is_right = batch['right'][n].cpu().numpy()
            verts[:, 0] = (2 * is_right - 1) * verts[:, 0]

            cam_t = pred_cam_t_full[n]

            all_verts.append(verts)
            all_cam_t.append(cam_t)
            all_right.append(is_right)

        return all_verts, all_cam_t, all_right

    def step(self, payload: dict) -> dict:
        image = payload["image"]

        # detect
        boxes, is_right = self.detect_hands(image)
        if len(boxes) == 0:
            return {"status": "ok", "hands": 0}

        # preprocess
        dataloader = self.preprocess(image, boxes, is_right, rescale_factor=2.0)

        # process all batches
        verts_list, cams_list, right_list = [], [], []

        for batch in dataloader:
            out = self.infer(batch)
            verts, cams, rights = self.postprocess(out, batch)
            verts_list.extend(verts)
            cams_list.extend(cams)
            right_list.extend(rights)

        # return final structured output
        return {
            "status": "ok",
            "verts": verts_list,
            "cams": cams_list,
            "right": right_list
        }

def main(cfg: WilorConfig):
    model = WilorModel()
    image = cv2.imread(cfg.image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = model.step({"image": image})
    print("CLI Result:", result)

if __name__ == "__main__":
    cfg = tyro.cli(WilorConfig)
    main(cfg)