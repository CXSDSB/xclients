from __future__ import annotations

import logging

import cv2
import numpy as np
import tyro
from rich import print
from webpolicy.client import Client

from .common import Config, spec


def main(cfg: Config) -> None:
    client = Client(cfg.host, cfg.port)
    cap = cv2.VideoCapture(0)

    logging.info("Displaying frames from camera 0; press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        print(frame.shape)
        payload = {"image": [frame]}
        out = client.step(payload)
        if not out:
            logging.error("Failed to read frame from camera 0")
            continue

        print(spec(out))
        d = out["depth"][0]
        d = ((d - np.min(d)) / (np.max(d) - np.min(d)) * 255.0).astype(np.uint8)
        d = 255 - d

        print(out["extrinsics"])
        print(out["intrinsics"])

        if cfg.show:
            cv2.imshow("Camera 0", d)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Config))
