import sys
from pathlib import Path

# Ensure external/wilor is added early so wilor package imports succeed
PLUGIN_DIR = Path(__file__).parent
# WILOR_ROOT = (PLUGIN_DIR / "../../external/wilor").resolve()
# sys.path.append(str(WILOR_ROOT))
from plugins.server_wilor.src.server_wilor.server import WilorModel
from dataclasses import dataclass
import tyro


@dataclass
class WilorConfig:
    # 未来可以添加更多参数，比如模型路径、是否启用GPU等
    pass


def load(cfg: WilorConfig = None):
    """Factory function for xclients to load the policy."""
    # cfg 未来可以用于配置模型，现在先保留接口
    print("[main] Loading WilorModel...")
    return WilorModel()


def main(cfg: WilorConfig):
    """Standalone debug mode."""
    print("[main] Running WilorModel standalone...")
    model = load(cfg)

    # TODO：这里可以加入你自己的调试图片路径
    # image = cv2.imread("test.jpg")
    # result = model.step(image)
    # print(result)

    print("[main] WilorModel loaded successfully (debug mode).")


if __name__ == "__main__":
    main(tyro.cli(WilorConfig))