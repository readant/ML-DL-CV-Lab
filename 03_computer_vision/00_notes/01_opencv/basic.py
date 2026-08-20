import cv2
from pathlib import Path

# 通过 __file__ 上溯定位仓库根，不依赖当前工作目录（CWD）
# basic.py 位于 03_computer_vision/00_notes/01_opencv/，上溯 3 层到仓库根
PROJECT_ROOT = Path(__file__).resolve().parents[3]
IMG_PATH = PROJECT_ROOT / "data" / "images" / "test" / "logo.png"

# 读取图片，支持 jpg/png
# cv2.imread(路径, 模式) 模式：1彩色，0灰度，-1带透明通道
img = cv2.imread(str(IMG_PATH), 1)

# 判断图片是否读取成功
if img is None:
    print("图片读取失败，请检查文件路径！")
else:
    # 显示窗口，窗口名随便写
    cv2.imshow("My Image", img)

    # 保存图片
    cv2.imwrite(str(PROJECT_ROOT / "data" / "images" / "test" / "output.jpg"), img)

    # 等待按键，0代表无限等待，单位毫秒
    cv2.waitKey(0)
    # 销毁全部窗口，必须写，不然窗口卡死
    cv2.destroyAllWindows()
