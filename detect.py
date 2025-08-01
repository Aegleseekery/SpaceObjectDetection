import numpy as np
import cv2
import tifffile
import os
import sep
from astropy.io import fits
from skimage.measure import label, regionprops
from typing import List, Tuple, Dict, Any, Optional


def load_image(image_path: str) -> np.ndarray:
    """
    支持 fits, tif/tiff, png/jpg/jpeg/bmp。
    返回单通道 float64 灰度图，NaN/inf 置为 0。
    """
    ext = os.path.splitext(image_path)[1].lower()

    if ext in [".fits", ".fit", ".fts"]:
        with fits.open(image_path, memmap=False) as hdul:
            data = hdul[0].data
            if data is None:
                raise ValueError(f"FITS 文件 {image_path} 没有数据")
            if data.ndim > 2:
                data = data[0]
            image = np.array(data, dtype=np.float64)
    elif ext in [".tif", ".tiff"]:
        image = tifffile.imread(image_path)
        if image.ndim == 3:
            # 简单转灰度（加权）
            img = image[..., :3].astype(np.float64)
            image = 0.21 * img[..., 0] + 0.72 * img[..., 1] + 0.07 * img[..., 2]
        image = image.astype(np.float64)
    elif ext in [".png", ".jpg", ".jpeg", ".bmp"]:
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.astype(np.float64)
    else:
        # 通用尝试用 OpenCV
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"不支持的图像格式或无法加载: {image_path}")
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.astype(np.float64)

    # 清理 NaN / inf
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    return image

class Detector:
    def __init__(self,
                 method:str,
                 image_dir:str,
                 save_dir: str,
                 vis_dir:str,
                 min_objsize=10,
                 max_objsize=1000,
                 wh_ratio=5) -> None:

        self.method = method
        self.image_dir = image_dir
        self.save_dir = save_dir
        self.vis_dir = vis_dir
        self.min_objsize = min_objsize
        self.max_objsize = max_objsize
        self.wh_ratio = wh_ratio

    def _run(self):
        """
        检测的核心函数
        :return: 粗检测结果
        """
        pass
    def _parse(self):
        """
        解析检测结果，并转为标准格式
        :return:
        """

    def detect(self):
        """

        :param image_path: 图像所在目录
        :return: 返回形如[[cx,cy,area,w,h,gray_sum],...]的一个List,在save_dir不为None时,将结果的txt保存在save_dir下，在
        """
        eps = 1e-7
        image= load_image(self.image_dir)
        raw_output = self._run(image)
        results = self._parse(raw_output)

        '滤除异常目标'
        # 滤除大小异常的目标
        results = results[(results[:,-3] * results[:,-2] > self.min_objsize) *
                          (results[:,-2] * results[:,-3]) <self.max_objsize]
        # 去长条
        results = results[(results[:,-3] / (results[:,-2] + eps) < self.wh_ratio) *
                          (results[:,-2] / (results[:,-3] + eps) < self.wh_ratio)]

        print(f"current_method: {self.method}, detected: {len(results)} objects.")


        '可视化并保存'
        if self.vis_dir is not None:
            os.makedirs(self.vis_dir, exist_ok=True)

            # 规范 image 到 0-255 uint8（线性缩放避免溢出）
            img = image.copy()
            if img.dtype != np.uint8:
                mn, mx = np.min(img), np.max(img)
                if mx > mn:
                    img = (img - mn) / (mx - mn) * 255.0
                else:
                    img = np.zeros_like(img)
                img = np.clip(img, 0, 255).astype(np.uint8)
            else:
                img = image.astype(np.uint8)

            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            for cx, cy, area, w, h, *_ in results:
                x, y = int(cx - w / 2), int(cy - h / 2)
                cv2.rectangle(vis, (x, y), (x + int(w), y + int(h)), (0, 0, 255), 1)

            base_name = os.path.splitext(os.path.basename(self.image_dir))[0]
            save_path = os.path.join(self.vis_dir, f"{base_name}_{self.method}.png")
            cv2.imwrite(save_path, vis)

        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
            fname = os.path.splitext(os.path.basename(self.image_dir))[0]
            out_txt = os.path.join(self.save_dir, f"{fname}_{self.method}.txt")
            header = f"# columns: cx, cy, area, w, h, gray_sum\n"
            fmt = "{:10.4f} {:10.4f} {:10.4f} {:10.4f} {:10.4f} {:10.1f}\n"
            with open(out_txt, "w") as f:
                f.write(header)
                for row in results:
                    cx, cy, area, w, h, gray_sum = row
                    f.write(fmt.format(cx, cy, area, w, h, gray_sum))

        return results

class OtsuDetector(Detector):
    def __init__(self, *args, min_area: float = 5.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_area = min_area

    def _run(self, image: np.ndarray) -> Any:
        """
        用大津阈值做二值化，然后用连通区域提取候选。
        返回 (regions, image)，regions 是 regionprops 列表。
        """
        # 1. 线性缩放到 0-255 uint8（保留相对强度）
        img = image.astype(np.float64)
        mn, mx = np.min(img), np.max(img)
        if mx > mn:
            norm = (img - mn) / (mx - mn) * 255.0
        else:
            norm = np.zeros_like(img)
        norm_uint8 = np.clip(norm, 0, 255).astype(np.uint8)

        # 2. Otsu 阈值
        _, bw = cv2.threshold(norm_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 3. 连通区域分析（用原始 image 做 intensity_image）
        labeled = label(bw > 0)
        regions = regionprops(labeled, intensity_image=image)

        return regions, image  # 后面 parse 用

    def _parse(self, raw_output: Any) -> np.ndarray:
        regions, image = raw_output
        result_list = []
        h_img, w_img = image.shape[:2]
        for r in regions:
            cy, cx = r.centroid  # 注意 skimage 顺序 (row, col)
            area = float(r.area)
            minr, minc, maxr, maxc = r.bbox  # (min_row, min_col, max_row, max_col)
            w = float(maxc - minc)
            h = float(maxr - minr)
            if area < self.min_area:
                continue
            gray_sum = float(r.intensity_image.sum())
            result_list.append([cx, cy, area, w, h, gray_sum])

        if not result_list:
            return np.zeros((0, 6), dtype=float)
        return np.array(result_list, dtype=float)

class SExtractor(Detector):
    def __init__(self, *args, thresh: float = 1.5, min_area: float = 5.0, **kwargs):
        """
        :param thresh: 以背景 rms 的倍数作为阈值（类似 SExtractor 的 DETECT_THRESH）
        :param min_area: 过滤面积过小的（以 bbox 面积）
        """
        super().__init__(*args, **kwargs)
        self.thresh = thresh
        self.min_area = min_area

    def _run(self, image: np.ndarray) -> Any:
        """
        用 sep 做背景建模和提取，返回 objects 结构数组
        """
        # sep 需要 C-contiguous float32/float64
        data = image.astype(np.float64)
        # build background
        bkg = sep.Background(data)
        data_sub = data - bkg
        # 提取对象，阈值是背景 rms * thresh
        objects = sep.extract(data_sub, self.thresh, err=bkg.globalrms)
        return objects

    def _parse(self, raw_output: Any) -> List[List[float]]:
        """
        把 sep 输出转成 [cx, cy, area, w, h, gray_sum]
        这里近似用 2*a, 2*b 作为 bbox 宽高（不考虑旋转），area=w*h。
        gray_sum 取 bbox 内原图求和。
        """
        se_dets = np.array([[i[7],
                             i[8],
                             i[1],
                             i[4]-i[3],
                             i[6]-i[5],
                             i[-8]] for i in raw_output])
        return se_dets


if __name__ == '__main__':
    # det = OtsuDetector(
    #     method="otsu",
    #     image_dir="test_img/00001.tif",
    #     save_dir="results/",
    #     vis_dir="vis/",
    # )
    det = SExtractor(
        method="sep",
        image_dir="test_img/00001.tif",
        save_dir="results/",
        vis_dir="vis/",
        thresh=1.5
    )

    results = det.detect()


