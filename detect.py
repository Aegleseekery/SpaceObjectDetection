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
    Support fits, tif/tiff, png/jpg/jpeg/bmp.
    Return single-channel float64 grayscale image, with NaN/inf replaced by 0.
    """
    ext = os.path.splitext(image_path)[1].lower()

    if ext in [".fits", ".fit", ".fts"]:
        with fits.open(image_path, memmap=False) as hdul:
            data = hdul[0].data
            if data is None:
                raise ValueError(f"FITS File {image_path} not found.")
            if data.ndim > 2:
                data = data[0]
            image = np.array(data, dtype=np.float64)
    elif ext in [".tif", ".tiff"]:
        image = tifffile.imread(image_path)
        if image.ndim == 3:
            # Simple conversion to grayscale (weighted)
            img = image[..., :3].astype(np.float64)
            image = 0.21 * img[..., 0] + 0.72 * img[..., 1] + 0.07 * img[..., 2]
        image = image.astype(np.float64)
    elif ext in [".png", ".jpg", ".jpeg", ".bmp"]:
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.astype(np.float64)
    else:
        # Generic attempt using OpenCV
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"unsupported type of image: {image_path}")
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.astype(np.float64)

    # Clean NaN / inf values
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    return image


class Detector:
    def __init__(self,
                 method: str,
                 image_dir: str,
                 save_dir: str,
                 vis_dir: str,
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
        Core detection logic.
        :return: detections in standard format
        """
        pass

    def detect(self):
        """
        :return: List of [cx, cy, area, w, h, gray_sum]; if save_dir is not None, saves a txt file there
        """
        eps = 1e-7
        image = load_image(self.image_dir)
        dets = self._run(image)

        # === filter out abnormal objects ===
        # filter by size constraints
        dets = dets[(dets[:, -3] * dets[:, -2] > self.min_objsize) *
                    (dets[:, -2] * dets[:, -3] < self.max_objsize)]
        # remove elongated shapes
        dets = dets[(dets[:, -3] / (dets[:, -2] + eps) < self.wh_ratio) *
                    (dets[:, -2] / (dets[:, -3] + eps) < self.wh_ratio)]

        print(f"current_method: {self.method}, detected: {len(dets)} objects.")

        # === visualize and save ===
        if self.vis_dir is not None:
            os.makedirs(self.vis_dir, exist_ok=True)
            # normalize image to 0-255 uint8 (linear scaling to avoid overflow)
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
            for cx, cy, area, w, h, *_ in dets:
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
                for row in dets:
                    cx, cy, area, w, h, gray_sum = row
                    f.write(fmt.format(cx, cy, area, w, h, gray_sum))

        return dets


class OtsuDetector(Detector):
    def __init__(self, *args, min_area: float = 5.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_area = min_area

    def _run(self, image: np.ndarray) -> Any:
        """
        Apply Otsu thresholding and extract connected components.
        Returns (regions, image), where regions is a list of regionprops.
        """
        # 1. Linear scale to 0-255 uint8 (preserve relative intensity)
        mn, mx = np.min(image), np.max(image)
        if mx > mn:
            norm = (image - mn) / (mx - mn) * 255.0
        else:
            norm = np.zeros_like(image)
        norm_uint8 = np.clip(norm, 0, 255).astype(np.uint8)

        # 2. Otsu thresholding
        _, bw = cv2.threshold(norm_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 3. Connected component analysis (use original image as intensity image)
        labeled = label(bw > 0)
        regions = regionprops(labeled, intensity_image=image)

        result_list = []
        for r in regions:
            cy, cx = r.centroid  # skimage order: (row, col)
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
    def __init__(self, *args, thresh: float = 1.5, **kwargs):
        """
        Source extraction using sep (analogous to SExtractor).
        :param thresh: threshold multiplier of background RMS (similar to DETECT_THRESH)
        """
        super().__init__(*args, **kwargs)
        self.thresh = thresh

    def _run(self, image: np.ndarray) -> Any:
        """
        Build background model and extract sources using sep.
        """
        bkg = sep.Background(image)
        data_sub = image - bkg
        # Extract objects with threshold = background RMS * thresh
        objects = sep.extract(data_sub, self.thresh, err=bkg.globalrms)
        se_dets = np.array([[i[7],
                             i[8],
                             i[1],
                             i[4] - i[3],
                             i[6] - i[5],
                             i[-8]] for i in objects])
        return se_dets


class PoissonThresholding(Detector):
    def __init__(self, *args, binNums=256, criterion='gaussian', **kwargs):
        """
        An Improved Automatic Detection and Segmentation of Cell Nuclei in Histopathology Images
        https://ieeexplore.ieee.org/document/5306149
        :param binNums: number of histogram bins
        :param criterion: error criterion, default is gaussian
        """
        super().__init__(*args, **kwargs)
        self.binNums = binNums
        self.criterion = criterion

    def minerrthresh(self, image: np.ndarray):
        """
        Compute minimum error threshold using specified statistical model.
        """
        img_min, img_max = np.min(image), np.max(image)

        binMultiplier = self.binNums / (img_max - img_min)

        hist, _ = np.histogram(image, bins=self.binNums, range=(img_min, img_max))
        hist = hist / np.sum(hist)

        total_mean = np.sum(np.arange(1, self.binNums + 1) * hist)  # indices start from 1

        error_func = np.zeros(self.binNums)

        # iterate over candidate thresholds
        for i in range(2, self.binNums - 1):  # MATLAB range 2 to binNums-1
            prior_left = np.sum(hist[:i]) + np.finfo(float).eps
            prior_right = np.sum(hist[i:]) + np.finfo(float).eps

            mean_left = np.sum((np.arange(0, i) * hist[:i])) / prior_left
            mean_right = np.sum((np.arange(i, self.binNums) * hist[i:])) / prior_right

            if self.criterion == 'gaussian':
                # Gaussian model
                var_left = np.sum(((np.arange(0, i) - mean_left) ** 2) * hist[:i]) / prior_left
                var_right = np.sum(((np.arange(i, self.binNums) - mean_right) ** 2) * hist[i:]) / prior_right
                std_left = np.sqrt(var_left) + np.finfo(float).eps
                std_right = np.sqrt(var_right) + np.finfo(float).eps
                error_func[i] = 1 + 2 * (prior_left * np.log(std_left) + prior_right * np.log(std_right)) \
                                - 2 * (prior_left * np.log(prior_left) + prior_right * np.log(prior_right))
            elif self.criterion == 'poisson':
                # Poisson model
                error_func[i] = total_mean \
                                - prior_left * (np.log(prior_left) + mean_left * np.log(mean_left + np.finfo(float).eps)) \
                                - prior_right * (np.log(prior_right) + mean_right * np.log(mean_right + np.finfo(float).eps))

        t_star = np.argmin(error_func[2:self.binNums - 1]) + 2
        threshold = img_min + t_star / binMultiplier
        return threshold

    def _run(self, image: np.ndarray) -> Any:
        """
        Threshold using the minimum error criterion, then extract connected components.
        """
        # Binarization
        threshold = self.minerrthresh(image)
        print('threshold:', threshold)
        binary_image = image > threshold

        # Connected component analysis
        labeled_img = label(binary_image)
        regions = regionprops(labeled_img, intensity_image=image)

        pt_dets = []
        for r in regions:
            cy, cx = r.centroid
            area = r.area
            minr, minc, maxr, maxc = r.bbox
            w = maxc - minc
            h = maxr - minr
            gray_sum = r.intensity_image.sum()
            pt_dets.append([cx, cy, area, w, h, gray_sum])

        pt_dets = np.array(pt_dets, dtype=np.float32)
        return pt_dets


if __name__ == '__main__':
    # det = OtsuDetector(
    #     method="otsu",
    #     image_dir="test_img/00001.tif",
    #     save_dir="results/",
    #     vis_dir="vis/",
    # )
    # det = SExtractor(
    #     method="sep",
    #     image_dir="test_img/00001.tif",
    #     save_dir="results/",
    #     vis_dir="vis/",
    #     thresh=10
    # )
    det = PoissonThresholding(
        method="pt",
        image_dir="test_img/00001.tif",
        save_dir="results/",
        vis_dir="vis/",
        binNums=256,
        criterion="gaussian",
    )

    results = det.detect()
