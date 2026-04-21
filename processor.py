"""
processor.py
------------
Contains the ImageProcessor class with all image-processing logic.
No UI or file-path logic lives here — every method accepts a NumPy
array and returns a NumPy array (BGR colour space, matching OpenCV).
"""

import heapq
import cv2
import numpy as np
from collections import Counter
from scipy.ndimage import generic_filter


class ImageProcessor:
    """Encapsulates every image-processing technique as a standalone method."""

    # ------------------------------------------------------------------
    # 1. PADDING
    # ------------------------------------------------------------------

    def apply_padding(
        self,
        image: np.ndarray,
        top: int = 20,
        bottom: int = 20,
        left: int = 20,
        right: int = 20,
        mode: str = "zero",
    ) -> np.ndarray:
        """
        Add a border around the image.

        Parameters
        ----------
        image  : Input BGR image array.
        top, bottom, left, right : Border thickness in pixels.
        mode   : 'zero'    → constant black border (cv2.BORDER_CONSTANT)
                 'edge'    → replicate edge pixels  (cv2.BORDER_REPLICATE)
                 'reflect' → mirror-reflect         (cv2.BORDER_REFLECT_101)
                 'wrap'    → wrap-around            (cv2.BORDER_WRAP)
        """
        border_map = {
            "zero":    cv2.BORDER_CONSTANT,
            "edge":    cv2.BORDER_REPLICATE,
            "reflect": cv2.BORDER_REFLECT_101,
            "wrap":    cv2.BORDER_WRAP,
        }
        border_type = border_map.get(mode, cv2.BORDER_CONSTANT)
        return cv2.copyMakeBorder(
            image, top, bottom, left, right,
            borderType=border_type,
            value=[0, 0, 0],          # only used for BORDER_CONSTANT
        )

    # ------------------------------------------------------------------
    # 2. FILTERS  (blur / sharpen via custom kernels)
    # ------------------------------------------------------------------

    def apply_blur(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Gaussian blur using a square kernel.

        Parameters
        ----------
        kernel_size : Must be a positive odd integer (e.g. 3, 5, 7 …).
        """
        ksize = max(1, kernel_size | 1)   # ensure odd
        return cv2.GaussianBlur(image, (ksize, ksize), 0)

    def apply_sharpen(self, image: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """
        Sharpen the image with a Laplacian-based unsharp-mask kernel.

        Parameters
        ----------
        strength : Controls how aggressively details are enhanced (0.0 – 5.0).
        """
        # Unsharp-mask kernel: centre boosted, neighbours suppressed
        k = strength
        kernel = np.array(
            [
                [0,   -k,      0],
                [-k,  1 + 4*k, -k],
                [0,   -k,      0],
            ],
            dtype=np.float32,
        )
        sharpened = cv2.filter2D(image, ddepth=-1, kernel=kernel)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def apply_custom_kernel(
        self, image: np.ndarray, kernel: np.ndarray
    ) -> np.ndarray:
        """
        Apply an arbitrary user-supplied kernel to the image.

        Parameters
        ----------
        kernel : 2-D float32 NumPy array (must be square and odd-sized).
        """
        filtered = cv2.filter2D(image, ddepth=-1, kernel=kernel.astype(np.float32))
        return np.clip(filtered, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # 3. EDGE DETECTION
    # ------------------------------------------------------------------

    def apply_sobel(
        self, image: np.ndarray, ksize: int = 3, scale: float = 1.0
    ) -> np.ndarray:
        """
        Sobel edge detection (gradient magnitude in X + Y).

        Returns a single-channel (grayscale) image as a 3-channel BGR array
        so it can be shown alongside the original without confusion.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
        magnitude = cv2.magnitude(sobelx, sobely) * scale
        magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
        return cv2.cvtColor(magnitude, cv2.COLOR_GRAY2BGR)

    def apply_canny(
        self, image: np.ndarray, threshold1: int = 50, threshold2: int = 150
    ) -> np.ndarray:
        """
        Canny edge detection.

        Parameters
        ----------
        threshold1, threshold2 : Hysteresis thresholding values.
        """
        gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, threshold1, threshold2)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # ------------------------------------------------------------------
    # 4. SEGMENTATION
    # ------------------------------------------------------------------

    def apply_otsu_threshold(self, image: np.ndarray) -> np.ndarray:
        """
        Binary segmentation using Otsu's automatic thresholding.

        Returns a 3-channel BGR image (white foreground, black background).
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    def apply_kmeans(self, image: np.ndarray, k: int = 3) -> np.ndarray:
        """
        Colour quantisation / segmentation using K-Means clustering.

        Parameters
        ----------
        k : Number of colour clusters (segments).
        """
        h, w = image.shape[:2]
        pixels = image.reshape(-1, 3).astype(np.float32)

        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            100,
            0.2,
        )
        _, labels, centres = cv2.kmeans(
            pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )

        centres   = np.uint8(centres)
        segmented = centres[labels.flatten()]
        return segmented.reshape((h, w, 3))

    # ------------------------------------------------------------------
    # 5. COMPRESSION — Run-Length Encoding (lossless)
    # ------------------------------------------------------------------

    def rle_encode(self, image: np.ndarray) -> list[tuple[int, int]]:
        """
        Lossless Run-Length Encoding of a flattened image.

        Converts the image to grayscale, flattens it to 1-D, then encodes
        consecutive identical pixel values as (value, run_length) tuples.

        Returns
        -------
        List of (pixel_value, count) tuples.
        """
        gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        flat    = gray.flatten()
        encoded = []
        if flat.size == 0:
            return encoded

        current_val   = int(flat[0])
        current_count = 1

        for pixel in flat[1:]:
            val = int(pixel)
            if val == current_val:
                current_count += 1
            else:
                encoded.append((current_val, current_count))
                current_val   = val
                current_count = 1

        encoded.append((current_val, current_count))
        return encoded

    def rle_decode(
        self, encoded: list[tuple[int, int]], shape: tuple[int, int]
    ) -> np.ndarray:
        """
        Decode an RLE-encoded sequence back to a grayscale image array.

        Parameters
        ----------
        encoded : Output from rle_encode().
        shape   : (height, width) of the original image.
        """
        flat = np.concatenate(
            [np.full(count, value, dtype=np.uint8) for value, count in encoded]
        )
        gray = flat.reshape(shape)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def rle_compression_ratio(self, image: np.ndarray) -> float:
        """
        Compute the RLE compression ratio for the given image.

        Ratio = original_bytes / encoded_bytes.
        A value > 1.0 means the encoding is smaller than the original.
        """
        gray           = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        original_bytes = gray.nbytes
        encoded        = self.rle_encode(image)
        # Each run stored as two 4-byte ints
        encoded_bytes  = len(encoded) * 8
        if encoded_bytes == 0:
            return 0.0
        return original_bytes / encoded_bytes

    # ══════════════════════════════════════════════════════════════════════
    # ── NEW METHODS ───────────────────────────────────────────────────────
    # ══════════════════════════════════════════════════════════════════════

    # ------------------------------------------------------------------
    # FILTERS (new)
    # ------------------------------------------------------------------

    def apply_mean_filter(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Simple box / mean filter using cv2.blur."""
        ksize = max(1, kernel_size | 1)
        return cv2.blur(image, (ksize, ksize))

    def apply_median_filter(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Median filter — effective for salt-and-pepper noise."""
        ksize = max(1, kernel_size | 1)
        return cv2.medianBlur(image, ksize)

    def apply_laplacian(self, image: np.ndarray) -> np.ndarray:
        """
        Laplacian edge detection.
        Returns a 3-channel BGR image (grayscale edges promoted to BGR).
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lap  = cv2.Laplacian(gray, cv2.CV_64F)
        lap  = np.clip(np.abs(lap), 0, 255).astype(np.uint8)
        return cv2.cvtColor(lap, cv2.COLOR_GRAY2BGR)

    def apply_midpoint_filter(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Midpoint filter: (local_max + local_min) / 2 per sliding window.
        Applied channel-wise on the BGR image.
        """
        ksize = max(1, kernel_size | 1)
        half  = ksize // 2

        def _midpoint(values: np.ndarray) -> float:
            return (float(values.max()) + float(values.min())) / 2.0

        result_channels = []
        for ch in cv2.split(image):
            filtered = generic_filter(ch.astype(np.float64), _midpoint,
                                      size=ksize, mode="reflect")
            result_channels.append(np.clip(filtered, 0, 255).astype(np.uint8))
        return cv2.merge(result_channels)

    def apply_alpha_trimmed_mean(
        self, image: np.ndarray, kernel_size: int = 5, d: int = 2
    ) -> np.ndarray:
        """
        Alpha-trimmed mean filter.
        Sorts the kernel_size² pixel values, trims d//2 from each end,
        and replaces the centre pixel with the mean of the remaining values.
        Applied channel-wise.
        """
        ksize = max(1, kernel_size | 1)
        trim  = max(0, d // 2)

        def _alpha_mean(values: np.ndarray) -> float:
            s = np.sort(values)
            n = len(s)
            lo, hi = trim, n - trim
            if lo >= hi:
                return float(s[n // 2])
            return float(s[lo:hi].mean())

        result_channels = []
        for ch in cv2.split(image):
            filtered = generic_filter(ch.astype(np.float64), _alpha_mean,
                                      size=ksize, mode="reflect")
            result_channels.append(np.clip(filtered, 0, 255).astype(np.uint8))
        return cv2.merge(result_channels)

    def apply_harmonic_mean_filter(
        self, image: np.ndarray, kernel_size: int = 5
    ) -> np.ndarray:
        """
        Harmonic mean filter: n / sum(1/x) per window.
        Zero-valued pixels are replaced with 1 to avoid division by zero.
        Applied channel-wise.
        """
        ksize = max(1, kernel_size | 1)

        def _harmonic(values: np.ndarray) -> float:
            safe = np.where(values == 0, 1e-6, values.astype(np.float64))
            return float(len(safe) / np.sum(1.0 / safe))

        result_channels = []
        for ch in cv2.split(image):
            filtered = generic_filter(ch.astype(np.float64), _harmonic,
                                      size=ksize, mode="reflect")
            result_channels.append(np.clip(filtered, 0, 255).astype(np.uint8))
        return cv2.merge(result_channels)

    def apply_contraharmonic_mean_filter(
        self, image: np.ndarray, kernel_size: int = 5, Q: float = 1.5
    ) -> np.ndarray:
        """
        Contraharmonic mean filter: sum(x^(Q+1)) / sum(x^Q) per window.
        Q > 0 → eliminates pepper noise; Q < 0 → eliminates salt noise.
        Applied channel-wise.
        """
        ksize = max(1, kernel_size | 1)

        def _contra(values: np.ndarray) -> float:
            x   = values.astype(np.float64) + 1e-6
            num = np.sum(x ** (Q + 1))
            den = np.sum(x **  Q)
            return float(num / den) if den != 0 else 0.0

        result_channels = []
        for ch in cv2.split(image):
            filtered = generic_filter(ch.astype(np.float64), _contra,
                                      size=ksize, mode="reflect")
            result_channels.append(np.clip(filtered, 0, 255).astype(np.uint8))
        return cv2.merge(result_channels)

    def apply_box_lowpass_filter(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Normalised box (average) kernel applied via cv2.filter2D."""
        ksize  = max(1, kernel_size | 1)
        kernel = np.ones((ksize, ksize), dtype=np.float32) / (ksize * ksize)
        result = cv2.filter2D(image, ddepth=-1, kernel=kernel)
        return np.clip(result, 0, 255).astype(np.uint8)

    def apply_highpass_filter(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        High-pass filter: original image minus its low-pass (box) version.
        Result is the detail / edge layer, scaled into [0, 255].
        """
        lowpass = self.apply_box_lowpass_filter(image, kernel_size)
        diff    = image.astype(np.int16) - lowpass.astype(np.int16) + 128
        return np.clip(diff, 0, 255).astype(np.uint8)

    def apply_prewitt_filter(self, image: np.ndarray) -> np.ndarray:
        """
        Prewitt edge detection.
        Applies Prewitt kernels in X and Y, returns gradient magnitude
        as a 3-channel BGR image.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
        kx   = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
        ky   = kx.T
        gx   = cv2.filter2D(gray, ddepth=-1, kernel=kx)
        gy   = cv2.filter2D(gray, ddepth=-1, kernel=ky)
        mag  = np.clip(np.sqrt(gx**2 + gy**2), 0, 255).astype(np.uint8)
        return cv2.cvtColor(mag, cv2.COLOR_GRAY2BGR)

    # ------------------------------------------------------------------
    # SEGMENTATION (new)
    # ------------------------------------------------------------------

    def apply_global_threshold(
        self, image: np.ndarray, threshold: int = 127
    ) -> np.ndarray:
        """Manual binary threshold using cv2.threshold."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    def apply_adaptive_threshold(
        self, image: np.ndarray, block_size: int = 11, C: int = 2
    ) -> np.ndarray:
        """
        Adaptive Gaussian threshold using cv2.adaptiveThreshold.
        block_size must be odd and > 1.
        """
        gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bsize = max(3, block_size | 1)
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            bsize, C,
        )
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    def apply_multi_otsu(self, image: np.ndarray, classes: int = 3) -> np.ndarray:
        """
        Multi-level Otsu thresholding via skimage.
        Returns a colour-mapped image where each class gets a distinct grey level.
        """
        from skimage.filters import threshold_multiotsu

        gray      = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresholds = threshold_multiotsu(gray, classes=classes)
        regions    = np.digitize(gray, bins=thresholds)           # 0 … classes-1
        # Map each class to [0, 255] evenly
        scale      = 255 // (classes - 1) if classes > 1 else 255
        mapped     = (regions * scale).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(mapped, cv2.COLOR_GRAY2BGR)

    def apply_region_growing(
        self,
        image: np.ndarray,
        seed: tuple[int, int] = (0, 0),
        threshold: int = 15,
    ) -> np.ndarray:
        """
        Flood-fill style region growing from a seed pixel.
        Pixels within `threshold` intensity of the seed are grown into the region.
        Returns a BGR image with the grown region shown in white on black.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        sy, sx = int(seed[0]), int(seed[1])
        sy, sx = np.clip(sy, 0, h - 1), np.clip(sx, 0, w - 1)
        seed_val = int(gray[sy, sx])

        visited = np.zeros((h, w), dtype=bool)
        region  = np.zeros((h, w), dtype=np.uint8)

        stack = [(sy, sx)]
        visited[sy, sx] = True

        while stack:
            y, x = stack.pop()
            if abs(int(gray[y, x]) - seed_val) <= threshold:
                region[y, x] = 255
                for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))

        return cv2.cvtColor(region, cv2.COLOR_GRAY2BGR)

    def apply_watershed(self, image: np.ndarray) -> np.ndarray:
        """
        Marker-based watershed segmentation using cv2.watershed.
        Uses distance-transform peaks as foreground markers.
        Returns the segment boundaries overlaid in red on the original image.
        """
        gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Noise removal
        kernel  = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        # Sure background
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # Sure foreground via distance transform
        dist    = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
        sure_fg = sure_fg.astype(np.uint8)

        # Unknown region
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)
        markers   += 1
        markers[unknown == 255] = 0

        result  = image.copy()
        markers = cv2.watershed(result, markers)
        result[markers == -1] = [0, 0, 255]   # red boundaries
        return result

    def apply_slic_superpixels(
        self,
        image: np.ndarray,
        n_segments: int = 100,
        compactness: float = 10.0,
    ) -> np.ndarray:
        """
        SLIC superpixel segmentation via skimage.
        Returns the original image with superpixel boundaries drawn in yellow.
        """
        from skimage.segmentation import slic, mark_boundaries

        rgb    = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        labels = slic(rgb, n_segments=n_segments, compactness=compactness,
                      start_label=1, channel_axis=-1)
        marked = mark_boundaries(rgb / 255.0, labels, color=(1, 1, 0))
        marked = (marked * 255).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(marked, cv2.COLOR_RGB2BGR)

    def apply_active_contours(
        self,
        image: np.ndarray,
        center: tuple[int, int] = None,
        radius: int = 50,
    ) -> np.ndarray:
        """
        Snake / active contour segmentation via skimage.
        Initialises a circular contour and evolves it to fit edges.
        Returns the image with the final contour drawn in green.
        """
        from skimage.segmentation import active_contour
        from skimage.filters import gaussian

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255.0
        smooth = gaussian(gray, sigma=3)

        h, w = image.shape[:2]
        cy = center[0] if center else h // 2
        cx = center[1] if center else w // 2
        r  = max(5, radius)

        theta = np.linspace(0, 2 * np.pi, 400)
        snake_init = np.column_stack([
            cy + r * np.sin(theta),
            cx + r * np.cos(theta),
        ])

        snake = active_contour(smooth, snake_init,
                               alpha=0.015, beta=10, gamma=0.001)

        result = image.copy()
        pts    = snake[:, ::-1].astype(np.int32)   # (col, row)
        cv2.polylines(result, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        return result

    def compute_segmentation_metrics(
        self,
        ground_truth: np.ndarray,
        predicted: np.ndarray,
    ) -> dict:
        """
        Compute segmentation quality metrics.

        Returns
        -------
        dict with keys: pixel_accuracy, iou, dice, miou
        Both inputs must be single-channel or 3-channel binary masks.
        Non-zero pixels are treated as foreground.
        """
        def _to_binary(mask: np.ndarray) -> np.ndarray:
            if mask.ndim == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            return (mask > 0).astype(np.uint8)

        gt  = _to_binary(ground_truth).flatten()
        pr  = _to_binary(predicted).flatten()

        tp = int(np.logical_and(gt == 1, pr == 1).sum())
        tn = int(np.logical_and(gt == 0, pr == 0).sum())
        fp = int(np.logical_and(gt == 0, pr == 1).sum())
        fn = int(np.logical_and(gt == 1, pr == 0).sum())

        pixel_accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
        iou_fg  = tp / max(tp + fp + fn, 1)
        iou_bg  = tn / max(tn + fn + fp, 1)
        dice    = (2 * tp) / max(2 * tp + fp + fn, 1)
        miou    = (iou_fg + iou_bg) / 2

        return {
            "pixel_accuracy": round(pixel_accuracy, 4),
            "iou":            round(iou_fg, 4),
            "dice":           round(dice, 4),
            "miou":           round(miou, 4),
        }

    # ------------------------------------------------------------------
    # COMPRESSION (new)
    # ------------------------------------------------------------------

    # ── Huffman ───────────────────────────────────────────────────────

    @staticmethod
    def _build_huffman_tree(freq: dict) -> dict:
        """Build Huffman codebook from a symbol-frequency dict."""
        heap = [[w, [sym, ""]] for sym, w in freq.items()]
        heapq.heapify(heap)
        while len(heap) > 1:
            lo = heapq.heappop(heap)
            hi = heapq.heappop(heap)
            for pair in lo[1:]:
                pair[1] = "0" + pair[1]
            for pair in hi[1:]:
                pair[1] = "1" + pair[1]
            heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
        return {sym: code for sym, code in heap[0][1:]}

    def huffman_encode(
        self, image: np.ndarray
    ) -> tuple[str, dict, float]:
        """
        Huffman encode a grayscale version of the image.

        Returns
        -------
        (encoded_bits_string, codebook_dict, compression_ratio)
        """
        gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        flat  = gray.flatten().tolist()
        freq  = Counter(flat)

        if len(freq) == 1:
            # Edge case: single unique value
            sym      = next(iter(freq))
            codebook = {sym: "0"}
        else:
            codebook = self._build_huffman_tree(freq)

        encoded_bits = "".join(codebook[p] for p in flat)
        original_bits  = gray.nbytes * 8
        encoded_bits_n = len(encoded_bits)
        ratio = original_bits / max(encoded_bits_n, 1)

        return encoded_bits, codebook, round(ratio, 4)

    def huffman_decode(
        self,
        encoded: str,
        codebook: dict,
        shape: tuple[int, int],
    ) -> np.ndarray:
        """
        Decode a Huffman bit-string back to a grayscale image array.

        Parameters
        ----------
        encoded  : Bit string produced by huffman_encode.
        codebook : Symbol → code dict from huffman_encode.
        shape    : (height, width) of original image.
        """
        reverse  = {v: k for k, v in codebook.items()}
        pixels, buf = [], ""
        for bit in encoded:
            buf += bit
            if buf in reverse:
                pixels.append(reverse[buf])
                buf = ""
        gray = np.array(pixels, dtype=np.uint8).reshape(shape)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # ── LZW ───────────────────────────────────────────────────────────

    def lzw_encode(self, image: np.ndarray) -> tuple[list[int], float]:
        """
        LZW encoding on flattened grayscale pixels (values 0-255).

        Returns
        -------
        (codes_list, compression_ratio)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        flat = gray.flatten().tolist()

        # Initialise dictionary with all single-byte symbols
        dict_size = 256
        table     = {(i,): i for i in range(dict_size)}

        codes, w = [], (flat[0],)
        for c in flat[1:]:
            wc = w + (c,)
            if wc in table:
                w = wc
            else:
                codes.append(table[w])
                table[wc] = dict_size
                dict_size += 1
                w = (c,)
        codes.append(table[w])

        original_bits = gray.nbytes * 8
        # Approximate: each code needs ceil(log2(dict_size)) bits, use 12 as typical
        encoded_bits  = len(codes) * 12
        ratio = original_bits / max(encoded_bits, 1)
        return codes, round(ratio, 4)

    def lzw_decode(self, codes: list[int], shape: tuple[int, int]) -> np.ndarray:
        """
        Decode LZW codes back to a grayscale image array.

        Parameters
        ----------
        codes : List of int codes from lzw_encode.
        shape : (height, width) of original image.
        """
        dict_size = 256
        table     = {i: (i,) for i in range(dict_size)}

        result = []
        w      = table[codes[0]]
        result.extend(w)

        for code in codes[1:]:
            if code in table:
                entry = table[code]
            elif code == dict_size:
                entry = w + (w[0],)
            else:
                raise ValueError(f"Bad LZW code: {code}")
            result.extend(entry)
            table[dict_size] = w + (entry[0],)
            dict_size += 1
            w = entry

        n_pixels = shape[0] * shape[1]
        gray = np.array(result[:n_pixels], dtype=np.uint8).reshape(shape)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # ── DPCM ──────────────────────────────────────────────────────────

    def dpcm_encode(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """
        Differential Pulse-Code Modulation encoding.
        Each pixel is replaced by its difference from the previous pixel
        (scanning left-to-right, top-to-bottom).

        Returns
        -------
        (residuals_array, compression_ratio)
        Residuals are int16; ratio is based on 8-bit originals vs. typical
        entropy savings estimated from the residual range.
        """
        gray      = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        flat      = gray.flatten().astype(np.int16)
        residuals = np.empty_like(flat)
        residuals[0]  = flat[0]
        residuals[1:] = flat[1:] - flat[:-1]

        # Compression ratio estimate: 8 bits / average bits needed per residual
        unique_res = len(np.unique(residuals))
        bits_needed = max(1, np.ceil(np.log2(unique_res + 1)))
        ratio = 8.0 / bits_needed
        return residuals.reshape(gray.shape), round(ratio, 4)

    def dpcm_decode(
        self, residuals: np.ndarray, shape: tuple[int, int]
    ) -> np.ndarray:
        """
        Reconstruct image from DPCM residuals via cumulative sum.

        Parameters
        ----------
        residuals : int16 array from dpcm_encode.
        shape     : (height, width) of original image.
        """
        flat = residuals.flatten().astype(np.int16)
        reconstructed = np.cumsum(flat).clip(0, 255).astype(np.uint8)
        gray = reconstructed.reshape(shape)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # ── Arithmetic coding (model only) ────────────────────────────────

    def arithmetic_encode(
        self, image: np.ndarray
    ) -> tuple[float, dict, float]:
        """
        Arithmetic coding (pure-Python model-based implementation).

        Works on grayscale pixel values.  Performs an exact interval-narrowing
        encode in Python floats (sufficient for demonstration; real
        implementations use multi-precision integers).

        Returns
        -------
        (encoded_value, prob_model_dict, compression_ratio)
            encoded_value   : float in [0, 1)
            prob_model_dict : symbol → cumulative probability interval (lo, hi)
            compression_ratio : original_bits / estimated_encoded_bits
        """
        gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        flat  = gray.flatten().tolist()
        total = len(flat)

        freq  = Counter(flat)
        syms  = sorted(freq.keys())

        # Build cumulative probability model
        prob_model: dict[int, tuple[float, float]] = {}
        cumulative = 0.0
        for s in syms:
            p = freq[s] / total
            prob_model[s] = (cumulative, cumulative + p)
            cumulative    += p

        # Encode
        lo, hi = 0.0, 1.0
        for symbol in flat:
            rng = hi - lo
            s_lo, s_hi = prob_model[symbol]
            hi  = lo + rng * s_hi
            lo  = lo + rng * s_lo

        encoded_value = (lo + hi) / 2

        # Entropy-based compression ratio estimate
        entropy = -sum(
            (c / total) * np.log2(c / total)
            for c in freq.values() if c > 0
        )
        ratio = 8.0 / max(entropy, 1e-9)
        return encoded_value, prob_model, round(ratio, 4)

    # ------------------------------------------------------------------
    # PADDING (new)
    # ------------------------------------------------------------------

    def apply_symmetric_padding(
        self,
        image: np.ndarray,
        top: int = 20,
        bottom: int = 20,
        left: int = 20,
        right: int = 20,
    ) -> np.ndarray:
        """
        Symmetric (mirror-reflect) padding using cv2.BORDER_REFLECT.
        Unlike BORDER_REFLECT_101 this includes the edge pixel itself.
        """
        return cv2.copyMakeBorder(
            image, top, bottom, left, right,
            borderType=cv2.BORDER_REFLECT,
        )

    def apply_asymmetric_padding(
        self,
        image: np.ndarray,
        top: int = 10,
        bottom: int = 30,
        left: int = 10,
        right: int = 30,
    ) -> np.ndarray:
        """
        Asymmetric constant padding via numpy.pad.
        Each side can independently be a different thickness; the pad value
        is always 0.  Useful for demonstrating non-uniform border extension.
        """
        if image.ndim == 3:
            padded = np.pad(
                image,
                pad_width=((top, bottom), (left, right), (0, 0)),
                mode="constant",
                constant_values=0,
            )
        else:
            padded = np.pad(
                image,
                pad_width=((top, bottom), (left, right)),
                mode="constant",
                constant_values=0,
            )
        return padded.astype(np.uint8)

    def apply_color_padding(
        self,
        image: np.ndarray,
        top: int = 20,
        bottom: int = 20,
        left: int = 20,
        right: int = 20,
        color: tuple[int, int, int] = (255, 0, 0),
    ) -> np.ndarray:
        """
        Constant-colour padding.  Unlike the original zero-padding, you can
        specify any BGR colour tuple for the border.

        Parameters
        ----------
        color : BGR tuple, e.g. (255, 0, 0) for blue.
        """
        return cv2.copyMakeBorder(
            image, top, bottom, left, right,
            borderType=cv2.BORDER_CONSTANT,
            value=list(color),
        )

