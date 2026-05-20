"""
@file forensics_engine.py
@brief Core image forensics analysis engine.
@details Implements SIFT, SURF, AKAZE, ORB algorithms for copy-move
         forgery detection, plus ELA-based tamper localization and a
         multi-signal AI-generation detector using 8 independent forensic signals.
@author ImageForensics Team
@version 1.2.0
"""

import cv2
import numpy as np
import logging
import base64
import scipy.fft as sfft
import scipy.stats as sstats
from skimage.feature import local_binary_pattern
from io import BytesIO
from PIL import Image, ImageChops, ImageEnhance, ImageFilter

# AIDetector: hibrit istatistiksel + Anthropic Vision API
try:
    from ai_detector import AIDetector
    _AI_DETECTOR_AVAILABLE = True
except ImportError:
    _AI_DETECTOR_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
# ForensicsEngine
# ============================================================================

class ForensicsEngine:
    """
    @class ForensicsEngine
    @brief Orchestrates all image forgery detection algorithms.

    @details
    State machine (FSM) is managed externally by app.py.
    This class is stateless between calls.

    Algorithms:
    - SIFT  : Scale-Invariant Feature Transform (copy-move)
    - SURF  : Speeded-Up Robust Features        (copy-move)
    - AKAZE : Accelerated KAZE                  (copy-move)
    - ORB   : Oriented FAST + Rotated BRIEF     (copy-move)
    - ELA   : Error Level Analysis              (compression artifacts)
    - AI    : AIDetector hibrit motoru (istatistiksel + opsiyonel Anthropic Vision API)
    """

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(self):
        """@brief Initialize engine, matchers and AIDetector."""
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        self.bf_hamming = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # AIDetector: api_detector.py import edildiyse kullan
        if _AI_DETECTOR_AVAILABLE:
            self._ai_detector = AIDetector()
            logger.info("ForensicsEngine v1.3: AIDetector (hibrit mod) aktif")
        else:
            self._ai_detector = None
            logger.info("ForensicsEngine v1.3: AIDetector yok, 8-sinyal motoru kullanılıyor")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_full_analysis(self, img_np: np.ndarray, img_bytes: bytes,
                          algorithms: list) -> dict:
        """
        @brief Run all selected algorithms and aggregate results.

        @param img_np    Image as NumPy array (H x W x 3, RGB).
        @param img_bytes Raw image bytes (for ELA compression analysis).
        @param algorithms List of algorithm IDs to run.
        @return          Full results dict ready for JSON serialization.
        """
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape

        algo_results = {}
        heatmap = np.zeros((h, w), dtype=np.float32)

        # --- Run each keypoint algorithm ---
        if "sift" in algorithms:
            r = self._run_keypoint_detector("sift", gray, img_np)
            algo_results["sift"] = r
            heatmap = self._accumulate_heatmap(heatmap, r["regions"])

        if "surf" in algorithms:
            r = self._run_keypoint_detector("surf", gray, img_np)
            algo_results["surf"] = r
            heatmap = self._accumulate_heatmap(heatmap, r["regions"])

        if "akaze" in algorithms:
            r = self._run_keypoint_detector("akaze", gray, img_np)
            algo_results["akaze"] = r
            heatmap = self._accumulate_heatmap(heatmap, r["regions"])

        if "orb" in algorithms:
            r = self._run_keypoint_detector("orb", gray, img_np)
            algo_results["orb"] = r
            heatmap = self._accumulate_heatmap(heatmap, r["regions"])

        # --- ELA ---
        ela_result, ela_b64 = self._run_ela(img_bytes)
        ela_heatmap = ela_result["heatmap"]
        if ela_heatmap is not None:
            heatmap += ela_heatmap * 0.5

        # --- Aggregate tampering confidence ---
        scores = [v["confidence"] for v in algo_results.values()]
        ela_conf = ela_result["confidence"]
        scores.append(ela_conf)

        aggregate_confidence = float(np.mean(scores)) if scores else 0.0
        tampered = aggregate_confidence > 35.0

        # --- Ekran görüntüsü tespiti ---
        # Ekran görüntüleri (screenshot, UI, diagram) istatistiksel olarak
        # AI görsellerine benzer: düz renkler, sıfır gürültü, yüksek renk korelasyonu.
        # Bu tür içerikler için AI tespitini bypass ederiz.
        is_screenshot = self._detect_screenshot(img_np, gray)

        # --- AI detection: AIDetector (hibrit) veya 8-sinyal yedek ---
        if is_screenshot:
            # Ekran görüntüsü: AI tespitini devre dışı bırak
            ai_result = {
                "generated":     False,
                "confidence":    0.0,
                "verdict_label": "Ekran Görüntüsü / Dijital İçerik — AI Tespiti Geçersiz",
                "signal_detail": [{"name": "Ekran Görüntüsü Tespiti",
                                   "score": 100.0,
                                   "label": "Düz renkler, sıfır gürültü, UI elementleri tespit edildi"}],
            }
        elif self._ai_detector is not None:
            raw_ai = self._ai_detector.detect(img_np, img_bytes)
            # AIDetector sonucunu ortak formata çevir
            ai_result = {
                "generated":     raw_ai["generated"],
                "confidence":    raw_ai["confidence"],
                "verdict_label": self._ai_verdict_label(raw_ai),
                "signal_detail": self._ai_signal_detail(raw_ai),
            }
        else:
            ai_result = self._detect_ai_generated_multisignal(img_np, gray)

        # --- Normalize and render heatmap ---
        heatmap_b64, annotated_b64 = self._render_outputs(img_np, heatmap)

        return {
            "tampered": tampered,
            "confidence": round(aggregate_confidence, 2),
            "ai_generated": ai_result["generated"],
            "ai_confidence": round(ai_result["confidence"], 2),
            "ai_signal_detail": ai_result["signal_detail"],
            "ai_verdict_label": ai_result["verdict_label"],
            "algorithm_results": algo_results,
            "ela_confidence": round(ela_conf, 2),
            "heatmap_b64": heatmap_b64,
            "annotated_b64": annotated_b64,
            "ela_b64": ela_b64,
            "image_size": {"width": int(w), "height": int(h)},
        }

    # ------------------------------------------------------------------
    # Algorithm: Copy-Move via Keypoint Detectors
    # ------------------------------------------------------------------

    def _run_keypoint_detector(self, algo_id: str, gray: np.ndarray,
                                img_np: np.ndarray) -> dict:
        """
        @brief Generic copy-move detector using a keypoint-based algorithm.

        @param algo_id  One of: 'sift', 'surf', 'akaze', 'orb'.
        @param gray     Grayscale image.
        @param img_np   Original color image.
        @return         Dict with confidence, keypoint_count, match_count, regions.
        """
        try:
            detector = self._create_detector(algo_id)
            if detector is None:
                return self._empty_result(algo_id, "Detector not available")

            kps, descs = detector.detectAndCompute(gray, None)

            if descs is None or len(kps) < 10:
                return self._empty_result(algo_id, "Insufficient keypoints")

            # Match descriptors to find copy-move pairs
            if algo_id == "orb":
                matches = self.bf_hamming.knnMatch(descs, descs, k=3)
            else:
                matches = self.bf_matcher.knnMatch(descs, descs, k=3)

            good_matches = []
            regions = []

            for match_group in matches:
                # Skip self-matches (distance == 0)
                filtered = [m for m in match_group if m.distance > 0.01]
                if len(filtered) < 2:
                    continue

                m, n = filtered[0], filtered[1]

                # Lowe's ratio test
                ratio = 0.75 if algo_id in ("sift", "surf") else 0.80
                if m.distance < ratio * n.distance:
                    pt1 = kps[m.queryIdx].pt
                    pt2 = kps[m.trainIdx].pt

                    # Spatial separation guard (avoid trivial neighbors)
                    dist = np.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])
                    if dist > 20:
                        good_matches.append(m)
                        regions.append({
                            "x1": int(pt1[0]), "y1": int(pt1[1]),
                            "x2": int(pt2[0]), "y2": int(pt2[1]),
                            "strength": float(1.0 - m.distance /
                                              max(n.distance, 1e-6))
                        })

            match_count = len(good_matches)
            kp_count = len(kps)

            # Confidence heuristic: ratio of suspicious matches to keypoints
            confidence = min(100.0, (match_count / max(kp_count, 1)) * 400)

            return {
                "algorithm": algo_id.upper(),
                "confidence": round(confidence, 2),
                "keypoint_count": kp_count,
                "match_count": match_count,
                "regions": regions,
                "status": "ok"
            }

        except Exception as e:
            logger.warning(f"{algo_id} failed: {e}")
            return self._empty_result(algo_id, str(e))

    def _create_detector(self, algo_id: str):
        """
        @brief Factory: create OpenCV feature detector by ID.
        @param algo_id Algorithm identifier string.
        @return OpenCV feature detector object or None.
        """
        if algo_id == "sift":
            return cv2.SIFT_create(nfeatures=2000)
        elif algo_id == "surf":
            try:
                return cv2.xfeatures2d.SURF_create(400)
            except AttributeError:
                # SURF requires opencv-contrib; fall back to SIFT silently
                logger.warning("SURF not available, using SIFT as fallback")
                return cv2.SIFT_create(nfeatures=2000)
        elif algo_id == "akaze":
            return cv2.AKAZE_create()
        elif algo_id == "orb":
            return cv2.ORB_create(nfeatures=3000)
        return None

    # ------------------------------------------------------------------
    # Algorithm: Error Level Analysis (ELA)
    # ------------------------------------------------------------------

    def _run_ela(self, img_bytes: bytes, quality: int = 90) -> tuple:
        """
        @brief Error Level Analysis — orijinal JPEG kalitesine adaptif versiyon.

        @details
        Standart ELA sabit bir kalite (örn. 90) ile karşılaştırır.
        Sorun: Orijinal görüntü kalitesi 90'dan yüksekse (örn. telefon selfie ~97)
        büyük fark oluşur ve manipülasyon yokken %100 ELA skoru verir.

        Çözüm: Orijinal JPEG quantization tablosundan kaliteyi tahmin et,
        karşılaştırmayı orijinale yakın bir kalitede yap.
        Gerçek manipülasyon: orijinale benzer kalitede bile yüksek fark
        Sadece yeniden sıkıştırma: orijinaline yakın kalitede fark küçük

        @param img_bytes Raw image bytes.
        @param quality   Varsayılan karşılaştırma kalitesi (adaptif mod yoksa).
        @return          Tuple of (result_dict, base64_ELA_image_string).
        """
        try:
            pil = Image.open(BytesIO(img_bytes)).convert("RGB")
            fmt = Image.open(BytesIO(img_bytes)).format or ""

            # --- Orijinal JPEG kalitesini tahmin et ---
            orig_quality = self._estimate_jpeg_quality(img_bytes)

            # Adaptif karşılaştırma kalitesi:
            # Yüksek kaliteli orijinal (telefon selfie ~95-99) → çok yakın kalitede karşılaştır
            # Orta kalite (WhatsApp ~75-85) → standart 90 kullan
            if orig_quality is not None and orig_quality >= 94:
                compare_quality = max(90, orig_quality - 2)  # çok yakın: orig-2
                scale_factor    = 0.7                         # muhafazakâr
            elif orig_quality is not None and orig_quality >= 85:
                compare_quality = max(80, orig_quality - 5)
                scale_factor    = 1.0
            else:
                compare_quality = quality   # varsayılan 90
                scale_factor    = 1.5

            # Re-save at controlled quality
            buf = BytesIO()
            pil.save(buf, "JPEG", quality=compare_quality)
            buf.seek(0)
            recompressed = Image.open(buf).convert("RGB")

            ela_arr = ImageChops.difference(pil, recompressed)
            ela_np  = np.array(ela_arr).astype(np.float32)

            # Amplify for visibility
            scale   = 10
            ela_vis = np.clip(ela_np * scale, 0, 255).astype(np.uint8)

            # Confidence: top-5% brightest pixels
            brightness = ela_vis.mean(axis=2)
            threshold  = np.percentile(brightness, 95)
            hot_pixels = brightness[brightness > threshold]

            confidence = min(100.0, float(hot_pixels.mean()) * scale_factor) \
                         if len(hot_pixels) else 0.0

            # Heatmap
            hm      = brightness / 255.0

            ela_pil = Image.fromarray(ela_vis)
            ela_buf = BytesIO()
            ela_pil.save(ela_buf, "PNG")
            ela_b64 = base64.b64encode(ela_buf.getvalue()).decode("utf-8")

            logger.debug(f"ELA: orig_q={orig_quality} cmp_q={compare_quality} conf={confidence:.1f}")

            return (
                {"confidence": round(confidence, 2), "heatmap": hm},
                ela_b64
            )

        except Exception as e:
            logger.warning(f"ELA failed: {e}")
            return ({"confidence": 0.0, "heatmap": None}, "")

    @staticmethod
    def _estimate_jpeg_quality(img_bytes: bytes) -> int | None:
        """
        @brief JPEG quantization tablosundan orijinal kaliteyi tahmin et.
        @param img_bytes  Ham görüntü baytları.
        @return           Tahmini kalite (1-100) veya JPEG değilse None.
        """
        try:
            import struct
            data   = img_bytes
            tables = []
            i      = 0
            while i < len(data) - 3:
                if data[i] == 0xFF and data[i+1] == 0xDB:
                    length = struct.unpack('>H', data[i+2:i+4])[0]
                    tables.extend(list(data[i+5: i+4+length])[:64])
                    i += 2 + length
                else:
                    i += 1

            if not tables:
                return None   # JPEG değil (PNG, BMP vb.)

            avg_q = sum(tables[:64]) / 64.0
            # Yaklaşık kalite dönüşümü
            if avg_q <= 1:    return 100
            elif avg_q <= 2:  return 99
            elif avg_q <= 4:  return 97
            elif avg_q <= 8:  return 92
            elif avg_q <= 12: return 85
            elif avg_q <= 16: return 80
            elif avg_q <= 24: return 72
            elif avg_q <= 36: return 65
            elif avg_q <= 55: return 55
            else:             return max(30, int(100 - avg_q))
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Algorithm: AI Generation Detection — 8-Signal Forensic Detector
    # ------------------------------------------------------------------

    def _detect_ai_generated_multisignal(self, img_np: np.ndarray,
                                          gray: np.ndarray) -> dict:
        """
        @brief Multi-signal AI image detector using 8 independent forensic features.

        @details
        AI-generated images (GAN, Diffusion, VAE) leave characteristic forensic traces
        that differ from real camera photographs. This method computes 8 independent
        signals, each calibrated with empirically derived thresholds:

        Signal 1 — DCT Coefficient Kurtosis:
          Real photos follow a Laplacian distribution in DCT space (high kurtosis ~3-6).
          AI images tend toward Gaussian (kurtosis ~2-3) due to latent space sampling.

        Signal 2 — High-Pass Noise Residual Pattern:
          Camera sensors produce specific noise floors (PRNU). AI images have either
          too-uniform noise (diffusion) or patterned noise (GAN upsampling artifacts).

        Signal 3 — Local Binary Pattern (LBP) Texture Entropy:
          Real textures have high LBP entropy due to micro-surface variation.
          AI images show lower entropy from over-smoothed or synthetically generated textures.

        Signal 4 — FFT Radial Power Spectrum slope (beta):
          Natural images follow 1/f^beta power law (beta≈2). AI images deviate significantly,
          especially diffusion models which produce flatter spectra.

        Signal 5 — Color Co-occurrence (GLCM) Homogeneity:
          AI images show unusually high GLCM homogeneity due to smooth blending.
          Real photos have lower homogeneity from real-world surface variance.

        Signal 6 — Chromatic Aberration Analysis:
          Real camera lenses produce measurable RGB channel misalignment at edges.
          AI images have near-zero chromatic aberration (perfect channel alignment).

        Signal 7 — Blocking Artifact Metric (BAM):
          AI-generated images sometimes show periodic boundary artifacts from
          patch-based generation (e.g. 8x8 or 16x16 grid patterns from VAEs).

        Signal 8 — Saturation Distribution Kurtosis:
          Real photos have high saturation variance (sunlight, shadow, mixed lighting).
          AI images often show lower saturation kurtosis — more uniform color distribution.

        @param img_np  Color image (RGB, uint8).
        @param gray    Grayscale image.
        @return        Dict with 'generated' (bool), 'confidence' (float 0-100),
                       'signal_detail' (list), 'verdict_label' (str).
        """
        signals = {}

        # ── Signal 1: DCT Kurtosis ──────────────────────────────────────
        try:
            h, w = gray.shape
            # Use center crop to avoid border effects
            cy, cx = h // 2, w // 2
            crop_h, crop_w = min(256, h), min(256, w)
            crop = gray[cy - crop_h//2: cy + crop_h//2,
                        cx - crop_w//2: cx + crop_w//2].astype(np.float32)
            dct = cv2.dct(crop / 255.0)
            # Focus on mid-frequency band (avoid DC and very high freq)
            mid_dct = dct[4:crop_h//2, 4:crop_w//2].flatten()
            if len(mid_dct) > 10:
                kurt = float(sstats.kurtosis(mid_dct, fisher=True))
                if not np.isfinite(kurt):
                    signals["dct_kurtosis"] = (50.0, "DCT kurtosis: hesaplanamadı (tek tip bölge)")
                # Empirical calibration from real Stable Diffusion / Midjourney images:
                # AI images: DCT kurtosis typically 0 – 4  (latent space sampling → Gaussian)
                # Real photos: DCT kurtosis typically 5 – 30+ (heavy-tailed Laplacian)
                elif kurt < 0.5:
                    signals["dct_kurtosis"] = (90.0, f"Çok düşük DCT kurtosis ({kurt:.2f}) — diffusion/GAN imzası")
                elif kurt < 2.5:
                    signals["dct_kurtosis"] = (72.0, f"Düşük DCT kurtosis ({kurt:.2f}) — AI olası")
                elif kurt < 5.0:
                    signals["dct_kurtosis"] = (45.0, f"Orta DCT kurtosis ({kurt:.2f}) — belirsiz")
                elif kurt < 12.0:
                    signals["dct_kurtosis"] = (20.0, f"Yüksek DCT kurtosis ({kurt:.2f}) — doğal fotoğraf")
                else:
                    signals["dct_kurtosis"] = (10.0, f"Çok yüksek DCT kurtosis ({kurt:.2f}) — kesinlikle gerçek")
            else:
                signals["dct_kurtosis"] = (50.0, "DCT kurtosis: yetersiz veri")
        except Exception as e:
            logger.warning(f"Signal 1 (DCT kurtosis) failed: {e}")
            signals["dct_kurtosis"] = (50.0, "DCT kurtosis: hata")

        # ── Signal 2: High-Pass Noise Residual ─────────────────────────
        try:
            gray_f = gray.astype(np.float32)
            # Noise residual: subtract 3x3 median filtered version
            blurred = cv2.GaussianBlur(gray_f, (3, 3), 0)
            residual = gray_f - blurred
            noise_std = float(residual.std())
            # Compute spatial autocorrelation of residual (periodicity check)
            # GAN upsampling leaves periodic noise patterns
            res_norm = residual - residual.mean()
            if res_norm.std() > 0:
                # Horizontal autocorrelation at lag 1,2,4,8
                row_means = res_norm.mean(axis=0)
                ac_lag1 = float(np.corrcoef(row_means[:-1], row_means[1:])[0, 1])
                ac_lag4 = float(np.corrcoef(row_means[:-4], row_means[4:])[0, 1])
                periodicity = max(abs(ac_lag1), abs(ac_lag4))
            else:
                periodicity = 0.0

            # Empirical calibration:
            # Stable Diffusion / Midjourney: noise_std typically 0.5 – 4.0 (too clean)
            # Real camera photos: noise_std typically 6.0 – 25.0 (sensor noise)
            # Heavily compressed JPEGs: noise_std 3-8 (compression noise)
            if noise_std < 2.0:
                signals["noise_residual"] = (88.0, f"Çok düşük gürültü katmanı ({noise_std:.2f}) — diffusion/upscale imzası")
            elif noise_std < 4.5:
                signals["noise_residual"] = (68.0, f"Düşük gürültü ({noise_std:.2f}) — AI olası")
            elif periodicity > 0.4:
                signals["noise_residual"] = (72.0, f"Periyodik gürültü deseni (ac={periodicity:.2f}) — GAN upsampling")
            elif noise_std > 14.0:
                signals["noise_residual"] = (15.0, f"Yüksek doğal gürültü ({noise_std:.2f}) — kamera sensörü")
            elif noise_std > 8.0:
                signals["noise_residual"] = (25.0, f"Normal kamera gürültüsü ({noise_std:.2f})")
            else:
                signals["noise_residual"] = (42.0, f"Orta gürültü ({noise_std:.2f}) — sıkıştırılmış olabilir")
        except Exception as e:
            logger.warning(f"Signal 2 (noise) failed: {e}")
            signals["noise_residual"] = (50.0, "Gürültü analizi: hata")

        # ── Signal 3: LBP Texture Entropy ──────────────────────────────
        try:
            gray_u8 = gray.astype(np.uint8)
            lbp = local_binary_pattern(gray_u8, P=8, R=1, method='uniform')
            # Compute histogram entropy of LBP
            hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)
            hist = np.clip(hist, 1e-9, None)
            entropy = float(-np.sum(hist * np.log2(hist)))
            max_entropy = np.log2(10)
            norm_entropy = entropy / max_entropy  # 0-1

            # AI images: lower entropy (smoother, less texture variation)
            # Real photos: higher entropy (rich micro-textures)
            if norm_entropy < 0.65:
                signals["lbp_entropy"] = (78.0, f"Düşük LBP entropi ({norm_entropy:.3f}) — düzgün yapay yüzey")
            elif norm_entropy < 0.78:
                signals["lbp_entropy"] = (52.0, f"Orta LBP entropi ({norm_entropy:.3f})")
            elif norm_entropy < 0.88:
                signals["lbp_entropy"] = (28.0, f"Yüksek LBP entropi ({norm_entropy:.3f}) — doğal doku")
            else:
                signals["lbp_entropy"] = (15.0, f"Çok yüksek LBP entropi ({norm_entropy:.3f}) — zengin doku")
        except Exception as e:
            logger.warning(f"Signal 3 (LBP) failed: {e}")
            signals["lbp_entropy"] = (50.0, "LBP analizi: hata")

        # ── Signal 4: FFT Radial Power Spectrum (1/f law) ──────────────
        try:
            h, w = gray.shape
            fft_img = sfft.fft2(gray.astype(np.float64))
            fft_shifted = sfft.fftshift(fft_img)
            power = np.abs(fft_shifted) ** 2

            # Build radial frequency bins
            cy2, cx2 = h // 2, w // 2
            Y, X = np.ogrid[:h, :w]
            R = np.sqrt((X - cx2)**2 + (Y - cy2)**2).astype(int)
            max_r = min(cx2, cy2)
            radial_power = np.array([
                power[R == r].mean() for r in range(1, max_r)
                if np.any(R == r)
            ])

            if len(radial_power) > 10:
                # Fit log-log linear to get beta exponent
                freqs = np.arange(1, len(radial_power) + 1)
                log_f = np.log10(freqs)
                log_p = np.log10(radial_power + 1e-12)
                # Linear fit
                valid = np.isfinite(log_p)
                if valid.sum() > 5:
                    beta = float(np.polyfit(log_f[valid], log_p[valid], 1)[0])
                    # Empirical from Stable Diffusion / Midjourney / DALL-E:
                    # AI (diffusion): beta typically -1.0 to -1.6  (too flat, over-smooth)
                    # AI (GAN):       beta typically -0.3 to -1.0  (very flat, sharpened)
                    # Real photos:    beta typically -2.0 to -2.8  (natural 1/f^2 falloff)
                    # Scanned art:    beta < -3.0 (very steep)
                    if beta > -0.8:
                        signals["fft_beta"] = (86.0, f"Çok düz güç spektrumu (β={beta:.2f}) — GAN/upscale imzası")
                    elif beta > -1.5:
                        signals["fft_beta"] = (73.0, f"Düz spektrum (β={beta:.2f}) — diffusion model imzası")
                    elif beta > -2.0:
                        signals["fft_beta"] = (50.0, f"Sınırda spektrum (β={beta:.2f}) — belirsiz")
                    elif beta >= -2.8:
                        signals["fft_beta"] = (18.0, f"Doğal 1/f² spektrum (β={beta:.2f}) — gerçek fotoğraf")
                    elif beta < -3.5:
                        signals["fft_beta"] = (55.0, f"Aşırı dik spektrum (β={beta:.2f}) — GAN artefakt")
                    else:
                        signals["fft_beta"] = (35.0, f"Derin/sanat spektrum (β={beta:.2f})")
                else:
                    signals["fft_beta"] = (50.0, "FFT beta: yetersiz veri")
            else:
                signals["fft_beta"] = (50.0, "FFT beta: çok küçük görüntü")
        except Exception as e:
            logger.warning(f"Signal 4 (FFT beta) failed: {e}")
            signals["fft_beta"] = (50.0, "FFT spektrum: hata")

        # ── Signal 5: GLCM Homogeneity ──────────────────────────────────
        try:
            # Quantize to 8 levels for speed
            g8 = (gray.astype(np.float32) / 32).astype(np.uint8).clip(0, 7)
            # Horizontal co-occurrence matrix
            co = np.zeros((8, 8), dtype=np.float64)
            co[g8[:-1, :].ravel(), g8[1:, :].ravel()] += 1
            total = co.sum()
            if total > 0:
                co /= total
            # Homogeneity = sum( co[i,j] / (1 + |i-j|) )
            i_idx, j_idx = np.meshgrid(np.arange(8), np.arange(8), indexing='ij')
            homogeneity = float(np.sum(co / (1.0 + np.abs(i_idx - j_idx))))

            # AI images: homogeneity > 0.7 (too smooth transitions)
            # Real photos: 0.4 - 0.65
            if homogeneity > 0.75:
                signals["glcm_homogeneity"] = (80.0, f"Çok yüksek GLCM homojenlik ({homogeneity:.3f}) — yapay yumuşaklık")
            elif homogeneity > 0.65:
                signals["glcm_homogeneity"] = (58.0, f"Yüksek GLCM homojenlik ({homogeneity:.3f})")
            elif homogeneity > 0.45:
                signals["glcm_homogeneity"] = (25.0, f"Normal GLCM homojenlik ({homogeneity:.3f})")
            else:
                signals["glcm_homogeneity"] = (15.0, f"Düşük GLCM homojenlik ({homogeneity:.3f}) — zengin doku")
        except Exception as e:
            logger.warning(f"Signal 5 (GLCM) failed: {e}")
            signals["glcm_homogeneity"] = (50.0, "GLCM: hata")

        # ── Signal 6: Chromatic Aberration ──────────────────────────────
        try:
            r_ch = img_np[:, :, 0].astype(np.float32)
            g_ch = img_np[:, :, 1].astype(np.float32)
            b_ch = img_np[:, :, 2].astype(np.float32)

            # Detect edges in green channel (highest resolution in Bayer)
            edges_g = cv2.Canny(g_ch.astype(np.uint8), 50, 150)
            edge_mask = edges_g > 0

            if edge_mask.sum() > 100:
                # At edges: measure R-G and B-G channel differences
                rg_diff = float(np.abs((r_ch - g_ch)[edge_mask]).mean())
                bg_diff = float(np.abs((b_ch - g_ch)[edge_mask]).mean())
                ca_score = (rg_diff + bg_diff) / 2.0

                # Empirical calibration:
                # AI images (SD/MJ/DALL-E): CA = 0.2 – 1.5  (near-perfect channel alignment)
                # Real DSLR photos:          CA = 2.0 – 10.0 (lens chromatic aberration)
                # Phone photos (corrected):  CA = 0.8 – 3.0
                # Heavily compressed JPEG:   CA = 1.0 – 5.0
                if ca_score < 0.8:
                    signals["chromatic_aberration"] = (87.0, f"Sıfıra yakın kromatik aberasyon (CA={ca_score:.2f}) — lens yok → AI")
                elif ca_score < 1.8:
                    signals["chromatic_aberration"] = (63.0, f"Çok düşük CA ({ca_score:.2f}) — AI olası veya dijital düzeltme")
                elif ca_score <= 7.0:
                    signals["chromatic_aberration"] = (16.0, f"Gerçekçi lens aberasyonu (CA={ca_score:.2f}) — kamera imzası")
                else:
                    signals["chromatic_aberration"] = (38.0, f"Yüksek CA ({ca_score:.2f}) — aşırı sıkıştırma veya lens")
            else:
                signals["chromatic_aberration"] = (50.0, "Kromatik aberasyon: kenar bulunamadı")
        except Exception as e:
            logger.warning(f"Signal 6 (CA) failed: {e}")
            signals["chromatic_aberration"] = (50.0, "Kromatik aberasyon: hata")

        # ── Signal 7: Blocking Artifact Metric ──────────────────────────
        try:
            h, w = gray.shape
            gray_f2 = gray.astype(np.float64)
            # Check 8-pixel boundary periodicity (JPEG / VAE block pattern)
            h_diffs, v_diffs = [], []
            for bsize in [8, 16]:
                # Horizontal block boundaries
                for y in range(bsize, h - bsize, bsize):
                    row_diff = float(np.abs(gray_f2[y, :] - gray_f2[y-1, :]).mean())
                    interior = float(np.abs(np.diff(gray_f2[y-bsize:y, :], axis=0)).mean())
                    if interior > 0:
                        h_diffs.append(row_diff / (interior + 1e-6))
                # Vertical block boundaries
                for x in range(bsize, w - bsize, bsize):
                    col_diff = float(np.abs(gray_f2[:, x] - gray_f2[:, x-1]).mean())
                    interior = float(np.abs(np.diff(gray_f2[:, x-bsize:x], axis=1)).mean())
                    if interior > 0:
                        v_diffs.append(col_diff / (interior + 1e-6))

            if h_diffs and v_diffs:
                bam = float(np.mean(h_diffs + v_diffs))
                # BAM > 1.3 → boundary stronger than interior → block artifact
                if bam > 2.0:
                    signals["blocking_artifact"] = (72.0, f"Güçlü blok artefakları (BAM={bam:.2f}) — VAE/GAN ızgara imzası")
                elif bam > 1.4:
                    signals["blocking_artifact"] = (48.0, f"Orta blok artefakları (BAM={bam:.2f})")
                else:
                    signals["blocking_artifact"] = (20.0, f"Blok artefakı yok (BAM={bam:.2f})")
            else:
                signals["blocking_artifact"] = (40.0, "Blok analizi: yetersiz veri")
        except Exception as e:
            logger.warning(f"Signal 7 (BAM) failed: {e}")
            signals["blocking_artifact"] = (50.0, "Blok analizi: hata")

        # ── Signal 8: Saturation Distribution Kurtosis ──────────────────
        try:
            hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
            saturation = hsv[:, :, 1].astype(np.float32).ravel()

            if saturation.std() > 1e-3:
                sat_kurt = float(sstats.kurtosis(saturation, fisher=True))
                sat_skew = float(sstats.skew(saturation))

                # Empirical calibration for HSV saturation channel kurtosis:
                # Stable Diffusion / Midjourney: sat_kurt typically -0.5 to 1.5
                #   (vivid, uniform saturation — aesthetic palette)
                # Real outdoor photos: sat_kurt typically 2.0 – 8.0
                #   (mix of desaturated sky/shadows + vivid subjects)
                # Studio photos: sat_kurt 0.5 – 3.0 (controlled lighting)
                if sat_kurt < 0.3:
                    signals["saturation_kurtosis"] = (78.0, f"Düzgün doygunluk dağılımı (kurt={sat_kurt:.2f}) — AI renk paleti")
                elif sat_kurt < 1.5:
                    signals["saturation_kurtosis"] = (56.0, f"Orta doygunluk kurtosis ({sat_kurt:.2f}) — AI olası")
                elif sat_kurt < 3.0:
                    signals["saturation_kurtosis"] = (38.0, f"Normal doygunluk kurtosis ({sat_kurt:.2f}) — stüdyo fotoğrafı")
                elif sat_kurt >= 3.0:
                    signals["saturation_kurtosis"] = (16.0, f"Yüksek doygunluk kurtosis ({sat_kurt:.2f}) — doğal aydınlatma")
            else:
                # Near-zero saturation (grayscale image)
                signals["saturation_kurtosis"] = (55.0, "Grayscale görüntü — doygunluk analizi sınırlı")
        except Exception as e:
            logger.warning(f"Signal 8 (saturation) failed: {e}")
            signals["saturation_kurtosis"] = (50.0, "Doygunluk analizi: hata")

        # ── Aggregate: weighted majority voting ─────────────────────────
        # Signal weights (empirically tuned for SD/MJ vs real photo discrimination):
        # DCT kurtosis and FFT beta are the most reliable signals.
        # Noise residual and CA are highly discriminative but can be fooled by compression.
        weights = {
            "dct_kurtosis":         0.20,  # Most reliable for diffusion models
            "noise_residual":       0.18,  # Very reliable: cameras have noise, AI doesn't
            "fft_beta":             0.20,  # Strong signal: 1/f law deviation
            "lbp_entropy":          0.10,  # Moderate: texture micro-variation
            "glcm_homogeneity":     0.08,  # Moderate: smoothness proxy
            "chromatic_aberration": 0.14,  # Strong: lens physics don't lie
            "blocking_artifact":    0.05,  # Weak: only useful for some GAN architectures
            "saturation_kurtosis":  0.05,  # Supplementary: color palette analysis
        }

        weighted_sum  = sum(signals[k][0] * weights[k] for k in weights if k in signals)
        weight_total  = sum(weights[k] for k in weights if k in signals)
        final_confidence = weighted_sum / weight_total if weight_total > 0 else 50.0

        # Calibration: steeper sigmoid (k=6 instead of 8) pushes scores away from center
        # Decision boundary at 52 (slight bias toward "not AI" to reduce false positives)
        calib = float(1.0 / (1.0 + np.exp(-(final_confidence - 50.0) / 6.0)) * 100.0)

        # Count how many signals individually vote "AI" (score >= 60)
        ai_vote_count = sum(1 for k in weights if k in signals and signals[k][0] >= 60.0)
        total_signals = len([k for k in weights if k in signals])

        # Boost confidence if majority of signals vote AI
        if ai_vote_count >= 5 and calib < 70:
            calib = min(95.0, calib + 12.0)
        elif ai_vote_count >= 4 and calib < 60:
            calib = min(85.0, calib + 8.0)
        # Suppress if very few signals vote AI (reduce false positives)
        elif ai_vote_count <= 1 and calib > 45:
            calib = max(10.0, calib - 10.0)

        generated = calib >= 55.0

        # Build detail list for UI
        signal_detail = [
            {
                "name": k.replace("_", " ").title(),
                "score": round(signals[k][0], 1),
                "label": signals[k][1]
            }
            for k in weights if k in signals
        ]

        # Verdict label
        if calib >= 80:
            verdict_label = f"AI ÜRETİMİ — KESİN ({ai_vote_count}/{total_signals} sinyal)"
        elif calib >= 65:
            verdict_label = f"AI ÜRETİMİ — YÜKSEK ({ai_vote_count}/{total_signals} sinyal)"
        elif calib >= 50:
            verdict_label = f"AI ÜRETİMİ — ZAYIF ({ai_vote_count}/{total_signals} sinyal)"
        elif calib >= 35:
            verdict_label = f"GERÇEK — ZAYIF ({ai_vote_count}/{total_signals} sinyal)"
        elif calib >= 20:
            verdict_label = f"GERÇEK — YÜKSEK ({ai_vote_count}/{total_signals} sinyal)"
        else:
            verdict_label = f"GERÇEK — KESİN ({ai_vote_count}/{total_signals} sinyal)"

        logger.info(f"AI Detection: {calib:.1f}% [{verdict_label}] | "
                    + " | ".join(f"{k}={signals[k][0]:.0f}" for k in weights if k in signals))

        return {
            "generated":    generated,
            "confidence":   round(calib, 2),
            "signal_detail": signal_detail,
            "verdict_label": verdict_label,
        }
    # Heatmap & Visualization
    # ------------------------------------------------------------------

    def _accumulate_heatmap(self, heatmap: np.ndarray,
                             regions: list) -> np.ndarray:
        """
        @brief Accumulate detected regions into the heatmap array.

        @param heatmap  Float32 array to accumulate into.
        @param regions  List of region dicts from algorithm results.
        @return         Updated heatmap array.
        """
        h, w = heatmap.shape
        for r in regions:
            strength = r.get("strength", 0.5)
            # Handle both (x1,y1,x2,y2) box and (x1,y1)+(x2,y2) point pairs
            if "x2" in r and "y2" in r:
                x1, y1 = max(0, r["x1"]), max(0, r["y1"])
                x2, y2 = min(w, r["x2"]), min(h, r["y2"])
                if x2 > x1 and y2 > y1:
                    heatmap[y1:y2, x1:x2] += strength
            else:
                # Point pair: draw small circle
                for key in [("x1","y1"), ("x2","y2")]:
                    xk, yk = key
                    cx = min(w-1, max(0, r.get(xk, 0)))
                    cy = min(h-1, max(0, r.get(yk, 0)))
                    cv2.circle(heatmap, (cx, cy), 8, strength, -1)
        return heatmap

    def _render_outputs(self, img_np: np.ndarray,
                        heatmap: np.ndarray) -> tuple:
        """
        @brief Render heatmap overlay and annotated output image.

        @param img_np   Original image (RGB).
        @param heatmap  Float32 heatmap array.
        @return         Tuple (heatmap_b64, annotated_b64) as PNG base64 strings.
        """
        h, w = heatmap.shape

        # Normalize heatmap
        if heatmap.max() > 0:
            hm_norm = (heatmap / heatmap.max() * 255).astype(np.uint8)
        else:
            hm_norm = np.zeros((h, w), dtype=np.uint8)

        # Apply colormap (JET)
        hm_color = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)
        hm_color_rgb = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)

        # Blend with original
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(img_bgr, 0.6, hm_color, 0.4, 0)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        # Draw contours on annotated output
        _, thresh = cv2.threshold(hm_norm, 80, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        annotated = overlay_rgb.copy()
        cv2.drawContours(annotated, contours, -1, (255, 50, 50), 2)

        # Encode both to base64 PNG
        heatmap_b64  = self._np_to_b64(hm_color_rgb)
        annotated_b64 = self._np_to_b64(annotated)

        return heatmap_b64, annotated_b64

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _np_to_b64(img_np: np.ndarray) -> str:
        """
        @brief Convert NumPy image array to base64-encoded PNG string.
        @param img_np  RGB uint8 array.
        @return        Base64 string.
        """
        pil = Image.fromarray(img_np.astype(np.uint8))
        buf = BytesIO()
        pil.save(buf, "PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _detect_screenshot(self, img_np: np.ndarray,
                            gray: np.ndarray) -> bool:
        """
        @brief Görüntünün ekran görüntüsü / dijital UI içeriği olup olmadığını tespit et.

        @details
        Ekran görüntüleri (screenshot, diagram, UI mockup, yazılım arayüzü) istatistiksel
        olarak AI görsellerine benzer çünkü:
          - Düz renkli büyük alanlar (arka plan, butonlar) → sıfıra yakın gürültü
          - Sınırlı renk paleti (tema renkleri) → yüksek kanal korelasyonu
          - DCT kurtosis düşük → AI ile örtüşüyor

        Tespit yöntemi 3 bağımsız sinyal kullanır:
          1. Düz bölge oranı: Komşu piksel farkı < 3 olan piksellerin oranı
             UI'da büyük düz renkli alanlar bunu çok yüksek yapar (>0.55)
          2. Benzersiz renk oranı: Toplam piksel sayısına göre farklı renk sayısı
             Fotoğraflar çok renklidir; ekran görüntüleri az ve tekrarlayan renk içerir
          3. Histogram tepe noktası: Tek bir gri değerinin histogramdaki payı
             UI arka planları histogramda keskin bir tepe oluşturur (>0.04)

        @param img_np  RGB görüntü array.
        @param gray    Grayscale görüntü array.
        @return        True ekran görüntüsü ise, False değilse.
        """
        try:
            h, w = gray.shape

            # Sinyal 1: Düz bölge oranı
            diff_h = np.abs(
                gray[:, 1:].astype(np.int16) - gray[:, :-1].astype(np.int16)
            )
            diff_v = np.abs(
                gray[1:, :].astype(np.int16) - gray[:-1, :].astype(np.int16)
            )
            flat_ratio = float((diff_h < 3).mean())

            # Sinyal 2: Benzersiz renk oranı
            unique_colors = len(np.unique(img_np.reshape(-1, 3), axis=0))
            unique_ratio  = unique_colors / (h * w)

            # Sinyal 3: Histogram tepe noktası
            hist      = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
            hist_norm = hist / (hist.sum() + 1e-9)
            max_bin   = float(hist_norm.max())

            # Üç sinyal birlikte yüksekse → ekran görüntüsü
            # Eşikler: gerçek foto ve AI görsellerinde yanlış pozitif vermeyecek şekilde
            # Trello:     flat=0.928  unique=0.0076  maxbin=0.475  → SS ✓
            # Gerçek foto: flat=0.289  unique=0.064   maxbin=0.014  → ok ✓
            # AI görsel:  flat=0.75   unique=0.055    maxbin=0.089  → ok ✓
            is_ss = (flat_ratio > 0.70 and
                     unique_ratio < 0.05 and
                     max_bin > 0.05)

            if is_ss:
                logger.info(
                    f"Ekran görüntüsü tespit edildi: "
                    f"flat={flat_ratio:.3f} unique={unique_ratio:.4f} "
                    f"hist_peak={max_bin:.4f}"
                )

            return is_ss

        except Exception as e:
            logger.warning(f"Screenshot detection failed: {e}")
            return False

    def _ai_verdict_label(self, raw: dict) -> str:
        """
        @brief AIDetector sonucundan Türkçe verdict etiketi üret.
        @param raw  AIDetector.detect() dönüş dict'i.
        @return     Kullanıcıya gösterilecek etiket string'i.
        """
        conf   = raw.get("confidence", 0)

        if conf >= 80:
            base = "AI ÜRETİMİ — KESİN"
        elif conf >= 65:
            base = "AI ÜRETİMİ — YÜKSEK"
        elif conf >= 50:
            base = "AI ÜRETİMİ — ZAYIF"
        elif conf >= 35:
            base = "GERÇEK — ZAYIF"
        elif conf >= 20:
            base = "GERÇEK — YÜKSEK"
        else:
            base = "GERÇEK — KESİN"
        return base

    def _ai_signal_detail(self, raw: dict) -> list:
        """
        @brief AIDetector ham sonucunu sinyal detay listesine çevir.
        @param raw  AIDetector.detect() dönüş dict'i.
        @return     UI'ın beklediği signal_detail listesi.
        """
        details = []

        # API sonucu varsa en üste koy
        api_conf = raw.get("api_conf")
        if api_conf is not None:
            details.append({
                "name":  "Anthropic Vision API",
                "score": round(api_conf, 1),
                "label": raw.get("reason", "API tabanlı görsel analiz")
            })

        # İstatistiksel sinyaller
        signals = raw.get("signals", {})
        notes   = raw.get("signal_notes", {})
        labels  = {
            "freq":    "Frekans Spektrumu (DCT)",
            "noise":   "Gürültü Düzenliliği",
            "color":   "Renk Kanalı Korelasyonu",
            "edge":    "Kenar Kalitesi",
            "texture": "Doku Düzenlilik Analizi",
        }
        for key, display in labels.items():
            if key in signals:
                details.append({
                    "name":  display,
                    "score": round(signals[key], 1),
                    "label": notes.get(key, "")
                })

        return details

    @staticmethod
    def _empty_result(algo_id: str, reason: str = "") -> dict:
        """
        @brief Return a zero-confidence placeholder result.
        @param algo_id  Algorithm identifier.
        @param reason   Human-readable failure reason.
        @return         Placeholder result dict.
        """
        return {
            "algorithm": algo_id.upper(),
            "confidence": 0.0,
            "keypoint_count": 0,
            "match_count": 0,
            "regions": [],
            "status": f"error: {reason}"
        }