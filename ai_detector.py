"""
@file ai_detector.py
@brief AI-Generated Image Detector — Üç Katmanlı Hibrit Sistem
@details
  Katman 1: CLIP Lokal Model (clip_weights.pth varsa) — %85-90 doğruluk
  Katman 2: Anthropic Vision API (ANTHROPIC_API_KEY varsa) — %90+ doğruluk
  Katman 3: İstatistiksel Analiz (her zaman aktif) — yedek/destekleyici

  Öncelik: CLIP > Anthropic API > İstatistiksel
  CLIP dosyaları yoksa istatistiksel çalışır.

@author ImageForensics Team
@version 3.0.0
"""

import os
import cv2
import base64
import logging
import numpy as np
import requests
from io import BytesIO
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)

_BASE_DIR      = Path(__file__).parent
CLIP_WEIGHTS   = _BASE_DIR / "clip_weights.pth"
CLIP_PROCESSOR = _BASE_DIR / "clip_processor"

ANTHROPIC_API   = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = "claude-opus-4-5"

# Lazy-loaded globals
_clip_model     = None
_clip_processor = None
_clip_loaded    = False   # False=denenmedi, True=OK, None=başarısız


def _load_clip():
    """@brief CLIP modelini lokal dosyalardan yükle (tek seferlik)."""
    global _clip_model, _clip_processor, _clip_loaded

    if _clip_loaded is not False:
        return _clip_loaded is True

    if not CLIP_PROCESSOR.exists():
        logger.info("CLIP dosyaları yok — istatistiksel mod aktif")
        _clip_loaded = None
        return False

    # Yöntem 1: clip_processor klasörü içinde tam model varsa (önerilen)
    model_safe = CLIP_PROCESSOR / "model.safetensors"
    model_bin  = CLIP_PROCESSOR / "pytorch_model.bin"
    has_full_model = model_safe.exists() or model_bin.exists()

    # Yöntem 2: ayrı clip_weights.pth dosyası varsa
    has_weights = CLIP_WEIGHTS.exists()

    if not has_full_model and not has_weights:
        logger.info("CLIP ağırlık dosyası bulunamadı — istatistiksel mod aktif")
        _clip_loaded = None
        return False

    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor as CP

        logger.info(f"CLIP processor yükleniyor: {CLIP_PROCESSOR}")
        proc = CP.from_pretrained(str(CLIP_PROCESSOR))

        if has_full_model:
            # Tam model clip_processor içinde — doğrudan yükle (internet gerekmez)
            logger.info("CLIP tam model yükleniyor (clip_processor/)...")
            model = CLIPModel.from_pretrained(str(CLIP_PROCESSOR))
        else:
            # Sadece weights.pth var — mimari için internete bağlanır
            logger.info(f"CLIP weights yükleniyor: {CLIP_WEIGHTS}")
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            state     = torch.load(str(CLIP_WEIGHTS), map_location="cpu")
            model_keys = set(model.state_dict().keys())
            filtered   = {k: v for k, v in state.items() if k in model_keys}
            model.load_state_dict(filtered, strict=False)

        model.eval()
        _clip_processor = proc
        _clip_model     = model
        _clip_loaded    = True
        logger.info("CLIP modeli başarıyla yüklendi")
        return True

    except Exception as e:
        logger.warning(f"CLIP yüklenemedi: {e}")
        _clip_loaded = None
        return False


class AIDetector:
    """
    @class AIDetector
    @brief Üç katmanlı hibrit AI görüntü detektörü.

    @details
    Öncelik sırası:
      1. CLIP lokal model — internet gerekmez, evde/okulda çalışır
      2. Anthropic Vision API — ANTHROPIC_API_KEY env var gerekli
      3. İstatistiksel analiz — her durumda aktif (yedek)

    Semantik kaynak varsa: 75% semantik + 25% istatistiksel
    """

    def __init__(self):
        self._api_key       = os.environ.get("ANTHROPIC_API_KEY", "")
        self._api_available = bool(self._api_key)
        self._clip_ready    = _load_clip()

        if self._clip_ready:
            logger.info("AIDetector v3: CLIP lokal model aktif")
        elif self._api_available:
            logger.info("AIDetector v3: Anthropic API aktif")
        else:
            logger.info("AIDetector v3: Sadece istatistiksel mod")

    # ------------------------------------------------------------------
    # Ana metod
    # ------------------------------------------------------------------

    def detect(self, img_np: np.ndarray, img_bytes: bytes) -> dict:
        """
        @brief Görüntünün AI üretimi olup olmadığını tespit et.
        @param img_np    RGB uint8 numpy array.
        @param img_bytes Ham görüntü baytları.
        @return          generated, confidence, method, signals içeren dict.
        """
        gray     = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        stat_res = self._statistical_analysis(img_np, gray)

        if self._clip_ready:
            clip_res = self._clip_inference(img_np)
            if clip_res is not None:
                return self._combine(stat_res, clip_res, source="CLIP Lokal Model")

        if self._api_available:
            api_res = self._anthropic_vision(img_bytes)
            if api_res is not None:
                return self._combine(stat_res, api_res, source="Anthropic Vision API")

        return self._combine(stat_res, None, source="İstatistiksel Analiz")

    # ------------------------------------------------------------------
    # Katman 1: CLIP Zero-Shot
    # ------------------------------------------------------------------

    def _clip_inference(self, img_np: np.ndarray) -> dict | None:
        """
        @brief CLIP zero-shot ile AI/gerçek sınıflandırma.

        @details
        CLIP görüntüyü text açıklamalarıyla cosine similarity üzerinden karşılaştırır.
        3 gerçek + 3 AI metin promptu kullanılır, sonuçlar ensemble ile birleştirilir.
        Bu yöntem istatistiksel sinyallerin göremediği semantik özellikleri yakalar:
        yapay cilt, fizik hataları, hayal ürünü sahneler, garip anatomik bozukluklar.

        @param img_np  RGB görüntü array.
        @return        confidence ve generated içeren dict, hata durumunda None.
        """
        try:
            import torch

            pil = Image.fromarray(img_np)

            real_texts = [
                "a real photograph taken with a camera",
                "a genuine photo with natural camera noise and real lighting",
                "a photo taken by a person with a smartphone or DSLR",
            ]
            ai_texts = [
                "an AI generated image created by a neural network",
                "a synthetic image made by Midjourney or Stable Diffusion",
                "a digitally generated photorealistic image that is not real",
            ]
            all_texts = real_texts + ai_texts

            inputs = _clip_processor(
                text=all_texts, images=pil,
                return_tensors="pt", padding=True, truncation=True
            )
            with torch.no_grad():
                out    = _clip_model(**inputs)
                probs  = out.logits_per_image.softmax(dim=1).squeeze().tolist()

            real_p = sum(probs[:len(real_texts)])
            ai_p   = sum(probs[len(real_texts):])
            total  = real_p + ai_p + 1e-9
            ai_pct = (ai_p / total) * 100.0

            logger.info(f"CLIP: AI={ai_pct:.1f}% Gerçek={real_p/total*100:.1f}%")

            return {
                "ai_generated": ai_pct >= 52.0,
                "confidence":   round(ai_pct, 2),
                "reason":       f"CLIP: AI={ai_pct:.1f}% / Gerçek={real_p/total*100:.1f}%",
            }
        except Exception as e:
            logger.warning(f"CLIP hatası: {e}")
            return None

    # ------------------------------------------------------------------
    # Katman 2: Anthropic Vision API
    # ------------------------------------------------------------------

    def _anthropic_vision(self, img_bytes: bytes) -> dict | None:
        """@brief Anthropic Claude Vision ile AI tespiti."""
        try:
            pil = Image.open(BytesIO(img_bytes)).convert("RGB")
            ratio = min(768/pil.width, 768/pil.height, 1.0)
            if ratio < 1.0:
                pil = pil.resize((int(pil.width*ratio), int(pil.height*ratio)), Image.LANCZOS)
            buf = BytesIO()
            pil.save(buf, "JPEG", quality=85)
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            prompt = (
                "Bu görüntüyü incele. SADECE JSON döndür:\n"
                '{"ai_generated": true/false, "confidence": 0-100, '
                '"reason": "max 80 karakter"}\n'
                "AI işaretleri: yapay cilt, garip eller, fizik hataları, "
                "hayal ürünü sahne. Sadece JSON."
            )
            resp = requests.post(
                ANTHROPIC_API,
                headers={"x-api-key": self._api_key,
                         "anthropic-version": "2023-06-01",
                         "content-type": "application/json"},
                json={"model": ANTHROPIC_MODEL, "max_tokens": 120,
                      "messages": [{"role": "user", "content": [
                          {"type": "image", "source": {"type": "base64",
                           "media_type": "image/jpeg", "data": img_b64}},
                          {"type": "text", "text": prompt}
                      ]}]},
                timeout=20
            )
            if resp.status_code != 200:
                return None
            import json
            raw = resp.json()["content"][0]["text"].strip()
            if "```" in raw:
                raw = raw.split("```")[1].lstrip("json").strip()
            p = json.loads(raw)
            return {"ai_generated": bool(p.get("ai_generated", False)),
                    "confidence":   float(p.get("confidence", 50)),
                    "reason":       str(p.get("reason", ""))}
        except Exception as e:
            logger.warning(f"Anthropic API hatası: {e}")
            return None

    # ------------------------------------------------------------------
    # Katman 3: İstatistiksel Analiz
    # ------------------------------------------------------------------

    def _statistical_analysis(self, img_np: np.ndarray, gray: np.ndarray) -> dict:
        """@brief 5 bağımsız sinyal ile JPEG/karanlık toleranslı istatistiksel analiz."""
        signals = {
            "freq":    self._freq_analysis(gray),
            "noise":   self._noise_analysis(gray),
            "color":   self._color_analysis(img_np),
            "edge":    self._edge_analysis(gray),
            "texture": self._texture_analysis(gray),
        }
        weights = {"freq": 0.40, "noise": 0.30, "color": 0.10,
                   "edge": 0.12, "texture": 0.08}
        raw = sum(signals[k]["score"] * weights[k] for k in signals)

        import re
        bm = re.search(r'bright=([\d.]+)', signals["freq"].get("note", ""))
        brightness = float(bm.group(1)) if bm else 100.0
        n_note = signals["noise"].get("note", "")
        jpeg   = "jpeg=True" in n_note
        lm     = re.search(r'lap_s=([\d.]+)', n_note)
        lap_s  = float(lm.group(1)) if lm else 99.0

        if brightness < 80.0 and not jpeg and lap_s < 12.0:
            raw *= 0.72
        elif brightness < 100.0 and not jpeg and lap_s < 12.0:
            raw *= 0.88

        return {
            "confidence": round(self._sigmoid(raw, 47.0, 9.0), 2),
            "raw":        round(raw, 2),
            "signals":    {k: round(v["score"], 1) for k, v in signals.items()},
            "notes":      {k: v.get("note", "")    for k, v in signals.items()},
        }

    def _freq_analysis(self, gray: np.ndarray) -> dict:
        try:
            crop = cv2.resize(gray, (256, 256)).astype(np.float32)
            dct  = cv2.dct(crop / 255.0)
            mid  = float(np.abs(dct[32:96, 32:96]).sum()) / (np.abs(dct).sum()+1e-9)
            high = float(np.abs(dct[96:, 96:]).sum()) / (np.abs(dct).sum()+1e-9)
            from scipy.stats import kurtosis as sk
            kurt = float(sk(dct[4:64, 4:64].flatten(), fisher=True))
            brightness = float(gray.mean())
            dark = brightness < 85.0
            s = 0.0
            if not np.isfinite(kurt): s = 25
            elif dark:
                s = 80 if kurt<3 else 55 if kurt<10 else 30 if kurt<25 else 8 if kurt<60 else 0
            else:
                s = 85 if kurt<5 else 65 if kurt<18 else 42 if kurt<45 else 12 if kurt<100 else 0
            hm = high/(mid+1e-9)
            if hm > 1.5: s += 12
            return {"score": min(100.0,s), "note": f"kurt={kurt:.1f} bright={brightness:.0f} dark={dark}"}
        except Exception as e:
            return {"score": 40.0, "note": f"err:{e}"}

    def _noise_analysis(self, gray: np.ndarray) -> dict:
        try:
            med   = cv2.medianBlur(gray, 3)
            noise = gray.astype(np.float32) - med.astype(np.float32)
            n_std = float(noise.std())
            lap_s = float(cv2.Laplacian(gray, cv2.CV_64F).std())
            dnr   = n_std / (lap_s + 1e-6)
            jpeg  = n_std > 1.5 and lap_s > 15.0
            h, w  = gray.shape
            step  = max(32, min(h,w)//8)
            rs    = [float(noise[y:y+step,x:x+step].std())
                     for y in range(0,h-step,step) for x in range(0,w-step,step)]
            rcv   = float(np.std(rs)/(np.mean(rs)+1e-6)) if rs else 1.0
            s = 0.0
            if dnr < 0.10:   s += 70
            elif dnr < 0.18: s += 50
            elif dnr < 0.28: s += (15 if jpeg else 35)
            if n_std < 1.5 and not jpeg:   s += 20
            elif n_std < 2.5 and not jpeg: s += 10
            if rcv < 0.20: s += 18
            elif rcv < 0.40: s += 6
            return {"score": min(100.0,s), "note": f"dnr={dnr:.3f} n={n_std:.2f} lap_s={lap_s:.1f} jpeg={jpeg}"}
        except Exception as e:
            return {"score": 40.0, "note": f"err:{e}"}

    def _color_analysis(self, img_np: np.ndarray) -> dict:
        try:
            h, w = img_np.shape[:2]
            idx  = np.random.choice(h*w, min(8000,h*w), replace=False)
            r    = img_np[:,:,0].flatten()[idx].astype(np.float64)
            g    = img_np[:,:,1].flatten()[idx].astype(np.float64)
            b    = img_np[:,:,2].flatten()[idx].astype(np.float64)
            corr = (abs(float(np.corrcoef(r,g)[0,1]))+abs(float(np.corrcoef(r,b)[0,1]))+
                    abs(float(np.corrcoef(g,b)[0,1])))/3.0
            rm,gm,bm = img_np[:,:,0].mean(),img_np[:,:,1].mean(),img_np[:,:,2].mean()
            dom    = max(rm,gm,bm)/(rm+gm+bm+1e-6)
            bypass = dom>0.50 or (rm+gm+bm)/3<40
            def sm(h):
                h=h/(h.sum()+1e-9); return float(np.mean(np.abs(np.diff(h))))
            hs = (sm(cv2.calcHist([img_np],[0],None,[256],[0,256]).flatten())+
                  sm(cv2.calcHist([img_np],[1],None,[256],[0,256]).flatten()))/2
            s = 0.0
            if not bypass:
                s += 50 if corr>0.92 else 30 if corr>0.85 else 15 if corr>0.75 else 0
            elif corr>0.95: s += 10
            s += 25 if hs<0.0015 else 10 if hs<0.003 else 0
            return {"score": min(100.0,s), "note": f"corr={corr:.3f} bypass={bypass}"}
        except Exception as e:
            return {"score": 40.0, "note": f"err:{e}"}

    def _edge_analysis(self, gray: np.ndarray) -> dict:
        try:
            ratio = float(cv2.Canny(gray,100,200).mean())/(float(cv2.Canny(gray,50,150).mean())+1e-6)
            gx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)
            gy = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)
            mag = np.sqrt(gx**2+gy**2).flatten()
            mcv = mag.std()/(mag.mean()+1e-6)
            s = (35 if ratio>0.75 else 30 if ratio<0.20 else 10)
            s += (35 if mcv<1.5 else 15 if mcv<2.5 else 0)
            return {"score": min(100.0,s), "note": f"ratio={ratio:.2f} mag_cv={mcv:.2f}"}
        except Exception as e:
            return {"score": 40.0, "note": f"err:{e}"}

    def _texture_analysis(self, gray: np.ndarray) -> dict:
        try:
            dh = np.abs(gray[:,1:].astype(np.int16)-gray[:,:-1].astype(np.int16))
            dv = np.abs(gray[1:,:].astype(np.int16)-gray[:-1,:].astype(np.int16))
            dhm,dvm = float(dh.mean()),float(dv.mean())
            iso  = 1.0-abs(dhm-dvm)/(dhm+dvm+1e-6)
            sm   = 1.0/(1.0+(float(dh.std())+float(dv.std()))/2.0)
            jiso = iso>0.98 and dhm>3 and dvm>3
            s = 0.0
            if iso>0.97 and not jiso:   s += 28
            elif iso>0.95 and not jiso: s += 15
            elif iso>0.90:              s += 8
            s += 30 if sm>0.05 else 12 if sm>0.02 else 0
            return {"score": min(100.0,s), "note": f"iso={iso:.3f} jpeg_iso={jiso}"}
        except Exception as e:
            return {"score": 40.0, "note": f"err:{e}"}

    # ------------------------------------------------------------------
    # Birleştirme & Yardımcılar
    # ------------------------------------------------------------------

    def _combine(self, stat: dict, semantic: dict | None, source: str) -> dict:
        """@brief Semantik ve istatistiksel sonuçları birleştir (75/25)."""
        if semantic is not None:
            combined  = semantic["confidence"] * 0.75 + stat["confidence"] * 0.25
            generated = combined > 50.0
            reason    = semantic.get("reason", "")
        else:
            combined  = stat["confidence"]
            generated = combined > 50.0
            reason    = self._build_reason(stat)
        return {
            "generated":    generated,
            "confidence":   round(combined, 2),
            "method":       source,
            "reason":       reason,
            "stat_conf":    stat["confidence"],
            "api_conf":     semantic["confidence"] if semantic else None,
            "signals":      stat.get("signals", {}),
            "signal_notes": stat.get("notes", {}),
        }

    def _build_reason(self, stat: dict) -> str:
        s = stat.get("signals", {})
        n = [k for k in ["freq","noise","color","edge","texture"] if s.get(k,0)>60]
        labels = {"freq":"frekans yapay","noise":"gürültü uniform",
                  "color":"renk yapay","edge":"kenar yapay","texture":"doku izotropik"}
        return ", ".join(labels[k] for k in n) if n else "belirgin AI işareti yok"

    @staticmethod
    def _sigmoid(x: float, c: float = 50.0, s: float = 10.0) -> float:
        return float(100.0 / (1.0 + np.exp(-(x - c) / s)))
