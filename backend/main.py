"""
PixScale API — image resize & AI upscale service
FastAPI backend. OpenCV DNN SuperRes (EDSR/ESPCN/FSRCNN) + Pillow Lanczos.
"""
import io
import os
import time
import json
import uuid
import logging
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageOps
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

BASE = Path(__file__).parent.parent
UPLOAD_DIR = BASE / "uploads"
OUTPUT_DIR = BASE / "outputs"
MODELS_DIR = BASE / "models"
LOGS_DIR = BASE / "logs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# ── Логирование ──────────────────────────────────────────────────────────
# Пишем в файл /opt/pixscale/logs/pixscale.log (ротация 5 × 2 МБ)
# и в stdout (systemd journal).  Формат включает уровень и миллисекунды.
_fmt = logging.Formatter(
    "%(asctime)s.%(msecs)03d %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_file_h = RotatingFileHandler(LOGS_DIR / "pixscale.log", maxBytes=2_000_000, backupCount=5, encoding="utf-8")
_file_h.setFormatter(_fmt)
_stream_h = logging.StreamHandler()
_stream_h.setFormatter(_fmt)

logging.basicConfig(level=logging.INFO, handlers=[_file_h, _stream_h], force=True)
log = logging.getLogger("pixscale")
client_log = logging.getLogger("pixscale.client")

MAX_UPLOAD_MB = 40
MAX_OUT_PIXELS = 60_000_000   # ~60 MP protection

# Tile-based super-resolution:
# Разбиваем вход на тайлы с overlap, гоним SR на каждом, склеиваем с фидингом.
# Память зависит только от размера тайла, а не от размера изображения.
# На сервере 3.6 GB RAM без GPU — это единственный способ обрабатывать
# реальные фотографии (FullHD и выше).
TILE_SIZE = {
    # (model, scale) -> tile edge in pixels (input space)
    ("fsrcnn", 2): 384, ("fsrcnn", 3): 320, ("fsrcnn", 4): 256,
    ("espcn",  2): 384, ("espcn",  3): 320, ("espcn",  4): 256,
    # EDSR — тяжёлая, меньшие тайлы
    ("edsr",   2): 192, ("edsr",   3): 160, ("edsr",   4): 128,
}
TILE_OVERLAP = 16   # пикселей, прячет стыки между тайлами
ALLOWED_MIME = {
    "image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp", "image/tiff",
}

# AI model cache: {(model, scale): loaded_sr}
_model_cache: dict[tuple[str, int], "cv2.dnn_superres.DnnSuperResImpl"] = {}


def _load_sr(model: str, scale: int):
    """Load (or reuse cached) super-resolution model."""
    key = (model, scale)
    if key in _model_cache:
        return _model_cache[key]

    path = MODELS_DIR / f"{model.upper()}_x{scale}.pb"
    if not path.exists():
        raise FileNotFoundError(f"model not found: {path.name}")

    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(str(path))
    sr.setModel(model.lower(), scale)
    _model_cache[key] = sr
    log.info("loaded SR model %s x%d", model, scale)
    return sr


def _save_output(img_np_bgr: np.ndarray, fmt: str, quality: int = 92) -> Path:
    """Save BGR numpy image to outputs folder in given format."""
    fmt = fmt.lower()
    if fmt not in {"png", "jpg", "jpeg", "webp"}:
        fmt = "png"
    ext = "jpg" if fmt == "jpeg" else fmt

    name = f"pixscale_{uuid.uuid4().hex[:12]}.{ext}"
    out_path = OUTPUT_DIR / name

    # cv2 uses BGR; convert & use PIL to control quality/format for PNG/WEBP/JPG uniformly
    rgb = cv2.cvtColor(img_np_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)

    if ext == "png":
        pil.save(out_path, format="PNG", optimize=True)
    elif ext == "webp":
        pil.save(out_path, format="WEBP", quality=quality, method=6)
    else:  # jpg
        pil.save(out_path, format="JPEG", quality=quality, optimize=True, progressive=True)
    return out_path


def _pil_to_bgr(pil: Image.Image) -> np.ndarray:
    if pil.mode not in ("RGB", "RGBA"):
        pil = pil.convert("RGB")
    if pil.mode == "RGBA":
        bg = Image.new("RGB", pil.size, (255, 255, 255))
        bg.paste(pil, mask=pil.split()[3])
        pil = bg
    arr = np.array(pil)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _tiled_upsample(bgr: np.ndarray, model: str, scale: int) -> np.ndarray:
    """Tiled super-resolution — обрабатываем изображение кусками,
    чтобы не уложить сервер OOM-kill'ом. Память ограничена размером тайла.

    Возвращает апскейл BGR numpy-массив (h*scale, w*scale, 3).
    """
    sr = _load_sr(model, scale)
    h, w = bgr.shape[:2]
    tile = TILE_SIZE.get((model, scale), 256)
    ov = TILE_OVERLAP

    # Если изображение маленькое — один проход, без тайлов
    if h <= tile and w <= tile:
        return sr.upsample(bgr)

    out = np.zeros((h * scale, w * scale, 3), dtype=np.uint8)

    # Идём по сетке с перекрытием
    y = 0
    while y < h:
        x = 0
        # Границы тайла во входном пространстве (с overlap для стыков)
        y0 = max(0, y - ov)
        y1 = min(h, y + tile + ov)
        while x < w:
            x0 = max(0, x - ov)
            x1 = min(w, x + tile + ov)

            patch = bgr[y0:y1, x0:x1]
            up = sr.upsample(patch)

            # Позиции без overlap — что кладём в итоговый буфер
            dst_y0 = y * scale
            dst_y1 = min((y + tile), h) * scale
            dst_x0 = x * scale
            dst_x1 = min((x + tile), w) * scale

            # Смещения внутри апскейленного тайла (убираем overlap-поля)
            src_y0 = (y - y0) * scale
            src_y1 = src_y0 + (dst_y1 - dst_y0)
            src_x0 = (x - x0) * scale
            src_x1 = src_x0 + (dst_x1 - dst_x0)

            out[dst_y0:dst_y1, dst_x0:dst_x1] = up[src_y0:src_y1, src_x0:src_x1]

            x += tile
        y += tile

    return out


app = FastAPI(title="PixScale API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://pixscale.gornich.fun", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Логируем КАЖДЫЙ запрос: метод, путь, статус, длительность, IP."""
    start = time.perf_counter()
    rid = uuid.uuid4().hex[:8]
    ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "?").split(",")[0].strip()
    request.state.rid = rid

    log.info("→ %s %s %s %s", rid, ip, request.method, request.url.path)
    try:
        response = await call_next(request)
    except Exception:
        dt = (time.perf_counter() - start) * 1000
        log.exception("✗ %s crashed after %.0fms", rid, dt)
        return JSONResponse({"ok": False, "error": "internal server error", "rid": rid}, status_code=500)
    dt = (time.perf_counter() - start) * 1000
    level = logging.WARNING if response.status_code >= 400 else logging.INFO
    log.log(level, "← %s %d %.0fms %s", rid, response.status_code, dt, request.url.path)
    response.headers["X-Request-ID"] = rid
    return response


@app.exception_handler(HTTPException)
async def http_exc_handler(request: Request, exc: HTTPException):
    log.warning("HTTPException %s %s → %d: %s", request.method, request.url.path, exc.status_code, exc.detail)
    return JSONResponse({"ok": False, "error": exc.detail, "status": exc.status_code}, status_code=exc.status_code)


@app.post("/api/client-log")
async def client_log_endpoint(request: Request):
    """Принимает логи от браузера: ошибки, действия пользователя, состояние.

    Поле `level` — info|warn|error, `event` — короткий код, `data` — произвольный объект.
    Пишем в тот же файл, чтобы можно было tail-ом видеть связку клиент↔сервер.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    level = str(body.get("level", "info")).lower()
    event = str(body.get("event", "client"))[:60]
    data = body.get("data")
    ua = request.headers.get("user-agent", "?")[:140]
    ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "?").split(",")[0].strip()
    try:
        data_str = json.dumps(data, ensure_ascii=False, default=str)[:1500]
    except Exception:
        data_str = str(data)[:1500]
    msg = f"[CLIENT] ip={ip} ua={ua!r} event={event} data={data_str}"
    if level == "error":
        client_log.error(msg)
    elif level in ("warn", "warning"):
        client_log.warning(msg)
    else:
        client_log.info(msg)
    return {"ok": True}


@app.get("/api/health")
def health():
    return {"ok": True, "models": sorted({p.stem for p in MODELS_DIR.glob("*.pb")})}


@app.get("/api/info")
def info():
    """Available options for the frontend."""
    return {
        "max_upload_mb": MAX_UPLOAD_MB,
        "max_output_pixels": MAX_OUT_PIXELS,
        "models": [
            {"id": "fsrcnn", "label": "FSRCNN · быстрый", "scales": [2, 3, 4], "size": "tiny"},
            {"id": "espcn",  "label": "ESPCN · сбалансированный", "scales": [2, 3, 4], "size": "tiny"},
            {"id": "edsr",   "label": "EDSR · высокое качество (медленно)", "scales": [2, 3, 4], "size": "large"},
        ],
        "formats": ["png", "jpg", "webp"],
        "lanczos_modes": ["width", "height", "both", "percent", "ratio"],
    }


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    """Return image metadata after upload (preview endpoint)."""
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"file too large (max {MAX_UPLOAD_MB} MB)")
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)
        w, h = img.size
    except Exception as e:
        raise HTTPException(400, f"cannot decode image: {e}")

    # save temp
    tmp_name = f"in_{uuid.uuid4().hex[:12]}.{(img.format or 'png').lower()}"
    tmp_path = UPLOAD_DIR / tmp_name
    img.save(tmp_path)
    log.info("analyze ok size=%dx%d bytes=%d fmt=%s token=%s", w, h, len(data), img.format, tmp_name)
    return {
        "token": tmp_name,
        "width": w,
        "height": h,
        "ratio": round(w / h, 4) if h else 1,
        "format": (img.format or "").lower(),
        "size_bytes": len(data),
    }


def _load_upload(token: str) -> Image.Image:
    path = UPLOAD_DIR / token
    if not path.exists() or ".." in token or "/" in token:
        raise HTTPException(404, "upload not found")
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return img


@app.post("/api/resize")
async def resize_endpoint(
    token: str = Form(...),
    width: Optional[int] = Form(None),
    height: Optional[int] = Form(None),
    keep_ratio: bool = Form(True),
    fmt: str = Form("png"),
    quality: int = Form(92),
    resample: str = Form("lanczos"),   # lanczos | bicubic | bilinear | nearest
):
    """Classic resize via Pillow."""
    img = _load_upload(token)
    orig_w, orig_h = img.size

    if width is None and height is None:
        raise HTTPException(400, "width or height required")

    if keep_ratio:
        if width and not height:
            height = round(width * orig_h / orig_w)
        elif height and not width:
            width = round(height * orig_w / orig_h)
        else:
            # both given — pick the axis that produces smallest image inside box
            ratio_w = width / orig_w
            ratio_h = height / orig_h
            r = min(ratio_w, ratio_h)
            width = max(1, round(orig_w * r))
            height = max(1, round(orig_h * r))
    else:
        width = width or orig_w
        height = height or orig_h

    width = max(1, int(width))
    height = max(1, int(height))

    if width * height > MAX_OUT_PIXELS:
        raise HTTPException(400, f"output too large (> {MAX_OUT_PIXELS:,} px)")

    resample_map = {
        "lanczos": Image.Resampling.LANCZOS,
        "bicubic": Image.Resampling.BICUBIC,
        "bilinear": Image.Resampling.BILINEAR,
        "nearest": Image.Resampling.NEAREST,
    }
    r_filter = resample_map.get(resample, Image.Resampling.LANCZOS)

    # RGBA → flatten for jpg
    if fmt.lower() in ("jpg", "jpeg") and img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    elif img.mode not in ("RGB", "RGBA", "L"):
        img = img.convert("RGB")

    log.info("resize token=%s %dx%d→%dx%d filter=%s fmt=%s", token, orig_w, orig_h, width, height, resample, fmt)
    resized = img.resize((width, height), r_filter)

    # save
    bgr = _pil_to_bgr(resized.convert("RGB"))
    out = _save_output(bgr, fmt, quality)
    return {
        "ok": True,
        "file": out.name,
        "url": f"/api/download/{out.name}",
        "width": width,
        "height": height,
        "size_bytes": out.stat().st_size,
    }


@app.post("/api/upscale")
async def upscale_endpoint(
    token: str = Form(...),
    model: str = Form("fsrcnn"),   # fsrcnn | espcn | edsr
    scale: int = Form(2),          # 2 | 3 | 4
    fmt: str = Form("png"),
    quality: int = Form(92),
):
    """AI upscale через OpenCV DNN Super-Resolution (tiled, memory-safe)."""
    model = model.lower()
    if model not in {"fsrcnn", "espcn", "edsr"}:
        raise HTTPException(400, "invalid model")
    if scale not in (2, 3, 4):
        raise HTTPException(400, "scale must be 2, 3 or 4")

    img = _load_upload(token)
    w, h = img.size

    out_w, out_h = w * scale, h * scale
    if out_w * out_h > MAX_OUT_PIXELS:
        raise HTTPException(
            400,
            f"Результат был бы {out_w}×{out_h} (> {MAX_OUT_PIXELS/1e6:.0f} MP). "
            f"Уменьшите исходник или выберите меньший ×.",
        )

    log.info("upscale token=%s src=%dx%d model=%s x%d → %dx%d", token, w, h, model, scale, out_w, out_h)
    bgr = _pil_to_bgr(img)
    t0 = time.perf_counter()

    try:
        result = _tiled_upsample(bgr, model, scale)
    except MemoryError:
        raise HTTPException(
            507,
            "Не хватило памяти даже с тайловым режимом. "
            "Попробуйте FSRCNN вместо EDSR или меньший множитель.",
        )
    except Exception as e:
        log.exception("upscale failed token=%s model=%s x%d", token, model, scale)
        raise HTTPException(500, f"upscale failed: {e}")

    out = _save_output(result, fmt, quality)
    log.info("upscale ok %dx%d in %.1fs → %s (%d bytes)", result.shape[1], result.shape[0], time.perf_counter()-t0, out.name, out.stat().st_size)
    return {
        "ok": True,
        "file": out.name,
        "url": f"/api/download/{out.name}",
        "width": int(result.shape[1]),
        "height": int(result.shape[0]),
        "size_bytes": out.stat().st_size,
        "model": model,
        "scale": scale,
    }


@app.post("/api/enhance")
async def enhance_endpoint(
    token: str = Form(...),
    model: str = Form("fsrcnn"),        # fsrcnn | espcn  (для mode=ai)
    strength: str = Form("medium"),     # sharpen | denoise | ai
    fmt: str = Form("png"),
    quality: int = Form(92),
):
    """Улучшить качество БЕЗ изменения размера.

    Три режима (strength):
      - sharpen : мягкая резкость (UnsharpMask). Быстро, для слегка мягких фото.
      - denoise : сильный денойз + резкость (bilateral + UnsharpMask). Убирает
                  зернистость/компрессию, потом добавляет чёткости.
      - ai      : AI-восстановление (SR ×2 → Lanczos down → UnsharpMask).
                  Честно помогает на реально низкокачественных / пиксельных.
    """
    from PIL import ImageFilter

    img = _load_upload(token)
    orig_w, orig_h = img.size

    if orig_w * orig_h > 12_000_000:
        raise HTTPException(
            413,
            f"Для улучшения качества — макс. 12 MP ({orig_w}×{orig_h}). "
            f"Сначала уменьшите через вкладку 'Ресайз'.",
        )

    # Нормализуем режим
    if strength not in ("sharpen", "denoise", "ai"):
        # обратная совместимость со старыми значениями
        strength = {"light": "sharpen", "medium": "denoise", "strong": "ai"}.get(strength, "denoise")
    log.info("enhance token=%s %dx%d mode=%s model=%s", token, orig_w, orig_h, strength, model)
    t0 = time.perf_counter()

    pil = img
    if pil.mode not in ("RGB", "RGBA"):
        pil = pil.convert("RGB")

    try:
        if strength == "sharpen":
            result_pil = pil.filter(
                ImageFilter.UnsharpMask(radius=1.2, percent=110, threshold=2)
            )

        elif strength == "denoise":
            # Bilateral filter — сглаживает шум, сохраняя границы.
            bgr = _pil_to_bgr(pil)
            # sigmaColor/sigmaSpace среднеагрессивные, d=7 — быстро на CPU
            denoised = cv2.bilateralFilter(bgr, d=7, sigmaColor=35, sigmaSpace=35)
            rgb = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
            tmp = Image.fromarray(rgb)
            result_pil = tmp.filter(
                ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=3)
            )

        else:  # ai
            model = model.lower()
            if model not in {"fsrcnn", "espcn"}:
                model = "fsrcnn"
            bgr = _pil_to_bgr(pil)
            try:
                up = _tiled_upsample(bgr, model, 2)
            except MemoryError:
                raise HTTPException(507, "Не хватило памяти.")
            rgb = cv2.cvtColor(up, cv2.COLOR_BGR2RGB)
            pil_up = Image.fromarray(rgb)
            tmp = pil_up.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
            result_pil = tmp.filter(
                ImageFilter.UnsharpMask(radius=1.0, percent=60, threshold=2)
            )
    except HTTPException:
        raise
    except Exception as e:
        log.exception("enhance failed")
        raise HTTPException(500, f"enhance failed: {e}")

    final_bgr = _pil_to_bgr(result_pil)
    out = _save_output(final_bgr, fmt, quality)
    log.info("enhance ok mode=%s in %.1fs → %s (%d bytes)", strength, time.perf_counter()-t0, out.name, out.stat().st_size)
    return {
        "ok": True,
        "file": out.name,
        "url": f"/api/download/{out.name}",
        "width": orig_w,
        "height": orig_h,
        "size_bytes": out.stat().st_size,
        "mode": strength,
    }


@app.get("/api/download/{name}")
def download(name: str):
    if "/" in name or ".." in name:
        raise HTTPException(400, "bad name")
    path = OUTPUT_DIR / name
    if not path.exists():
        raise HTTPException(404, "not found")
    media = "image/png"
    if name.endswith(".jpg"):
        media = "image/jpeg"
    elif name.endswith(".webp"):
        media = "image/webp"
    return FileResponse(path, media_type=media, filename=name)


@app.on_event("startup")
def cleanup_on_start():
    """Remove files older than 1 hour from uploads/outputs."""
    import time
    now = time.time()
    for d in (UPLOAD_DIR, OUTPUT_DIR):
        for p in d.glob("*"):
            try:
                if now - p.stat().st_mtime > 3600:
                    p.unlink()
            except Exception:
                pass
    log.info("startup cleanup done")
