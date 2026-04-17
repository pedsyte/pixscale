# PixScale

**Live:** https://pixscale.gornich.fun
**Part of:** [Gornich](https://gornich.fun)

Сервис для изменения размера и AI-апскейла изображений прямо в браузере.
Бесплатно, без регистрации, файлы удаляются автоматически через 1 час.

---

## Возможности

### 📐 Классический ресайз (Pillow)
- Произвольные размеры в пикселях
- Замок пропорций (aspect ratio lock)
- **Пресеты пропорций:** `1:1`, `4:3`, `3:2`, `16:9`, `9:16`, `21:9`, `2:3`
- **Пресеты процентов:** `25%`, `50%`, `75%`, `150%`, `200%`, `300%`, `400%`, `500%`
- **Алгоритмы:** Lanczos (по умолчанию, лучшее качество), Bicubic, Bilinear, Nearest
- Лимит на выходе: 60 мегапикселей

### ⚡ AI Upscale (OpenCV DNN Super-Resolution)
Три предобученных нейросетевых модели, CPU-only:

| Модель    | Качество | Скорость | Размер  |
|-----------|----------|----------|---------|
| `FSRCNN`  | хорошее  | ⚡ быстрая | ~40 KB  |
| `ESPCN`   | отличное | 🏃 средняя | ~90 KB  |
| `EDSR`    | максимум | 🐢 медленная | ~37 MB |

**Множители:** ×2, ×3, ×4

### 🎨 Форматы
- **Вход:** JPG, PNG, WebP, BMP, TIFF
- **Выход:** PNG (без потерь), JPG (с качеством), WebP

### 🔧 UX-фишки
- Drag & drop, выбор файла через кнопку
- **Before/After слайдер** с перетаскиваемой ручкой
- Метаданные исходника (размер, пропорция, размер файла)
- Toast-уведомления
- Автоматическая очистка старых файлов через 1 час

---

## Стек

- **Backend:** Python 3.11, FastAPI, Uvicorn
- **Image processing:** Pillow 12, OpenCV-Contrib (dnn_superres), NumPy
- **Frontend:** Vanilla HTML + CSS + JavaScript (no build step, легковесный)
- **Fonts:** DM Sans, JetBrains Mono
- **Палитра:** amber (`#f59e0b`) + teal (`#14b8a6`) на тёмном фоне
- **Server:** systemd + nginx + Let's Encrypt

---

## API

### `GET /api/health`
Статус сервиса и список доступных моделей.

### `GET /api/info`
Информация о доступных моделях, форматах и лимитах.

### `POST /api/analyze`
Загрузка файла. Возвращает `token`, размеры, пропорции.

**Form fields:** `file` (multipart)

**Response:**
```json
{"token":"in_abc123.png","width":1920,"height":1080,"ratio":1.7778,"format":"png","size_bytes":245760}
```

### `POST /api/resize`
Классический ресайз через Pillow.

**Form fields:**
- `token` — от `/analyze`
- `width` (int, opt)
- `height` (int, opt)
- `keep_ratio` (bool, default `true`)
- `fmt` (`png` / `jpg` / `webp`)
- `quality` (int, default 92)
- `resample` (`lanczos` / `bicubic` / `bilinear` / `nearest`)

### `POST /api/upscale`
AI-апскейл через OpenCV DNN.

**Form fields:**
- `token` — от `/analyze`
- `model` (`fsrcnn` / `espcn` / `edsr`)
- `scale` (`2` / `3` / `4`)
- `fmt`, `quality`

### `GET /api/download/{name}`
Скачать готовый файл.

---

## Структура

```
/opt/pixscale/
├── backend/
│   └── main.py          — FastAPI app
├── frontend/
│   └── index.html       — SPA (vanilla)
├── models/              — .pb файлы SR-моделей (~111 MB)
├── uploads/             — временные загрузки (auto-clean 1h)
├── outputs/             — готовые файлы (auto-clean 1h)
├── ROADMAP.md
├── BACKLOG.md
└── README.md
```

## Deploy

```bash
# Служба
systemctl restart pixscale

# Nginx
nginx -t && systemctl reload nginx

# SSL (уже установлен)
certbot renew
```

**Порт бэкенда:** `127.0.0.1:8093`

---

## Источники моделей

- FSRCNN: https://github.com/Saafke/FSRCNN_Tensorflow
- ESPCN: https://github.com/fannymonori/TF-ESPCN
- EDSR: https://github.com/Saafke/EDSR_Tensorflow

## Лицензия

MIT — используй свободно.
