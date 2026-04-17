# PixScale — Backlog / Changelog

Формат: от нового к старому. Каждая запись — дата + краткое описание + почему.

---

## 2026-04-17 — Инициализация проекта PixScale

**Что сделано:**
- Создан сервис `pixscale.gornich.fun` для ресайза и AI-апскейла изображений
- Развёрнут на порт `127.0.0.1:8093`, проксируется nginx с SSL (Let's Encrypt)
- Стек: FastAPI (Python 3.11) + OpenCV-Contrib (dnn_superres) + Pillow + vanilla HTML/CSS/JS
- 9 pretrained моделей супер-резолюции: FSRCNN / ESPCN / EDSR (×2, ×3, ×4)
- Уникальный UI: amber (#f59e0b) + teal (#14b8a6) палитра на тёмном фоне, шрифты DM Sans + JetBrains Mono
- Фичи фронта: drag&drop, превью, лок пропорций, пресеты сторон и процентов, before/after слайдер с ручкой, тосты, выбор алгоритма и модели, выбор формата вывода

**Почему так:**
- **OpenCV DNN вместо Real-ESRGAN**: на сервере нет Vulkan/CUDA, всего 2 ядра и 3.6 GB RAM. Real-ESRGAN на CPU с PyTorch съел бы всю память. OpenCV DNN работает на CPU, модели весят от 40 KB (FSRCNN) до 37 MB (EDSR), апскейл занимает секунды вместо минут.
- **Vanilla JS вместо React**: утилита простая (один экран), React не нужен, лишний билд-степ. Плюс уникальная архитектура vs MusicBox (чтобы не повторяться).
- **FSRCNN по умолчанию**: почти неотличим от EDSR на большинстве фото, но в 10× быстрее.
- **Отдельный сабдомен**: соответствует правилам проекта — каждый сервис на своём поддомене, в своём `/opt/<name>/` и отдельном GitHub-репозитории.
- **Лимиты 40 MB upload / 60 MP output**: защита от DoS на слабой машине.
- **Startup cleanup**: файлы старше 1 часа удаляются при рестарте сервиса — не копится мусор в `uploads/` и `outputs/`.

**E2E проверено:**
- `/api/health` → 200, модели перечислены
- `/api/analyze` → токен и метаданные
- `/api/resize` (200×150 → 100×75 JPG) → OK
- `/api/upscale` (FSRCNN ×2, 200×150 → 400×300 PNG) → 0.23 сек

**Инфраструктура:**
- `/etc/systemd/system/pixscale.service` — uvicorn, enabled, Active
- `/etc/nginx/conf.d/pixscale.conf` — proxy + SPA fallback + client_max_body_size 50m
- SSL сертификат выписан, истекает 2026-07-16 (auto-renew через certbot)
