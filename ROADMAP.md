# PixScale — Roadmap

## Фаза 0: Инициализация ✅
- [x] Создана структура `/opt/pixscale/` (backend, frontend, models, uploads, outputs)
- [x] Git init
- [x] Выбран стек: FastAPI + OpenCV DNN + Pillow + vanilla JS
- [x] Определены порт `8093` и поддомен `pixscale.gornich.fun`

## Фаза 1: Backend ✅
- [x] FastAPI приложение (`backend/main.py`)
- [x] Эндпоинт `GET /api/health` — статус
- [x] Эндпоинт `GET /api/info` — доступные опции
- [x] Эндпоинт `POST /api/analyze` — загрузка и метаданные
- [x] Эндпоинт `POST /api/resize` — Pillow Lanczos/Bicubic/Bilinear/Nearest
- [x] Эндпоинт `POST /api/upscale` — AI через OpenCV dnn_superres
- [x] Эндпоинт `GET /api/download/{name}` — скачивание
- [x] Автоочистка старых файлов (1 час) на старте
- [x] CORS, лимит 40 MB на upload, лимит 60 MP на выход
- [x] EXIF-aware поворот через `ImageOps.exif_transpose`
- [x] Кэш моделей в памяти (не перечитывать с диска)

## Фаза 2: AI модели ✅
- [x] FSRCNN ×2, ×3, ×4 (мелкие, быстрые)
- [x] ESPCN ×2, ×3, ×4 (баланс)
- [x] EDSR ×2, ×3, ×4 (качественные, 37 MB каждая)

## Фаза 3: Frontend ✅
- [x] Vanilla HTML/CSS/JS (без билд-степа)
- [x] Уникальный дизайн: amber + teal палитра
- [x] Drag & drop зона загрузки
- [x] Sticky header + hero с чипами
- [x] Превью загруженного изображения с chess-pattern фоном
- [x] Метаданные (размер, пропорции, размер файла)
- [x] Две вкладки: Ресайз / AI Upscale
- [x] Ресайз: ширина × высота, замок пропорций
- [x] Пресеты пропорций (1:1, 4:3, 3:2, 16:9, 9:16, 21:9, 2:3, ориг)
- [x] Пресеты процентов (25/50/75/150/200/300/400/500)
- [x] Выбор алгоритма (Lanczos/Bicubic/Bilinear/Nearest)
- [x] AI: кнопки ×2/×3/×4 с hint'ами скорости
- [x] Выбор модели (FSRCNN/ESPCN/EDSR)
- [x] Превью размера результата до запуска AI
- [x] Выбор формата (PNG/JPG/WebP)
- [x] **Before/After слайдер** с drag-ручкой
- [x] Кнопка скачивания с размером файла
- [x] Toast-уведомления (error/info)
- [x] Секция features (4 карточки)
- [x] Спиннеры загрузки
- [x] Сброс (reset) состояния

## Фаза 4: Инфраструктура ✅
- [x] systemd service `/etc/systemd/system/pixscale.service`
- [x] nginx конфиг `/etc/nginx/conf.d/pixscale.conf`
- [x] Let's Encrypt SSL-сертификат (certbot --nginx)
- [x] HTTP → HTTPS редирект
- [x] Запуск и enable автостарта

## Фаза 5: Документация ✅
- [x] README.md с описанием API, стека, структуры
- [x] ROADMAP.md (этот файл)
- [x] BACKLOG.md с историей изменений
- [x] .gitignore

## Фаза 6: GitHub
- [ ] Создать репо `pedsyte/pixscale`
- [ ] Первый коммит + push

## Фаза 7: Добавить в Gornich App Center
- [ ] Добавить карточку PixScale на https://gornich.fun
- [ ] Возможно добавить в промо-слайдер

---

## Идеи на будущее
- [ ] Batch upload (несколько файлов сразу)
- [ ] History — список последних обработок в localStorage
- [ ] Кроп перед ресайзом (рамка выделения)
- [ ] Поворот и отражение
- [ ] Добавить Real-ESRGAN (если станет доступен Vulkan/GPU)
- [ ] Webhook/URL source (обработать картинку по URL)
- [ ] Сравнение моделей на одном изображении side-by-side
- [ ] PWA (установка как приложение)
- [ ] Экспорт в ICO, HEIC, AVIF
- [ ] Метрики (сколько файлов обработано)
- [ ] Rate limit по IP
