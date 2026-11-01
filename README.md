# AI UGC Creator

Один FastAPI-проект. Этапы пайплайна живут в отдельных папках как **независимые группы роутов**. Склейка 0→8 идёт через session-слой `/v1/pipeline` (JSON на диске, без БД). Переиспользуемые файлы — `/v1/assets` (`uploads/{asset_id}/…`).

Локальный demo UI: **http://localhost:8000/demo**

## Структура

```
AI UGC Creator/
├── app/
│   ├── main.py
│   ├── core/
│   ├── storage/
│   ├── api/
│   ├── pipeline/                # session orchestration 0→8 + assets
│   ├── static/demo.html         # временный UI для тестов
│   └── stages/
│       ├── video_ingestion/     # этап 1
│       ├── hook_analyzer/       # этап 2
│       ├── face_replacement/    # этап 3 — face + text removal на кадре
│       ├── hook_video_generator/ # этап 4 — image-to-video хука 2–4 сек
│       ├── hook_text_engine/    # этап 5 — OCR/ASR + адаптация хука
│       ├── text_renderer/       # этап 6 — burn-in текста
│       ├── product_demo_composer/ # этап 7 — [AI Hook] → [Product Demo]
│       └── music_overlay/        # этап 8 — музыка
├── tests/
├── requirements.txt
└── .env.example
```

Новый этап = папка в `app/stages/<name>/` + `include_router` в `app/api/router.py`. Этапы **не импортируют** друг друга; `app/pipeline` оркестрирует сверху.

## API (группы)

| Группа | Endpoints |
|--------|-----------|
| **Health** | `GET /health` |
| **Demo UI** | `GET /demo` |
| **Media** | `GET /media/{path}` |
| **Assets** | `POST /v1/assets/upload`, `GET /v1/assets/{asset_id}` |
| **Pipeline** | `POST /v1/pipeline/sessions`, steps `…/product`, `…/ingest/*`, `…/hook/*`, `…/face/*`, `…/hook-video/*`, `…/hook-text/*`, `…/render`, `…/compose`, `…/music` |
| **Video Ingestion** | `POST /v1/ingestion/upload`, `POST /v1/ingestion/url` (прямой MP4/MOV **или** Instagram Reel / TikTok URL через yt-dlp) |
| **Hook Analyzer** | `POST /v1/hook-analyzer/candidates`, `/motion-plan`, `/reanalyze` |
| **Hook Text Engine** | `POST /v1/hook-text-engine/extract`, `/adapt`, `/select`; `GET /sessions/{id}` |
| **Face Replacement** | `POST /v1/face-replacement/generate`; `GET|POST /generations/{id}` (+ `/rerun`) |
| **Hook Video** | `POST /v1/hook-video/generate`, `/compare`; `GET /jobs/{id}`; `POST /jobs/{id}/rerun` (можно `motion_prompt`) |
| **Text Renderer** | `POST /v1/text-renderer/render`, `/preview`; `GET /renders/{id}`; `POST /renders/{id}/rerender` |
| **Product Demo Composer** | `POST /compose` (upload), `POST /compose/refs` (URL reuse); `GET /jobs/{id}`; `POST /jobs/{id}/rerun` |
| **Music Overlay** | `POST /v1/music-overlay/overlay`, `/overlay/upload`; `GET /jobs/{id}`; `POST /jobs/{id}/rerun` |

---

### Video Ingestion

Принимает MP4/MOV (upload или прямая ссылка на файл), либо **Instagram Reel / TikTok URL** (через `yt-dlp`). Сохраняет оригинал без изменений, возвращает `video_id` + метаданные.

---

### Hook Analyzer

Анализирует короткий hook-интервал: кандидаты первого кадра + motion plan.  
Неизвестные детали помечаются как `not clearly observed` / `undetermined` — без выдумок.

#### `POST /v1/hook-analyzer/candidates`

```json
{
  "video_id": "uuid",
  "video_url": "http://localhost:8000/media/originals/<uuid>/source.mp4",
  "hook_start_seconds": 0,
  "hook_end_seconds": 4,
  "candidate_count": 5
}
```

Ответ:

```json
{
  "video_id": "uuid",
  "analysis_key": "0.000_4.000",
  "candidates": [
    {
      "frame_id": "uuid",
      "timestamp_seconds": 0.4,
      "image_url": "http://localhost:8000/media/hooks/.../candidates/<frame_id>.jpg",
      "quality_score": 0.72,
      "reason": "face visible; face size suitable for hook; ..."
    }
  ]
}
```

Превью доступны через `/media/...` для UI. Наблюдения и кадры кэшируются под `hooks/{video_id}/{analysis_key}/`.

#### `POST /v1/hook-analyzer/motion-plan`

```json
{
  "video_id": "uuid",
  "start_seconds": 0,
  "duration_seconds": 3
}
```

Достаточно **`video_id` или `video_url`** (можно оба). По `video_id` файл ищется в `originals/{id}/source.*` / кэше.

`start_seconds` / `duration_seconds` — окно обрезки (по умолчанию `0` и `3`). Клип режется через ffmpeg, уходит в **Gemini 3.8 Flash**, в ответе:

- `motionPrompt` — natural-language промпт под Kling 3.0
- `trimmed_video_url` — обрезанный клип для ручной проверки
- плюс прежний heuristic motion plan (timeline / emotion / gaze)

Нужен `GOOGLE_AI_API_KEY` в `.env`.

#### `POST /v1/hook-analyzer/reanalyze`

После выбора кадра:

```json
{
  "video_id": "uuid",
  "selected_frame_id": "uuid",
  "start_seconds": 0,
  "duration_seconds": 3
}
```

Observations берутся из кэша; `motionPrompt` пересчитывается по уже обрезанному клипу.

---

### Hook Text Engine

Независимый этап: извлекает hook-текст (OCR + ASR) и адаптирует его под продукт.  
Видео не изменяется, изображения не генерируются. On-screen и spoken тексты **никогда не сливаются** без явного указания источника.

В **pipeline** (`POST …/hook-text/extract-adapt`) — **один** Gemini video-запрос (`HOOK_TEXT_MODEL`, fallback `HOOK_TEXT_FALLBACK_MODEL`): клип → on-screen + spoken + product variants. Отдельные `/extract` и `/adapt` остаются для ручного API; `/adapt` без видео использует дешёвый `GEMINI_TEXT_MODEL`.

Хранение сессии адаптации: `original.json`, `variants.json`, `selected.json` раздельно под `hook_text/adaptations/{adaptation_id}/`.

Приоритет текста: **manual_text → variant → original**.

#### `POST /v1/hook-text-engine/extract`

```json
{
  "video_id": "uuid",
  "video_url": "string",
  "start_seconds": 0,
  "end_seconds": 4,
  "languages": ["en", "ru"]
}
```

Ответ:

```json
{
  "video_id": "uuid",
  "extraction_id": "uuid",
  "on_screen_text": [
    {
      "text": "WAIT FOR IT",
      "start_seconds": 0,
      "end_seconds": 0.5,
      "bbox": [10, 20, 200, 40],
      "confidence": 0.91,
      "source": "on_screen"
    }
  ],
  "spoken_text": [
    {
      "text": "you need to see this",
      "start_seconds": 0.2,
      "end_seconds": 1.0,
      "language": "en",
      "confidence": 0.88,
      "source": "spoken"
    }
  ]
}
```

OCR/ASR через Gemini (`GOOGLE_AI_API_KEY`). Без ключа: ASR пустой; OCR — локальный fallback (pytesseract, если установлен).

#### `POST /v1/hook-text-engine/adapt`

```json
{
  "original_text": "Wait… nobody talks about this?",
  "product_name": "AquaBottle",
  "product_description": "Insulated bottle that keeps drinks cold for hours.",
  "target_audience": "commuters",
  "language": "en",
  "hook_goal": "curiosity",
  "variants_count": 5,
  "manual_text": null,
  "skip_adaptation": false,
  "extraction_id": null
}
```

- `manual_text` перекрывает `original_text`.
- `skip_adaptation: true` — без вариантов, `selected_text` = оригинал.
- Claims review отклоняет выдуманные % / отзывы / «proven», которых нет во входном описании продукта.

#### `POST /v1/hook-text-engine/select`

```json
{
  "adaptation_id": "uuid",
  "manual_text": null,
  "variant_id": "uuid",
  "use_original": false
}
```

#### `GET /v1/hook-text-engine/sessions/{adaptation_id}`

Возвращает `original_text`, `variants`, `selected_text` из хранилища.

---

### Face Replacement

Независимый этап: создаёт **первый кадр** для image-to-video с новой внешностью.
Генерация видео в этот этап **не входит**.

- Основа: выбранный кадр (`selected_frame_url`)
- Источник внешности: файл `face_reference` (upload)
- Сохраняются поза / композиция / освещение / фон (через prompt-ограничения провайдера)
- Опционально удаляет overlay-текст и **проверяет** остатки (warning, не hard-fail)
- Оригиналы архивируются без изменений; повторный вызов = новая генерация
- Вызов модели отделён adapter’ом (`gemini` | `stub`; расширяется через registry)

#### `POST /v1/face-replacement/generate`

`multipart/form-data`

| Field | Required | Default / example | Description |
|-------|----------|-------------------|-------------|
| `video_id` | **yes** | `8ffb1165-1fba-4762-a0b5-eab238819d0e` | Id из ingestion |
| `video_url` | no | `http://localhost:8000/media/originals/8ffb1165-1fba-4762-a0b5-eab238819d0e/source.mp4` | Контекст пайплайна (не для swap) |
| `selected_frame_url` | **yes** | `http://localhost:8000/media/hooks/3fa85f64-5717-4562-b3fc-2c963f66afa6/0.000_4.000/candidates/1a9bb1d1-341a-4e21-b2fa-56c325ef532b.jpg` | Master scene (FIRST image) |
| `face_reference` | **yes** | — | Файл identity (SECOND image) |
| `remove_overlay_text` | no | `true` | Снять overlay-текст |
| `preserve_original_composition` | no | `true` | Сохранить композицию |
| `preserve_pose` | no | `true` | Сохранить позу |
| `preserve_lighting` | no | `true` | Сохранить освещение |
| `model_provider` | no | `gemini` | `gemini` / `stub` |
| `model_name` | no | `gemini-3-pro-image` (Nano Banana Pro) | Secondary: `gemini-3.1-flash-image` (Nano Banana 2); aliases `nano-banana-pro` / `nano-banana-2` |

Пример (curl):

```bash
curl -X POST "http://localhost:8000/v1/face-replacement/generate" \
  -F "video_id=8ffb1165-1fba-4762-a0b5-eab238819d0e" \
  -F "video_url=http://localhost:8000/media/originals/8ffb1165-1fba-4762-a0b5-eab238819d0e/source.mp4" \
  -F "selected_frame_url=http://localhost:8000/media/hooks/3fa85f64-5717-4562-b3fc-2c963f66afa6/0.000_4.000/candidates/1a9bb1d1-341a-4e21-b2fa-56c325ef532b.jpg" \
  -F "face_reference=@./face.jpg;type=image/jpeg" \
  -F "remove_overlay_text=true" \
  -F "model_provider=gemini"
```

Ответ:

```json
{
  "video_id": "uuid",
  "generated_frame_url": "http://localhost:8000/media/face_replacement/<video_id>/generations/<id>/frame.png",
  "width": 1080,
  "height": 1920,
  "model_provider": "gemini",
  "model_name": "gemini-3-pro-image",
  "text_removal_requested": true,
  "status": "success",
  "warnings": [],
  "generation_id": "uuid",
  "usage": {
    "prompt_tokens": 1200,
    "candidates_tokens": 400,
    "thoughts_tokens": 0,
    "total_tokens": 1600,
    "input_cost_usd": 0.0024,
    "output_cost_usd": 0.006,
    "estimated_cost_usd": 0.0084,
    "currency": "USD",
    "processing_time_seconds": 4.82,
    "provider_latency_seconds": 4.51,
    "pricing_note": "Paid-tier estimate for gemini-3-pro-image ..."
  }
}
```

Нужен `GOOGLE_AI_API_KEY` для провайдера `gemini`. Для локальных тестов: `model_provider: "stub"`.

---

### Hook Video Generator

Независимый этап: image-to-video из первого кадра и motion plan. Длительность клипа 2–4 секунды. Бизнес-логика не вызывает API конкретной модели: провайдер выбирается полями `model_provider` и `model_name`.

`POST /generate` возвращает **202** и статус `queued` или `processing`. Это постановка задачи, не готовое видео. Статус `success` и `generated_video_url` появляются только после того, как файл скачан и проверен: контейнер mp4/mov, длительность рядом с запрошенной и в окне 2–4 секунды, размер кадра равен `output_resolution`.

Опрос: `GET /v1/hook-video/jobs/{job_id}`.  
Callback вендора: `POST /v1/hook-video/jobs/{job_id}/callback`. Даже `provider_status: succeeded` не становится `success`, пока файл не прошёл проверку.

Исходный кадр и параметры (включая собранный motion prompt) хранятся в `hook_video/jobs/{job_id}/request.json`. Повторный запуск: `POST /jobs/{job_id}/rerun` (можно сменить только модель). Сравнение моделей: `POST /compare`, затем `GET /comparisons/{id}`.

#### `POST /v1/hook-video/generate`

```json
{
  "video_id": "uuid",
  "first_frame_url": "http://localhost:8000/media/hooks/<video_id>/.../frame.jpg",
  "motion_plan": {
    "emotion": "curious",
    "emotion_intensity": 0.7,
    "gaze_direction": "toward camera",
    "facial_expression": "slight smile",
    "motion_timeline": [
      {
        "start_seconds": 0,
        "end_seconds": 1.5,
        "head_motion": "slight tilt right",
        "eye_motion": "looks at camera",
        "mouth_motion": "closed",
        "body_motion": "still",
        "emotion": "curious"
      }
    ]
  },
  "duration_seconds": 3,
  "aspect_ratio": "9:16",
  "output_resolution": "1080x1920",
  "fps": 30,
  "model_provider": "stub",
  "model_name": "stub-hold"
}
```

`motion_plan` принимает форму ответа hook analyzer (`emotion`, `motion_timeline`, `motionPrompt`). Этапы друг друга не импортируют.

Ответ 202:

```json
{
  "job_id": "uuid",
  "video_id": "uuid",
  "status": "queued",
  "model_provider": "stub",
  "model_name": "stub-hold",
  "warnings": []
}
```

Готовый ролик (`GET /jobs/{job_id}`):

```json
{
  "job_id": "uuid",
  "video_id": "uuid",
  "generated_video_url": "http://localhost:8000/media/hook_video/jobs/<job_id>/output.mp4",
  "duration_seconds": 3,
  "width": 1080,
  "height": 1920,
  "fps": 30,
  "model_provider": "stub",
  "model_name": "stub-hold",
  "status": "success",
  "warnings": []
}
```

Провайдеры:

- `stub` — локальный hold первого кадра через ffmpeg (тесты)
- `kling` — Kling 3.0 / 3.0 Turbo image-to-video (`KLING_API_KEY`; пакет General, без Kling 4.0). Для Kling ставь `duration_seconds: 3` или `4`
- `http` — generic async JSON (`HOOK_VIDEO_HTTP_SUBMIT_URL`, `HOOK_VIDEO_HTTP_STATUS_URL` с `{provider_job_id}`)

Свой вендор — класс `VideoGenerationProvider` и `register_provider`, без изменений API.

Пример Kling:

```json
{
  "model_provider": "kling",
  "model_name": "kling-3.0",
  "duration_seconds": 3,
  "output_resolution": "1080x1920"
}
```

---

### Text Renderer

Независимый этап: детерминированный burn-in пользовательского текста на сгенерированное видео через FFmpeg + ASS (libass).  
Эмодзи: в `fontsdir` подключается bundled **Noto Emoji** (монохром), а emoji-руны в ASS помечаются `{\fnNoto Emoji}` — иначе Arial даёт «тофу» □. Полноцветные Apple Color Emoji через libass недоступны (формат sbix).  
Исходный текст **не переписывается**. Исходное видео архивируется и **не перезаписывается**. Повторный рендер — без повторной генерации ролика.

- Перенос строк, выравнивание, размер шрифта, обводка, вертикаль через `padding_top` (%)
- Несколько timed-сегментов
- Вертикальные и произвольные разрешения
- `skip_text: true` — экспорт копии без overlay
- Preview-кадр и `rerender` по `render_id`

#### `POST /v1/text-renderer/render`

```json
{
  "video_id": "uuid",
  "generated_video_url": "http://localhost:8000/media/hook_video/jobs/<id>/output.mp4",
  "text": "WAIT FOR IT",
  "language": "en",
  "style": {
    "font_family": "Arial",
    "font_size": 72,
    "font_weight": "bold",
    "font_color": "#FFFFFF",
    "outline_color": "#000000",
    "outline_width": 3.5,
    "padding_top": 72
  },
  "segments": [
    { "start_seconds": 0, "end_seconds": null, "text": "WAIT FOR IT" }
  ],
  "output_format": "mp4",
  "skip_text": false
}
```

`end_seconds: null` (или поле опущено) — текст держится **до конца видео** при любой длительности.  
Либо передайте только `text` без `segments` — то же поведение на весь клип.

`padding_top` — отступ текста от верха кадра в **процентах** (0–95). Примеры: `8` ≈ верх, `45` ≈ центр, `70–80` ≈ нижний хук как в UGC.

Для вида как в UGC-референсе: `font_size` 64–84, `font_weight: "bold"` или `"black"`, `padding_top` ≈ `72`.

Ответ:

```json
{
  "video_id": "uuid",
  "render_id": "uuid",
  "rendered_video_url": "http://localhost:8000/media/text_renderer/renders/<id>/output.mp4",
  "duration_seconds": 3,
  "width": 1080,
  "height": 1920,
  "format": "mp4",
  "text_segments_rendered": 1,
  "status": "success",
  "preview_image_url": "http://localhost:8000/media/text_renderer/renders/<id>/preview.jpg",
  "warnings": []
}
```

Дополнительно: `POST /preview`, `GET /renders/{render_id}`, `GET /renders/{render_id}/request`, `POST /renders/{render_id}/rerender`.

---

### Product Demo Composer

Независимый этап склейки двух видео в фиксированном порядке:

**[AI HOOK] → [PRODUCT DEMO]**

Сервис **не генерирует** видео и **не меняет** содержание сцен. Он нормализует в **1080×1920 (9:16)** — формат Instagram Reels / TikTok — и склеивает клипы. По умолчанию `fit_mode=blur`: полный кадр по центру, пустые зоны заполняются **размытой копией** того же клипа (без чёрных полос). Также: `cover` (кроп) / `contain` (letterbox). Музыкальный оверлей — отдельный этап [Music Overlay](#music-overlay).

Исходники архивируются и **никогда не перезаписываются**. Между модулями передаются только JSON-контракты и URL артефактов (`project_id`, `asset_id`, `job_id`, `video_url`).

#### `POST /v1/product-demo-composer/compose` → `202`

`multipart/form-data` (в Swagger demo выбирается как **файл**):

| Field | Type | Описание |
|-------|------|----------|
| `project_id` | form | UUID проекта |
| `hook_asset_id` | form | asset id AI Hook |
| `hook_video_url` | form | URL AI Hook (`/media/...` или http(s)) |
| `demo_video` | **file** | Product Demo — MP4/MOV |
| `demo_asset_id` | form | опционально; иначе генерируется |
| `output_format` | form | `mp4` / `mov` |
| `aspect_ratio` | form | `9:16` / `16:9` / `1:1` |
| `resolution` | form | по умолчанию `1080x1920` (Reels/TikTok) |
| `fps` | form | по умолчанию `30` |
| `fit_mode` | form | `blur` (по умолчанию) / `cover` / `contain` |
| `transition` | form | `none` / `fade` / `crossfade` |
| `transition_duration_seconds` | form | `0` для `none` |
| `audio_mode` | form | `both` / `hook` / `demo` / `none` |

- `transition`: `none` (по умолчанию, жёсткая склейка), `fade`, `crossfade`
- `audio_mode`: `both` | `hook` | `demo` | `none` (тишина; без наложения музыки)

Ответ:

```json
{
  "project_id": "uuid",
  "job_id": "uuid",
  "status": "queued",
  "warnings": []
}
```

#### `GET /v1/product-demo-composer/jobs/{job_id}`

После успеха:

```json
{
  "project_id": "uuid",
  "job_id": "uuid",
  "asset_id": "uuid",
  "video_url": "http://localhost:8000/media/product_demo_composer/jobs/<id>/output.mp4",
  "duration_seconds": 15,
  "width": 1080,
  "height": 1920,
  "fps": 30,
  "segments": [
    {"type": "hook", "start_seconds": 0, "end_seconds": 3},
    {"type": "product_demo", "start_seconds": 3, "end_seconds": 15}
  ],
  "status": "success",
  "warnings": []
}
```

Дополнительно: `GET /jobs/{job_id}/request`, `POST /jobs/{job_id}/rerun` (повтор с новыми settings из архивных исходников).

---

### Music Overlay

Независимый этап: кладёт выбранную музыку на **уже готовое** видео. Не знает, откуда видео взялось (AI, монтаж, загрузка). Исходные видео и аудио архивируются и не перезаписываются. Видеоряд копируется (`-c:v copy`): FPS, разрешение и кодек картинки не меняются.

Два смещения независимы:

- `music_start_offset_seconds` — с какой секунды **исходного** аудио начать
- `video_start_seconds` — на какой секунде **видео** музыка становится слышна

Если фрагмент короче оставшегося видео, `music_end_behavior`:

- `silence` — дальше видео без музыки
- `trim` — то же самое: доступный фрагмент один раз, без повтора (видео не обрезается)
- `loop` — повтор фрагмента от `music_start_offset_seconds` до конца окна

`silence` и `trim` звучат одинаково. Различие — в коде предупреждения (`music_ended_before_video` / `music_shorter_than_video_no_loop`). Если музыка длиннее окна, любой режим обрезает её по видео.

Громкость линейная (`volume=0.3` = 30%). Дорожки сводятся через `amix` с `normalize=0`, затем `alimiter` (потолок 0.95, без автоподнятия тихих мест). Разные sample rate приводятся к 48 kHz stereo.

#### `POST /v1/music-overlay/overlay` → `202`

```json
{
  "project_id": "uuid",
  "video": { "asset_id": "uuid", "video_url": "http://localhost:8000/media/..." },
  "music": { "asset_id": "uuid", "audio_url": "http://localhost:8000/media/..." },
  "settings": {
    "music_start_offset_seconds": 35,
    "video_start_seconds": 0,
    "music_volume": 0.3,
    "original_audio_enabled": true,
    "original_audio_volume": 1.0,
    "fade_in_seconds": 0,
    "fade_out_seconds": 1.5,
    "loop_music": false,
    "music_end_behavior": "silence",
    "output_format": "mp4"
  }
}
```

Ответ постановки в очередь: `project_id`, `job_id`, `status: queued`, `warnings`.

#### `POST /v1/music-overlay/overlay/upload` → `202`

`multipart/form-data`: те же поля settings плюс файл `music` (MP3, WAV, AAC, M4A, FLAC, OGG и другие форматы, которые читает FFmpeg). Видео по-прежнему передаётся URL.

#### `GET /v1/music-overlay/jobs/{job_id}`

```json
{
  "project_id": "uuid",
  "job_id": "uuid",
  "asset_id": "uuid",
  "video_url": "http://localhost:8000/media/music_overlay/jobs/<id>/output.mp4",
  "duration_seconds": 15,
  "audio": {
    "music_start_offset_seconds": 35,
    "video_start_seconds": 0,
    "music_volume": 0.3,
    "original_audio_enabled": true,
    "fade_in_seconds": 0,
    "fade_out_seconds": 1.5
  },
  "status": "success",
  "warnings": []
}
```

Дополнительно: `GET /jobs/{job_id}/request`, `POST /jobs/{job_id}/rerun`.

Оффсет больше длительности музыки, `video_start_seconds` за концом ролика и слишком длинные fade отклоняются до постановки задачи (`422`, `invalid_overlay_settings`).

---

## Запуск

Нужны Python 3.10+, FFmpeg (`ffmpeg`/`ffprobe` в `PATH`).

```bash
cd "AI UGC Creator"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- Docs: http://localhost:8000/docs  
- Тесты: `pytest -v`
