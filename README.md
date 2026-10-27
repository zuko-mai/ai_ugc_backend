# AI UGC Creator

Один FastAPI-проект. Этапы пайплайна живут в отдельных папках и подключаются как **независимые группы роутов**. Позже их можно собрать в единый pipeline.

## Структура

```
AI UGC Creator/
├── app/
│   ├── main.py                 # FastAPI entrypoint
│   ├── core/                   # общий config / logging / exceptions
│   ├── storage/                # абстрактное файловое хранилище
│   ├── api/                    # health, media, сборка роутеров
│   └── stages/
│       └── video_ingestion/    # этап 1 (логика + свой router)
├── tests/
│   └── stages/video_ingestion/
├── requirements.txt
└── .env.example
```

Новый этап = новая папка в `app/stages/<name>/` + `include_router` в `app/api/router.py`. Этапы не импортируют друг друга.

## API (группы)

| Группа (OpenAPI tag) | Endpoints |
|----------------------|-----------|
| **Health** | `GET /health` |
| **Media** | `GET /media/{path}` |
| **Video Ingestion** | `POST /v1/ingestion/upload`, `POST /v1/ingestion/url` |

### Video Ingestion — успешный ответ

```json
{
  "video_id": "uuid",
  "video_url": "http://localhost:8000/media/originals/<uuid>/source.mp4",
  "duration_seconds": 12.5,
  "fps": 30.0,
  "width": 1080,
  "height": 1920,
  "has_audio": true,
  "audio_url": null,
  "status": "success"
}
```

`audio_url` на этом этапе всегда `null` — исходник не меняется.

## Запуск

Нужны Python 3.10+ и FFmpeg (`ffprobe` в `PATH`).

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

```bash
curl -X POST http://localhost:8000/v1/ingestion/upload -F "file=@./sample.mp4"
curl -X POST http://localhost:8000/v1/ingestion/url \
  -H "Content-Type: application/json" \
  -d '{"url":"https://example.com/tmp/clip.mp4"}'
```
