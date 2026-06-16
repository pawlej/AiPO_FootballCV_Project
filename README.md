# AiPO Project — Football Computer Vision Locally

# Polish version below

Project author: Paweł Lejczak

## Directory structure

```text
AiPO_FootballCV_Project/
├── content/                 # input videos, e.g. match.mp4
├── models/                  # YOLO models, e.g. best.pt
├── outputs/                 # generated results
├── stubs/                   # ByteTrack/YOLO track cache
├── configs/                 # example configuration files
├── notebooks/               # inference and testing notebooks
├── main.py                  # main launcher: python main.py run/infer
└── src/aipo_football_cv/    # project source code
```

The `content/` directory is used for input match recordings. If this directory is not present after cloning the repository, it must be created manually:

```bash
mkdir content
```

The recordings should be downloaded from the DFL Bundesliga Data Shootout dataset:

```text
https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout/data?select=clips
```

The `models/` directory is used for YOLO model files. A trained model file, for example `best.pt`, should be placed there. Custom trained models can also be added to this directory.

If the `models/` directory is not present, it must also be created manually:

```bash
mkdir models
```

## Installation

A virtual environment should be created first:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

Alternatively, dependencies can be installed directly from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Minimal full pipeline execution

After adding a video file to `content/` and a model file to `models/`, the full pipeline can be executed with:

```bash
python main.py run \
  --video content/example_match.mp4 \
  --model models/best.pt \
  --output-dir outputs \
  --confidence 0.3
```

On Windows PowerShell, the same command can be executed in one line:

```powershell
python main.py run --video content/example_match.mp4 --model models/best.pt --output-dir outputs --confidence 0.3
```

Generated results:

* `outputs/<video_name>_annotated.mp4` — annotated video,
* `outputs/<video_name>_team_stats.csv` — team statistics,
* `outputs/<video_name>_player_stats.csv` — player statistics,
* `stubs/tracks.pkl` — optional tracking cache.

The project can also be executed as an installed command:

```bash
aipo-football-cv run --video content/example_match.mp4 --model models/best.pt
```

It can also be executed directly as a Python module:

```bash
python -m aipo_football_cv.cli run --video content/example_match.mp4 --model models/best.pt
```

## Quick test on a short video fragment

For debugging purposes, it is recommended to process only a short fragment first:

```bash
python main.py run \
  --video content/example_match.mp4 \
  --model models/best.pt \
  --max-frames 200
```

## Model-only inference

To check how the model performs without tracking and without the full statistics pipeline:

```bash
python main.py infer \
  --video content/example_match.mp4 \
  --model models/best.pt \
  --output-csv outputs/model_detections.csv \
  --max-frames 200
```

The same workflow is available in the notebook:

```text
notebooks/model_inference_check.ipynb
```

## View transform / perspective transformation

Perspective transformation is optional. The original workflow used a Roboflow model and the `sports` package, which require an API key. By default, this part is disabled.

To enable it, the required packages should be installed:

```bash
pip install inference
pip install git+https://github.com/roboflow/sports.git
```

Then the Roboflow API key should be set.

Windows PowerShell:

```powershell
$env:ROBOFLOW_API_KEY="your_api_key"
```

Linux/macOS:

```bash
export ROBOFLOW_API_KEY="your_api_key"
```

The pipeline can then be executed with perspective transformation enabled:

```bash
python -m aipo_football_cv.cli run \
  --video content/example_match.mp4 \
  --model models/best.pt \
  --enable-view-transform
```

---


# AiPO projekt — lokalny system Football Computer Vision

Autor projektu: Paweł Lejczak

## Struktura katalogów

```text
AiPO_FootballCV_Project/
├── content/                 # nagrania wejściowe, np. match.mp4
├── models/                  # modele YOLO, np. best.pt
├── outputs/                 # wygenerowane wyniki
├── stubs/                   # cache tracków ByteTrack/YOLO
├── configs/                 # przykładowa konfiguracja
├── notebooks/               # notebooki do inferencji i testów
├── main.py                  # główny launcher: python main.py run/infer
└── src/aipo_football_cv/    # kod źródłowy projektu
```

Folder `content/` służy do przechowywania nagrań meczowych. Jeżeli po sklonowaniu repozytorium folder nie istnieje, należy utworzyć go ręcznie:

```bash
mkdir content
```

Nagrania należy pobrać ze zbioru DFL Bundesliga Data Shootout:

```text
https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout/data?select=clips
```

Folder `models/` służy do przechowywania plików modeli YOLO. W tym folderze należy umieścić wytrenowany model, np. `best.pt`. Można również dodać własne wytrenowane modele.

Jeżeli folder `models/` nie istnieje, należy również utworzyć go ręcznie:

```bash
mkdir models
```

## Instalacja

Najpierw należy utworzyć środowisko wirtualne:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

Alternatywnie zależności można zainstalować z pliku `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Minimalne uruchomienie pełnego pipeline'u

Po dodaniu nagrania do folderu `content/` oraz modelu do folderu `models/`, pełny pipeline można uruchomić poleceniem:

```bash
python main.py run \
  --video content/example_match.mp4 \
  --model models/best.pt \
  --output-dir outputs \
  --confidence 0.3
```

W Windows PowerShell można użyć jednej linii:

```powershell
python main.py run --video content/example_match.mp4 --model models/best.pt --output-dir outputs --confidence 0.3
```

Wyniki działania systemu:

* `outputs/<nazwa_filmu>_annotated.mp4` — nagranie z adnotacjami,
* `outputs/<nazwa_filmu>_team_stats.csv` — statystyki drużynowe,
* `outputs/<nazwa_filmu>_player_stats.csv` — statystyki zawodników,
* `stubs/tracks.pkl` — opcjonalny cache tracków.

Projekt można również uruchomić jako zainstalowaną komendę:

```bash
aipo-football-cv run --video content/example_match.mp4 --model models/best.pt
```

Możliwe jest także uruchomienie bezpośrednio przez moduł Pythona:

```bash
python -m aipo_football_cv.cli run --video content/example_match.mp4 --model models/best.pt
```

## Szybki test na krótkim fragmencie

Do debugowania zalecane jest najpierw przetworzenie krótkiego fragmentu nagrania:

```bash
python main.py run \
  --video content/example_match.mp4 \
  --model models/best.pt \
  --max-frames 200
```

## Inferencja samego modelu

Do sprawdzenia działania modelu bez trackingu i bez pełnego pipeline'u statystyk można użyć:

```bash
python main.py infer \
  --video content/example_match.mp4 \
  --model models/best.pt \
  --output-csv outputs/model_detections.csv \
  --max-frames 200
```

Ten sam workflow znajduje się również w notebooku:

```text
notebooks/model_inference_check.ipynb
```

## View transform / transformacja perspektywy

Transformacja perspektywy jest opcjonalna. Oryginalny workflow wykorzystywał model Roboflow oraz pakiet `sports`, które wymagają klucza API. Domyślnie ta część jest wyłączona.

Aby ją włączyć, należy zainstalować wymagane pakiety:

```bash
pip install inference
pip install git+https://github.com/roboflow/sports.git
```

Następnie należy ustawić klucz Roboflow API.

Windows PowerShell:

```powershell
$env:ROBOFLOW_API_KEY="twoj_klucz"
```

Linux/macOS:

```bash
export ROBOFLOW_API_KEY="twoj_klucz"
```

Pipeline można wtedy uruchomić z włączoną transformacją perspektywy:

```bash
python -m aipo_football_cv.cli run \
  --video content/example_match.mp4 \
  --model models/best.pt \
  --enable-view-transform
```
