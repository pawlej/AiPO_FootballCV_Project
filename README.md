# AiPO projekt — Football Computer Vision lokalnie
Autor projektu: Paweł Lejczak
## Struktura katalogów

```text
AiPO_FootballCV_Project/
├── content/                 # tu wrzuć nagrania, np. match.mp4
├── models/                  # tu wrzuć modele YOLO, np. best.pt
├── outputs/                 # tutaj zapiszą się wyniki
├── stubs/                   # cache tracków ByteTrack/YOLO
├── configs/                 # przykładowa konfiguracja
├── notebooks/               # notebooki do inferencji/testów
├── scripts/                 # przykładowe komendy dla Windows/Linux
├── main.py                  # główny launcher: python main.py run/infer
└── src/aipo_football_cv/    # właściwy kod projektu
```

Do folderu `content` należy pobrać nagrania z: https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout/data?select=clips
Folder `models` zawier plik wyuczonego modelu YOLO `best.pt`. Dodaj tam własne modele.

## Instalacja


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

Alternatywnie:

```bash
pip install -r requirements.txt
```

## Minimalne uruchomienie pełnego pipeline'u

Po dodaniu nagrania do `content/` i modelu do `models/`:

```bash
python main.py run \
  --video content/example_match.mp4 \
  --model models/best.pt \
  --output-dir outputs \
  --confidence 0.3
```

Na Windows możesz użyć jednej linii:

```powershell
python main.py run --video content/example_match.mp4 --model models/best.pt --output-dir outputs --confidence 0.3
```

Wyniki:

- `outputs/<nazwa_filmu>_annotated.mp4` — nagranie z adnotacjami,
- `outputs/<nazwa_filmu>_team_stats.csv` — statystyki drużynowe,
- `outputs/<nazwa_filmu>_player_stats.csv` — statystyki zawodników,
- `stubs/tracks.pkl` — opcjonalny cache tracków.


Możesz też odpalać projekt jako zainstalowaną komendę:

```bash
aipo-football-cv run --video content/example_match.mp4 --model models/best.pt
```

Albo bez `main.py` przez moduł Pythona:

```bash
python -m aipo_football_cv.cli run --video content/example_match.mp4 --model models/best.pt
```

## Szybki test na krótkim fragmencie

Do debugowania warto najpierw przetworzyć mały fragment:

```bash
python main.py run \
  --video content/example_match.mp4 \
  --model models/best.pt \
  --max-frames 200
```

## Model-only inference

Do sprawdzenia, jak model się nauczył, bez trackingu i bez całego systemu:

```bash
python main.py infer \
  --video content/example_match.mp4 \
  --model models/best.pt \
  --output-csv outputs/model_detections.csv \
  --max-frames 200
```

Ten sam workflow jest też w notebooku:

```text
notebooks/model_inference_check.ipynb
```

## View transform / transformacja perspektywy

Transformacja pozycji na współrzędne boiska jest opcjonalna. Oryginalny notebook korzystał z modelu Roboflow i pakietu `sports`, co wymaga API key. Domyślnie ta część jest wyłączona.

Aby ją włączyć:

```bash
pip install inference
pip install git+https://github.com/roboflow/sports.git
```

Następnie ustaw klucz:

Windows PowerShell:

```powershell
$env:ROBOFLOW_API_KEY="twoj_klucz"
```

Linux/macOS:

```bash
export ROBOFLOW_API_KEY="twoj_klucz"
```

I uruchom:

```bash
python -m aipo_football_cv.cli run \
  --video content/example_match.mp4 \
  --model models/best.pt \
  --enable-view-transform
```

