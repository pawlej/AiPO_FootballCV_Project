# AiPO projekt — Football Computer Vision lokalnie

Ten projekt jest lokalną, uporządkowaną wersją notebooka Colab. Kod został rozdzielony na moduły, usunięto zależność od Google Drive/Colaba, a wczytywanie klatek przeniesiono na `supervision.get_video_frames_generator`, dzięki czemu łatwiej uruchamiać analizę na własnym komputerze.

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

Foldery `content` i `models` są celowo puste. Dodaj tam własne pliki lokalnie.

## Instalacja

Najwygodniej utworzyć osobne środowisko:

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

Transformacja pozycji na współrzędne boiska jest opcjonalna. Oryginalny notebook korzystał z modelu Roboflow i pakietu `sports`, co wymaga API key. Domyślnie ta część jest wyłączona, żeby projekt działał lokalnie bez Colaba.

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

## Co zostało poprawione względem notebooka

1. Usunięto Colaba, Google Drive i komórki `pip install` z kodu wykonawczego.
2. Zmieniono twarde ścieżki na argumenty CLI i lokalne foldery `content/`, `models/`, `outputs/`.
3. Wczytywanie filmu przeniesiono na generator klatek z `supervision`.
4. Kod rozdzielono na moduły: tracking, przypisanie drużyn, przypisanie piłki, statystyki, zapis wideo, inferencja modelu.
5. Poprawiono niespójność `has ball` → `has_ball`.
6. Dodano odporniejsze interpolowanie pozycji piłki z `ffill/bfill`.
7. Dodano notebook do samej inferencji modelu i zapis wyników detekcji do CSV.
8. Zmieniono opis projektu na `AiPO projekt`.

## Typowe problemy

### `ModuleNotFoundError: No module named 'aipo_football_cv'`

Uruchom projekt po instalacji editable:

```bash
pip install -e .
```

albo uruchamiaj komendy z głównego katalogu projektu.

### Model nie znajduje klas `player`, `goalkeeper`, `referee`, `ball`

Pipeline zakłada takie nazwy klas, bo tak działał pierwotny notebook. Jeśli Twój model ma inne nazwy klas, zmień mapowanie w `src/aipo_football_cv/tracker.py`.

### Statystyki prędkości są dziwne

Bez transformacji perspektywy prędkość jest przybliżeniem z pozycji pikselowych. Do sensowniejszych wartości w metrach włącz `--enable-view-transform` albo dostosuj `--meter-scale`.
