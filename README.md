# RoadSense — Windows + VS Code setup

Smart Road Damage Monitoring dashboard powered by your trained **STCrackNet**
(F1 0.7199, IoU 0.5624) plus a satellite-imagery analysis pipeline.

This guide assumes **zero prior setup** on your Windows machine.

---

## What you will build

A local web dashboard at `http://localhost:8000`:

- **Click anywhere on the map** → satellite analysis for that point
- **Draw a region** → grid of analyses with summary stats
- **Upload a close-up road photo** → runs your STCrackNet model
- Classification (NORMAL / MODERATE / SEVERE), RDI score, damage overlays,
  history, report export — all in a dark-themed dashboard

---

## Prerequisites (one-time)

### 1 · Install Python 3.10+

1. Go to <https://www.python.org/downloads/>
2. Download the **Windows 64-bit installer**
3. Run it — **tick "Add python.exe to PATH"** at the bottom of the first screen
4. Install

To verify: open **Command Prompt** (Win+R → `cmd`) and type:

```bat
python --version
```

You should see `Python 3.11.x` or similar. If you get "not recognized",
Python wasn't added to PATH — re-install with the checkbox ticked.

### 2 · Install VS Code

1. <https://code.visualstudio.com/>
2. Install the **Python extension** (click Extensions icon on left, search "Python", install the Microsoft one)

### 3 · Get a Google Maps API key

1. Go to <https://console.cloud.google.com/>
2. Create a new project (name it anything, e.g. "roadsense")
3. Search "Maps Static API" in the top search bar → click it → **Enable**
4. Go to **APIs & Services → Credentials** → **Create Credentials → API Key**
5. Copy the key (looks like `AIzaSy...`)

Google gives you $200 of free credit monthly — plenty for this project.

---

## Project setup

### Step 1 · Unzip the project

Unzip `roadsense.zip` somewhere clean, e.g. `C:\Users\YOU\Desktop\roadsense`.

Folder should look like:

```
roadsense\
  backend\
  frontend\
  weights\
  setup.bat
  run.bat
  .env.example
  requirements.txt
  README.md
```

### Step 2 · Put your model weights in place

From your Colab notebook, `STCrackNet_final.pth` is saved in Google Drive at
`/content/drive/MyDrive/STCrackNet_final.pth`.

1. Open [Google Drive](https://drive.google.com) in your browser
2. Find `STCrackNet_final.pth`
3. Right-click → **Download**
4. Move the downloaded file into the `roadsense\weights\` folder

After this, the path should be: `roadsense\weights\STCrackNet_final.pth`

> Skipping this step is fine — satellite mode still works without it. The
> "Upload Pavement" button will just be disabled.

### Step 3 · Open the project in VS Code

1. Launch VS Code
2. `File → Open Folder` → pick the `roadsense` folder
3. Open a terminal inside VS Code: `Terminal → New Terminal`
   (or press `` Ctrl+` ``)

The terminal opens in the `roadsense` folder automatically.

### Step 4 · Run the setup script

In the VS Code terminal, type:

```bat
setup.bat
```

This will:
- check your Python install
- create a virtual environment (the `.venv` folder)
- install FastAPI, PyTorch, OpenCV, etc. (3–5 minutes, ~500 MB)
- copy `.env.example` → `.env`
- verify your model weights

When it finishes, press any key to close the prompt.

### Step 5 · Add your API key

1. In VS Code, open the file `.env` (it's at the project root)
2. Paste your Google Maps key after the `=` sign:

```env
GOOGLE_MAPS_API_KEY=AIzaSyDWx7FyuKDhrtoixEkzX2V0-qWCXsWVEOs
```

3. Save the file (Ctrl+S)

### Step 6 · Run the app

In the VS Code terminal:

```bat
run.bat
```

You should see output like:

```
========================================================
  Starting RoadSense server...
========================================================
  Dashboard:  http://localhost:8000
  API docs:   http://localhost:8000/docs
...
[OK] STCrackNet loaded from weights/STCrackNet_final.pth on cpu
INFO: Uvicorn running on http://127.0.0.1:8000
```

Your browser opens automatically to the dashboard.

To **stop the server**, press `Ctrl+C` in the terminal.

---

## Using the dashboard

**Click-to-analyse**: just click anywhere on the satellite map. RoadSense
fetches the tile from Google, extracts the road regions, detects cracks,
and shows metrics on the right.

**Region analysis**: click **Draw Region**, drag a rectangle on the map.
A 3×3 grid of samples runs and summary stats update.

**Pavement upload**: click **Upload Pavement** (top right). Pick any
close-up road photo — your STCrackNet runs and highlights cracks.

**Export**: click **Export Report** to download a JSON of everything
analysed in the current session.

---

## Troubleshooting

### "python is not recognized"

Python isn't on PATH. Re-install Python and tick the "Add to PATH" box,
or follow [this guide](https://docs.python.org/3/using/windows.html#finding-the-python-executable).

### `setup.bat` fails on `pip install`

Your network might be blocking pypi. Try from a different network, or use
a VPN. If you're behind a proxy, set `HTTP_PROXY` and `HTTPS_PROXY` in the
terminal before running setup.

### Server starts but browser shows "Backend offline"

- Make sure `run.bat` is still running in the terminal
- Try <http://127.0.0.1:8000/health> directly — you should see JSON
- Windows Firewall may have blocked it; allow Python when prompted

### "tile fetch failed: ..." in the browser

Your Google key is wrong, missing, or the Maps Static API isn't enabled.
Open `.env`, check the key, make sure **Maps Static API** is enabled in
the Google Cloud Console for your project.

### "STCrackNet offline" chip in the top bar

`weights\STCrackNet_final.pth` is missing. Satellite mode still works —
but you can't upload pavement photos. Download the weights from Drive
(see Step 2).

### Torch install is very slow / fails

PyTorch is ~200 MB. On slow connections, run:

```bat
.venv\Scripts\activate.bat
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

---

## Architecture

```
roadsense\
├── backend\
│   ├── model.py        STCrackNet (your trained model)
│   ├── analysis.py     Road mask · crack detect · metrics · overlays
│   ├── providers.py    Google Static Maps · Sentinel Hub
│   └── server.py       FastAPI endpoints
├── frontend\
│   └── index.html      Dashboard — Leaflet + Leaflet.draw
├── weights\
│   └── STCrackNet_final.pth   (you place this)
├── .env.example
├── requirements.txt
├── setup.bat           first-time setup
├── run.bat             launch server
└── README.md           this file
```

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET  | `/` | Dashboard |
| GET  | `/health` | Status |
| POST | `/api/analyze` | Single (lat, lon) → metrics + images |
| POST | `/api/analyze-region` | n×n grid around a point |
| POST | `/api/analyze-upload` | Pavement photo via STCrackNet |
| GET  | `/api/history` | Last 200 analyses |
| GET  | `/api/image/{id}/{key}` | Stored PNG |

### Damage metrics

- `crack_coverage` — cracks / road pixels
- `crack_density`  — cracks / total pixels
- `mean_width_px`  — mean crack width (distance transform)
- `connectivity`   — largest component / total
- `severity_score` — 0–1 composite
- `rdi`            — 0–100 Road Degradation Index
- `classification` — `NORMAL` / `MODERATE` / `SEVERE`

---

## Note on the satellite pipeline

STCrackNet was trained on 256×256 close-up pavement crops, so it would
not work well on ~30 cm/pixel Google or Sentinel tiles. Satellite mode
therefore uses a two-stage classical pipeline:

1. HSV-based road extraction isolates asphalt regions
2. Black-hat + Canny + elongation filter detects crack-like lines,
   masked to the road region

This gives honest, interpretable results for the demo. When you later
train a proper satellite-damage model, swap it into `analyze_satellite()`
in `backend/analysis.py` — everything else stays the same.
