# Urban degradation detection 

Smart Road Damage Monitoring dashboard powered by your trained **STCrackNet**
(F1 0.7199, IoU 0.5624) plus a satellite-imagery analysis pipeline.

A local web dashboard at `http://localhost:8000`:

- **Click anywhere on the map** → satellite analysis for that point
- **Draw a region** → grid of analyses with summary stats
- **Upload a close-up road photo** → runs your STCrackNet model
- Classification (NORMAL / MODERATE / SEVERE), RDI score, damage overlays,
  history, report export — all in a dark-themed dashboard

## Architecture

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


Damage metrics
 `crack_coverage` — cracks / road pixels
 `crack_density`  — cracks / total pixels
 `mean_width_px`  — mean crack width (distance transform)
 `connectivity`   — largest component / total
 `severity_score` — 0–1 composite
 `rdi`            — 0–100 Road Degradation Index
 `classification` — `NORMAL` / `MODERATE` / `SEVERE`

<img width="1313" height="777" alt="image" src="https://github.com/user-attachments/assets/d56e6a38-e387-43ff-bc90-0deb5d498c3e" />
<img width="971" height="677" alt="image" src="https://github.com/user-attachments/assets/26987728-3603-40e9-a8d8-3aa68514d0c8" />
<img width="1278" height="788" alt="image" src="https://github.com/user-attachments/assets/b92c9809-b8e4-4617-ba04-34b11958c551" />
