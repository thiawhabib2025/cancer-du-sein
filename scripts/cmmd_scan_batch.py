"""Scan CMMD par lots reprenables.

Le montage de fichiers est trop lent pour tout traiter en un appel : ce script
traite un budget de temps donné puis s'arrête proprement. Le relancer reprend
où il s'était arrêté (progression dans data/cache/cmmd_headers.csv).

Usage :  python scripts/cmmd_scan_batch.py [budget_secondes]
"""
import sys, time, csv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pydicom

BUDGET = int(sys.argv[1]) if len(sys.argv) > 1 else 130
ROOT = Path("data/raw/cmmd/CMMD/cmmd")
CACHE = Path("data/cache/cmmd_headers.csv")

t0 = time.time()

done: set[str] = set()
if CACHE.exists():
    with open(CACHE) as f:
        done = {row.split(",")[0] for row in f.read().splitlines()[1:] if row}
else:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text("patient,path,side,view\n")

patients = sorted(d.name for d in ROOT.iterdir() if d.is_dir())
todo = [p for p in patients if p not in done]
print(f"{len(patients)} patientes au total, {len(done)} faites, {len(todo)} restantes")


def scan_patient(pid: str) -> list[tuple[str, str, str, str]]:
    out = []
    for dcm in (ROOT / pid).rglob("*.dcm"):
        try:
            ds = pydicom.dcmread(str(dcm), stop_before_pixels=True)
            side = str(getattr(ds, "ImageLaterality",
                               getattr(ds, "Laterality", ""))).strip().upper()[:1]
            view = str(getattr(ds, "ViewPosition", "")).strip().upper()
        except Exception:
            side, view = "", ""
        out.append((pid, str(dcm.relative_to(ROOT.parent.parent)), side, view))
    return out


n_done = 0
with open(CACHE, "a", newline="") as f:
    w = csv.writer(f)
    with ThreadPoolExecutor(max_workers=16) as ex:
        # par petits blocs pour pouvoir s'arrêter à temps
        BLOCK = 40
        for i in range(0, len(todo), BLOCK):
            if time.time() - t0 > BUDGET:
                break
            block = todo[i:i + BLOCK]
            for rows in ex.map(scan_patient, block):
                for r in rows:
                    w.writerow(r)
            n_done += len(block)
            f.flush()

remaining = len(todo) - n_done
print(f"+{n_done} patientes en {time.time()-t0:.0f}s — reste {remaining}")
print("TERMINE" if remaining <= 0 else "RELANCER")
