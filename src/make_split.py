import argparse, random, re, shutil
from collections import defaultdict
from config import SOURCE_DIR, DATASET_DIR, SEED

def part_id(stem: str) -> str:
    """Which physical part does this image belong to?"""
    if "_aug_" in stem: # NG_2_aug_031 -> NG_2
        return stem.split("_aug_")[0]
    m = re.match(r"(.+_p\d+)_\d+$", stem) #ok_p012_03 -> ok_p012
    return m.group(1) if m else stem # NG_2

def collect(folder):
    groups = defaultdict(list)
    for p in sorted(folder.rglob("*.jpg")):
        groups[part_id(p.stem)].append(p)
    return groups

def copy_all(files, dst):
    dst.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, dst / f.name)

ap = argparse.ArgumentParser()
ap.add_argument("--train_frac", type=float, default=0.7)
ap.add_argument("--smoke", action="store_true",
                help="split oOK images by IMAGE (leaky). Pipeline test only!")
                
args = ap.parse_args()

random.seed(SEED)
if DATASET_DIR.exists():
    shutil.rmtree(DATASET_DIR)

ok = collect(SOURCE_DIR / "ok")
ng = collect(SOURCE_DIR / "ng")
print(f"OK parts: {len(ok)} | NG parts: {len(ng)}")

if args.smoke or len(ok) < 5:
    print("WARNING: too few OK parts, spliting by image. Results are NOT trustworthy.")
    files = [f for fs in ok.values() for f in fs]
    random.shuffle(files)
    n = int(len(files) * args.train_frac)
    train, test = files[:n], files[n:]
else:
    parts = sorted(ok)
    random.shuffle(parts)
    n = max(1, int(len(parts) * args.train_frac))
    train = [f for p in parts[:n] for f in ok[p]]
    test = [f for p in parts[n:] for f in ok[p]]

copy_all(train, DATASET_DIR / "train" / "good")
copy_all(test, DATASET_DIR / "test" / "good")
copy_all([f for fs in ng.values() for f in fs], DATASET_DIR / "test" / "defect")
print(f"train/good={len(train)} test/good{len(test)} "
      f"test/defect={sum(len(v) for v in ng.values())}")