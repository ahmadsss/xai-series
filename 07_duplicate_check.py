# Exact-duplicate and readability check for data/brain_mri (Section 3.3 of the report).
# Exact duplicates = identical file bytes (MD5). Near-duplicates (resized/re-encoded copies) are NOT detected.
import hashlib, os, collections, json
from PIL import Image

ROOT = "data/brain_mri"
hashes = collections.defaultdict(list)
unreadable = []
for split in ("training", "testing"):
    for c in sorted(os.listdir(os.path.join(ROOT, split))):
        for f in sorted(os.listdir(os.path.join(ROOT, split, c))):
            p = os.path.join(ROOT, split, c, f)
            hashes[hashlib.md5(open(p, "rb").read()).hexdigest()].append((split, c, f))
            try:
                Image.open(p).verify()
            except Exception as e:
                unreadable.append((p, str(e)))

files = sum(len(v) for v in hashes.values())
groups = [v for v in hashes.values() if len(v) > 1]
print(f"files {files} | unique contents {len(hashes)} | unreadable {len(unreadable)}")
print(f"duplicate groups {len(groups)} | label conflicts {sum(len({c for _, c, _ in v}) > 1 for v in groups)}")

train = {m for m, v in hashes.items() if any(s == "training" for s, _, _ in v)}
leak = [f"{c}/{f}" for m, v in hashes.items() if m in train for s, c, f in v if s == "testing"]
per_class = collections.Counter(x.split("/")[0] for x in leak)
totals = collections.Counter((c) for v in hashes.values() for s, c, _ in v if s == "testing")
for c in sorted(totals):
    print(f"test {c}: {per_class[c]} / {totals[c]} have an identical copy in training")
print(f"total test images with a training duplicate: {len(leak)} / {sum(totals.values())}")

# Duplicates inside the same split and class folder (extra copies beyond the first) and unique images
for split in ("training", "testing"):
    print(f"\n{split}: class | images | unique | extra copies within this folder")
    for c in sorted(totals):
        files = [m for m, v in hashes.items() for s, cc, _ in v if s == split and cc == c]
        n, u = len(files), len(set(files))
        print(f"  {c}: {n} | {u} | {n - u}")
    allf = [m for m, v in hashes.items() for s, _, _ in v if s == split]
    print(f"  total: {len(allf)} | {len(set(allf))} | {len(allf) - len(set(allf))}")
json.dump(sorted(leak), open("leaked_test_files.json", "w"), indent=0)
