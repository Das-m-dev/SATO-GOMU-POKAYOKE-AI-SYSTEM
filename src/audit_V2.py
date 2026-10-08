import hashlib
import pandas as pd

df = pd.read_csv("outputs/audit.csv")
df["size"] = df["w"].astype(str) + "x" + df["h"].astype(str)

# 1. Stats per label AND size (never mix sizes)
for (label, size), g in df.groupby(["label","size"]):
    print(f"\n=== {label} | {size} | n={len(g)} ===")
    print(g[["brightness", "sharpness"]]
            .describe(percentiles=[.05, .5, .95]).round(1))

#2. Lowest-sharpness images (blur suspects), main size only
main = df[df["size"] == "4032x3024"]
print("\nBlurriest 10 images:")
print(main.nsmallest(10, "sharpness")[["path", "sharpness"]].to_string(index=False))

#3. Exact duplicate files
def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

df["md5"] = df["path"].apply(md5)
print("\nExact duplicate files:", df.duplicated("md5").sum())

#4. The odd-size images
print("\nNon-standard size files:")
print(df[df["size"] != "2048x1536"]["path"].to_string(index=False))