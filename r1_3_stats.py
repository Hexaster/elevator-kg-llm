"""R1-3 statistics: per-item Key Entity Recall, paired significance tests, bootstrap CIs.

Recomputes Key Entity Recall per item (exactly matching utils.calculate_keyword_recall),
validates aggregate means against the published table, then pairs baseline vs KG by `id`
and reports n, mean+-SD, the mean gain with a bootstrap 95% CI, and a paired Wilcoxon
signed-rank test. Read-only; prints to stdout.
"""
import json
import os
import re
import numpy as np
from scipy import stats

BASE = os.environ.get("DATA_DIR", "./data")

# model -> (baseline file, KG file, published baseline mean, published KG mean)
MODELS = {
    "DeepSeek-R1": (f"{BASE}/output/fault_qa_deepseek-r1.jsonl",
                    f"{BASE}/fault_qa_deepseek-r1_kgrag_multi.jsonl", 35.79, 53.47),
    "DeepSeek-V3": (f"{BASE}/output/fault_qa_deepseek-v3.jsonl",
                    f"{BASE}/fault_qa_deepseek-v3_kgrag_multi.jsonl", 39.26, 56.69),
    "GPT-4o":      (f"{BASE}/output/fault_qa_gpt-4o.jsonl",
                    f"{BASE}/fault_qa_4o_kgrag_multi.jsonl", 41.31, 57.23),
    "GPT-o1":      (f"{BASE}/output/fault_qa_o1.jsonl",
                    f"{BASE}/fault_qa_o1_kgrag_multi.jsonl", 29.76, 57.94),
}


def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    return re.sub(r"\s+", " ", text).lower().strip()


def item_recall(keywords, output):
    if not keywords:
        return 0.0
    out = clean_text(output)
    matched = sum(1 for kw in keywords if re.search(re.escape(clean_text(kw)), out))
    return matched / len(keywords)


def load_recall_by_id(path):
    """Return {id: recall} for one file."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[str(r["id"])] = item_recall(r.get("keywords", []), r.get("output", ""))
    return out


def main():
    print("=" * 78)
    print("STEP 1 — recomputed Recall means vs published table")
    print("=" * 78)
    print(f"{'Model':14s} {'base(calc)':>10s} {'base(tab)':>9s} "
          f"{'kg(calc)':>9s} {'kg(tab)':>8s}  match")

    paired = {}  # model -> (base_arr, kg_arr) aligned by id
    for m, (bf, kf, btab, ktab) in MODELS.items():
        base = load_recall_by_id(bf)
        kg = load_recall_by_id(kf)
        ids = sorted(set(base) & set(kg), key=lambda x: int(x))
        b = np.array([base[i] for i in ids])
        k = np.array([kg[i] for i in ids])
        paired[m] = (b, k, ids)
        bc, kc = b.mean() * 100, k.mean() * 100
        ok = abs(bc - btab) < 0.05 and abs(kc - ktab) < 0.05
        print(f"{m:14s} {bc:10.2f} {btab:9.2f} {kc:9.2f} {ktab:8.2f}  "
              f"{'OK' if ok else 'MISMATCH'}  (n={len(ids)})")

    print()
    print("=" * 78)
    print("STEP 2 — paired statistics (KG - baseline), per model, n items")
    print("=" * 78)
    rng = np.random.default_rng(0)
    for m, (b, k, ids) in paired.items():
        n = len(ids)
        diff = k - b
        gain = diff.mean()
        # bootstrap 95% CI of the mean paired gain (percentile, 10k resamples)
        boot = np.array([rng.choice(diff, size=n, replace=True).mean()
                         for _ in range(10000)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        # paired Wilcoxon signed-rank (two-sided); zeros dropped (default 'wilcox')
        nonzero = int(np.count_nonzero(diff))
        if nonzero > 0:
            w = stats.wilcoxon(k, b)
            pstr = f"{w.pvalue:.2e}"
        else:
            pstr = "n/a"
        dz = gain / diff.std(ddof=1) if diff.std(ddof=1) > 0 else float("nan")
        impr = int((diff > 0).sum())
        wors = int((diff < 0).sum())
        tied = int((diff == 0).sum())
        print(f"\n{m} (n={n})")
        print(f"  baseline recall : {b.mean()*100:5.2f}% +- {b.std(ddof=1)*100:5.2f}")
        print(f"  KG recall       : {k.mean()*100:5.2f}% +- {k.std(ddof=1)*100:5.2f}")
        print(f"  mean gain       : {gain*100:+5.2f} pp  "
              f"95% CI [{lo*100:+.2f}, {hi*100:+.2f}]")
        print(f"  Wilcoxon p      : {pstr}  (nonzero pairs={nonzero})")
        print(f"  Cohen's d_z     : {dz:.2f}   improved/tied/worsened = "
              f"{impr}/{tied}/{wors}")


if __name__ == "__main__":
    main()
