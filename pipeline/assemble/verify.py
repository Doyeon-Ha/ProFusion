#!/usr/bin/env python
"""Check array by array that a rebuild (results/rebuild/) matches the data cache (data/).

npz files carry zip timestamps, so they cannot be compared by file hash; the arrays are compared
key by key instead.
- The DMS sets must match (194 ProteinGym, 49 external).
- Every key on the rebuild side must match the cache exactly in dtype, shape and value (including
  the positions of NaN).
  One exception: the external ddg_available. The cache was built by a version of the script from
  before that key existed, so the cache does not have it.
- Keys present only in the cache are reported as a list. Those were added after assembly (for
  example llr_TranceptionL on both sides, and llr_S2F on ProteinGym). Neither the paper numbers nor
  the figures use them.

run: python pipeline/assemble/verify.py   (exits 1 if anything fails)
"""
from __future__ import annotations

import sys

import numpy as np

from plmfusion import paths

SIDES = {
    "proteingym": (paths.RESULTS / "rebuild/proteingym/two_dim", paths.pg_cache(), set()),
    "mavedb_external": (paths.RESULTS / "rebuild/mavedb_external/two_dim", paths.ext_cache(), {"ddg_available"}),
}


def same(a: np.ndarray, b: np.ndarray) -> bool:
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.dtype.kind in "fc":
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def main() -> int:
    bad = 0
    for side, (new_dir, old_dir, skip) in SIDES.items():
        new = {p.stem: p for p in new_dir.glob("*.npz")}
        old = {p.stem: p for p in old_dir.glob("*.npz")}
        print(f"\n[{side}] rebuild {len(new)} / cache {len(old)}")
        if set(new) != set(old):
            bad += 1
            print(f"  x DMS sets differ: only in rebuild {sorted(set(new) - set(old))[:5]}, only in cache {sorted(set(old) - set(new))[:5]}")
        n_keys, cache_only, bad0 = 0, set(), bad
        for stem in sorted(set(new) & set(old)):
            a, b = np.load(new[stem], allow_pickle=True), np.load(old[stem], allow_pickle=True)
            cache_only |= set(b.files) - set(a.files)
            for k in a.files:
                if k in skip:
                    continue
                n_keys += 1
                if k not in b.files:
                    bad += 1; print(f"  x {stem}: key {k} is not in the cache"); continue
                if not same(a[k], b[k]):
                    bad += 1; print(f"  x {stem}: {k} values differ")
        print(f"  compared {n_keys:,} (DMS, key) pairs, {bad - bad0} mismatches")
        if cache_only:
            print(f"  keys only in the cache (added after assembly, unused by the paper): {sorted(cache_only)}")
    print("\nresult:", "all match" if bad == 0 else f"{bad} mismatches")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
