"""Check that the inference path (predict) produces **the same scores** as the evaluation path
(external.ERec).

Why it matters
    The fixed weights were fitted on the channel convention of the evaluation path. If the
    inference path builds its channels even slightly differently, those weights refer to
    different quantities and the result is silently wrong.

What is checked
    The same cached npz is fed to both paths and the scores compared.
    1) Feeding the channels stored in the npz (ch_*) must give **character-for-character** equality.
    2) Starting from raw values (ddg, rsa) must agree to within floating-point error.
       (build_channels z has no epsilon while common.zs adds +1e-9, giving a 1e-9 difference.)
    3) BLOSUM and Delta charge rebuilt from wt/mut must match the npz.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from plmfusion import paths
from plmfusion.common import llr_key
from plmfusion.external import ERec
from plmfusion.predict import AA20, Channels, axis_of, build_matrix, load_weights, predict

BB = "ESM2"
# The actual key in the npz. ESM2 uses the masked-marginal variant (llr_ESM2mask), via
# common.LLR_ALIAS. ERec reads the same key, so picking llr_ESM2 directly here would make the
# two paths look at different values.
LLR_KEY = f"llr_{llr_key(BB)}"


def _cases(n=3):
    try:
        files = sorted(paths.ext_cache().glob("*.npz"))[:n]
    except FileNotFoundError:
        return []
    return files


CASES = _cases()
if not CASES:
    pytest.skip("no external-set cache (see scripts/link_data.sh)", allow_module_level=True)


def _channels_from_npz(d, use_raw: bool):
    pos1 = d["pos1"].astype(np.int64)
    wt = np.array([AA20[i] for i in d["wt_i"].astype(int)])
    mut = np.array([AA20[i] for i in d["mut_i"].astype(int)])
    if use_raw:
        # Start from raw values. rsa is stored per position, so expand it per variant.
        gemme, ddg = d["ch_gemme"], d["ddg"].astype(float)
        rsa = d["rsa"].astype(float)[d["pidx"].astype(int)]
    else:
        # The stored channels as they are. Same values ERec is fed.
        # predict flips the sign of ddg internally, so -ch_ddg is passed.
        gemme, ddg, rsa = d["ch_gemme"], -d["ch_ddg"].astype(float), d["ch_rsa"]
    return Channels(pos1=pos1, wt=wt, mut=mut, gemme=gemme, rsa=rsa, ddg=ddg,
                    llr=d[LLR_KEY].astype(float))


@pytest.mark.parametrize("f", CASES, ids=[f.stem for f in CASES])
def test_same_as_evaluation_path(f):
    """Feeding the stored channels must reproduce the evaluation path exactly."""
    d = np.load(f, allow_pickle=True)
    if LLR_KEY not in d.files:
        pytest.skip(f"{f.stem} has no {LLR_KEY}")
    w = load_weights(str(d["axis"]), BB)
    want = ERec(f, BB).s(w)
    got = predict(_channels_from_npz(d, use_raw=False), w)
    assert got.shape == want.shape
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("f", CASES, ids=[f.stem for f in CASES])
def test_from_raw_values(f):
    """Starting from raw values (ΔΔG, RSA) must agree to within floating-point error."""
    d = np.load(f, allow_pickle=True)
    if LLR_KEY not in d.files:
        pytest.skip(f"{f.stem} has no {LLR_KEY}")
    w = load_weights(str(d["axis"]), BB)
    want = ERec(f, BB).s(w)
    got = predict(_channels_from_npz(d, use_raw=True), w)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-6)


@pytest.mark.parametrize("f", CASES[:1], ids=[f.stem for f in CASES[:1]])
def test_substitution_tables_match(f):
    """BLOSUM and Δcharge rebuilt from wt/mut must match the cache."""
    d = np.load(f, allow_pickle=True)
    ch = _channels_from_npz(d, use_raw=False)
    from plmfusion.predict import _blosum20, _CHG, AAI
    wt_i = np.array([AAI[a] for a in ch.wt])
    mt_i = np.array([AAI[a] for a in ch.mut])
    np.testing.assert_array_equal(_blosum20()[wt_i, mt_i], d["blosum"])
    np.testing.assert_allclose(_CHG[mt_i] - _CHG[wt_i], d["dchg"].astype(float), atol=1e-6)


def test_weights_and_axis_lookup():
    w = load_weights("function", "ESM2")
    assert w.shape == (7,) and w[0] > 0
    assert axis_of("Binding") == "function"
    assert axis_of("Expression") == "stability"
    with pytest.raises(ValueError, match="Unknown assay type"):
        axis_of("Stability")          # folding ΔG, never validated with these weights
    with pytest.raises(ValueError, match="comp7 weight rows"):
        load_weights("stability_ddg", "ESM2")


def test_matrix_shape_and_residual_degeneracy():
    """With one variant per position the residual components (b_*) are 0, as documented."""
    n = 10
    rng = np.random.default_rng(0)
    ch = Channels(pos1=np.arange(1, n + 1), wt=np.array(list("ACDEFGHIKL")),
                  mut=np.array(list("LKIHGFEDCA")), gemme=rng.normal(size=n),
                  rsa=rng.random(n), ddg=rng.normal(size=n), llr=rng.normal(size=n))
    CH = build_matrix(ch)
    assert CH.shape == (n, 7)
    for k in (1, 3, 5, 6):                      # b_GEMME, b_ddG, b_BLOSUM, b_dchg
        assert np.abs(CH[:, k]).max() < 1e-12


@pytest.mark.parametrize("f", CASES, ids=[f.stem for f in CASES])
def test_deleterious_direction_against_labels(f):
    """The top of deleterious_rank must **actually** be the variants with low labels.

    This pins the sign to the labels rather than to the code. predict_table was once inverted,
    and every other test still passed, because the scores themselves were right and only the
    rank labelling was flipped.

    ⚠ On a DMS where the model has no signal the direction cannot be measured (if the score is
      uncorrelated with the label, both means are the same). So only DMS with rho > 0.1 are
      checked. This tests the **sign convention**, not performance.
    """
    from plmfusion.predict import predict_table
    d = np.load(f, allow_pickle=True)
    if LLR_KEY not in d.files:
        pytest.skip(f"{f.stem} has no {LLR_KEY}")
    w = load_weights(str(d["axis"]), BB)
    rec = ERec(f, BB)
    ch = _channels_from_npz(d, use_raw=False)
    T = predict_table(ch, w)

    # Mechanical properties, which must hold whether or not there is signal
    assert T.deleterious_rank.iloc[0] == 1
    assert T.fused.iloc[0] == T.fused.min()
    assert T.fused.is_monotonic_increasing

    rho = rec.rho(w)
    if rho <= 0.1:
        pytest.skip(f"{f.stem}: fused rho {rho:+.3f}, no signal so the direction is unmeasurable")
    lab = pd.Series(d["ry"].astype(float),
                    index=[f"{a}{p}{b}" for a, p, b in zip(ch.wt, ch.pos1, ch.mut)])
    k = max(50, len(T) // 10)
    worst = lab.reindex(T.head(k).mutant).mean()          # predicted damaging
    best = lab.reindex(T.tail(k).mutant).mean()           # predicted tolerated
    assert worst < best, (
        f"{f.stem}: mean label at the top of deleterious_rank is {worst:.4f}, which should be "
        f"below the bottom at {best:.4f} (the sign may be inverted)")
