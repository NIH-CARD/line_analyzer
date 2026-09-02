"""PCA of the clone panel, coloured by genotype class.

    python -m proteomics_revertant.pca --data data --outdir results

What this is for. Before trusting any per-protein result you want to know how the
libraries actually arrange themselves. The useful question is not "do the
genotypes separate" -- with one clone per genotype they always will, because
clone identity and genotype are the same thing. It is *which* axis separates
them, and whether the revertant lands where the design says it should.

    If PC1 separates wild type from edit AND the revertant returns toward
    wild type, the dominant axis of variation is tracking allele dosage.

    If PC1 separates by plex, by passage, or puts the revertant further from
    wild type than the edit is, the dominant axis is technical or clonal and
    the per-protein results are carrying that load.

Three things are drawn:

  scores          one point per library, coloured by genotype class, with a 50%
                  transparent cloud per class (a 95% normal-theory ellipse where
                  there are enough libraries, a padded convex hull otherwise).
  loading arrows  the proteins pulling hardest on the two axes being shown.
  loading table   for PC1-10, the features with the largest absolute loadings,
                  written to TSV so they can be followed up. A second table
                  ranks up to the top 100 features by variance-weighted
                  influence across the retained components.

Only the three panels among PC1, PC2 and PC3 are plotted (1v2, 1v3, 2v3). The
loading TABLE still covers PC1-10, because a component can matter for
interpretation long after it has stopped being worth a scatter plot.

No imputation. PCA needs a complete matrix, so it runs on the proteins detected
in every library; the count kept and dropped is printed and written into the
figure subtitle, because a PCA on 60% of the proteome is a different statement
from one on 95%.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Ellipse, Polygon  # noqa: E402

# genotype classes, in legend order
CLASS_COLOURS = {
    "wild type": "#4C72B0",
    "edited": "#C44E52",
    "revertant": "#55A868",
    "other": "#8C8C8C",
}


def genotype_class(state: str, dose: int, max_dose: int) -> str:
    s = state.lower()
    if "rev" in s:
        return "revertant"
    if dose == 0:
        return "wild type"
    if dose > 0:
        return "edited"
    return "other"


# ---------------------------------------------------------------------------
# the decomposition
# ---------------------------------------------------------------------------


def run_pca(expr: pd.DataFrame, n_comp: int = 10, scale: bool = False):
    """SVD on complete-case proteins, libraries as observations."""
    arr = expr.to_numpy(float)
    complete = np.isfinite(arr).all(axis=1)
    X = arr[complete].T                      # libraries x proteins
    proteins = expr.index.to_numpy()[complete]

    centre = X.mean(axis=0)
    Xc = X - centre
    if scale:
        sd = Xc.std(axis=0, ddof=1)
        sd[sd == 0] = 1.0
        Xc = Xc / sd

    k = int(min(n_comp, min(Xc.shape) - 1))
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    scores = U[:, :k] * S[:k]
    loadings = Vt[:k].T                      # proteins x components
    var = (S ** 2) / (S ** 2).sum()
    return dict(
        scores=scores, loadings=loadings, proteins=proteins,
        var_explained=var[:k], n_complete=int(complete.sum()),
        n_total=int(arr.shape[0]), n_components=k,
        libraries=expr.columns.to_numpy(),
    )


def top_loadings(pca, n_top=15, n_influence=100):
    """Largest absolute loading per component, plus an overall influence ranking.

    Two tables, answering two different questions:

      per_pc      for each component, the `n_top` features pulling hardest on
                  that specific axis.
      overall     the `n_influence` features contributing most to the retained
                  variation as a whole, ranked. "Up to": a dataset with fewer
                  complete-case features than the cap yields correspondingly
                  fewer rows.

    The overall influence score is variance-weighted, sum_j loading_ij^2 * var_j.
    Squaring makes the direction of the loading irrelevant -- what is being
    measured is how much variation the feature accounts for, not which way it
    moves -- and weighting by the variance each component explains stops a large
    loading on a negligible component from outranking a moderate loading on the
    dominant one. Because the loadings of each component are unit-normalised,
    these scores sum across all features to the total retained variance, so
    `influence_pct` reads directly as a percentage of it.
    """
    L = pca["loadings"]
    var = pca["var_explained"]
    rows = []
    for j in range(pca["n_components"]):
        order = np.argsort(-np.abs(L[:, j]))[:n_top]
        for rank, i in enumerate(order, 1):
            rows.append(dict(component=f"PC{j + 1}", rank=rank,
                             protein_id=pca["proteins"][i],
                             loading=round(float(L[i, j]), 5),
                             abs_loading=round(float(abs(L[i, j])), 5),
                             pc_var_explained=round(float(var[j]), 5)))
    per_pc = pd.DataFrame(rows)

    # variance-weighted influence across PC1-10: how much of the retained
    # variation does this protein actually account for
    contrib = L ** 2 * var[None, :]
    infl = contrib.sum(axis=1)
    total = float(infl.sum())
    dominant = np.argmax(contrib, axis=1)
    overall = (pd.DataFrame(dict(
        protein_id=pca["proteins"],
        influence=np.round(infl, 6),
        influence_pct=np.round(100.0 * infl / total, 4) if total > 0
        else np.zeros_like(infl),
        dominant_component=[f"PC{j + 1}" for j in dominant],
        loading_on_dominant=np.round(L[np.arange(L.shape[0]), dominant], 5)))
        .sort_values("influence", ascending=False)
        .head(n_influence).reset_index(drop=True))
    overall.insert(0, "rank", np.arange(1, len(overall) + 1))
    overall["cumulative_influence_pct"] = np.round(
        overall["influence_pct"].cumsum(), 4)
    return per_pc, overall


# ---------------------------------------------------------------------------
# clouds
# ---------------------------------------------------------------------------


def _ellipse(ax, pts, colour, n_std=2.0, alpha=0.5, floor=None):
    """95% normal-theory ellipse.

    Regularised on purpose. Four libraries from one clone are often almost
    collinear in a given plane, and the raw covariance then draws a sliver that
    reads as a confident, precise cluster when it is the opposite. Adding a
    small isotropic ridge -- a few percent of the overall spread -- keeps the
    cloud honest about how little it knows in the thin direction.
    """
    if len(pts) < 3:
        return False
    cov = np.cov(pts.T)
    if not np.all(np.isfinite(cov)):
        return False
    if floor is not None and floor > 0:
        cov = cov + np.eye(2) * floor ** 2
    if np.linalg.matrix_rank(cov) < 2:
        return False
    vals, vecs = np.linalg.eigh(cov)
    if np.any(vals <= 0):
        return False
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w, h = 2 * n_std * np.sqrt(vals)
    ax.add_patch(Ellipse(pts.mean(axis=0), w, h, angle=angle,
                         facecolor=colour, alpha=alpha, edgecolor=colour,
                         linewidth=1.2, zorder=1))
    return True


def _hull(ax, pts, colour, pad=0.18, alpha=0.5):
    """Padded convex hull, for classes too small or too flat for an ellipse."""
    c = pts.mean(axis=0)
    if len(pts) == 1:
        r = 0.05 * max(np.ptp(ax.get_xlim()), 1.0)
        ax.add_patch(Ellipse(c, r, r, facecolor=colour, alpha=alpha,
                             edgecolor=colour, zorder=1))
        return
    try:
        from scipy.spatial import ConvexHull
        h = ConvexHull(pts)
        poly = pts[h.vertices]
    except Exception:
        poly = pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]
    poly = c + (poly - c) * (1.0 + pad)
    ax.add_patch(Polygon(poly, closed=True, facecolor=colour, alpha=alpha,
                         edgecolor=colour, linewidth=1.2, zorder=1))


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------


def score_panel(ax, pca, meta, i, j, n_arrows=6, n_arrow_labels=5,
                annotate_clones=True):
    S = pca["scores"]
    var = pca["var_explained"]
    x, y = S[:, i], S[:, j]
    floor = 0.05 * float(np.hypot(np.ptp(x), np.ptp(y)))

    for cls, colour in CLASS_COLOURS.items():
        m = (meta["class"] == cls).to_numpy()
        if not m.any():
            continue
        pts = np.column_stack([x[m], y[m]])
        if not _ellipse(ax, pts, colour, floor=floor):
            _hull(ax, pts, colour)
        ax.scatter(pts[:, 0], pts[:, 1], s=46, color=colour, edgecolor="white",
                   linewidth=0.8, zorder=3, label=f"{cls} (n={m.sum()})")

    if annotate_clones and meta["clone"].nunique() <= 8:
        pad = 0.045 * (np.ptp(y) or 1.0)
        for clone, sub in meta.groupby("clone", sort=False):
            k = sub.index.to_numpy()
            ax.annotate(f"{clone} ({sub['dose'].iloc[0]}n)",
                        (x[k].mean(), y[k].mean() + pad),
                        fontsize=6.8, ha="center", va="bottom", zorder=6,
                        color="#222222",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                  ec="none", alpha=0.80))

    # loading arrows for the proteins pulling hardest on this pair of axes
    L = pca["loadings"]
    strength = np.hypot(L[:, i], L[:, j])
    pick = np.argsort(-strength)[:n_arrows]
    half = 0.5 * min(np.ptp(x) or 1.0, np.ptp(y) or 1.0)
    span = 0.62 * half / (strength[pick].max() or 1.0)
    for rank, prot in enumerate(pick):
        dx, dy = L[prot, i] * span, L[prot, j] * span
        ax.annotate("", xy=(dx, dy), xytext=(0, 0), zorder=5,
                    arrowprops=dict(arrowstyle="-|>", color="#2F2F2F",
                                    linewidth=1.1, alpha=0.8))
        if rank < n_arrow_labels:
            ax.annotate(pca["proteins"][prot], (dx * 1.06, dy * 1.06),
                        fontsize=6.2, zorder=7, color="#2F2F2F",
                        ha="left" if dx >= 0 else "right",
                        va="bottom" if dy >= 0 else "top",
                        bbox=dict(boxstyle="round,pad=0.10", fc="white",
                                  ec="none", alpha=0.65))

    ax.axhline(0, color="#BBBBBB", linewidth=0.6, zorder=0)
    ax.axvline(0, color="#BBBBBB", linewidth=0.6, zorder=0)
    ax.set_xlabel(f"PC{i + 1}  ({100 * var[i]:.1f}%)")
    ax.set_ylabel(f"PC{j + 1}  ({100 * var[j]:.1f}%)")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def scree_panel(ax, pca):
    var = 100 * pca["var_explained"]
    k = len(var)
    ax.bar(np.arange(1, k + 1), var, color="#B0B0B0", width=0.7)
    ax.bar(np.arange(1, 4), var[:3], color="#4C72B0", width=0.7)
    for n in range(k):
        ax.text(n + 1, var[n] + 0.6, f"{var[n]:.0f}", ha="center", fontsize=6.5,
                color="#444444")
    ax.set_xticks(np.arange(1, k + 1))
    ax.set_xlabel("component")
    ax.set_ylabel("% variance")
    ax.set_title("scree (blue = plotted above)", fontsize=8.5, loc="left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def figure_for(design, expr, samples, pca, meta, outpath: Path):
    fig = plt.figure(figsize=(15.5, 5.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.62], wspace=0.30)

    pairs = [(0, 1), (0, 2), (1, 2)]
    for n, (i, j) in enumerate(pairs):
        if max(i, j) >= pca["n_components"]:
            continue
        ax = fig.add_subplot(gs[0, n])
        score_panel(ax, pca, meta, i, j)
        ax.set_title(f"PC{i + 1} vs PC{j + 1}", fontsize=10, loc="left")
        if n == 0:
            ax.legend(frameon=True, framealpha=0.85, edgecolor="none",
                      fontsize=7.5, loc="upper left")

    scree_panel(fig.add_subplot(gs[0, 3]), pca)

    kept, total = pca["n_complete"], pca["n_total"]
    fig.suptitle(
        f"{design.label}\n"
        f"{len(samples)} libraries, {design.n_clones} clones, "
        f"{kept} of {total} proteins complete in every run "
        f"({100 * kept / total:.0f}%) -- no imputation; clouds are 95% "
        f"ellipses at 50% opacity",
        fontsize=10, ha="left", x=0.008, y=0.99, va="top")
    fig.subplots_adjust(top=0.80)
    fig.savefig(outpath, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# interpretation
# ---------------------------------------------------------------------------


def interpret(pca, meta) -> str:
    """Does the leading axis track allele dosage, or something else?"""
    S = pca["scores"]
    var = pca["var_explained"]
    lines = ["PCA READING", "=" * 62,
             f"  proteins used            : {pca['n_complete']} of "
             f"{pca['n_total']} complete in every library",
             f"  variance in PC1-3        : "
             f"{100 * var[:3].sum():.1f}%  "
             f"(PC1 {100 * var[0]:.1f}, PC2 {100 * var[1]:.1f}, "
             f"PC3 {100 * var[2]:.1f})", ""]

    dose = meta["dose"].to_numpy(float)
    for j in range(min(3, pca["n_components"])):
        r = float(np.corrcoef(S[:, j], dose)[0, 1]) if np.std(dose) > 0 else np.nan
        by = []
        for col in ("plex", "lineage"):
            if col in meta and meta[col].nunique() > 1:
                g = [S[(meta[col] == v).to_numpy(), j] for v in meta[col].unique()]
                spread = float(np.ptp([x.mean() for x in g]) / (S[:, j].std() or 1))
                by.append(f"{col} spread {spread:.2f} SD")
        lines.append(f"  PC{j + 1}: correlation with allele dosage r = {r:+.2f}"
                     + (f"   [{'; '.join(by)}]" if by else ""))

    # where does the revertant sit on PC1, relative to wild type and edit?
    cls = meta["class"].to_numpy()
    def centre(c):
        m = cls == c
        return float(S[m, 0].mean()) if m.any() else np.nan
    wt, ed, rv = centre("wild type"), centre("edited"), centre("revertant")
    lines.append("")
    if np.all(np.isfinite([wt, ed, rv])):
        span = abs(ed - wt)
        frac = (abs(rv - wt) / span) if span > 0 else np.nan
        lines.append(f"  PC1 positions: wild type {wt:+.1f}, edited {ed:+.1f}, "
                     f"revertant {rv:+.1f}")
        if np.isfinite(frac):
            lines.append(f"  revertant sits {frac:.2f} of the way from wild type "
                         f"to edited on PC1")
            if frac < 0.35:
                lines.append("  -> the reversion returns the dominant axis toward "
                             "wild type, which is what the design predicts")
            elif frac > 0.8:
                lines.append("  -> WARNING: the revertant is as far from wild type "
                             "as the edit is. PC1 is more likely to be tracking "
                             "clonal drift than allele dosage.")
            else:
                lines.append("  -> partial return; PC1 is a mixture of dosage and "
                             "clone-specific variation")
    lines.append("")
    lines.append("  Note: with one clone per genotype, separation on any component")
    lines.append("  is expected and is NOT evidence of a variant effect. What is")
    lines.append("  informative is the ORDERING -- whether the revertant returns.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_meta(samples, design):
    max_dose = max(design.doses)
    state_of = {c.name: c.state for c in design.clones}
    meta = samples.copy().reset_index(drop=True)
    meta["state"] = meta["clone"].map(state_of)
    meta["class"] = [genotype_class(s, d, max_dose)
                     for s, d in zip(meta["state"], meta["dose"])]
    return meta


def run_one(prefix, outdir: Path, n_comp=10, n_top=15, scale=False,
             n_influence=100):
    from .datasets import load_dataset

    expr, samples, design, _ = load_dataset(prefix)
    expr = expr[samples["library"].to_numpy()]
    pca = run_pca(expr, n_comp=n_comp, scale=scale)
    meta = build_meta(samples, design)

    key = design.key
    png = outdir / f"{key}_pca.png"
    figure_for(design, expr, samples, pca, meta, png)

    per_pc, overall = top_loadings(pca, n_top=n_top, n_influence=n_influence)
    per_pc.to_csv(outdir / f"{key}_pca_loadings.tsv", sep="\t", index=False)
    overall.to_csv(outdir / f"{key}_pca_influence.tsv", sep="\t", index=False)

    sc = pd.DataFrame(pca["scores"],
                      columns=[f"PC{n + 1}" for n in range(pca["n_components"])])
    sc.insert(0, "library", pca["libraries"])
    sc = sc.merge(meta[["library", "clone", "state", "dose", "class", "plex"]],
                  on="library")
    sc.to_csv(outdir / f"{key}_pca_scores.tsv", sep="\t", index=False)

    story = interpret(pca, meta)
    (outdir / f"{key}_pca.txt").write_text(story + "\n")
    return design, pca, story, png


def main():
    from .run import find_datasets

    ap = argparse.ArgumentParser(description="PCA of the clone panel")
    ap.add_argument("--data", default="data")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--components", type=int, default=10,
                    help="components to retain for the loading table (plots "
                         "always show PC1-3 only)")
    ap.add_argument("--top", type=int, default=15,
                    help="features per component in the loading table")
    ap.add_argument("--top-influence", type=int, default=100,
                    help="features in the ranked overall influence table")
    ap.add_argument("--scale", action="store_true",
                    help="standardise each protein before decomposition")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for prefix in find_datasets(Path(args.data)):
        design, pca, story, png = run_one(prefix, outdir, args.components,
                                          args.top, args.scale,
                                          args.top_influence)
        print("=" * 70)
        print(design.label)
        print(story)
        print(f"  -> {png.name}, {design.key}_pca_loadings.tsv, "
              f"{design.key}_pca_influence.tsv, {design.key}_pca_scores.tsv")
        print()


if __name__ == "__main__":
    main()
