"""Extract the plotted data of the published figure files (.fig) into JSON.

A .fig figure file holds the figure's graphics tree; every
plotted series keeps its XData/YData (and LData/UData for error bars). This
script walks that tree and writes one JSON file per panel, so the figure
scripts can overlay the published curves and the tests can check the replica
theory against them.

Run:  python -m scripts.extract_published_figures --fig-dir <dir with pre_fig_*.fig> \
          --out reference/published
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import scipy.io as sio

SERIES_TYPES = {"graph2d.lineseries": "line", "specgraph.errorbarseries": "errorbar"}


def _field(node: np.ndarray, name: str):
    """Field of a struct record, unwrapping 1x1 object cells; None if absent."""
    if name not in node.dtype.names:
        return None
    x = node[name]
    while isinstance(x, np.ndarray) and x.dtype == object and x.size == 1:
        x = x.flat[0]
    return x


def _props(node: np.ndarray) -> dict:
    p = _field(node, "properties")
    if not (isinstance(p, np.ndarray) and p.dtype.names):
        return {}
    out = {}
    for k in p.dtype.names:
        v = p[k].flat[0] if p[k].dtype == object else p[k]
        while isinstance(v, np.ndarray) and v.dtype == object and v.size == 1:
            v = v.flat[0]
        out[k] = v
    return out


def _text(v) -> str:
    a = np.asarray(v).squeeze()
    if a.dtype.kind in "US":
        return str(a)
    return ""


def _vec(v) -> list[float | None]:
    a = np.asarray(v, dtype=float).ravel()
    return [None if np.isnan(t) else float(t) for t in a]


def _walk(node: np.ndarray, series: list[dict], texts: list[str]) -> None:
    kind = _text(_field(node, "type"))
    p = _props(node)
    if kind in SERIES_TYPES:
        s = {"kind": SERIES_TYPES[kind],
             "label": _text(p["DisplayName"]) if "DisplayName" in p else "",
             "color": _vec(p["Color"]) if "Color" in p else None,
             "linestyle": _text(p["LineStyle"]) if "LineStyle" in p else "",
             "marker": _text(p["Marker"]) if "Marker" in p else "",
             "x": _vec(p["XData"]), "y": _vec(p["YData"])}
        if kind == "specgraph.errorbarseries":
            s["err_lo"] = _vec(p["LData"])
            s["err_hi"] = _vec(p["UData"])
        series.append(s)
        return                                  # children of an errorbar series are its drawn lines
    if kind == "text" and "String" in p:
        t = _text(p["String"])
        if t:
            texts.append(t)
    ch = _field(node, "children")
    if isinstance(ch, np.ndarray) and ch.dtype.names:
        for c in ch.flat:
            _walk(c, series, texts)


def extract(fig_path: Path) -> dict:
    d = sio.loadmat(str(fig_path), struct_as_record=True)
    series: list[dict] = []
    texts: list[str] = []
    for n in d["hgS_070000"].flat:
        _walk(n, series, texts)
    if not series:
        raise ValueError(f"no plotted series found in {fig_path}")
    return {"source": fig_path.name, "texts": texts, "series": series}


def panel_name(fig_path: Path) -> str:
    """pre_fig_2_a.fig -> fig02a; other names are kept (lower-cased)."""
    m = re.fullmatch(r"pre_fig_(\d+)_?([a-z]?)", fig_path.stem)
    if m:
        return f"fig{int(m.group(1)):02d}{m.group(2)}"
    return fig_path.stem.lower()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fig-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    figs = sorted(args.fig_dir.glob("pre_fig_*.fig"))
    if not figs:
        raise FileNotFoundError(f"no pre_fig_*.fig files in {args.fig_dir}")
    args.out.mkdir(parents=True, exist_ok=True)
    written: dict[str, dict] = {}
    for f in figs:
        name = panel_name(f)
        data = extract(f)
        if name in written:
            # pre_fig_7_b.fig and pre_fig_7b.fig are two copies of one panel
            if written[name]["series"] != data["series"]:
                raise ValueError(f"{f.name} and {written[name]['source']} both map to {name} "
                                 "but differ")
            print(f"{f.name:32s} == {written[name]['source']} (skipped)")
            continue
        written[name] = data
        (args.out / f"{name}.json").write_text(json.dumps(data, indent=1))
        print(f"{f.name:32s} -> {name}.json  ({len(data['series'])} series)")


if __name__ == "__main__":
    main()
