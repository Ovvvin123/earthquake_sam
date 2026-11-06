"""Review-only USGS land-fault buffers within the active V5 footprint.

Does not create V6, export training masks, edit HDF5, or import training code.
Run with the existing earthquake-sam Python environment.
"""
from pathlib import Path
import argparse
import hashlib
import json

import geopandas as gpd
import h5py
import numpy as np
from shapely import box, contains_xy, union_all, length, buffer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[1]


def land_geometry(mask, x0, y0, res):
    """Exact union of positive-elevation pixel cells, with merged row runs."""
    active = {}
    rectangles = []
    for row in range(mask.shape[0] + 1):
        if row < mask.shape[0]:
            edges = np.flatnonzero(np.diff(np.r_[False, mask[row], False]))
            runs = set(zip(edges[::2].tolist(), edges[1::2].tolist()))
        else:
            runs = set()
        for run in active.keys() - runs:
            start = active[run]
            left, right = run
            rectangles.append(box(x0 + (left - .5) * res,
                                  y0 + (start - .5) * res,
                                  x0 + (right - .5) * res,
                                  y0 + (row - .5) * res))
        active = {run: active.get(run, row) for run in runs}
    return union_all(rectangles)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=CODE / "diagnostics/v6_land_envelope_preview_v1")
    parser.add_argument("--buffer-km", type=int, nargs=3, default=[3, 5, 8],
                        help="Three positive, strictly increasing buffer distances in km")
    args = parser.parse_args()
    distances = args.buffer_km
    if any(km <= 0 for km in distances) or distances != sorted(set(distances)):
        parser.error("Buffer distances must be positive and strictly increasing")
    distance_label = " / ".join(map(str, distances))
    if args.output.exists():
        parser.error(f"Refusing to overwrite existing preview: {args.output}")
    data = CODE / "earth_dataset_v5"
    source = ROOT / "Qfaults_GIS/SHP/Qfaults_US_Database.shp"
    source_stat = (data / "faults.h5").stat()
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    split_keys = {s: (data / f"key_{s}.lst").read_text().split()
                  for s in ("train", "val", "test")}
    keys = set().union(*split_keys.values())
    assert len(keys) == sum(map(len, split_keys.values())), "Duplicate split keys"
    res = manifest["provenance"]["res_deg"]
    tile_size = manifest["provenance"]["tile"]
    first = manifest["tiles"][0]
    ii, jj = map(int, first["key"].split("_")[1:])
    x0 = first["lon_c"] - ((tile_size - 1) / 2 + jj) * res
    y0 = first["lat_c"] - ((tile_size - 1) / 2 + ii) * res
    height = max(int(k.split("_")[1]) + tile_size for k in keys)
    width = max(int(k.split("_")[2]) + tile_size for k in keys)
    z = np.full((height, width), np.nan, np.float32)
    used = np.zeros(z.shape, bool)
    print(f"Reading {len(keys)} active V5 tiles...", flush=True)
    with h5py.File(data / "faults.h5", "r") as f:
        for key in sorted(keys):
            i, j = map(int, key.split("_")[1:])
            region = np.s_[i:i + tile_size, j:j + tile_size]
            image = f[key]["image"][:]
            assert image.shape == (tile_size, tile_size)
            overlap = used[region]
            assert np.array_equal(z[region][overlap], image[overlap], equal_nan=True), key
            z[region] = image
            used[region] = True
    valid = used & np.isfinite(z) & (z > -2147483647)
    land = valid & (z > 0)
    assert land.any()
    xs = x0 + np.arange(width) * res
    ys = y0 + np.arange(height) * res
    extent = [xs[0] - res / 2, xs[-1] + res / 2,
              ys[0] - res / 2, ys[-1] + res / 2]
    print("Building exact positive-elevation land footprint...", flush=True)
    land_geom = land_geometry(land, x0, y0, res)
    assert land_geom.is_valid
    assert np.isclose(land_geom.area, land.sum() * res * res, rtol=1e-8)
    print("Reading and clipping existing USGS fault lines...", flush=True)
    faults = gpd.read_file(source).to_crs(4326)
    # Spatial index excludes the nationwide catalog outside this footprint.
    candidates = faults.iloc[faults.sindex.query(box(*[extent[0], extent[2],
                                                       extent[1], extent[3]]),
                                                predicate="intersects")].copy()
    candidates.geometry = candidates.geometry.intersection(land_geom)
    clipped = candidates.loc[~candidates.geometry.is_empty &
                             (length(candidates.geometry.to_numpy()) > 0)].copy()
    assert len(clipped), "No USGS land faults in active footprint"
    metric = clipped.to_crs(3310)
    line_union = union_all(metric.geometry.to_numpy())
    print(f"Retained {len(clipped)} source features; buffering in EPSG:3310...", flush=True)
    masks = {}
    statistics = {}
    # Spherical cell areas are only descriptive; mask distances use projected meters.
    radius_km = 6371.0088
    row_area = radius_km ** 2 * np.deg2rad(res) * (
        np.sin(np.deg2rad(ys + res / 2)) - np.sin(np.deg2rad(ys - res / 2)))
    for km in distances:
        # Buffer individual source segments in bounded batches. Buffering the
        # entire dense MultiLineString at once creates excessive GEOS intermediates.
        pieces = []
        geometries = metric.geometry.to_numpy()
        for start in range(0, len(geometries), 256):
            pieces.append(union_all(buffer(geometries[start:start + 256], km * 1000,
                                           quad_segs=16)))
            if start % 12800 == 0:
                print(f"{km} km: buffered {min(start + 256, len(geometries))}/{len(geometries)} segments", flush=True)
        envelope = union_all(pieces)
        geometry = gpd.GeoSeries([envelope], crs=3310).to_crs(4326).iloc[0]
        mask = np.zeros(z.shape, bool)
        for start in range(0, height, 128):
            xx, yy = np.meshgrid(xs, ys[start:start + 128])
            mask[start:start + 128] = contains_xy(geometry, xx, yy)
        mask &= land
        assert not (mask & (~valid | (z <= 0))).any()
        masks[km] = mask
        statistics[str(km)] = {
            "candidate_pixels": int(mask.sum()),
            "fraction_of_positive_elevation_land": float(mask.sum() / land.sum()),
            "approximate_area_km2": float(np.sum(mask.sum(axis=1) * row_area)),
            "ocean_or_zero_elevation_pixels": int((mask & (z <= 0)).sum()),
            "outside_active_footprint_pixels": int((mask & ~used).sum()),
        }
        print(f"{km} km: {statistics[str(km)]}", flush=True)
    for smaller, larger in zip(distances, distances[1:]):
        assert not (masks[smaller] & ~masks[larger]).any()

    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    font_path = next((p for p in (Path("C:/Windows/Fonts/msyh.ttc"),
                                 Path("/mnt/c/Windows/Fonts/msyh.ttc")) if p.exists()), None)
    font = FontProperties(fname=str(font_path)) if font_path else FontProperties()
    plt.rcParams["axes.unicode_minus"] = False
    base = np.full(z.shape, np.nan, dtype=np.float32)
    base[valid & (z < 0)] = 0
    base[land] = 1
    base[valid & (z == 0)] = 2
    cmap = ListedColormap(["#b8d6e6", "#e9dcc5", "#c5c5c5"])
    overlays = {}
    for km, mask in masks.items():
        rgba = np.zeros((*z.shape, 4), dtype=np.uint8)
        rgba[mask] = [56, 156, 92, 170]
        overlays[km] = rgba

    def draw(ax, km, xlim, ylim):
        ax.imshow(base, origin="lower", extent=extent, cmap=cmap, vmin=0, vmax=2,
                  interpolation="nearest", rasterized=True)
        ax.imshow(overlays[km], origin="lower", extent=extent, interpolation="nearest")
        # Crop source lines for readable detail views and lower render cost.
        local = clipped.clip(box(xlim[0], ylim[0], xlim[1], ylim[1]))
        if len(local):
            local.plot(ax=ax, color="#b51f44", linewidth=.38, zorder=4)
        ax.set(xlim=xlim, ylim=ylim, xlabel="Longitude", ylabel="Latitude")
        ax.set_aspect(1 / np.cos(np.deg2rad(np.mean(ylim))))
        ax.tick_params(labelsize=9)
        ax.grid(alpha=.13)

    handles = [Patch(facecolor="#e9dcc5", label="高程 > 0 的陆地"),
               Patch(facecolor="#b8d6e6", label="高程 < 0（含海洋；不进入 mask）"),
               Patch(facecolor="#73b584", label="候选监督 mask"),
               Line2D([0], [0], color="#b51f44", label="保留的 USGS 陆地断层"),
               Patch(facecolor="white", edgecolor="#999999", label="V5 未使用范围 / 无效数据")]
    fig, axs = plt.subplots(1, 3, figsize=(17, 10))
    for ax, km in zip(axs, masks):
        draw(ax, km, extent[:2], extent[2:])
        pct = statistics[str(km)]["fraction_of_positive_elevation_land"] * 100
        ax.set_title(f"外扩 {km} km · 陆地覆盖 {pct:.1f}%", fontproperties=font, fontsize=14)
    fig.suptitle(f"V6 陆地监督范围预览 · {distance_label} 公里对比", fontproperties=font, fontsize=19, y=.98)
    fig.legend(handles=handles, loc="lower center", ncol=3, prop=font, bbox_to_anchor=(.5, .04))
    fig.text(.5, .018, "红线先按 V5 有效陆地截取，再外扩；绿色仅限高程 > 0。未做闭运算或凸包填充，尚未生成训练 mask。",
             ha="center", fontproperties=font, fontsize=10)
    fig.subplots_adjust(left=.05, right=.99, top=.92, bottom=.14, wspace=.22)
    fig.savefig(out / "land_mask_comparison.png", dpi=180)
    plt.close(fig)

    views = [("北段：门多西诺及邻近陆地", [-124.7, -122.1], [38.7, 40.7]),
             ("中段：旧金山湾及周边", [-123.0, -120.5], [36.5, 38.5]),
             ("南段：洛杉矶至圣迭戈", [-119.5, -116.7], [32.3, 34.5])]
    for name, xlim, ylim in views:
        fig, axs = plt.subplots(1, 3, figsize=(17, 6.7))
        for ax, km in zip(axs, masks):
            draw(ax, km, xlim, ylim)
            ax.set_title(f"外扩 {km} km", fontproperties=font, fontsize=15)
        fig.suptitle(name + " · 同范围对比", fontproperties=font, fontsize=18)
        fig.legend(handles=handles, loc="lower center", ncol=3, prop=font, bbox_to_anchor=(.5, .025))
        fig.subplots_adjust(left=.05, right=.99, top=.88, bottom=.20, wspace=.23)
        filename = {"北": "north", "中": "central", "南": "south"}[name[0]]
        fig.savefig(out / f"land_mask_{filename}.png", dpi=180)
        plt.close(fig)

    hash_inputs = [data / "manifest.json", *[data / f"key_{s}.lst" for s in split_keys],
                   *[source.with_suffix(ext) for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg")]]
    report = {
        "status": "PREVIEW_ONLY_AWAITING_USER_APPROVAL",
        "source_dataset": "earth_dataset_v5",
        "source_fault_catalog": str(source.relative_to(ROOT)),
        "split_counts": {s: len(v) for s, v in split_keys.items()},
        "active_tile_count": len(keys),
        "valid_unique_pixels": int(valid.sum()),
        "positive_elevation_land_pixels": int(land.sum()),
        "zero_elevation_pixels_excluded": int((valid & (z == 0)).sum()),
        "retained_source_feature_count": len(clipped),
        "unique_clipped_fault_length_km": float(line_union.length / 1000),
        "recipe": {
            "buffer_km": distances, "metric_crs": "EPSG:3310",
            "land_definition": "finite, non-nodata V5 image elevation > 0 meters",
            "order": "clip existing USGS lines to union of positive-elevation pixel cells in active V5 tiles; buffer in meters; intersect raster pixel centers with valid land",
            "closing_km": 0, "convex_hull": False,
            "catalog_attribute_filter": None,
            "coastline_vector_or_CA_state_boundary_filter": False,
            "pixel_size_degrees": res,
        },
        "variants": statistics,
        "checks": {"overlapping_tile_elevations_identical": True,
                   "nested_masks_in_increasing_buffer_order": True,
                   "all_mask_pixels_positive_elevation": True},
        "notes": ["Counts deduplicate overlapping tiles; fraction is relative to positive-elevation land in active footprint.",
                  "Area is approximate spherical pixel-cell area, not cadastral area.",
                  "Positive elevation is a land proxy, not a vector coastline; below-sea-level land and zero-elevation cells are excluded.",
                  "Buffer is a candidate supervision region, not evidence that all faults within it have been cataloged.",
                  "No source confidence/class filters or external catalogs are introduced.",
                  "Ocean remains present as input context, excluded only from these candidate supervised masks."],
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in hash_inputs},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "writes_hdf5_or_training_mask": False,
        "creates_earth_dataset_V6": False, "starts_training": False,
    }
    after = (data / "faults.h5").stat()
    assert (source_stat.st_size, source_stat.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    report["checks"]["source_hdf5_size_and_mtime_unchanged"] = True
    (out / "preview_recipe.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = "".join(f"<tr><td>{km} km</td><td>{s['candidate_pixels']:,}</td>"
                   f"<td>{s['fraction_of_positive_elevation_land']:.1%}</td>"
                   f"<td>{s['approximate_area_km2']:,.0f}</td><td>0</td></tr>"
                   for km, s in statistics.items())
    (out / "index.html").write_text("""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>V6 陆地 mask 预览</title><style>body{font:17px/1.7 sans-serif;max-width:1600px;margin:30px auto;padding:20px;color:#20302b}img{width:100%;border:1px solid #ddd}table{border-collapse:collapse}td,th{padding:9px 20px;border:1px solid #ccc}a{color:#146b47}</style>
<h1>V6 陆地监督 mask：BUFFER_DISTANCES 公里预览</h1>
<p><b>待确认。尚未制作 earth_dataset_V6，未修改训练代码，未启动训练。</b></p>
<p>使用原 V5 的 216 个 train/val/test 图块的联合范围；仅采用现有 USGS Qfaults_US_Database 目录。先将断层截取到有效高程 &gt; 0 的陆地区域，再在 EPSG:3310 中外扩对应公里数，最后将 mask 限定到同一有效陆地范围。不做额外闭运算、凸包或区域填充。</p>
<p>绿色表示候选监督范围，不是断层标签；红色表示已保留的陆地断层。海洋不进入监督 mask，但没有从输入数据中删除；之后仍可按方案参与伪标签和一致性训练。白色来自原 V5 图块范围的空缺或无效数据。</p>
<table><tr><th>外扩距离</th><th>去重像元数</th><th>占有效正高程陆地</th><th>近似面积 km²</th><th>高程 ≤ 0 的像元</th></tr>""".replace("BUFFER_DISTANCES", distance_label) + rows + """</table>
<p>高程 &gt; 0 是这次采用的陆地近似，低于海平面的陆地和零高程像元会被排除。缓冲区并不证明目录在该范围内完整；扩大 mask 也会扩大未来可能被当作背景的区域。</p>
<h2>全范围</h2><img src="land_mask_comparison.png">
<h2>北段</h2><img src="land_mask_north.png">
<h2>中段</h2><img src="land_mask_central.png">
<h2>南段</h2><img src="land_mask_south.png">
<p><a href="preview_recipe.json">参数、来源哈希、核验结果与完整统计</a></p></html>""", encoding="utf-8")
    print(f"Preview complete: {out}", flush=True)


if __name__ == "__main__":
    main()
