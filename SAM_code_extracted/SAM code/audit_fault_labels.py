"""Trace original catalog records to saved labels and retained prediction branches."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import h5py
import geopandas as gpd
from scipy import ndimage as ndi
from count_fault_branches import thin, trace_branches

def disk_offsets(radius, strict=False):
    n=int(np.ceil(radius))
    yy,xx=np.mgrid[-n:n+1,-n:n+1]
    mask=yy*yy+xx*xx < radius*radius if strict else yy*yy+xx*xx <= radius*radius
    return np.column_stack([yy[mask],xx[mask]]).astype(int)

def expand(points, offsets, shape):
    points=np.asarray(points,dtype=int).reshape(-1,2)
    if not len(points): return np.empty((0,2),int)
    a=(points[:,None,:]+offsets[None,:,:]).reshape(-1,2)
    a=a[(a[:,0]>=0)&(a[:,0]<shape[0])&(a[:,1]>=0)&(a[:,1]<shape[1])]
    return np.unique(a,axis=0)

def native(value):
    if value is None or (isinstance(value,float) and not np.isfinite(value)): return None
    if isinstance(value,np.generic): return value.item()
    return str(value)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run",type=Path,default=Path("train_v5_online"))
    ap.add_argument("--catalog",type=Path,default=Path("../../Qfaults_GIS/SHP/ca_offshore.shp"))
    ap.add_argument("--data",type=Path,default=Path("earth_dataset_v3"))
    args=ap.parse_args()
    out=args.run/"segment_counts"/"label_audit"
    if out.exists() and any(out.iterdir()): raise ValueError("Output exists")
    out.mkdir(parents=True,exist_ok=True)
    export=json.loads((args.run/"predictions_seen_unseen/manifest.json").read_text())
    branches=json.loads((args.run/"segment_counts/branches/branch_counts.json").read_text())
    res=json.loads((args.data/"manifest.json").read_text())["provenance"]["res_deg"]
    g=gpd.read_file(args.catalog).to_crs(4326).reset_index(drop=True)
    points,owners=[],[]
    part_count=0
    for fid,geom in enumerate(g.geometry):
        for line in (geom.geoms if geom.geom_type.startswith("Multi") else [geom]):
            part_count+=1
            xy=np.asarray(line.coords)
            for a,b in zip(xy[:-1],xy[1:]):
                n=max(2,int(np.hypot(*(b-a))/(res*.5))+1)
                points.append(np.linspace(a,b,n)); owners.append(np.full(n,fid,int))
    points,owners=np.concatenate(points),np.concatenate(owners)
    band_offsets=disk_offsets(export["d0"]/res,strict=True)
    tolerance=branches["tolerance_pixels"]
    near_offsets=disk_offsets(tolerance)
    report=dict(model=export["model"],epoch=export["epoch"],d0=export["d0"],threshold=export["threshold"],
        catalog=str(args.catalog.resolve()),catalog_records=len(g),catalog_line_parts=part_count,
        catalog_name_values=int(g.FAULT_NAME.nunique()),
        catalog_non_unspecified_name_values=int(g.loc[g.FAULT_NAME.str.lower()!="unspecified","FAULT_NAME"].nunique()),
        catalog_unspecified_name_records=int(g.FAULT_NAME.str.lower().eq("unspecified").sum()),
        counting_unit="Shapefile row/feature; repeated names and multipart records are not independent geological faults.",
        label_record_match="ANY pixel in that record's original d0 label band is <=3 grid pixels from a retained predicted centerline; many-to-many.",
        centerline_coverage="Fraction of distinct rasterized source centerline pixels within original d0-band-plus-3-pixel tolerance of retained predicted centerlines; grid-point fraction, not physical length.",
        sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.catalog.parent.glob(args.catalog.stem+".*")},
        results={})
    with h5py.File(args.data/"faults.h5","r") as h5:
        for split in ["train_seen","test_unseen","all_exported"]:
            with np.load(args.run/f"segment_counts/{split}_mosaic.npz") as z:
                truth=z["truth"]; valid=z["valid"]; origin=np.array([int(z["row_origin"]),int(z["col_origin"])])
            shape=truth.shape; width=shape[1]
            seeds={}; bands={}
            h5_union=np.zeros(shape,bool); reconstructed_union=np.zeros(shape,bool)
            rows=[r for r in export["images"] if split=="all_exported" or r["split"]==split]
            source_xor=0; band_xor=0
            for row in rows:
                key=row["key"]; group=h5[key]; hh,ww=group["k0"].shape
                y,x=map(int,key.split("_")[1:])
                lat0=float(group.attrs["lat_c"])-(hh-1)*res/2
                lon0=float(group.attrs["lon_c"])-(ww-1)*res/2
                selected=(points[:,0]>=lon0)&(points[:,0]<=lon0+(ww-1)*res)&(points[:,1]>=lat0)&(points[:,1]<=lat0+(hh-1)*res)
                xy=points[selected]; ids=owners[selected]
                local=np.column_stack([np.rint((xy[:,1]-lat0)/res),np.rint((xy[:,0]-lon0)/res)]).astype(int)
                raw=np.zeros((hh,ww),bool)
                if len(local): raw[local[:,0],local[:,1]]=True
                distances=group["k0"][:]; expected=distances==0
                source_xor+=int(np.count_nonzero(raw!=expected))
                assert np.array_equal(raw,expected),key
                region=np.s_[y-origin[0]:y-origin[0]+hh,x-origin[1]:x-origin[1]+ww]
                with np.load(args.run/"predictions_seen_unseen"/row["split"]/(key+".npz")) as z:
                    tile_valid=z["valid"]; tile_truth=z["truth"]
                union_local=np.zeros((hh,ww),bool)
                offset=np.array([y,x])-origin
                for fid in np.unique(ids):
                    pp=np.unique(local[ids==fid],axis=0)
                    core=pp[tile_valid[pp[:,0],pp[:,1]]]
                    bp=expand(pp,band_offsets,(hh,ww))
                    bp=bp[tile_valid[bp[:,0],bp[:,1]]]
                    if len(core):
                        cp=core+offset
                        seeds.setdefault(int(fid),[]).append(cp[:,0]*width+cp[:,1])
                    if len(bp):
                        union_local[bp[:,0],bp[:,1]]=True
                        bp=bp+offset
                        bands.setdefault(int(fid),[]).append(bp[:,0]*width+bp[:,1])
                band_xor+=int(np.count_nonzero(union_local!=tile_truth))
                assert np.array_equal(union_local,tile_truth),("label band mismatch",key)
                h5_union[region]|=expected&tile_valid
                reconstructed_union[region]|=union_local
            assert np.array_equal(reconstructed_union,truth)
            seeds={fid:np.unique(np.concatenate(a)) for fid,a in seeds.items()}
            bands={fid:np.unique(np.concatenate(a)) for fid,a in bands.items()}
            # Include edge cases where only a record's label band reaches valid cells.
            active_ids=sorted(bands)
            metadata=json.loads((args.run/f"segment_counts/branches/{split}_segments.json").read_text())
            paths=json.loads((args.run/f"segment_counts/branches/{split}_paths.json").read_text())
            pathmap={p["id"]:np.array(p["global_row_col"])-origin for p in paths}
            retained=[r for r in metadata if r["retained"]]
            label_to_pred={fid:[] for fid in active_ids}
            pred_to_label={}
            all_near=np.zeros(shape,bool)
            for pred in retained:
                path=pathmap[pred["id"]]
                near=expand(path,near_offsets,shape)
                reach=np.zeros(shape,bool); reach[near[:,0],near[:,1]]=True
                all_near|=reach
                rr=reach.ravel()
                matches=[fid for fid in active_ids if rr[bands[fid]].any()]
                pred_to_label[pred["id"]]=matches
                assert bool(matches)==pred["matched"],("prediction match inconsistent",split,pred["id"])
                for fid in matches: label_to_pred[fid].append(pred["id"])
            # Reverse tolerance, to quantify how much of each actual source trace is approached.
            size=int(np.ceil(export["d0"]/res))
            structure=np.zeros((size*2+1,size*2+1),bool)
            structure[band_offsets[:,0]+size,band_offsets[:,1]+size]=True
            core_near=ndi.binary_dilation(all_near,structure=structure)
            detail=[]
            for fid in active_ids:
                sr=g.iloc[fid]; core=seeds.get(fid,np.empty(0,int))
                coverage=float(core_near.ravel()[core].mean()) if len(core) else None
                detail.append(dict(catalog_row=fid+1,fault_name=native(sr.FAULT_NAME),fault_zone=native(sr.FAULT_ZONE),
                    fault_id=native(sr.FAULT_ID),line_type=native(sr.LINE_TYPE),source=native(sr.FLT_SOURCE),
                    original_geometry_type=sr.geometry.geom_type,centerline_grid_points=len(core),
                    label_band_pixels=len(bands[fid]),matched=bool(label_to_pred[fid]),
                    matched_prediction_ids=label_to_pred[fid],centerline_coverage_fraction=coverage))
            matched=sum(r["matched"] for r in detail)
            total_core=int(h5_union.sum()); covered_core=int((h5_union&core_near).sum())
            # A separately labeled raster-topology count, using the same algorithm as predictions.
            label_components,nlabel=ndi.label(truth,np.ones((3,3)))
            label_sizes=np.bincount(label_components.ravel())
            take=label_sizes>=branches["minimum_component_pixels"];take[0]=False
            label_skeleton=thin(take[label_components])
            label_paths,_=trace_branches(label_skeleton)
            label_paths_kept=[p for p in label_paths if len(np.unique(p,axis=0))>=branches["minimum_centerline_points"]]
            result=dict(tile_count=len(rows),catalog_records_in_labels=len(active_ids),
                records_with_actual_centerline=len(seeds),records_matched=matched,records_not_matched=len(detail)-matched,
                records_centerline_at_least_half_covered=sum(r["centerline_coverage_fraction"] is not None and r["centerline_coverage_fraction"]>=.5 for r in detail),
                records_centerline_at_least_80_percent_covered=sum(r["centerline_coverage_fraction"] is not None and r["centerline_coverage_fraction"]>=.8 for r in detail),
                unique_source_centerline_pixels=total_core,source_centerline_pixels_covered=covered_core,
                source_centerline_coverage_fraction=covered_core/total_core if total_core else None,
                label_mask_connected_regions=nlabel,label_mask_centerline_branches=len(label_paths_kept),
                label_mask_raw_centerline_branches=len(label_paths),
                prediction_segments=len(retained),prediction_matched=sum(r["matched"] for r in retained),
                prediction_unmatched=sum(not r["matched"] for r in retained),
                prediction_at_least_half_near_label=sum(r["fraction_near_label"]>=.5 for r in retained),
                prediction_at_least_80_percent_near_label=sum(r["fraction_near_label"]>=.8 for r in retained),
                verified_source_raster_xor_pixels=source_xor,verified_label_band_xor_pixels=band_xor)
            report["results"][split]=result
            (out/f"{split}_catalog_records.json").write_text(json.dumps(detail,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
            (out/f"{split}_prediction_to_catalog.json").write_text(json.dumps(
                [dict(prediction_id=int(k),catalog_rows=[fid+1 for fid in v]) for k,v in pred_to_label.items()],indent=2),encoding="utf-8")
            print(split,json.dumps(result),flush=True)
    all_details=json.loads((out/"all_exported_catalog_records.json").read_text())
    groups={}
    for r in all_details:
        a=groups.setdefault(r["fault_name"],dict(label_records=0,touched=0,half_covered=0))
        a["label_records"]+=1;a["touched"]+=int(r["matched"])
        a["half_covered"]+=int(r["centerline_coverage_fraction"] is not None and r["centerline_coverage_fraction"]>=.5)
    report["groups_by_catalog_name"]=groups
    (out/"label_audit.json").write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    names={"train_seen":"训练区域","test_unseen":"未见测试区域","all_exported":"全部已导出区域"}
    lines=["# d0=0.02、无 UniMatch V2：原标签与预测详细统计","",
        "模型为 train_v5_online/best_student，第186轮；概率阈值0.5。先按地理坐标拼接去重，再提取预测中心线，在交汇处分段。","",
        "## 原始标签到底有多少？","",
        f"- 原始 ca_offshore.shp：{len(g):,}条目录记录，展开 MultiLineString 后为{part_count:,}条几何线部分。",
        f"- FAULT_NAME 有{report['catalog_name_values']}种取值，其中 unspecified 涉及{report['catalog_unspecified_name_records']}条记录；其余{report['catalog_non_unspecified_name_values']}种名称也包含断层带，不能理解为独立断层数量。",
        "- 目录记录不是独立地质断层。同名断层可有多条记录，1条记录也可含多个几何线部分。",
        "- 以下仅统计确实生成了本次区域内原标签的目录记录。源文件记录号从1开始，仅对本次文件指纹有效。","",
        "| 范围 | 产生标签的目录记录 | 被预测触及 | 完全未触及 | 至少半条中心线被接近 |",
        "|---|---:|---:|---:|---:|"]
    for name,r in report["results"].items():
        lines.append(f"| {names[name]} | {r['catalog_records_in_labels']} | {r['records_matched']} | {r['records_not_matched']} | {r['records_centerline_at_least_half_covered']} |")
    lines+=["","“触及”仍为宽松判定：该记录原始d0标签带与保留预测中心线的3像素邻域有任意交集。不能理解为整条恢复。中心线覆盖只在本次有数据的范围内计算，统计栅格点比例，不是公里长度比例。","",
        "## 预测结果与匹配程度","","| 范围 | 预测段数 | 任意部分匹配 | 未匹配 | 至少半段匹配 | 至少80%匹配 |",
        "|---|---:|---:|---:|---:|---:|"]
    for name,r in report["results"].items():
        lines.append(f"| {names[name]} | {r['prediction_segments']} | {r['prediction_matched']} | {r['prediction_unmatched']} | {r['prediction_at_least_half_near_label']} | {r['prediction_at_least_80_percent_near_label']} |")
    lines+=["","沿用原口径：去掉小于16像素的独立区域；保留至少5个中心线点的预测段；距原标签带≤3像素算匹配。原标签带本身使用d0=0.02。","",
        "## 若将标签也按同一栅格分段算法处理","","| 范围 | 原标签带连通区域 | 标签带中心线段（同样过滤） | 预测中心线段 |",
        "|---|---:|---:|---:|"]
    for name,r in report["results"].items():
        lines.append(f"| {names[name]} | {r['label_mask_connected_regions']} | {r['label_mask_centerline_branches']} | {r['prediction_segments']} |")
    lines+=["","这组标签段数来自d0=0.02的宽标签带：相邻原始断层可能合并，交汇处又会分段。它与目录记录数不同，不应混为同一种“条数”。","",
        "## 原始中心线覆盖（补充）","","| 范围 | 原始中心线栅格点 | 被接近的栅格点 | 覆盖比例 |","|---|---:|---:|---:|"]
    for name,r in report["results"].items():
        lines.append(f"| {names[name]} | {r['unique_source_centerline_pixels']} | {r['source_centerline_pixels_covered']} | {r['source_centerline_coverage_fraction']:.1%} |")
    lines+=["","此处用原标签中心线点到保留预测中心线的d0标签带+3像素容差计算；它只是补充覆盖指标，不能代替像素AP或独立断层召回率。","",
        "## 按目录名称分组（全部已导出区域）","","| 名称 | 标签记录 | 被触及 | 至少半条被接近 |","|---|---:|---:|---:|"]
    for name,r in sorted(groups.items(),key=lambda kv:-kv[1]["label_records"]):
        lines.append(f"| {name} | {r['label_records']} | {r['touched']} | {r['half_covered']} |")
    lines+=["","## 验证和限制","",
        "- 已对全部导出图块验证：按原始目录重建中心线与HDF5的k0=0逐像素一致；重建d0=0.02标签带与导出truth逐像素一致。",
        "- 对每个保留预测段，逐条检查它匹配的原目录记录；重新得到的匹配状态与此前35/58统计完全一致。",
        "- 预测段与目录记录为多对多关系：一段预测可经过多条标签，一条标签也可被多段预测触及。",
        "- 全部范围按地理并集重新计数。跨训练/测试范围的标签只算一次，因此目录总数不等于两组相加；预测交汇关系也会变化。",
        "- 未匹配预测并未经过地质核查，不能直接叫假断层；未触及标签也可能在输入中缺乏可见证据。",
        "- 这里是训练区域与独立测试区域的汇总，不代表整个目录全域性能；优先用未见测试区域描述泛化。",
        "- *_catalog_records.json 保存逐条目录记录、来源、覆盖比例与匹配预测编号。",
        "- *_prediction_to_catalog.json 保存每个预测段匹配哪些原始目录记录，可按编号追溯。",
        "- label_audit.json 保存完整结果与原目录文件SHA256。",""]
    (out/"详细统计说明.md").write_text("\n".join(lines),encoding="utf-8")
if __name__=="__main__": main()

