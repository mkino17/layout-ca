import json
from pathlib import Path
from .rod_sort import sort_shelf

MIN_GAP = 50

# util
def find_all_horizontal_gaps(boxes, min_gap=MIN_GAP):
    ys = sorted({s["bbox"][1] for s in boxes} | {s["bbox"][3] for s in boxes})
    gaps = []
    for a, b in zip(ys, ys[1:]):
        if b-a < min_gap: continue
        overlaps = any(s["bbox"][1] < b and s["bbox"][3] > a for s in boxes)
        if not overlaps: gaps.append((a+b)/2)
    return gaps

def find_all_vertical_gaps(boxes, min_gap=MIN_GAP):
    xs = sorted({s["bbox"][0] for s in boxes} | {s["bbox"][2] for s in boxes})
    gaps = []
    for a, b in zip(xs, xs[1:]):
        if b-a < min_gap: continue
        overlaps = any(s["bbox"][0] < b and s["bbox"][2] > a for s in boxes)
        if not overlaps: gaps.append((a+b)/2)
    return gaps

def partition_by_gaps(boxes, rect, cuts, axis):
    groups = []
    if axis == "H":
        bounds = [rect[1]] + cuts + [rect[3]]
        for y_start, y_end in zip(bounds, bounds[1:]):
            sub = [b for b in boxes if (b["bbox"][1]+b["bbox"][3])/2 >= y_start and (b["bbox"][1]+b["bbox"][3])/2 <= y_end]
            if sub: groups.append(([rect[0], y_start, rect[2], y_end], sub))
    else:
        bounds = [rect[0]] + cuts + [rect[2]]
        for x_start, x_end in zip(bounds, bounds[1:]):
            sub = [b for b in boxes if (b["bbox"][0]+b["bbox"][2])/2 >= x_start and (b["bbox"][0]+b["bbox"][2])/2 <= x_end]
            if sub: groups.append(([x_start, rect[1], x_end, rect[3]], sub))
    return groups

def xy_cut_group(boxes, rect, reading_direction, isvertical):
    if not isvertical:
        v_cuts = find_all_vertical_gaps(boxes)
        if v_cuts:
            parts = partition_by_gaps(boxes, rect, v_cuts, axis="V")
            if reading_direction == "right-left":
                parts = parts[::-1]
            return [xy_cut_group(sub_boxes, sub_rect, reading_direction, isvertical)
                    for sub_rect, sub_boxes in parts]

        h_cuts = find_all_horizontal_gaps(boxes)
        if h_cuts:
            parts = partition_by_gaps(boxes, rect, h_cuts, axis="H")
            return [xy_cut_group(sub_boxes, sub_rect, reading_direction, isvertical)
                    for sub_rect, sub_boxes in parts]

    else:
        h_cuts = find_all_horizontal_gaps(boxes)
        if h_cuts:
            parts = partition_by_gaps(boxes, rect, h_cuts, axis="H")
            return [xy_cut_group(sub_boxes, sub_rect, reading_direction, isvertical)
                    for sub_rect, sub_boxes in parts]

        v_cuts = find_all_vertical_gaps(boxes)
        if v_cuts:
            parts = partition_by_gaps(boxes, rect, v_cuts, axis="V")
            if reading_direction == "right-left":
                parts = parts[::-1]
            return [xy_cut_group(sub_boxes, sub_rect, reading_direction, isvertical)
                    for sub_rect, sub_boxes in parts]

    return sort_shelf(boxes, rect, reading_direction, x_ratio=0.05, r_over=0.25, eps=4)


def get_page_rect(segs):
    xs = [c for s in segs for c in (s["bbox"][0], s["bbox"][2])]
    ys = [c for s in segs for c in (s["bbox"][1], s["bbox"][3])]
    if not xs or not ys:
        return None
    return [min(xs), min(ys), max(xs), max(ys)]

def load_segments(path: Path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    segs = []
    for i, s in enumerate(data):
        x1, y1, x2, y2 = [float(v) for v in s["bbox"]]
        segs.append({
            "id": i, "label": s.get("label",""), "bbox":[x1,y1,x2,y2],
            "confidence": s.get("confidence"),
            "text": s.get("text",""),
            "hratio_to_text": s.get("hratio_to_text")
        })
    return segs

def flatten_groups(g):
    if isinstance(g, list) and g and isinstance(g[0], dict):
        return g
    elif isinstance(g, list):
        result = []
        for sub in g: result.extend(flatten_groups(sub))
        return result
    return []

def move_header_footer(ordered, header_label="Page-header", footer_label="Page-footer"):
    headers = [s for s in ordered if s.get("label")==header_label]
    footers = [s for s in ordered if s.get("label")==footer_label]
    middle  = [s for s in ordered if s.get("label") not in (header_label, footer_label)]
    return headers + middle + footers

# process
def process_reading_order(in_file: Path, out_file: Path, all_pages_list, reading_direction, isvertical, global_order_start=0):
    out, local_order, global_order = [], 0, global_order_start
    segs = load_segments(in_file)
    page_rect = get_page_rect(segs)
    if page_rect is None:
        out_file.write_text("[]", encoding="utf-8")
        return global_order
    grouped = xy_cut_group(segs, page_rect, reading_direction, isvertical)
    ordered = move_header_footer(flatten_groups(grouped))
    
    for s in ordered:
        seg_data = {
            "order": global_order,
            "local-order": local_order,
            "label": s["label"],
            "bbox": [round(v, 2) for v in s["bbox"]],
            "confidence": s.get("confidence"),
            "text": s.get("text", ""),
            "id": s["id"],
            "hratio_to_text": s.get("hratio_to_text"),
            "page": in_file.stem
        }
        out.append(seg_data); all_pages_list.append(seg_data)
        global_order += 1; local_order += 1

    out_file.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return global_order

def run_reading_order_detection(in_dir, out_dir, reading_direction, isvertical):
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    all_pages_ordered, global_order = [], 0

    for seg_file in sorted(in_dir.glob("*.json")):
        out_file = out_dir / seg_file.name
        global_order = process_reading_order(seg_file, out_file, all_pages_ordered, reading_direction, isvertical, global_order)
    all_file = out_dir / "all_pages_ordered.json"
    all_file.write_text(json.dumps(all_pages_ordered, ensure_ascii=False, indent=2), encoding="utf-8")
    return all_file