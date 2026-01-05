from math import hypot


def sort_shelf(boxes, rect, reading_direction, x_ratio=0.1, r_over=0.3, eps=4):
    def x1(b): return b["bbox"][0]
    def y1(b): return b["bbox"][1]
    def x2(b): return b["bbox"][2]
    def y2(b): return b["bbox"][3]
    def w(b):  return x2(b)-x1(b)

    def is_header(b):
        lbl = (b.get("label") or "").lower().replace("_", "-")
        return lbl in {"section-header","title"}

    def support_ok(upper, cand):
        if y1(cand) < y1(upper) - eps: return False
        if is_header(upper): return True
        bw = max(1.0, w(upper))
        over_r = max(0.0, x2(cand) - x2(upper))
        over_l = max(0.0, x1(upper) - x1(cand))
        return (over_r <= r_over*bw) and (over_l <= r_over*bw)

    page_w = rect[2]-rect[0]
    x_tol = page_w * x_ratio

    cols = []
    for b in sorted(boxes, key=lambda s: (x1(s), y1(s))):
        lx = x1(b)
        for col in cols:
            xs = [x1(bb) for bb in col['items']]
            if (min(xs)-x_tol) <= lx <= (max(xs)+x_tol):
                col['items'].append(b); break
        else:
            cols.append({'items':[b]})
    for col in cols:
        col['items'].sort(key=lambda s: (y1(s), x1(s)))

    cols.sort(key=lambda c: min(x1(bb) for bb in c['items']) if c['items'] else float('inf'))

    if reading_direction == "right-left":
        cols.reverse()

    def any_border_ok(items, border_y):
        by = border_y + eps
        return any(y2(b) <= by for b in items)

    def find_right_col_idx(start_idx, border_y):
        start = (start_idx if start_idx is not None else -1) + 1
        if border_y is None:
            for k in range(start, len(cols)):
                if cols[k]['items']: return k
            return None
        for k in range(start, len(cols)):
            if any_border_ok(cols[k]['items'], border_y): return k
        return None

    def peek_top(col_idx, border_y):
        items = cols[col_idx]['items']
        if not items: return None
        if border_y is None: return items[0]
        by = border_y + eps
        for b in items:
            if y2(b) <= by: return b
        return None

    def find_special_loc(special_bp):
        if special_bp is None: return None, None
        for k, col in enumerate(cols):
            for b in col['items']:
                if b is special_bp: return k, b
        return None, None

    def accept(b):
        for col in cols:
            if b in col['items']:
                col['items'].remove(b); break
        if b in remaining: remaining.remove(b)

    def nearest_below(prev, candidates, border_y=None, special=None, reading_direction="left-right"):
        if reading_direction == "left-right":
            anchor = (x1(prev), y2(prev))
            near = []
            for b in candidates:
                if border_y is not None and (b is not special) and not (y2(b) <= border_y + eps):
                    continue
                if y1(b) < y1(prev) - eps: continue
                if abs(x1(prev) - x1(b)) > w(prev): continue
                near.append(b)
            if not near: return special if special else None
            best = min(near, key=lambda b: hypot(anchor[0]-x1(b), anchor[1]-y1(b)))
            if abs(x1(prev)-x1(best)) > w(prev): return None
        else: 
            anchor = (x2(prev), y2(prev))
            near = []
            for b in candidates:
                if border_y is not None and (b is not special) and not (y2(b) <= border_y + eps):
                    continue
                if y1(b) < y1(prev) - eps: continue
                if abs(x2(prev) - x2(b)) > w(prev): continue
                near.append(b)
            if not near: return special if special else None
            best = min(near, key=lambda b: hypot(anchor[0]-x2(b), anchor[1]-y1(b)))
            if abs(x2(prev)-x2(best)) > w(prev): return None
        return best

    ordered, remaining = [], boxes[:]
    current, last_col_idx, border_y, special_bp = None, None, None, None

    while remaining:
        if current is None:
            k = find_right_col_idx(last_col_idx, border_y)
            if k is not None:
                seed = peek_top(k, border_y)
                if seed is None:
                    sk, sb = find_special_loc(special_bp)
                    if sb is None: break
                    k, seed = sk, sb
            else:
                sk, sb = find_special_loc(special_bp)
                if sb is None: break
                k, seed = sk, sb

            ordered.append(seed); accept(seed)
            current = seed; last_col_idx = k
            if seed is special_bp: border_y, special_bp = None, None
            continue

        cand_down = nearest_below(current, remaining, border_y=border_y, special=special_bp, reading_direction=reading_direction)
        if cand_down and support_ok(current, cand_down):
            ordered.append(cand_down); accept(cand_down)
            current = cand_down
            if special_bp is cand_down: border_y, special_bp = None, None
            continue
        elif cand_down and not support_ok(current, cand_down):
            border_y = cand_down["bbox"][1]; special_bp = cand_down
        else:
            border_y, special_bp = None, None

        current = None

    return ordered