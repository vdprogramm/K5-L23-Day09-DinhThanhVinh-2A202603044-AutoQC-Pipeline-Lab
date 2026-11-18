from collections import defaultdict
import yaml
from ..common import ERROR, parse_yolo_line

DROP_CODES = {"BAD_FORMAT", "BAD_CLASS_ID", "BOX_ZERO", "DUP_BOX"}
REMOVE_IMG_PREFIXES = ("DUP_IMAGE_", "LEAK_TEST", "CORRUPT_IMAGE", "BAD_SIZE")


def fix(ds, issues, cfg):
    fixes = []
    error_issues = [iss for iss in issues if iss.severity == ERROR]

    # 1. Xử lý lỗi cấp dataset (BAD_CLASS_NAME)
    for iss in error_issues:
        if iss.code == "BAD_CLASS_NAME":
            before_names = ds.yaml_names()
            after_names = list(cfg["classes"])
            if ds.yaml_path.exists():
                data = yaml.safe_load(ds.yaml_path.read_text(encoding="utf-8")) or {}
            else:
                data = {}
            data["names"] = {i: name for i, name in enumerate(after_names)}
            ds.yaml_path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
            ds.classes = after_names
            fixes.append({
                "image": iss.image or "data.yaml",
                "line": "",
                "action": "rename_classes",
                "before": str(before_names),
                "after": str(after_names),
            })

    # 2. Xác định và xoá các ảnh vi phạm nghiêm trọng
    removed_stems = set()
    for iss in error_issues:
        if iss.code.startswith(REMOVE_IMG_PREFIXES):
            stem = iss.image
            if stem in removed_stems:
                continue
            img_p = ds.image_path(stem)
            lbl_p = ds.label_path(stem)
            if img_p is not None and img_p.exists():
                img_p.unlink()
                if lbl_p.exists():
                    lbl_p.unlink()
                removed_stems.add(stem)
                fixes.append({
                    "image": stem,
                    "line": "",
                    "action": "remove_image",
                    "before": stem,
                    "after": "(đã xoá ảnh và nhãn)",
                })

    # 3. Xử lý MISSING_LABEL_FILE (không tự tạo file, chỉ ghi log chuyển người gán)
    logged_missing = set()
    for iss in error_issues:
        if iss.code == "MISSING_LABEL_FILE":
            stem = iss.image
            if stem not in removed_stems and stem not in logged_missing:
                logged_missing.add(stem)
                fixes.append({
                    "image": stem,
                    "line": "",
                    "action": "needs_label",
                    "before": "(không có file)",
                    "after": "chuyển người gán",
                })

    # 4. Gom lỗi cấp dòng theo từng file nhãn
    line_issues_by_stem = defaultdict(lambda: defaultdict(set))
    for iss in error_issues:
        stem = iss.image
        if stem in removed_stems or iss.line is None:
            continue
        if iss.code in DROP_CODES or iss.code == "BOX_OUT_OF_IMAGE":
            line_issues_by_stem[stem][iss.line].add(iss.code)

    for stem, lines_map in line_issues_by_stem.items():
        lbl_p = ds.label_path(stem)
        if not lbl_p.exists():
            continue

        raw_lines = lbl_p.read_text(encoding="utf-8", errors="replace").splitlines()
        new_lines = []
        modified = False

        for line_idx, raw_line in enumerate(raw_lines, start=1):
            codes = lines_map.get(line_idx)
            if not codes:
                new_lines.append(raw_line)
                continue

            stripped = raw_line.strip()
            # Ưu tiên drop_line nếu dòng dính cả lỗi xoá lẫn lỗi tràn viền
            if codes & DROP_CODES:
                modified = True
                fixes.append({
                    "image": stem,
                    "line": line_idx,
                    "action": "drop_line",
                    "before": stripped,
                    "after": "",
                })
                continue

            if "BOX_OUT_OF_IMAGE" in codes:
                try:
                    cls_id, cx, cy, w, h = parse_yolo_line(stripped)
                    x1 = max(0.0, min(1.0, cx - w / 2.0))
                    y1 = max(0.0, min(1.0, cy - h / 2.0))
                    x2 = max(0.0, min(1.0, cx + w / 2.0))
                    y2 = max(0.0, min(1.0, cy + h / 2.0))
                    nw = x2 - x1
                    nh = y2 - y1

                    if nw <= 0 or nh <= 0:
                        modified = True
                        fixes.append({
                            "image": stem,
                            "line": line_idx,
                            "action": "drop_line",
                            "before": stripped,
                            "after": "",
                        })
                    else:
                        ncx = (x1 + x2) / 2.0
                        ncy = (y1 + y2) / 2.0
                        after_str = f"{cls_id} {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}"
                        new_lines.append(after_str)
                        modified = True
                        fixes.append({
                            "image": stem,
                            "line": line_idx,
                            "action": "clip_box",
                            "before": stripped,
                            "after": after_str,
                        })
                except ValueError:
                    modified = True
                    fixes.append({
                        "image": stem,
                        "line": line_idx,
                        "action": "drop_line",
                        "before": stripped,
                        "after": "",
                    })

        if modified:
            content = "\n".join(new_lines) + ("\n" if new_lines else "")
            lbl_p.write_text(content, encoding="utf-8")

    return fixes
