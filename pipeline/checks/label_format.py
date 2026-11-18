from ..common import ERROR, WARN, Issue, iou_xywh, parse_yolo_line


def run(ds, cfg):
    issues = []
    lf_cfg = cfg["label_format"]
    overflow_tol = lf_cfg["overflow_tol"]
    dup_box_iou = lf_cfg["dup_box_iou"]
    min_box_px = lf_cfg["min_box_px"]
    max_box_frac = lf_cfg["max_box_frac"]
    num_classes = len(ds.classes)

    for img_path in ds.image_files():
        stem = img_path.stem
        lines = ds.read_label_lines(stem)
        if lines is None:
            continue

        img_size = ds.image_size(stem)
        seen_boxes = []  # Lưu (cls_id, (cx, cy, w, h)) của các dòng hợp lệ trước đó

        for line_idx, line_text in lines:
            try:
                cls_id, cx, cy, w, h = parse_yolo_line(line_text)
            except ValueError as e:
                issues.append(Issue(
                    check="label_format",
                    severity=ERROR,
                    code="BAD_FORMAT",
                    image=stem,
                    line=line_idx,
                    message=f"Sai định dạng dòng YOLO: {e}"
                ))
                continue

            if not (0 <= cls_id < num_classes):
                issues.append(Issue(
                    check="label_format",
                    severity=ERROR,
                    code="BAD_CLASS_ID",
                    image=stem,
                    line=line_idx,
                    message=f"ID lớp không hợp lệ: {cls_id}"
                ))
                continue

            if w <= 0 or h <= 0:
                issues.append(Issue(
                    check="label_format",
                    severity=ERROR,
                    code="BOX_ZERO",
                    image=stem,
                    line=line_idx,
                    message=f"Kích thước box không dương: w={w}, h={h}"
                ))
                continue

            # Kiểm tra tràn viền ảnh
            left = cx - w / 2.0
            right = cx + w / 2.0
            top = cy - h / 2.0
            bottom = cy + h / 2.0
            overflow = max(0.0, -left, right - 1.0, -top, bottom - 1.0)
            if overflow > overflow_tol:
                issues.append(Issue(
                    check="label_format",
                    severity=ERROR,
                    code="BOX_OUT_OF_IMAGE",
                    image=stem,
                    line=line_idx,
                    message=f"Box tràn ra ngoài ảnh (mức tràn: {overflow:.4f})",
                    extra={"overflow": round(overflow, 6)}
                ))

            # Kiểm tra trùng box với các dòng trước đó
            curr_box = (cx, cy, w, h)
            for prev_cls, prev_box in seen_boxes:
                if prev_cls == cls_id and iou_xywh(curr_box, prev_box) > dup_box_iou:
                    issues.append(Issue(
                        check="label_format",
                        severity=ERROR,
                        code="DUP_BOX",
                        image=stem,
                        line=line_idx,
                        message="Box trùng lặp với dòng trước đó"
                    ))
                    break
            seen_boxes.append((cls_id, curr_box))

            # Kiểm tra kích thước (WARN)
            if img_size is not None:
                img_w, img_h = img_size
                short_side_px = min(w * img_w, h * img_h)
                if short_side_px < min_box_px:
                    issues.append(Issue(
                        check="label_format",
                        severity=WARN,
                        code="BOX_TINY",
                        image=stem,
                        line=line_idx,
                        message=f"Box quá nhỏ ({short_side_px:.1f}px < {min_box_px}px)"
                    ))

            if (w * h) > max_box_frac:
                issues.append(Issue(
                    check="label_format",
                    severity=WARN,
                    code="BOX_HUGE",
                    image=stem,
                    line=line_idx,
                    message=f"Box quá lớn (tỉ lệ {w*h:.2f} > {max_box_frac})"
                ))

    return issues
