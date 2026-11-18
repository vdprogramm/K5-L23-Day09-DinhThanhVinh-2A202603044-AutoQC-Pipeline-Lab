from ..common import WARN, Issue, iou_xywh, parse_yolo_line
from ..model_cache import get_preds


def run(ds, cfg, preds=None):
    if preds is None:
        preds = get_preds(ds, cfg, "checker")

    so_cfg = cfg["second_opinion"]
    conf_thresh = so_cfg["conf"]
    min_area_px = so_cfg["min_area_px"]
    fp_conf = so_cfg["fp_conf"]
    iou_missing = so_cfg["iou_missing"]
    iou_match = so_cfg["iou_match"]
    iou_loose = so_cfg["iou_loose"]

    issues = []

    for img_path in ds.image_files():
        stem = img_path.stem
        if stem not in preds:
            continue
        lines = ds.read_label_lines(stem)
        if lines is None:
            continue

        img_size = ds.image_size(stem)
        if img_size is None:
            continue
        img_w, img_h = img_size
        raw_preds = preds[stem]

        # Lấy nhãn hợp lệ kèm số dòng gốc từ read_label_lines
        valid_labels = []  # (line_idx, cls_id, (cx, cy, w, h))
        for line_idx, line_text in lines:
            try:
                cls_id, cx, cy, w, h = parse_yolo_line(line_text)
                if 0 <= cls_id < len(ds.classes) and w > 0 and h > 0:
                    valid_labels.append((line_idx, cls_id, (cx, cy, w, h)))
            except ValueError:
                continue

        # 1. Duyệt từng dự đoán của model checker theo đúng thứ tự
        for p in raw_preds:
            p_cls, p_conf, p_cx, p_cy, p_w, p_h = p
            p_cls = int(p_cls)
            area_px = (p_w * img_w) * (p_h * img_h)
            if p_conf < conf_thresh or area_px < min_area_px:
                continue

            p_box = (p_cx, p_cy, p_w, p_h)
            same_cls_ious = []
            diff_cls_ious = []

            for _, l_cls, l_box in valid_labels:
                iou = iou_xywh(p_box, l_box)
                if iou >= iou_missing:
                    if l_cls == p_cls:
                        same_cls_ious.append(iou)
                    else:
                        diff_cls_ious.append(iou)

            if not same_cls_ious and not diff_cls_ious:
                issues.append(Issue(
                    check="second_opinion",
                    severity=WARN,
                    code="POSSIBLE_MISSING",
                    image=stem,
                    message=f"Model phát hiện lớp {p_cls} (conf={p_conf:.2f}) nhưng thiếu nhãn",
                    extra={"conf": p_conf, "pred_class": p_cls}
                ))
            elif same_cls_ious:
                best_same = max(same_cls_ious)
                if best_same < iou_loose:
                    issues.append(Issue(
                        check="second_opinion",
                        severity=WARN,
                        code="POSSIBLE_LOOSE_BOX",
                        image=stem,
                        message=f"Box nhãn lớp {p_cls} khớp lỏng/lệch (IoU={best_same:.2f})",
                        extra={"conf": p_conf, "iou": round(best_same, 4)}
                    ))
            else:
                best_diff = max(diff_cls_ious)
                if best_diff >= iou_match:
                    issues.append(Issue(
                        check="second_opinion",
                        severity=WARN,
                        code="POSSIBLE_WRONG_CLASS",
                        image=stem,
                        message=f"Nghi ngờ gán nhầm lớp (model dự đoán {p_cls}, IoU={best_diff:.2f})",
                        extra={"conf": p_conf, "iou": round(best_diff, 4), "pred_class": p_cls}
                    ))

        # 2. Kiểm tra POSSIBLE_FALSE_POS trên từng dòng nhãn hợp lệ
        fp_preds = [(p[2], p[3], p[4], p[5]) for p in raw_preds if p[1] >= fp_conf]
        for line_idx, l_cls, l_box in valid_labels:
            max_iou = max((iou_xywh(l_box, pb) for pb in fp_preds), default=0.0)
            if max_iou < iou_missing:
                issues.append(Issue(
                    check="second_opinion",
                    severity=WARN,
                    code="POSSIBLE_FALSE_POS",
                    image=stem,
                    line=line_idx,
                    message=f"Nhãn lớp {l_cls} không khớp dự đoán nào của model",
                    extra={"iou": round(max_iou, 4)}
                ))

    return issues
