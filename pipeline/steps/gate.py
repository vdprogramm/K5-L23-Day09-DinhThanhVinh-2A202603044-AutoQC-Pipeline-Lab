from ..common import ERROR, WARN


def decide(issues, image_stems, cfg, reviewed=frozenset()):
    valid_stems = set(image_stems)
    n_images = len(valid_stems)
    hold_thresh = cfg["gate"]["warn_ratio_hold"]

    error_issues = [iss for iss in issues if iss.severity == ERROR]
    n_errors = len(error_issues)

    warn_images = set()
    for iss in issues:
        if iss.severity == WARN and iss.image in valid_stems and iss.image not in reviewed:
            warn_images.add(iss.image)

    n_warn_images = len(warn_images)
    warn_ratio = round(n_warn_images / n_images, 4) if n_images > 0 else 0.0

    reasons = []
    if n_errors >= 1:
        decision = "Rework"
        reasons.append(f"Còn {n_errors} lỗi bắt buộc (ERROR) chưa được xử lý.")
    elif warn_ratio >= hold_thresh:
        decision = "Hold"
        reasons.append(
            f"Tỉ lệ ảnh còn cảnh báo là {warn_ratio:.2%} ({n_warn_images}/{n_images} ảnh), "
            f"đạt hoặc vượt ngưỡng chặn {hold_thresh:.2%}."
        )
    else:
        decision = "Release"
        reasons.append(
            f"Không còn lỗi bắt buộc; tỉ lệ ảnh cảnh báo {warn_ratio:.2%} ({n_warn_images}/{n_images} ảnh) "
            f"nằm dưới ngưỡng {hold_thresh:.2%}."
        )

    return {
        "decision": decision,
        "n_errors": n_errors,
        "n_warn_images": n_warn_images,
        "n_images": n_images,
        "warn_ratio": warn_ratio,
        "reasons": reasons,
    }
