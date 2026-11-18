# Báo cáo Day 09 — Đinh Thành Vinh, 2A202603044

## 1. Kết luận trước
Pipeline kết luận **Hold**. Lô dữ liệu đầu vào có **224 ảnh**, bước kiểm tra cửa vào (Intake) loại bỏ **21 ảnh** vi phạm (còn **203/224 ảnh** hợp lệ đưa vào xử lý và xuất ra bộ dữ liệu `clean_v1`). Lý do chính dẫn đến trạng thái Hold là dù toàn bộ **36 lỗi bắt buộc (ERROR)** đã được xử lý sạch (`n_errors = 0`), vẫn còn **38/203 ảnh** mang cảnh báo ngữ nghĩa (**18.72%**) vượt quá ngưỡng chặn **10.00%** do ngân sách duyệt thủ công giới hạn ở **80 ảnh**.

## 2. Lỗi và phân việc
Tổng hợp từ `submission/reports/02_error_list_v1` (trước khi sửa tự động có **36 lỗi bắt buộc** và **149 cảnh báo** trên **203 ảnh**):

| Mã lỗi | Số ảnh/dòng | Ai sửa (máy/người) | Vì sao |
|---|---:|---|---|
| `BAD_CLASS_NAME` | 1 file | Máy | Quy tắc cố định: tự động ghi lại danh sách tên lớp chuẩn trong `data.yaml`. |
| `BAD_FORMAT` | 4 dòng | Máy | Lỗi cú pháp (thiếu cột/sai kiểu số) không thể khôi phục tọa độ nên xóa dòng an toàn. |
| `BAD_CLASS_ID` | 3 dòng | Máy | ID lớp nằm ngoài phạm vi hợp lệ `0..3`, xóa dòng để tránh lỗi khi huấn luyện. |
| `BOX_ZERO` | 2 dòng | Máy | Box có chiều rộng hoặc chiều cao `<= 0` không có diện tích, xóa dòng. |
| `DUP_BOX` | 5 dòng | Máy | Box trùng lặp cùng lớp với dòng trước đó (`IoU > 0.9`), xóa dòng dư thừa. |
| `BOX_OUT_OF_IMAGE` | 15 dòng | Máy | Mép box tràn nhẹ ra ngoài biên `[0, 1]`, máy tự động cắt gọn (clip) tọa độ về khung ảnh. |
| `MISSING_LABEL_FILE` | 6 ảnh | Người | Ảnh thiếu hoàn toàn file `.txt`; máy không được phép tự bịa nhãn nên chuyển người gán. |
| `EMPTY_LABEL` | 4 ảnh | Người | File nhãn rỗng; cần người kiểm tra xem là ảnh nền sạch hay bị bỏ sót nhãn. |
| `BOX_TINY` / `BOX_HUGE` | 10 dòng | Người | Cảnh báo kích thước bất thường (`< 12px` hoặc `> 95%` diện tích ảnh), cần người xác nhận ngữ cảnh. |
| `POSSIBLE_MISSING` | 42 cảnh báo | Người | Model `yolo26l` phát hiện vật thể nhưng thiếu nhãn; cần người duyệt tránh nhận diện nhầm. |
| `POSSIBLE_FALSE_POS` | 48 cảnh báo | Người | Box nhãn không khớp dự đoán của model; cần người kiểm tra tránh xóa nhầm vật thể thật. |
| `POSSIBLE_LOOSE_BOX` | 36 cảnh báo | Người | Box nhãn lệch/lỏng so với dự đoán (`IoU < 0.5`); cần người căn chỉnh lại viền. |
| `POSSIBLE_WRONG_CLASS` | 9 cảnh báo | Người | Nghi ngờ gán nhầm lớp (`IoU >= 0.5` nhưng khác lớp); cần người đối chiếu ý nghĩa. |

Hàng đợi `05_review_queue.csv` sắp xếp ưu tiên theo mức độ nghiêm trọng và độ tự tin (`conf`) của model kiểm tra: đưa các lỗi bắt buộc cần con người xử lý (`MISSING_LABEL_FILE`) lên đầu, tiếp theo là các cảnh báo có độ tự tin cao nhất (như `POSSIBLE_WRONG_CLASS`, `POSSIBLE_MISSING` với `conf` cao) để tối ưu hóa hiệu quả sửa lỗi trên mỗi lượt duyệt. Với ngân sách **80 ảnh**, người duyệt xử lý được trọn vẹn **6 ảnh thiếu file nhãn** và **74 ảnh** có cảnh báo ngữ nghĩa ưu tiên cao nhất trong tổng số **125 ảnh** nằm trong hàng đợi.

## 3. Calibration (2 ngưỡng)

| Tham số | Khởi đầu | Đã chọn | Precision / Recall (trước → sau) | Lý do (đánh đổi) |
|---|---|---|---|---|
| `duplicates.phash_dist` | 2 | **6** | `DUP_IMAGE_NEAR`: `1.00 / 0.50` → `1.00 / 0.67` (F1: `0.67 → 0.80`) | Tăng khoảng cách Hamming từ 2 lên 6 giúp bắt thêm ảnh gần trùng bị nén/resize mà vẫn giữ Precision tuyệt đối `1.00` (không xóa nhầm ảnh khác). |
| `second_opinion.conf` | 0.25 | **0.6** | `POSSIBLE_MISSING`: `0.50 / 1.00` → `0.57 / 1.00` (F1: `0.67 → 0.73`)<br>`POSSIBLE_WRONG_CLASS`: `0.56 / 0.83` → `0.62 / 0.83` (F1: `0.67 → 0.71`) | Nâng ngưỡng tự tin từ 0.25 lên 0.6 giúp loại bỏ đáng kể báo động giả (giảm áp lực cho hàng đợi 80 ảnh) trong khi vẫn giữ nguyên tỷ lệ bắt đúng lỗi thật (Recall). |

**Chỗ calibration có thể đánh lừa:** Trong bảng sweep `duplicates.phash_dist`, tại giá trị `16`, chỉ số F1 của `DUP_IMAGE_NEAR` đạt đỉnh (`0.91`, Recall `0.83`). Tuy nhiên, nếu chọn ngưỡng này sẽ sập bẫy quá khớp (overfitting) trên tập `calib` nhỏ (39 ảnh): bằng chứng là ngay tại mức `20`, Recall của `DUP_IMAGE_EXACT` sụt giảm nghiêm trọng từ `1.00` xuống `0.33` do các ảnh khác nội dung nhưng có bố cục nền tương tự bị gom nhầm vào cùng một cụm trùng lặp. Khi kiểm tra thực tế trên lô 224 ảnh với `phash_dist = 6`, bước Intake loại bỏ **21 ảnh** (gồm ảnh hỏng, kích thước nhỏ, rò rỉ tập test/val và các cặp ảnh thực sự trùng lặp), đảm bảo không làm hao hụt dữ liệu sạch.

## 4. Cửa chặn và điều pipeline KHÔNG bắt được
* **(a) Chứng minh cửa chặn hoạt động đúng:** Khi chạy lệnh `python -m pipeline.run --budget 0 --reports work/rep_nb --export-dir work/exp_nb` (ngân sách duyệt bằng 0), **6 ảnh** mắc lỗi `MISSING_LABEL_FILE` không có người gán nhãn bổ sung nên vẫn tồn tại dưới dạng lỗi bắt buộc (`n_errors = 6`). Cửa chặn `gate` lập tức trả về kết luận **Rework**, trả mã thoát **`20`** cho hệ thống CI/CD và **không xuất** thư mục dữ liệu `work/exp_nb`, ngăn chặn triệt để dữ liệu lỗi lọt sang bước huấn luyện.
* **(b) Hai điểm mù của pipeline tự động:**
  1. **Box bị lệch nhẹ nhưng vẫn đạt ngưỡng IoU (`0.5 <= IoU < 0.75`):** Các nhãn có khung bao hơi rộng hoặc cắt lẹm nhẹ một phần đầu/đuôi xe nhưng vẫn có `IoU >= 0.5` so với dự đoán của `yolo26l` sẽ vượt qua bước `second_opinion` mà không bị gắn cờ `POSSIBLE_LOOSE_BOX`.
  2. **Lỗi đồng thuận sai giữa người gán và model lớn:** Những đối tượng ở xa, bị che khuất một phần hoặc thuộc lớp dễ nhầm lẫn (ví dụ `car` và `truck` kích thước nhỏ) mà cả nhãn gốc lẫn model `yolo26l` đều nhận diện sai giống nhau hoặc cùng bỏ sót thì pipeline không thể phát hiện ra.

## 5. Train: raw vs clean (tùy chọn, chấm riêng)
Chưa chạy (do ưu tiên hoàn thiện và kiểm thử toàn bộ quy trình kiểm soát chất lượng dữ liệu lõi trước khi tiến hành huấn luyện so sánh trên GPU).

## 6. Quyết định và ghi chú AI
* **(a) Quyết định cuối & hướng xử lý để đạt Release:** Chấp nhận quyết định **Hold** hiện tại cho lô `clean_v1` vì tỷ lệ ảnh còn cảnh báo là **18.72%** (`38/203 ảnh`), vượt ngưỡng **10.00%**. Để chuyển trạng thái từ **Hold** sang **Release**, cần cấp thêm ngân sách duyệt thủ công cho tối thiểu **18 ảnh** đứng đầu trong danh sách `38 ảnh` cảnh báo còn lại (đưa số ảnh cảnh báo chưa duyệt về mức `<= 20/203 ảnh`, tương đương `9.85% < 10%`).
* **(b) Ghi chú sử dụng AI:**
  * **Prompt chính:** Cung cấp docstring đặc tả đầu vào/đầu ra của 4 module (`label_format`, `gate`, `autofix`, `second_opinion`) cùng cấu trúc lớp `Dataset`, `Issue` trong `pipeline/common.py` để sinh mã nguồn theo chuẩn TDD.
  * **Cách kiểm code AI:** Đối chiếu từng dòng code với các ca biên trong `GUIDE.md` và chạy bộ kiểm thử tự động `python -m pytest -q` (đạt 79/79 test passed) cũng như kiểm tra thực tế trên lô 224 ảnh.
  * **Một lần AI sai:** Ở lần sinh code đầu tiên cho `second_opinion.py`, AI viết thiếu chỉ số trích xuất tọa độ box trong danh sách dự đoán `fp_preds` (`[(p, p, p, p)...]` thay vì `[(p, p, p, p)...]`), đã được phát hiện khi rà soát code và sửa lại chính xác trước khi chạy test.
