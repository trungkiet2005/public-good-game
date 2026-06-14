# Kịch bản thuyết trình
### "LLM agents có phải là *silicon samples* hợp lệ cho hành vi hợp tác của con người?"
**Hai paper đồng hành: CRSD/Climate (Milinski 2008) + PGG/Punishment (Herrmann 2008)**

---

## 0. Hướng dẫn sử dụng kịch bản

- **Thời lượng mục tiêu:** ~22–25 phút nói + 5–10 phút hỏi đáp.
- **Phân bổ:** Mở đầu + bối cảnh ~5 phút · Paper A ~7 phút · Paper B ~7 phút · Tổng hợp + kết luận ~4 phút.
- **Quy ước trong file:**
  - 🗣️ = *lời nói* (đọc gần như nguyên văn, hoặc diễn đạt lại tự nhiên).
  - 📌 = ý bắt buộc phải nhấn.
  - 📊 = cách đọc hình.
  - 🧮 = cách tính / giải thích phương pháp (dùng khi thầy hỏi sâu).
  - ❓ = câu thầy có thể hỏi + gợi ý trả lời.
- **2 slide quan trọng nhất** (nói chậm, dừng lại): **A2** (vô cảm rủi ro) và **B2** (trừng phạt không nâng hợp tác). Đây là 2 "kết quả quyết định".
- **Thông điệp xuyên suốt — lặp lại 3 lần** (mở đầu, giữa, kết): *LLM tái tạo **bề mặt** của hợp tác, nhưng đánh mất **cấu trúc chẩn đoán** — phản ứng với biến thí nghiệm.*

> 💡 Bản đồ slide: deck có 24 trang = 1 title + 18 slide đánh số + 4 trang "Phần …" (section divider, chỉ đọc 1 câu chuyển ý) + 1 phụ lục. Số "X/18" ở góc dưới phải là số slide nội dung.

---

## PHẦN MỞ ĐẦU

### 🟦 Slide title (trang 1)
🗣️ "Em chào thầy. Hôm nay em xin báo cáo tiến độ **hai paper đồng hành** mà em đang làm. Cả hai cùng trả lời một câu hỏi: *liệu các tác nhân LLM (LLM agents) có thể đóng vai 'silicon samples' — mẫu người nhân tạo — một cách hợp lệ trong các thí nghiệm về hợp tác hay không.* Paper thứ nhất dùng trò chơi **rủi ro khí hậu** của Milinski 2008; paper thứ hai dùng trò chơi **trừng phạt trong public-goods** của Herrmann 2008."

📌 Nói rõ ngay đây là *2 paper, 1 buổi*, và chúng **bổ trợ nhau** (companion).

---

### 🟦 Slide 1/18 — Nội dung báo cáo (trang 2)
🗣️ "Bố cục gồm 4 phần: (1) bối cảnh chung và tiêu chí đánh giá; (2) Paper A — rủi ro khí hậu; (3) Paper B — trừng phạt; và (4) tổng hợp. Nếu thầy chỉ nhớ một câu, thì đây ạ:" *(chỉ vào ô bên phải)* "**cả hai paper cho cùng kết luận — LLM tái tạo được *bề mặt* của hợp tác, nhưng đánh mất *cấu trúc chẩn đoán* làm nên ý nghĩa khoa học của thí nghiệm.**"

📌 Đọc to ô "Thông điệp một câu". Đây là luận điểm trung tâm.

---

## PHẦN 1 — BỐI CẢNH & CÂU HỎI CHUNG

### ⬜ Section divider "Phần 1" (trang 3)
🗣️ "Trước hết, vì sao câu hỏi này lại quan trọng."

---

### 🟦 Slide 2/18 — LLM làm "silicon samples" thay người (trang 4)
🗣️ "Gần đây có một làn sóng nghiên cứu dùng LLM làm **người chơi nhân tạo** trong các trò chơi kinh tế — người ta gọi là *homo silicus* hay *generative agents*. Sức hấp dẫn rất rõ: **rẻ, nhanh, mở rộng được**; có thể chạy thử thiết kế thí nghiệm trước khi tốn tiền tuyển người thật. Thậm chí có báo cáo nói phân bố phản hồi của LLM 'vượt qua được Turing test hành vi'."

🗣️ *(chỉ vào ô block)* "Nhưng tiền đề cốt lõi là cái gọi là **behavioural validity** — tính hợp lệ về hành vi. Một LLM chỉ hữu ích khi nó tái tạo được **cách con người *phản ứng*** với các đòn bẩy thí nghiệm — chứ không phải chỉ giống ở **mức trung bình**. Đây chính là điều cả hai paper của em đi kiểm chứng."

📊 Sơ đồ bên phải: con người phản ứng với biến thí nghiệm (mũi tên) → câu hỏi là LLM agent có tái tạo *đúng phản ứng đó* (comparative statics) không.

❓ *"Comparative statics nghĩa là gì?"* → "Là cách một đại lượng đầu ra thay đổi **khi ta thay đổi một biến điều khiển**. Ví dụ: khi giảm xác suất thảm họa, đầu tư thay đổi ra sao. Validity nằm ở *độ dốc của phản ứng*, không phải ở giá trị trung bình."

---

### 🟦 Slide 3/18 — Tiêu chí hợp lệ: phản ứng, không phải mức trung bình (trang 5)
🗣️ "Đây là **luận điểm thống nhất** của cả hai paper, và cũng là đóng góp lý thuyết: *một surrogate hợp lệ phải tái tạo **phản ứng với biến thao tác / cấu trúc quan hệ** — không phải mức hành vi trung bình mà nó tình cờ thể hiện.*"

🗣️ "Áp vào hai trò chơi: Paper A có **một** biến chẩn đoán là *loss probability* — xác suất mất trắng; ta hỏi: hợp tác có **giảm** khi rủi ro giảm như ở người không. Paper B có **hai** chữ ký chẩn đoán: (1) trừng phạt có **nâng** hợp tác không, và (2) cấu trúc **xuyên văn hóa** của antisocial punishment có còn không."

📌 Đây là 'kính lúp' để soi toàn bộ kết quả phía sau. Nói chậm.

---

### 🟦 Slide 4/18 — Phương pháp chung (trang 6)
🗣️ "Cả hai paper dùng chung một khung. Mỗi agent nhận **một prompt** gồm luật chơi + lịch sử ván + khối tính cách (disposition), rồi xuất một dòng quyết định **parse được** như `CONTRIBUTION = X`. Mỗi vòng, quyết định của tất cả agent được thu **đồng bộ một lô** nên ván chơi thực sự *đồng thời*."

🗣️ "Em dùng **3 model mở** cỡ 7–9B: Qwen2.5-7B, Gemma-2-9B, Llama-3.1-8B; chạy bằng vLLM, temperature 0.8, seed cố định để tái lập, offline trên một card RTX PRO 6000. Thiết kế giai thừa: **5 ngôn ngữ**, **4 disposition** — lưu ý disposition thứ tư khác nhau: Paper A có *risk-averse*, Paper B có *vengeful* — và **10 nhóm độc lập mỗi điều kiện**. Riêng Paper B thêm **16 'society personas'** là 16 thành phố trong nghiên cứu gốc của Herrmann."

🧮 *temperature 0.8* = mức ngẫu nhiên khi sinh chữ; >0 để có biến thiên giữa các nhóm. *seed cố định* = cùng đầu vào cho cùng đầu ra → tái lập được.

❓ *"Vì sao 10 nhóm/điều kiện?"* → "Khớp với cỡ mẫu của thí nghiệm gốc (Milinski/Herrmann đều ~10 nhóm/ô), để so sánh công bằng."

---

## PHẦN 2 — PAPER A: COLLECTIVE-RISK CLIMATE DILEMMA

### ⬜ Section divider "Phần 2" (trang 7)
🗣️ "Bắt đầu với Paper A — trò chơi rủi ro khí hậu."

---

### 🟦 Slide 5/18 — Trò chơi CRSD (Milinski 2008) (trang 8)
🗣️ "Luật chơi: **6 người**, mỗi người có vốn riêng **€40**, chơi **10 vòng**. Mỗi vòng mỗi người bí mật góp **€0, €2 hoặc €4** vào 'quỹ khí hậu' chung. Mục tiêu của cả nhóm là tích lũy đủ **€120**. Nếu đạt, mỗi người giữ phần còn lại của €40. Nếu **không đạt**, sẽ có một **xổ số**: với xác suất *p*, **tất cả mất sạch** — payoff về 0."

🗣️ "Biến chẩn đoán là **xác suất mất trắng** *p*, đặt ở 3 mức: **90%, 50%, 10%**. Tổng tối đa có thể góp là €240 (6 người × €4 × 10 vòng)."

📊 Biểu đồ cột bên phải (vẽ tay minh họa số liệu người): khi *p* giảm 90→50→10, đầu tư của **con người** rơi mạnh **€118 → €92 → €73**. Đó là 'chữ ký' của ra quyết định nhạy rủi ro.

🧮 **Final group total tính thế nào:** cộng toàn bộ tiền góp của 6 người qua 10 vòng. *Success* = (total ≥ €120).

📌 Nhấn: cái làm thí nghiệm này *có giá trị* không phải mức hợp tác, mà là **độ dốc của đầu tư theo p**.

❓ *"Sao không cho agent xem tổng tích lũy?"* → "Trung thành với bản gốc Milinski — người chơi cũng phải tự cộng. Em sẽ quay lại điểm này như một hạn chế ở slide tóm tắt Paper A."

---

### 🟦 Slide 6/18 — A1: Over-cooperation ở mọi mức rủi ro (trang 9, BẢNG)
🗣️ "Kết quả đầu tiên: cả 3 model **hợp tác quá mức**. Nhìn bảng: con người đạt target chỉ 50% / 10% / 0% số nhóm khi rủi ro giảm dần, và đầu tư rơi 118 → 92 → 73. Còn cả 3 model **đạt target 100% ở MỌI mức rủi ro**, đầu tư dồn ở **€158–187** — tức khoảng **66–78% của trần €240** — cao hơn người rất nhiều."

🗣️ "Một cách nhìn khác là **fair-sharers** — số người góp ít nhất €2/vòng trung bình. Nhóm người chỉ có 1–3; nhóm model gần như **6/6**. Mọi so sánh model–người đều có ý nghĩa thống kê."

🧮 **fair-sharer** = người góp trung bình ≥ €2/vòng (≥ €20 trong 10 vòng) — đúng định nghĩa 'fair share' của Milinski.
🧮 So sánh ở đây dùng **Welch t-test** (cho tổng tiền) và **Fisher exact test** (cho tỉ lệ đạt target) — giải thích chi tiết ở Phụ lục A.

📌 Nhưng đây *chưa* phải kết quả quan trọng nhất. "Mức cao" thì có thể chỉ là 'model hào phóng hơn người'. Câu hỏi thật nằm ở slide sau.

---

### 🟦 Slide 7/18 — A2 (QUYẾT ĐỊNH): Vô cảm với rủi ro thảm họa (trang 10, HÌNH `fig1_risk`)
> ⏸️ **Slide quan trọng nhất của Paper A — nói chậm, dừng sau câu chốt.**

🗣️ "Đây là kết quả quyết định. Vấn đề không phải *mức* hợp tác, mà là **độ nhạy với rủi ro**."

📊 Cách đọc hình:
- **Panel (a)** — trục hoành là vòng 1→10, trục tung là **tiền góp tích lũy** của nhóm (Gemma). Ba đường ứng với 3 mức rủi ro 90/50/10%. "Thầy thấy **ba đường gần như chồng lên nhau** và đều **vượt mốc €120** — bất kể rủi ro cao hay thấp, model vẫn đổ tiền y như nhau." Đường chấm là 'fair-share path', đường ngang đứt là target €120.
- **Panel (b)** — trục hoành là 3 mức rủi ro, trục tung là **tổng cuối**. Cột xám là người: **rơi dốc** khi rủi ro giảm. Ba cột màu là 3 model: **gần như phẳng** và nằm cao trên target.

🗣️ "Định lượng: ở người, ANOVA cho **F = 13.78, p < 0.0001** — phản ứng theo rủi ro rất mạnh và đơn điệu. Ở model, cùng phép ANOVA chỉ cho **F = 1.76 / 2.23 / 3.64** — gần như phẳng."

🗣️ *(câu chốt, dừng lại)* "**Cái đòn bẩy làm cho thí nghiệm này có ý nghĩa — xác suất thảm họa — gần như không tác động lên model.**"

🧮 **ANOVA (phân tích phương sai) tính thế nào — nói khi thầy hỏi:**
> "ANOVA một chiều so sánh trung bình của ≥3 nhóm. Nó tính tỉ số **F = phương sai *giữa* các nhóm / phương sai *trong* nhóm**. Nếu 3 mức rủi ro thật sự cho đầu tư khác nhau, biến thiên giữa nhóm sẽ lớn hơn nhiều so với nhiễu trong nhóm → F lớn. Bậc tự do ở đây là (2, 27): 3 nhóm nên df₁ = 3−1 = 2; 30 nhóm tổng nên df₂ = 30−3 = 27. F = 13.78 với p < 0.0001 nghĩa là xác suất thấy chênh lệch này nếu rủi ro *không* ảnh hưởng là dưới 0.01%. F ≈ 1.8 thì gần như không phân biệt được với nhiễu. Kèm theo là **η² (eta-squared)** = tỉ lệ phương sai được rủi ro giải thích: người ≈ 0.51, model chỉ 0.12–0.21."

🗣️ "Vì luận điểm của em là một **NULL** — *không có phản ứng* — em không thể chỉ dựa vào 'không bác bỏ được'. Em chứng minh null một cách **tích cực** bằng hai công cụ:"

🧮 **TOST — kiểm định tương đương:**
> "TOST = *Two One-Sided Tests*. Ý tưởng: đặt trước một **ngưỡng hiệu ứng nhỏ nhất đáng quan tâm (SESOI)** — ở đây là **một nửa mức giảm của người, ±€22.6**. Rồi chạy hai kiểm định một phía: (1) hiệu ứng có *lớn hơn* −22.6 không, (2) có *nhỏ hơn* +22.6 không. Nếu **cả hai** đều có ý nghĩa, ta kết luận hiệu ứng thật nằm gọn trong dải ±22.6 → 'tương đương với không đáng kể'. Kết quả: phản ứng 90%→10% của mọi model đều < ½ mức người (chênh chỉ +€2.0 / +€10.2 / +€7.4 so với €45.2 của người), tất cả **TOST p ≤ 0.014**."

🧮 **Bayes factor BF₀₁:**
> "BF₀₁ là tỉ số khả năng (likelihood) của dữ liệu dưới giả thuyết **không có hiệu ứng** so với **có hiệu ứng**. BF₀₁ = 4.8 nghĩa là dữ liệu *khả dĩ gấp 4.8 lần* dưới mô hình 'không có hiệu ứng'. Quy ước: 3–10 là bằng chứng **vừa phải** cho null. Qwen 4.8, Gemma 3.0 ủng hộ null; **Llama 0.8 < 1 nên *equivocal* — hơi nghiêng về 'có hiệu ứng' nhẹ**, em nói rõ chỗ này để khỏi nói quá."

❓ *"Tại sao Llama lại khác?"* → "Llama có F = 3.64 (p = 0.04) — hiệu ứng danh nghĩa yếu và **không đơn điệu** (không giảm đều theo rủi ro), η² = 0.21 vẫn dưới một nửa mức người 0.51. Nên dù BF nghiêng nhẹ, TOST vẫn cho thấy phản ứng < ½ mức người."

---

### 🟦 Slide 8/18 — A3: Hợp tác là "núm vặn", bất biến ngôn ngữ (trang 11, HÌNH `fig2_moderators`)
🗣️ "Vậy mức hợp tác cao này là *bản tính cố định* hay *điều khiển được*? Câu trả lời: nó là một **núm vặn**."

📊 Cách đọc hình:
- **Panel (a)** — trục tung là tỉ lệ đạt target (%), nhóm cột theo 4 disposition. "Chỉ cần đổi **một câu** disposition sang **'selfish'**, tỉ lệ thành công **sụp về 0%** với Qwen và Gemma (27% với Llama). 'Risk-averse' ở mức trung gian 73–93%. 'Neutral' và 'cooperative' kịch trần."
- **Panel (b)** — cột tím là tổng cuối theo 5 ngôn ngữ (Gemma); đường đỏ là **tỉ lệ lỗi định dạng (parse non-compliance)**. "Nội dung hợp tác **gần như không đổi** giữa các ngôn ngữ; cái nhảy vọt là **độ tuân thủ định dạng** — Arabic lên ~13.7%."

🗣️ "Diễn giải: over-cooperation là một **default persona điều khiển được bằng prompt** — *a knob, not a trait*. Một câu mô tả vai trò có thể đưa hành vi đi gần hết dải từ phản bội đến hợp tác hoàn toàn."

🧮 **parse-fallback là gì:** khi đầu ra của model không chứa dòng quyết định đúng định dạng, framework phải dùng giá trị dự phòng. Tỉ lệ này được log như **tín hiệu chất lượng dữ liệu & công bằng** (vì nó cao bất thường ở ngôn ngữ ít tài nguyên). Quan trọng: ở **tiếng Anh ≈ 0%**, nên kết quả chính *không* phải do lỗi parse.

❓ *"Vậy kết quả vô cảm rủi ro có phải do model đọc sai đề không?"* → "Không. Tiếng Anh — nơi mang kết quả chính — gần như 0% lỗi parse. Model đọc đúng luật mà vẫn đổ tiền bất kể rủi ro."

---

### 🟦 Slide 9/18 — Paper A: Tóm tắt (trang 12)
🗣️ "Tóm lại Paper A: ba model là những **kẻ hợp tác quá mức, vô cảm với rủi ro, điều khiển được bằng prompt**. Chúng tái tạo *bề mặt* của hành động khí hậu — đạt mục tiêu, chia sẻ gánh nặng — nhưng **thiếu cơ chế** kết nối với rủi ro thảm họa, thứ tạo ra và điều biến hành vi đó ở người."

🗣️ *(chủ động nêu hạn chế lớn nhất trước khi thầy hỏi)* "Có một điểm em muốn **phản biện trước**: do trung thành với bản gốc, agent **không được xem tổng tích lũy** nên về lý thuyết, sự 'phẳng' có thể một phần do hạn chế *tính nhẩm*. Nhưng em lập luận điều này **không giải thích được** kết quả: gánh nặng số học là **như nhau** ở cả ba mức rủi ro, nên nó không thể giải thích vì sao hành vi **không khác nhau *giữa* các mức rủi ro** — vốn mới là hiệu ứng em quan tâm."

📌 Đây là câu hỏi thầy dễ hỏi nhất về Paper A — đã có sẵn câu trả lời.

---

## PHẦN 3 — PAPER B: PGG VỚI TRỪNG PHẠT

### ⬜ Section divider "Phần 3" (trang 13)
🗣️ "Sang Paper B — cùng một tiêu chí, nhưng áp lên một thể chế khác: **trừng phạt tốn kém**."

---

### 🟦 Slide 10/18 — Trò chơi PGG với trừng phạt (Herrmann 2008) (trang 14)
🗣️ "Luật chơi: **4 người** chơi cùng nhau (partner matching) suốt **10 vòng**; mỗi vòng nhận **20 token** mới. Họ góp vào quỹ chung với **MPCR = 0.4**. Có **2 treatment**: **N** chỉ góp; **P** thêm một **giai đoạn trừng phạt tốn kém** — trả 1 token để trừ 3 token của người khác (tỉ lệ 1:3)."

🗣️ "Cách phân loại trừng phạt rất quan trọng: phạt người góp **ít hơn** mình = **prosocial** (phạt free-rider, đúng đắn); phạt người góp **bằng hoặc nhiều hơn** mình = **antisocial** (phạt người tử tế)."

🧮 **MPCR (marginal per-capita return) = 0.4 nghĩa là:** mỗi token em góp sẽ sinh 0.4 token cho **mỗi** trong 4 người → tổng xã hội 0.4 × 4 = 1.6 (nên góp là tối ưu cho cả nhóm), nhưng bản thân em chỉ nhận lại 0.4 (< 1) → **động cơ cá nhân là free-ride**. Đây chính là 'tragedy of the commons' thu nhỏ.

🗣️ *(chỉ ô bên phải)* "Ở người có **2 chữ ký chẩn đoán**: (1) mở trừng phạt **nâng** hợp tác — tăng ~+4.3 token; và (2) **cấu trúc xuyên văn hóa** — qua 16 xã hội, antisocial punishment và hợp tác có tương quan âm rất mạnh, **ρ ≈ −0.90**: xã hội nào hay phạt người tử tế thì hợp tác sụp."

📌 16 personas là **nhãn thành phố trung lập** ("Bạn là người chơi đến từ <Thành phố>") — *không* gán định kiến hành vi. Cố ý để hỏi: chỉ một nhãn địa lý đơn thuần có gợi ra biến thiên văn hóa không.

---

### 🟦 Slide 11/18 — B1: Góp hào phóng & trừng phạt, nhưng đa số antisocial (trang 15, BẢNG)
🗣️ "Kết quả nền: ở treatment N, cả 3 model góp **trên mức người** — 14.26, 13.41, 10.67 token so với grand-mean người là 8.6. Vậy model **có** hợp tác và **có** trừng phạt."

🗣️ "Nhưng nhìn cột cuối: **54–59%** số điểm trừng phạt là **antisocial** — nhắm vào người góp **bằng hoặc hơn** kẻ phạt. Tỉ lệ này ngang với những pool người **trừng phạt nặng nhất** trong Herrmann."

🧮 **antisocial share tính thế nào:** = (số điểm trừng phạt antisocial) / (tổng số điểm trừng phạt). Antisocial = các điểm nhắm vào mục tiêu có *độ lệch đóng góp ≥ 0* (góp ≥ kẻ phạt).

🗣️ *(chú thích bảng)* "Ở người, antisocial là **thiểu số** ở các pool phương Tây — nhưng **cao** ở một số xã hội khác; chính sự biến thiên đó là payload của Herrmann, và là cái em kiểm ở slide B3."

❓ *"Cột Δ (P−N) ở bảng nghĩa là gì?"* → "Là **hiệu** đóng góp khi *có* trừng phạt trừ khi *không* có. Ở người là +4.3 (trừng phạt giúp ích). Ở model là −0.36 / −3.55 / +0.08 — em phân tích kỹ ở slide sau."

---

### 🟦 Slide 12/18 — B2 (QUYẾT ĐỊNH): Trừng phạt KHÔNG nâng hợp tác (trang 16, HÌNH `fig1_punishment`)
> ⏸️ **Slide quan trọng nhất của Paper B — nói chậm.**

🗣️ "Đây là chữ ký chẩn đoán thứ nhất, và là kết quả quyết định của Paper B."

📊 Cách đọc hình:
- **Panel (a)** — trục tung là đóng góp trung bình (của 20), trục hoành là vòng 1→10 (Gemma). Đường đỏ là **N (không trừng phạt)**, đường xanh là **P (có trừng phạt)**. "Điều ngược đời: **đường P nằm DƯỚI đường N** — có trừng phạt mà hợp tác lại **thấp hơn**, ngược hẳn người."
- **Panel (b)** — trục hoành là **độ lệch đóng góp của mục tiêu so với kẻ phạt**, trục tung là mức trừng phạt. Cột xanh lá = phạt free-rider (prosocial), cột đỏ = antisocial. "Phần lớn chi tiêu rơi vào cột đỏ."

🗣️ "Định lượng: ở người mở trừng phạt làm hợp tác **tăng +4.3**. Ở model, **P−N = −0.36 / −3.55 / +0.08** — phẳng, hoặc với Gemma là **giảm có ý nghĩa**."

🧮 **Mann–Whitney U test (dùng cho P−N):**
> "Vì chỉ có 10 nhóm mỗi treatment và không giả định phân phối chuẩn, em dùng kiểm định **phi tham số Mann–Whitney U**: nó gộp tất cả giá trị của hai nhóm, **xếp hạng**, rồi xem hạng của nhóm này có lệch hệ thống so với nhóm kia không. Gemma cho **p < 0.001** → P thật sự thấp hơn N. Qwen p = 0.71, Llama p = 0.91 → không khác 0."

🗣️ "Hai trường hợp 'không khác 0' được củng cố bằng **TOST p = 0.010 / 0.002** và **Bayes factor BF₀₁ = 2.3 / 2.5** (nghiêng về 'không có hiệu ứng'). Còn ở panel (b), chỉ **Gemma** giữ được logic người — phạt free-rider nặng hơn (1.40 → 3.71 điểm theo độ lệch); Qwen và Llama phạt **gần như phẳng**, đánh người hợp tác gần ngang free-rider."

🗣️ *(chú thích †, nói chủ động)* "Một lưu ý phương pháp: mức +4.3 của người là **within-subject** (mỗi người chơi cả N lẫn P), còn agent của em chơi N và P ở **các nhóm độc lập** (between-group), nên không hoàn toàn so trực tiếp — nhưng kết luận định tính 'trừng phạt không nâng hợp tác' không phụ thuộc vào điểm này."

❓ *"BF₀₁ của Gemma đâu?"* → "Gemma ngược lại: BF₀₁ = 1.8×10⁻⁴ — bằng chứng *áp đảo* rằng **có** khác biệt, nhưng theo hướng **có hại** (trừng phạt làm hợp tác giảm). Nên em không gộp Gemma vào nhóm 'null'."

---

### 🟦 Slide 13/18 — B3: Cấu trúc xuyên văn hóa sụp đổ (trang 17, HÌNH `fig2_cross`)
🗣️ "Chữ ký chẩn đoán thứ hai — và là payload khoa học của Herrmann — cũng biến mất."

📊 Cách đọc hình:
- **Panel (a)** — mỗi điểm là **một thành phố** (16 persona, Gemma). Trục hoành là tổng điểm trừng phạt antisocial, trục tung là đóng góp trung bình. "Ở người, hai đại lượng này tương quan âm mạnh **ρ ≈ −0.90** — đám mây điểm dốc xuống. Ở model, đám mây gần như **không có hướng**: ρ = −0.15."
- **Panel (b)** — trục hoành là đóng góp của **người** (Herrmann) theo từng thành phố, trục tung là đóng góp của **model**. "Nếu model tái tạo được 'địa lý hợp tác', các điểm phải nằm trên một đường chéo tăng. Thực tế chúng **dẹt thành một dải ~1 token** và không khớp thứ hạng người (rank ρ = +0.06)."

🗣️ "Định lượng cả 3 model: tương quan antisocial–hợp tác là **−0.09 / −0.15 / +0.30** — gần bằng 0, Llama thậm chí đảo dấu. Mỗi cái đều **khác có ý nghĩa** so với −0.90 của người. 'Geography of cooperation' biến mất: 16 persona chỉ làm đóng góp dịch khoảng **1 token**. Xếp hạng 16 xã hội so với người chỉ tương quan **+0.33 / +0.06 / +0.17** — gần như không khớp."

🧮 **Spearman ρ (tương quan hạng):**
> "Spearman đo **mức độ đơn điệu** của quan hệ: thay vì dùng giá trị, nó xếp hạng rồi tính tương quan trên hạng. Bền với ngoại lệ và quan hệ phi tuyến. ρ = −0.90 nghĩa là 'xã hội phạt người tử tế càng nhiều thì hợp tác càng thấp' rất chặt. ρ ≈ 0 nghĩa là không có quan hệ."

🧮 **Fisher z-test (so ρ với −0.90):**
> "Để kiểm 'ρ của model có khác −0.90 của người không', em biến đổi **Fisher z**: z = arctanh(ρ), khi đó z xấp xỉ phân phối chuẩn và có sai số chuẩn đã biết, nên so sánh được hai tương quan. Kết quả p ≤ 0.0011 cho cả ba → khác biệt có ý nghĩa. (Em dùng biến thể *Fieller-corrected* để xử lý việc so với một hằng số tham chiếu.)"

❓ *"Sao không thử persona giàu hơn (có chuẩn mực, văn hóa)?"* → "Đúng là một hướng tiếp theo. Em cố ý dùng nhãn địa lý *trung lập* để trả lời câu hỏi hẹp: liệu một nhãn thành phố đơn thuần — kiểu một nghiên cứu silicon-sample hay dùng — có đủ gợi biến thiên văn hóa không. Kết quả null trả lời đúng câu đó; persona giàu hơn là bước sau."

---

### 🟦 Slide 14/18 — Paper B: Tóm tắt (trang 18)
🗣️ "Tóm lại Paper B: model tái tạo **sự tồn tại** của trừng phạt (cả pro- lẫn antisocial), nhưng **không** tái tạo **chức năng thể chế** (trừng phạt nâng hợp tác) lẫn **cấu trúc văn hóa**. Marginals trông giống người — kể cả mức antisocial cao bất ngờ — nhưng hai thuộc tính **quan hệ** làm nên giá trị khoa học đều mất. Hệ quả kỹ thuật: trao quyền trừng phạt cho một xã hội agent có thể thành **gánh nặng** chứ không phải lá chắn cho hợp tác."

---

## PHẦN 4 — TỔNG HỢP & KẾT LUẬN

### ⬜ Section divider "Phần 4" (trang 19)
🗣️ "Ghép hai paper lại."

---

### 🟦 Slide 15/18 — Một tiêu chí, hai dissociation song song (trang 20, BẢNG)
🗣️ "Đây là slide xương sống. Hai paper cho **cùng một mẫu hình**:" *(chỉ bảng)*
- "Cột trái — **bề mặt được tái tạo**: Paper A hợp tác cao, đạt target 100%; Paper B góp hào phóng và có trừng phạt."
- "Cột phải — **cấu trúc chẩn đoán bị mất**: Paper A mất *phản ứng với loss probability*; Paper B mất *cả hai* — trừng phạt nâng hợp tác, và cấu trúc xuyên văn hóa."

🗣️ "Một câu: *Cooperation is **real as output**, **absent as mechanism**.* Bề mặt giống người **không** bảo chứng cho cơ chế giống người."

📌 Nếu thầy chỉ chụp 1 slide, nên là slide này.

---

### 🟦 Slide 16/18 — Tiêu chí "behavioural validity" (đóng góp lý thuyết) (trang 21)
🗣️ "Đóng góp lý thuyết: phải test một 'silicon subject' **không phải bằng *mức* hành vi trung bình**, mà bằng **phản ứng với biến thao tác** và **cấu trúc điều kiện / quan hệ**. Một nghiên cứu chỉ đọc *marginals* sẽ kết luận **sai** rằng agent 'tái tạo' được kết quả người."

🗣️ "Hệ quả: với **computational social science**, phải validate trên chữ ký chẩn đoán, không phải vẻ ngoài hợp lý. Với **multi-agent LLM systems**, các thể chế hợp tác của con người **không tự động chuyển giao**. Và nếu dùng agent để mô phỏng đàm phán/chính sách khí hậu, ta dễ **dự báo sai** comparative statics — điều mà chính sách thật sự quan tâm."

---

### 🟦 Slide 17/18 — Hạn chế (chung cho cả 2 paper) (trang 22)
🗣️ "Em xin nêu thẳng các hạn chế:"
1. "**Phạm vi model** — chỉ 3 model mở 7–9B, một temperature; chưa kết luận cho model frontier lớn hơn."
2. "**Confound thiết kế** — CRSD giấu tổng tích lũy (self-summation); PGG so sánh between-group trong khi người là within-subject."
3. "**Cross-language** — lỗi định dạng không đều (PGG: Gemma/Arabic tới 31–43%), nên một số ô mẫu nhỏ hơn."
4. "**Persona tối giản** (PGG) — chỉ nhãn thành phố, chưa thử persona giàu chuẩn mực."
5. "**Prompt-sensitive & alignment training** — RLHF nhiều khả năng thổi phồng hợp tác và làm phẳng trừng phạt."

📌 Trình bày hạn chế một cách tự tin = thể hiện hiểu sâu, không phải điểm yếu.

---

### 🟦 Slide 18/18 — Kết luận (trang 23)
🗣️ "Kết luận: hai thí nghiệm hợp tác kinh điển, **một kết luận** — open LLM agents tái tạo **bề mặt** nhưng **mất cấu trúc chẩn đoán**. CRSD: hợp tác cao nhưng **vô cảm với rủi ro thảm họa**. PGG: trừng phạt tồn tại (đa phần antisocial) nhưng **không nâng hợp tác, không cấu trúc văn hóa**. Test validity đúng là **phản ứng với manipulation** — và trên đó, các agent này **chưa thay thế được con người**."

🗣️ "Hướng tiếp theo: thử **model frontier** xem chữ ký có hồi phục theo scale; **persona giàu chuẩn mực** và cho phép giao tiếp/reciprocity; **tách risk-response khỏi running-total** ở CRSD; và agent huấn luyện bằng **RL** ngay trong game. Em cảm ơn thầy và rất mong nhận góp ý ạ."

---

### 🟦 Phụ lục (trang 24, HÌNH `fig_alignment_posthoc`) — *chỉ mở nếu thầy hỏi sâu*
🗣️ "Em có một hình so sánh trực tiếp người (đường đen) với 3 model ở 4 góc nhìn:"
📊
- **Conditional punishment** (trên trái): mức phạt theo độ lệch — chỉ Gemma bám hình dạng người.
- **Contribution trajectories** (trên phải): quỹ đạo đóng góp — model không có cú nâng nhờ punishment như người.
- **City alignment** (dưới trái): điểm model lệch khỏi đường chéo → không khớp thứ hạng người.
- **Next-period reaction** (dưới phải): phản ứng ở vòng sau khi *bị* phạt — khác người.

---

## PHỤ LỤC A — GIẢI THÍCH CHI TIẾT CÁC PHƯƠNG PHÁP THỐNG KÊ
*(Để trả lời khi thầy hỏi sâu. Mỗi mục: dùng để làm gì → tính thế nào → đọc kết quả.)*

### 1. One-way ANOVA (F-test) — *CRSD, độ nhạy rủi ro*
- **Để làm gì:** kiểm trung bình của ≥3 nhóm có khác nhau không (3 mức rủi ro).
- **Cách tính:** F = MS_between / MS_within = (phương sai giữa các nhóm) / (phương sai trong nhóm). df = (k−1, N−k) = (2, 27).
- **Đọc:** F lớn + p nhỏ → các mức rủi ro cho đầu tư khác nhau. Người F=13.78 (p<0.0001); model F=1.76/2.23/3.64 (gần phẳng). Kèm **η² = SS_between/SS_total** = % phương sai do rủi ro giải thích (người 0.51; model 0.12–0.21).

### 2. TOST — kiểm định tương đương — *chứng minh "null" tích cực*
- **Để làm gì:** chứng minh hiệu ứng **nhỏ tới mức không đáng kể**, thay vì chỉ 'không bác bỏ được'.
- **Cách tính:** đặt **SESOI** (ngưỡng nhỏ nhất đáng quan tâm): CRSD ±€22.6 (½ mức giảm người), PGG ±2.14 token (½ mức tăng người). Chạy 2 kiểm định một phía: H₀ rằng hiệu ứng ≤ −SESOI, và ≥ +SESOI. Nếu **bác bỏ cả hai** → kết luận tương đương.
- **Đọc:** cả hai p có ý nghĩa (≤ 0.05) → hiệu ứng nằm trong dải bỏ qua được. CRSD p≤0.014; PGG p=0.010/0.002.

### 3. Bayes factor BF₀₁ — *cân nhắc bằng chứng cho/ngược null*
- **Để làm gì:** so trực tiếp 'không hiệu ứng' (H₀) với 'có hiệu ứng' (H₁).
- **Cách tính:** BF₀₁ = P(data|H₀)/P(data|H₁). CRSD dùng **xấp xỉ BIC**; PGG dùng **JZS** (Jeffreys–Zellner–Siow, prior chuẩn cho t-test).
- **Đọc (thang Jeffreys):** 1–3 yếu, 3–10 vừa, 10–30 mạnh (cho H₀). **BF₀₁ < 1 = bằng chứng *ngược* lại, ủng hộ H₁.** CRSD: 4.8/3.0 (ủng hộ null), Llama 0.8 (equivocal). PGG: 2.3/2.5 (Qwen/Llama); Gemma 1.8×10⁻⁴ = áp đảo *có* hiệu ứng (hại).

### 4. Welch's t-test — *so 2 trung bình, phương sai khác nhau*
- **Dùng:** so tổng tiền LLM vs người (CRSD).
- **Khác t-test thường:** không giả định 2 nhóm cùng phương sai (an toàn hơn). t lớn + p nhỏ → 2 trung bình khác nhau.

### 5. Fisher's exact test — *so tỉ lệ trên bảng 2×2 nhỏ*
- **Dùng:** so tỉ lệ đạt target (vd 100% model vs 50% người).
- **Cách tính:** tính *chính xác* xác suất của bảng đếm 2×2 (không xấp xỉ chuẩn) — phù hợp cỡ mẫu nhỏ.

### 6. Mann–Whitney U — *so 2 nhóm, phi tham số*
- **Dùng:** so đóng góp P vs N (PGG), mỗi nhóm 10 giá trị.
- **Cách tính:** gộp 2 nhóm, xếp hạng, xét tổng hạng của một nhóm lệch bao nhiêu so với kỳ vọng. Không cần phân phối chuẩn. Gemma p<0.001.

### 7. Spearman ρ — *tương quan hạng (đơn điệu)*
- **Dùng:** antisocial vs hợp tác qua 16 xã hội; xếp hạng LLM vs người.
- **Cách tính:** Pearson trên **hạng** thay vì giá trị. Bền với ngoại lệ/phi tuyến. Người −0.90; model ≈0.

### 8. Fisher z-test (Fieller-corrected) — *so một ρ với giá trị tham chiếu*
- **Dùng:** ρ của model có khác −0.90 của người không.
- **Cách tính:** z = arctanh(ρ) → xấp xỉ chuẩn, SE = 1/√(n−3); so sánh hai z. p≤0.0011.

### 9. Hedges g — *cỡ hiệu ứng chuẩn hóa (bias-corrected)*
- **Dùng:** đo độ lớn chênh lệch (kèm CI 95%).
- **Cách tính:** g = (chênh trung bình)/(SD gộp) × hệ số hiệu chỉnh mẫu nhỏ. g rất lớn (vd ~9.7) = hai phân phối tách hẳn nhau.

### 10. Holm–Bonferroni — *kiểm soát sai số khi test nhiều lần*
- **Dùng:** khi chạy nhiều kiểm định (vd 9 ô baseline), tránh dương tính giả tích lũy.
- **Cách tính:** sắp p tăng dần, so p nhỏ nhất với α/m, kế tiếp α/(m−1)… Mạnh hơn Bonferroni thường mà vẫn kiểm soát **family-wise error rate**.

### 11. Khái niệm trò chơi
- **Final group total (CRSD):** tổng tiền góp của 6 người × 10 vòng. Target €120, trần €240. Success nếu ≥ €120.
- **fair-sharer (CRSD):** góp ≥ €2/vòng trung bình (≥ €20/10 vòng).
- **MPCR = 0.4 (PGG):** mỗi token góp → 0.4 cho *mỗi* người (×4 = 1.6 cho nhóm; cá nhân chỉ nhận 0.4 → động cơ free-ride).
- **Cost ratio 1:3 (PGG):** trả 1 token để trừ 3 token của mục tiêu.
- **antisocial punishment:** phạt người góp ≥ mình (độ lệch ≥ 0). **antisocial share** = điểm antisocial / tổng điểm phạt.
- **P − N:** chênh đóng góp giữa treatment có và không có trừng phạt.
- **parse-fallback:** tỉ lệ đầu ra không đúng định dạng (chỉ báo chất lượng/công bằng).

---

## PHỤ LỤC B — BẢNG TRA SỐ LIỆU NHANH (đọc khi bị hỏi con số)

**CRSD — tổng cuối (€) [tỉ lệ đạt target], theo p = 90/50/10:**
| | 90% | 50% | 10% |
|---|---|---|---|
| Người | 118.2 [50%] | 92.2 [10%] | 73.0 [0%] |
| Qwen | 183.0 [100%] | 186.6 [100%] | 181.0 [100%] |
| Gemma | 168.2 [100%] | 162.8 [100%] | 158.0 [100%] |
| Llama | 177.4 [100%] | 165.2 [100%] | 170.0 [100%] |

- Risk ANOVA: người **F=13.78, p<0.0001** (η²≈0.51); model **F=1.76 / 2.23 / 3.64** (η² 0.12/0.14/0.21).
- TOST 90→10: chênh +€2.0 / +€10.2 / +€7.4 vs €45.2 người; p ≤ 0.014. BF₀₁ = 4.8 / 3.0 / 0.8.
- Disposition (success): selfish 0% / 0% / 27%; risk-averse 73–93%; neutral/coop ~100%.
- Parse-fallback: EN ≈0%; Gemma/AR 13.7%; Llama/VI 14.8%.

**PGG — đóng góp (của 20) & trừng phạt:**
| | Góp N | Góp P | Δ (P−N) | p | % antisocial |
|---|---|---|---|---|---|
| Người | 8.6 | 12.9 | +4.3 | — | thiểu số (phương Tây) |
| Qwen | 14.26 | 13.90 | −0.36 | 0.71 | 59% |
| Gemma | 13.41 | 9.86 | −3.55 | <0.001 | 55% |
| Llama | 10.67 | 10.75 | +0.08 | 0.91 | 54% |

- TOST (Qwen/Llama): p = 0.010 / 0.002; BF₀₁ = 2.3 / 2.5. Gemma BF₀₁ = 1.8×10⁻⁴ (có hại).
- Antisocial–hợp tác Spearman ρ: người ≈ **−0.90**; model **−0.09 / −0.15 / +0.30** (Fisher-z p ≤ 0.0011).
- Rank ρ (LLM vs người, 16 xã hội): +0.33 / +0.06 / +0.17. Geography: persona chỉ dịch ~1 token.
- Deviation gradient (Gemma): 1.40 (góp bằng) → 2.45 (free-ride nhẹ) → 3.71 (free-ride nặng). Qwen/Llama gần phẳng.
- Parse-fallback (PGG): EN ≤1.8% (góp)/6.8% (phạt); Gemma/AR 31%/43%; Llama/VI 15–19% (góp).

---

*Chúc buổi báo cáo thành công! Mẹo cuối: luyện nói 2–3 lần với đồng hồ; ở mỗi slide "QUYẾT ĐỊNH" (A2, B2) hãy dừng 2 giây sau câu chốt để thầy ngấm.*
