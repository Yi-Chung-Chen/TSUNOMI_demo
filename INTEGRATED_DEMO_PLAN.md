# Integrated MVP Demo — Approved v1

- 日期：2026-10-29
- 狀態：**2026-10-29 已審閱通過；依原計畫 §4.1.11 與最新指示執行 D0–D5。這輪禁止呼叫 advisor 腳本。**
- 基準：`2822d3a`，目前 `sygard-motion` 工作目錄乾淨。
- 依據：Yi-Chung 提供的 David 回覆，以及本對話確認的整合方向。

## 1. 要交付什麼

做一個可以收尾 project 的完整 demo：

> **帶雜訊的觀測 → 未來軌跡／行進意圖 → 現在與未來的 anomaly → threat level → 原因解釋**

重點是讓人看懂各元件如何合作。Prediction performance 不是這次 demo 的主要目標；800 scenarios 先維持，優先讓訓練與迭代成本可控。**先沿用現有 checkpoint，不重訓。**

觀眾應該看得出：

1. 我們偵測到一個不符合正常 aircraft traffic 的物件。
2. 模型預測它接下來會去哪裡。
3. 這條預測與保護區、時間等 context 有什麼關係。
4. 為什麼是 low／medium／high threat，以及警示何時出現。

**Anomaly 不等於 threat。** Hobbyist drone 可以同時是 high anomaly、low threat。

## 2. 保留的基礎與這次不做的事

| 元件 | 目前狀態／本次做法 |
|---|---|
| Dataset | 800 個五分鐘 aircraft reference scenarios；618 train、182 validation/evaluation。保留原 split。 |
| Trajectory model | 沿用 `epoch_016.ckpt`、既有 motion dictionary；6 s history → 30 s forecast。 |
| Grid detector | 保留五層高度、250 m cells，以及已採用的速度／方向平滑版本；不使用合併 `<100 m` 的實驗版。 |
| Camera adapter | 已完成 8 aircraft＋2 drone 的三路徑 demo，保留作補充展示。 |
| 新的主 demo 輸入 | Reference trajectory＋固定規格的 Gaussian noise，標成 simulated tracker output。 |
| Intent | v1 使用 forecast 和 predicted endpoint／zone relation 表達，不新增 intent head。 |
| Threat | 可檢視的規則系統，不訓練 threat classifier。 |

本次不等待 800 個 camera scenarios、不新增套件、不重建字典、不改模型架構、不追 SOTA，也不新增 weather／operating-condition 模型。保留原始資料、舊成果及 checkpoint。

目前 grid detector 的 context 是**位置／高度與運動狀態**。Day/night 和 protected zones 是本次要新增的 context；不能說天氣等已經接好了。

## 3. 建議的 v1 設計

### 3.1 Simulated tracker input

- Reference position 的 x/y/z 各加獨立、零均值 Gaussian noise，**建議 `sigma = 1 m`／axis**，即 variance = 1 m²／axis。用明確單位記錄 David 提到的 noise。
- 先用最簡單的逐時間點 noise，記錄它不是完整 tracker error model；不假裝具有真實 camera 的遮擋、ID switch 或相關誤差。
- 以 source／object／frame 和 seed 固定 noise。同一段觀測在重疊視窗中必須相同，不能每次 forecast 都重新抽一份過去。
- Observation stream 先生成、保存；推論每次只讀當下以前的 6 s。未來 reference 只供播放比較、檢查警示時序與計算誤差。
- Heading、速度、ground proxy 等衍生輸入從 noisy observations 因果估計，不偷用 clean heading／ground flags；type 統一 `UNK`，與 camera demo 接法一致。
- 重用既有單位轉換與 camera adapter 的 ground hysteresis（<10 m 進入地面、>20 m 離開），標示為 demo proxy。
- 提供 `sigma=0` 對照，讓 noise 對輸入及預測的影響可檢查。v1 不因畫面不好看而暗中平滑或修正預測。

### 3.2 Forecast 與 intent

- 以 3 Hz observation stream，**建議每 2 s 更新一次 forecast**。
- 每次使用 6 s history，產生 **32 條 30 s rollout**，沿用 top-k 5、temperature 1、固定 seed 規則。
- 主畫面顯示 32 條 joint rollout 的 3D medoid（與其他條平均距離最小）；可選擇淡化顯示其餘 rollout。**不使用 minADE、未來 GT 或事後挑選來決定展示哪條。**
- Intent 顯示為模型輸出的解讀，例如 `Remaining outside demo zone`、`Approaching demo zone`、`Forecast enters demo zone`，並畫出 predicted endpoint。
- 不宣稱識別操作者的真實意圖，也不新增 endpoint pseudo-label training。獨立 intent head 留待後續。
- 有 32 條時，zone relation／warning 用全部樣本統計，不讓畫面上的單一 sample 代表所有預測。

### 3.3 現在與未來的 anomaly

直接重用已凍結的 state-density detector：

`state NLL = location/height NLL + speed NLL + course NLL`

- Current score：由當下 noisy observation 與過去 2 s 位移計算。
- Future score：對每條 rollout 的未來狀態計算；v1 取 +2、+4、…、+30 s，速度／方向由該預測與已觀測的歷史接續計算，不能讀未來 GT。
- 沿用五層高度 smoothed baseline 的既有 state-score scaling／cutoff；來源和數值寫入 manifest，**不在這兩個故事場景重新校準**。使用 state NLL 的 cutoff；若顯示標準化 score，cutoff 要一起換到相同尺度，不能拿 combined cutoff 直接套用。
- Future score 是在「假設預測成真」時的 state plausibility，不等於真實未來已經異常。
- 顯示 current score、各 horizon 的 forecast score 摘要，以及 earliest forecast horizon exceeding cutoff。
- 建議只有某 horizon 至少 8/32 samples 超過 cutoff 才顯示 future anomaly warning；這個比例是待審的 demo 設定，不是校準過的機率。
- 現有 combined score 可保留作舊成果；本次先用 **state score** 串接 current／future anomaly，避免把 teacher-forced future-token NLL 當成當下已知的 online score。

### 3.4 Demo protected zones 與 context

- Zone 用 config 定義 ENU polygon＋height range。畫面顯示地面 footprint，判定時同時檢查高度，避免飛過建築上方就一律當成侵入。
- 先做一個 **Demo protected zone**；不宣稱是機場真實法規／禁航區，也不宣稱對應已確認的敏感建築。
- Context v1 只有明確指定的 daytime／nighttime。若改用 reference replay 呈現夜間故事，標示為 **scripted demo context**，不能把改寫時間說成原錄製 metadata。
- 僅有 reference＋noise 時，夜間是 threat-rule context，不是假裝已量測到夜間 camera 品質；v1 不因為 night label 就憑空加偵測漏失。
- 不填入沒有資料來源的 weather、授權狀態或 operating conditions。

### 3.5 Rule-based threat level（建議草案）

先用每個物件最近 10 s 的 current state scores 判定：任一有效時刻嚴格超過凍結 cutoff，就 latch 為 anomalous。只有 anomalous 物件計算下表；其餘顯示 `Normal traffic`，threat score=null（背景飛機同樣適用）。Future score 不可開啟 current anomaly latch。

保留每條 rule 的貢獻與 reason；分數不是風險機率。

| Evidence | Base points |
|---|---:|
| 當下已在 demo zone volume 內 | 80 |
| 目前在外面，至少 8/32 rollout 預測 30 s 內會進入 | 60 |
| 只有 1–7/32 rollout 預測會進入 | 30 |
| 無 predicted entry，但距 zone <500 m 且以 >2 m/s 接近 | 20 |
| 其餘 | 0 |

上表的 zone evidence **只取最高一項**，不重複累加。

Context modifiers：

- 不加 current anomaly 分數；anomaly 是進入 threat 計算的必要條件。
- Nighttime：+10，**只在已有 zone evidence 時加分**。
- 明顯加速、且朝 zone 接近：+10，只在已有 zone evidence 時加分。v1 建議由最近兩個 2 s 速度窗增加 >5 m/s 定義，不用未來真值。
- Clamp 到 0–100；`Low <30`、`Medium 30–59`、`High ≥60`。

因此只有 anomaly 或只有 nighttime 不會自動變成 high threat；預測侵入則提供主要 threat evidence。所有數值都在一個 config，標示為 **heuristic demo thresholds**，不是經獨立資料驗證的安全系統。

顯示 entering-rollout count／fraction、最早預測進入時間及不確定性。只有部分 rollout 進入時，使用 `Possible entry`／`Entry in sampled forecasts`，不把其中最早一條說成確定的 ETA。

缺少可用 history／forecast 時，顯示 `Insufficient data`，不能因為沒有證據就直接顯示 low threat。

### 3.6 Explanation

使用可追溯的模板，先不接額外 LLM：

- Current anomaly：哪個 location/height、speed、course 項貢獻較大。
- Forecast evidence：例如 `12/32 sampled futures enter the demo zone within 30 s`。
- Context evidence：例如 `Nighttime demo context`。
- Threat rule：列出實際觸發項與分數，不混成「NLL 高，所以危險」。

區分 current facts、model forecasts、scripted context、rule conclusions。不要將分數分解稱為因果證明，也不要編造物件是 hobbyist／惡意操作者的辨識結果。

## 4. 兩個故事場景

### A. Daytime hobbyist

- 明確標示 scripted hobbyist scenario：drone 在 demo zone 外活動，不朝 zone 前進。
- 可以是高 anomaly，但依 current／forecast zone relation 為 low threat。
- 顯示 intent：預測仍留在區外；解釋：速度／方向不像正常 aircraft，但沒有預測侵入。

### B. Nighttime approach

- Scripted drone 起初在區外活動，之後加速／改向、朝 demo zone 移動。
- 目標是展示：觀測到運動變化後，forecast 提供進入 zone 的 evidence，threat 由 low／medium 升到 high。
- 時間軸標記 motion change、first current anomaly、first future anomaly warning、first high threat，以及 reference zone entry（若有）。
- **不要求模型在沒有歷史線索時預知突然加速。** Future anomaly 也不保證早於 current anomaly；anomaly 與 protected-zone warning 分開驗收。

共同要求：

- 可使用現有 native drone route，或自建合理的 synthetic route；來源、waypoints、速度／加速度和 context 都留在 config／manifest，合成 route 不加入正常訓練集。
- Scenario A/B 是具名、人工設計的 qualitative demo，不是新的獨立 test set。
- 若加正常 aircraft 作背景，保留其原始 route；盡量沿用 joint inference，並確認真實不同物件不會因 tracker ID 重複而互相注意。
- 不將 synthetic drone 接進原始 native camera run 並聲稱它曾被 camera 觀測。
- 若模型對 B 預測失敗，保留該結果並回報；不能用 GT waypoint、CV 或手畫路徑替換後仍稱為模型 forecast。任何調整 scenario 的過程另記錄，不包裝成無偏評估。

## 5. 畫面與輸出

沿用目前已確認的 **深色地圖＋飛機／drone icon**，不以科學圖表 overview 取代 demo。

主畫面包含：

- Observation path、不使用 GT 的 medoid forecast、predicted endpoint、demo zone footprint。
- Object info card：`Anomaly score`、`Threat level`、`Predicted intent`、entry evidence／time。
- 簡短 explanation，最多列 2–3 個主要原因；原始數值保存在旁邊的 CSV／JSON。
- Demo context（日／夜）、當下時間、資料來源標籤。
- GT／reference 未來可作明確標示的 offline comparison toggle；不作預警輸入，正常播放時不要提前露出尚未發生的 reference 路徑。

每個場景交付：

1. 可重播 GIF；如現有工具可直接輸出 MP4，再提供 MP4，不為此裝新套件。
2. 代表時刻的獨立高解析 PNG：初始、motion change 附近、警示、結束。
3. 每個 observation／forecast 時刻的 scores、rule evidence、rollout arrays 和 manifest。
4. 一段可放報告的英文 caption，清楚註明 noisy-reference、scripted context、heuristic threat rules。

另提供一個離線可開的 demo viewer（play/pause、時間滑桿、GT／其他 rollout 顯示切換）；優先沿用現有工具或無新增依賴的輸出。若 viewer 成本影響交付，先以 GIF＋PNG 完成 v1。

## 6. 實作順序與 Done-when

每階段通過才往下；每個 milestone 做本機 commit，結果寫 `PROGRESS.md`。輸出放獨立 `outputs/integrated_demo/`，不覆寫先前成果。

### D0 — 固定範圍與 demo config

- 固定 noise／seed、模型／density artifact、zone geometry、context schema、rule thresholds、scenario source。
- 列出資料與模型 checksum；確認目前 repo 沒有未處理修改。
- 審閱通過後再同步新 scope 到原 `SYGARD_MOTION_PLAN.md`／`AGENTS.md`，保留舊 milestone 的歷史紀錄；顧問需讀到此份新 plan，不能只沿用舊 M8b 範圍。

**Done when：** 這份 plan 的待確認項已確認；config 可以描述兩個故事，並清楚區分 synthetic／reference／camera 資料。

### D1 — Noisy observation → forecast

- 實作可重現 observation stream 與模型 adapter，接凍結 checkpoint。
- 使用既有正常 aircraft 片段與 drone 片段各做 smoke run，保存 32 rollouts。

**Done when：** sigma=0 在同一 adapter 下還原 reference positions 並可重現 forecast；重疊視窗的 noise 一致；修改／刪除預測起點後的資料不改變當次 forecast；模型、字典、原資料 checksum 不變。

### D2 — Current／future anomaly 與 intent evidence

- 重用 state scorer、固定 scaling／cutoff，對 current 和 forecast states 計算。
- 輸出 per-factor、per-horizon、per-rollout evidence；產生 endpoint／zone relation。

**Done when：** 與現有 scorer 在相同輸入上結果一致；未來速度／方向只依 history＋該 rollout；單位、缺值和時間對齊測試通過；不讀 future GT、不重新 fit histogram。

### D3 — Threat rules 與 explanation

- 實作 polygon＋高度判定、forecast entry、closing／acceleration evidence、context modifiers 和 explanation。

**Done when：** 可手算的例子通過：異常但遠離區域=low；只有 nighttime 不會變 high；current／forecast entry 有對應高警示；缺資料=unknown；每個 explanation 都能對應到實際 evidence／rule。

### D4 — 兩個整合故事

- 用同一套 frozen inference/scoring/rules 跑 A/B，保存完整時序與預測。
- 檢查 forecast 在 B 是否實際提供 entry warning，並量出相對 reference entry 的 lead time。

**Done when：** A 呈現 anomaly 與 low threat 可以並存；B 至少一次在 reference entry 前有 model-based high warning，**若沒做到則回報限制、討論調整後才往下**。不為了通過而挑 best-of rollout 或塞入 GT forecast。

### D5 — Demo 成品

- 深色地圖播放、兩組 GIF／PNG、簡短 caption、說明檔與可重播指令。

**Done when：** 自己看過兩段完整播放；來源／context／rule 標示清楚；70 個既有測試及新增必要測試通過；受保護檔案不變；存下數值驗收、commit 並交 Yi-Chung 檢閱。

## 7. 審閱時優先確認

- [x] 主 demo 用 reference＋`sigma=1 m`／axis noise，現有 camera case 當補充。
- [x] v1 不重訓；每 2 s 用 32 rollout 更新 evidence，畫面顯示不看 GT 的 joint medoid。
- [x] Intent 先用 forecast／endpoint／zone relation，不新增獨立 head。
- [x] Protected zone 為明確標示的示意 polygon＋height range；day/night 可使用 scripted context。
- [x] 接受 §3.5 的 rule／threshold 草案作 demo 起點，後續調整明確記錄。
- [x] 先完成兩個故事的 GIF＋PNG；interactive viewer 視交付成本再加。

---

## 現有實作參考

- `src/rollout/closed_loop.py`：6 s history 的 causal rollout。
- `src/data/camera_trajectory.py`：camera input 對齊、heading／ground proxy。
- `src/eval/anomaly_state.py`：五層 smoothed state density 與分數。
- `src/eval/camera_rollout.py`、`src/eval/demo_gif.py`：既有深色地圖視覺。
- `outputs/anomaly_state_smoothed/`：應重用的 density、standardization、scores；需固定其既有 state cutoff 來源。
- `docs/camera_demo_acceptance.json`：目前 camera demo 的 70-test 驗收基準。

## D0 固定的四項取捨

1. 不做 time-conditioned density；沿用 frozen baseline。時段只在 scripted threat context，caption 不宣稱 detector 有時段／天氣特徵。
2. 採 joint 3D medoid；不看 GT。
3. 記錄 CV 數值對照；不畫、不替換 model forecast。
4. Day sigma=1m，night sigma=3m，各 axis；明確標示腳本設定。

參數與 synthetic routes／zone 在 configs/integrated_demo/v1.json 固定，D4 第一次驗收後不得自行修改以求通過。

## D6 — Approved presentation revision (2026-11-01)

Latest user review overrides the v1 10s latch and numeric threat rules for the
active demo; D0–D5 evidence remains preserved as historical acceptance.
- Object anomaly uses only the current state score > frozen cutoff, with an
  orange outline and a descriptive leading-factor reason. No temporal latch.
- Scenario status aggregates same-object evidence: Normal traffic; Unusual
  activity; Potential intrusion (currently anomalous + >=8/32 forecasts enter).
  An anomalous object already inside is Intrusion detected. Unknown observations
  and unavailable forecasts must be stated, never reported as reassuring evidence.
- Remove standalone forecast/intent and future-anomaly cards and numeric threat
  scores. Keep the actual model forecast, GT-free joint medoid and optional32
  samples. Day/night no longer adds points; retain scripted provenance labels.
- Read existing D4 observations/scores/forecasts only, no new inference/fitting.
  Export separately to outputs/integrated_demo/D6 and integrated_demo_v2.zip.
- Acceptance: current-only and same-object tests; no GT-future visual leakage;
  review both new scene media; frozen hashes and D4 files unchanged; local commit.
  Replay command: scripts/catk.sh -m scripts.integrated_d6 (d5 is a compatibility
  alias for this latest exporter, preserving the existing D5 output directory).

## D7 — Approved sampled-future groups preview (2026-11-01)

User approved a preview with representative futures and unequal sample shares.
Use saved32rollouts only. For the foreground drone, partition by whether each
remaining full3D trajectory ever enters the protected volume, then cluster by
mean pointwise3D distance with deterministic medoids, up to4representatives.
Each displayed path is an actual member; its weight is group_count/32. Exact
sample-share percentages sum to100%, not likelihood softmax or calibrated risk.
Identical paths may yield fewer groups. These are display groups, not learned
intent classes. Allocate slots proportionally, minimum one per populated entry
stratum. Red means all group members enter; amber means none do. Overall entry
share and scenario status still use all32 original samples, not representatives.
Recompute groups for remaining forecast between2s model updates, so past entries
are not counted as future entries. Labels A–D are local, not persistent identities.
Background aircraft retain a single medoid to keep the map readable. Keep the
viewer option showing all32 raw model samples and reference overlay (defaultoff).
No anomaly/threat/cutoff/zone/model/seed changes. Preserve D6/v2. New output:
outputs/integrated_demo/D7, integrated_demo_v3.zip. Replay:
scripts/catk.sh -m scripts.integrated_d7.
Acceptance: exact memberships/weights, medoid/sample identity, full-path and
height-aware entry, all32entry consistency, noGTfuture leakage, visual review,
unchanged frozen/D4 files and prior media, tests and local commit.

## D8 — Endpoint shares for every anomalous object (2026-11-01)

User feedback removes A–D badges and the entry/outside legend. Put group sample
percentages directly beside their endpoints, with short leaders and collision
avoidance. Apply the same up-to4 grouping independently to EVERY currently
anomalous object, including aircraft; weights sum1 per object, not per scene.
Normal objects retain a single medoid. Keep per-object zone-entry shares in the
object status card; remove the foreground-only entry summary from scenario card.
No current anomaly/entry/scenario rule changes. No artificial path bending,
forecast replacement, inference, fitting, new packages, network or advisor calls.
Audit saved raw model curvature to explain straight-looking futures. Draw all
remaining forecast vertices, not an endpoint chord. Preserve all prior versions;
export D8/v4 with an additional aircraft-anomaly snapshot per scene. Command:
scripts/catk.sh -m scripts.integrated_d8.
Acceptance: both objects' memberships/weights and entry shares verified against
original32samples; percentage-only labels and aircraft path geometry tests;
noGTfuture leakage; visual review/media/ZIP checks; frozen/D4 files unchanged;
local commit, report model limitations rather than cosmetic trajectory changes.

## D9 — Confirmed intrusion incident memory (2026-11-01)

User requests persistent high threat once intrusion has occurred. Add scenario
incident memory, separate from current-only object anomaly. Trigger only when a
currently anomalous object's observed position is inside the protected3D volume;
normal traffic or forecast-only entry cannot latch it. Capture object ID and
first observed intrusion time. Scenario stays High threat through exit, current
anomaly clearing or missing observations, until a new scenario/reset. Reasons
state historical entry time and current inside/exited/unknown position. Orange
boxes and multi-rollout eligibility remain strictly current-only. No numeric
threat score or model/score/forecast/zone changes. Historical D6–D8 remain intact.
Export D9/v5 from saved D4 only; command scripts/catk.sh -m scripts.integrated_d9.
Acceptance: entry/exit/missing data, forecast-only/normal-traffic exclusion,
causal prefix and scenario reset tests; B persists High through final frame;
A unaffected, all original anomaly/forecast evidence unchanged; media/ZIP review,
PROGRESS.md/QUESTIONS.md and local commit. No advisor/inference/training/network.

## D10 — Visual cleanup review (2026-11-01)

User-authorized presentation only. Read accepted D4 and assert exact D9 timeline,
groups/raw shares and map pixel bounds. No model, score/cutoff/rule/zone/extent
changes. Remove orange anomaly Rectangles; rotate the original icon with a
thicker orange marker edge (normal edge unchanged). Map object labels are names
only, orange when anomalous. All group percentages use deterministic
largest-remainder integer rounding and every group is shown, including0% if it
occurs; raw weights/grouping stay intact. WHY retains scenario reasons;
OBJECT STATUS is one physical line per object with state and raw zone-entry%.
Move DEMO PROTECTED ZONE outside the polygon, reserve it in label layout, give
an opaque backing and draw it last. Use measured label bounds for collision
checks, with no map reframing. Output separately to outputs/integrated_demo/D10;
retain prior versions. Replay scripts/catk.sh -m scripts.integrated_d10.
Review draft exports MP4/GIF/PNG and numeric layout/render receipts; existing
interactive D9 is untouched. Required review frames: B56s(frame168), B63.33s(190),
A62.33s(187). Compare against existing D8 PNGs at exactly those frames and save
three side-by-side comparison PNGs. Check all frames for label/zone clarity,
frozen hashes and prior-media equality; commit and STOP for Yi-Chung's review.
No advisor, inference, fitting, training, packages, network or further revision.

## D11 — Runway-anchor checkpoint in the demo (2026-11-02)

Yi-Chung approved (parent plan §4.1 item 13) replacing `epoch_016.ckpt` with the
M10 anchor checkpoint `outputs/runway_anchor/training/anchor/checkpoints/epoch_018.ckpt`
(best 10-03 val loss, runway_anchors=true). No extra seeds, no training.

What changes and what must not:
- Only the forecasts change: the 32 rollouts per forecast time, medoid, groups,
  endpoint shares, zone-entry counts, and the scenario statuses derived from them
  ("Potential intrusion" uses >=8/32 entry).
- The object anomaly score is the frozen M8b smoothed state density
  (`outputs/anomaly_state_smoothed`), not the network. Observations (same seed and
  noise), current state scores, cutoff, orange-marker decisions and the D9 observed
  intrusion latch must therefore be identical to D4/D10; assert this exactly.
  Any difference is a bug, not a result.
- Same config otherwise: seed 20261029, noise, routes, background segment, zone,
  rules, 6 s history / 30 s forecast / 2 s stride, 32 rollouts, top_k 5, temp 1.

Steps:
1. Add `configs/integrated_demo/v2_anchor.json`: a copy of `v1.json` with only
   `version` and `checkpoint` changed. Do not edit `v1.json`.
2. Let `scripts.integrated_d4` take `--config` and `--output-dir` (defaults keep
   current behaviour and paths byte-for-byte). `load_model` must build the model
   from the checkpoint's own hyperparameters (`runway_anchors`, anchor CSV) and
   strict-load it. Write the new run to `outputs/integrated_demo/D11/D4_anchor/`;
   never overwrite `outputs/integrated_demo/D4*`, `A_*`, `B_*` or `docs/integrated_demo_D4.json`.
3. Inference on CPU, or one job on GPU 1 or 3 after checking `nvidia-smi` (outside
   the sandbox). Anchor rollout is about 2.3x slower on CPU than epoch_016.
4. Re-check D4's acceptance on the new run and report it honestly: A shows anomaly
   without intrusion; B has a model-based high warning before the reference entry,
   with lead time vs the epoch_016 run (old D4). If B fails, still finish the media,
   state it in the README and PROGRESS.md, and do not tune anything to pass.
5. Let `scripts.integrated_d10` read a given D4 run directory; render the D10
   `readable_status` style unchanged from `D11/D4_anchor` into
   `outputs/integrated_demo/D11/` (MP4/GIF/PNG, same map extents). Save
   side-by-side comparisons D10 (epoch_016) vs D11 (anchor) at B56s (frame 168),
   B63.33s (190), A62.33s (187), plus the frame where B's first high warning occurs.
6. Write `outputs/integrated_demo/D11/comparison.md`: per scenario, old vs new first
   Potential-intrusion time, lead time before reference entry, zone-entry share
   over time (small plot or table), background-aircraft forecast change, and the
   asserted-identical anomaly/latch timelines. Caption states the checkpoint now uses
   16 runway-end anchors.
7. Package `outputs/integrated_demo/integrated_demo_v6.zip` (D11 media + README +
   caption + comparison). The interactive viewer stays the D9 one; do not rebuild it.

Done when: identical observation/anomaly/latch assertions pass; new and existing
tests pass; protected hashes (incl. epoch_016 and all earlier demo outputs)
unchanged; media self-reviewed; PROGRESS.md updated; local commit; STOP for
Yi-Chung's review. No advisor calls, packages, network, training or rule changes.

## D12 — Tracker history and display smoothing (2026-11-03)

Yi-Chung approved fading recent10s raw history over a faint full history,
causal display-only icon position smoothing and forecast-based icon heading.
Read D11 anchor data only, retain exact scores/forecast groups/threat timeline
and map bounds. Position: past7samples line fit at now; heading: next~2s joint
medoid tangent with0.6s circular easing, filtered-track fallback before6s.
Export D12/v7 separately, test causality/rendering, visually review, commit.
No model execution, training, advisor, packages, network or historical edits.
