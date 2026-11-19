# Demo 與輸出索引

從用途找影片。以下連結指向本機 checkout 的 `outputs/`，影片和模型權重在本機產出後才會出現。索引保留原路徑，方便既有腳本重播。

| 我想看 | 目前應該開哪裡 |
|---|---|
| 已認可的完整 scenario 介面 | [D12 夜間影片](../outputs/integrated_demo/D12/B_nighttime_approach/B_nighttime_approach.mp4)；有漸層尾跡、圖示平滑和預測朝向。[D12 全部成品](../outputs/integrated_demo/D12/) |
| 同一場景的白天／夜間異常對照 | [白天](../outputs/context_scenario_demo_history_heading/daytime/daytime_scenario.mp4) 和 [夜間](../outputs/context_scenario_demo_history_heading/nighttime/nighttime_scenario.mp4)；從 t=6 s 開始播放，開場朝向取自前 6 s history。 |
| 用 camera tracker 做軌跡預測 | [Camera demo](../outputs/camera_demo/)；10 個單物件案例，顯示 tracker、預測、reference 三條線。這是 camera 軌跡輸入的 BEV，Unreal 相機原始畫面是另一項素材。 |
| Joshua 的 Unreal 重播候選 | [重播請求與同步時間](unreal_replay_request.md)；[D12 候選影片](../outputs/unreal_replay_shortlist/d12_replay/) 每 2 秒更新預測，以 3 倍速播放。[初篩資料](../outputs/unreal_replay_shortlist/) 和 [舊版單次預測](../outputs/unreal_replay_shortlist/scenario_videos/) 可供比對。 |

## 早期預測展示與報告素材

| 目錄 | 內容 |
|---|---|
| [demo](../outputs/demo/) | M6 的六張靜態預測圖，按爬升、下降、平飛和地面選片。 |
| [demo_rollouts](../outputs/demo_rollouts/) | M6 的原始陣列，包含 BC 預測、constant velocity 和 GT replay，供評估與重畫。 |
| [eval](../outputs/eval/) | M6 的 ADE、FDE、miss rate 和 BC 對 constant velocity 的表格。 |
| [demo_gif](../outputs/demo_gif/) | M7 灰色底圖初版，10 支以飛機為主的預測 GIF。 |
| [demo_gif_v2](../outputs/demo_gif_v2/) | 同批片段的深色底圖與彩色標記版。兩版展示的 rollout 由 GT 誤差挑選，適合看舊版視覺效果。 |
| [quarterly_rollouts](../outputs/quarterly_rollouts/) | 季報用六組水平和垂直預測圖，有 PDF 和 PNG。 |
| [rollout_demo_stills](../outputs/rollout_demo_stills/) | 報告用的 10 張 +30 s 深色地圖靜態圖。 |
| [camera_demo](../outputs/camera_demo/) | 8 個 aircraft、2 個 drone 的 camera-conditioned 預測。看圖用 `gifs/` 或 `scenes/`，重算指標可讀 `rollouts/`。 |

## 完整 scenario 的版本

以下在 [integrated_demo](../outputs/integrated_demo/)；`D` 是實作階段，`v` 是 ZIP 發布包版本。

| 階段 | 變更 |
|---|---|
| D0–D4 | 固定規格、生成模擬 tracker 觀測，第一次跑完整情境；原始情境子目錄 `A_daytime_hobbyist/`、`B_nighttime_approach/` 和 D4 JSON 留在根目錄。 |
| D5 / v1 | 第一次正式整理成離線 viewer、MP4、GIF 與圖片。 |
| D6 / v2 | 分開呈現物件異常和 scenario 判定。 |
| D7 / v3 | 以代表性 futures 顯示 32 筆樣本中的分組比例。 |
| D8 / v4 | 百分比放在路徑終點，所有當下異常物件可顯示分組。 |
| D9 / v5 | 確認入區後維持 High threat。 |
| D10 | 橘色圖示外緣、百分比取整、原因卡和文字配置。 |
| D11 / v6 | 換用 Anchor checkpoint，沿用 D10 畫面。 |
| D12 / v7 | 歷史尾跡漸層、圖示平滑與預測朝向，目前的視覺基準。 |

## 異常與時段實驗

| 目錄 | 內容 |
|---|---|
| [anomaly](../outputs/anomaly/) | M8 的 teacher-forced NLL、token residual 指標及早期異常重播。 |
| [anomaly_state](../outputs/anomaly_state/) | M8b 原版的格點位置／高度、速度與方向分數，含重播。 |
| [anomaly_state_smoothed](../outputs/anomaly_state_smoothed/) | 五個高度分組及速度／方向平滑版。現有整合 demo 的 frozen scorer 從這裡讀取。 |
| [anomaly_state_height100](../outputs/anomaly_state_height100/) | 合併近地高度分組的已保留比較實驗。 |
| [anomaly_demo](../outputs/anomaly_demo/) | 11 個異常案例的獨立展示圖和 GIF。 |
| [anomaly_context_time](../outputs/anomaly_context_time/) | 依 Chicago 時段分組的 density，含比較曲線、資料覆蓋與評估。 |
| [context_flip_examples](../outputs/context_flip_examples/) | 同一狀態在白天／夜間跨過 cutoff 的靜態案例。 |
| [context_flip_animation](../outputs/context_flip_animation/) | 同一條合成 drone 軌跡的局部、左右同步動畫。 |
| [context_scenario_demo](../outputs/context_scenario_demo/) | 上述 drone 改成兩支完整 scenario 影片，保存兩支共用的 forecasts。 |
| [context_scenario_demo_heading_fixed](../outputs/context_scenario_demo_heading_fixed/) | 修正開場朝東的中間版。方向還沒出現時先顯示圓點。 |
| [context_scenario_demo_history_heading](../outputs/context_scenario_demo_history_heading/) | 用 6 s 歷史初始化兩個圖示，完整 scenario 時段對照的首選。 |
| [unreal_replay_shortlist](../outputs/unreal_replay_shortlist/) | 新混合資料候選的時間表、篩選數據、局部圖、rollouts、舊版單次預測影片和新版 D12 重播。 |

## 權重、技術評估與執行紀錄

| 目錄或檔案 | 內容 |
|---|---|
| [air_pre_bc](../outputs/air_pre_bc/) | 早期正式 BC 訓練，含 `epoch_016.ckpt` 和訓練紀錄。 |
| [runway_anchor](../outputs/runway_anchor/) | M10 對位、base20／Anchor 權重、模型比較及 attention 圖。 |
| [demo_smoke](../outputs/demo_smoke/) | M6 demo 開發時的小規模驗收輸出。 |
| [demo_run.log](../outputs/demo_run.log) | M6 demo 執行日誌。 |
| [m5_cpu_dryrun](../outputs/m5_cpu_dryrun/) | M5 CPU 試跑的設定和日誌。 |
| [m5_execution](../outputs/m5_execution/) | M5 執行收據、GPU 資訊和日誌。 |
| [m5_overfit](../outputs/m5_overfit/) | 小批資料的 overfit 驗收。 |
| [m5_smoke](../outputs/m5_smoke/) | M5 smoke 執行，含測試 checkpoint。 |
| [m6_cpu_dryrun](../outputs/m6_cpu_dryrun/) | M6 CPU 試跑的 rollouts、設定及日誌。 |
| [m6_execution](../outputs/m6_execution/) | M6 runner 的日誌和執行鎖。 |

往後的成品可以按用途分入口，例如「完整 scenario」、「camera 軌跡」、「報告素材」和「評估／權重」。遇到舊資料夾是目前腳本的輸入，例如 `anomaly_state_smoothed/` 或 `runway_anchor/training/`，先透過索引連結。每次產出新版時，記錄 scenario、時間區間、checkpoint、預測更新頻率和播放設定，再更新「目前應該看哪支」的連結。
