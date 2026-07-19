# Design Review: Badminton Action Recognition Pipeline

*Review date: July 2026. Scope: methodology audit before large-scale labeling investment.
Grounded in (a) 2022–2026 literature on skeleton-based and racket-sport action recognition,
and (b) experiments already run in this repository (leave-one-video-out logistic regression,
MediaPipe visibility analysis, detection-rate measurements).*

---

## 1. Executive Summary

The overall pipeline shape — video → pose → features → labels → classifier — is sound and
matches how the badminton literature works. **But three formulation choices, if not changed
now, will make a large fraction of future labeling effort unusable or misleading:**

1. **The labeling unit is wrong.** You label swing *ranges* and classify *individual frames*.
   The literature almost universally labels the **hit event** (contact instant) and classifies
   a **fixed window around it**. Range boundaries are ambiguous (your own data contains
   duplicate labels for the same swing disagreeing by 1–3 frames), slower to produce
   (two decisions per swing instead of one), and per-frame classification discards the
   temporal structure that *is* the signal.
2. **Player identity is untracked.** MediaPipe Pose is single-person. In match footage with
   two players visible, nothing guarantees the skeleton you extract belongs to the player
   whose swing you labeled — and nothing in `labels.csv` records which player swung. Any
   frame where the tracker locked onto the other player produces a **wrong-by-construction**
   training row that no amount of later modeling can fix.
3. **Motion features are not fps-normalized.** `wrist_displacement` is pixels *per frame*.
   Your two labeled videos run at 24 and 29.97 fps — the same physical swing reads ~25%
   slower per-frame in the 24 fps video. This silently corrupts exactly the feature family
   the classifier leans on most.

Fix those three (plus a handful of cheaper items below) **before** labeling more. None of
them require discarding existing labels — hit events can be derived from your existing
ranges (midpoint ≈ contact), and features can be re-extracted at any time because labels
are frame indices, not features.

The session's experiments support this diagnosis: leave-one-video-out AUC is unstable
(0.46–0.74) and improving pose quality alone did not move it, which is the signature of a
formulation/data problem, not a pose-noise problem.

---

## 2. Current Approach — Strengths

Worth keeping; several choices here are ahead of typical first attempts:

- **Leave-one-video-out evaluation from day one.** Most projects discover camera
  overfitting after months. You measured it immediately. Keep this; extend it to
  leave-one-*player*-out as data grows.
- **Scale-normalized motion** (`wrist_displacement_norm`). The instinct is correct and
  literature-standard; the reference length needs refinement (see §4.2).
- **Human-in-the-loop candidate review** (`find_swing_candidates.py` + `--candidates`
  navigation). This *is* the "weak supervision + human verification" pattern the labeling
  literature recommends — e.g. Netflix's Video Annotator framework
  ([arXiv:2402.06560](https://arxiv.org/abs/2402.06560)) is the same loop with a learned
  ranker. Your upgrade path is incremental, not a rewrite.
- **Honest negative accounting.** `build_dataset.py` refusing to treat unreviewed gaps as
  "not swing" avoids a label-noise trap most frame-level pipelines fall into.
- **Feature caching with staleness checks** (column and frame-count validation — added
  after being bitten once). Extend with a config hash (§7).

---

## 3. Current Approach — Weaknesses

Ordered by cost-if-ignored:

| # | Weakness | Why it's expensive later |
|---|---|---|
| W1 | Per-frame binary classification of range labels | Every hour of range-labeling buys less signal than event labels; per-frame formulation caps achievable accuracy regardless of model |
| W2 | No player identity in tracking or labels | Corrupted rows are undetectable after the fact; relabeling required |
| W3 | Motion features in px/frame, not per-second | Cross-video training learns fps differences, not swings |
| W4 | No stroke taxonomy (dropped early for speed) | Adding shot types later = re-watching every labeled swing again; recording type at label time costs one keystroke |
| W5 | Full-frame single-person MediaPipe on match footage | Hitting-arm landmarks are the *least* reliable (measured: wrist visibility passes 0.5 in only ~10% of frames — motion blur peaks during swings); small/distant subjects degrade further |
| W6 | Shoulder-width as scale reference | Foreshortens with trunk rotation — which peaks mid-swing, injecting swing-correlated noise into the normalizer |
| W7 | Duplicate/overlapping labels in `labels.csv` | Same swing labeled 2× (e.g. frames 274–282 and 276–281); harmless for OR-ed ranges, double-counts once you move to events |
| W8 | Handedness hardcoded (`ARM_SIDE = "RIGHT"`) | Left-handed players either mislabeled or excluded; standard fix (mirroring) is trivial now, painful after features ship |
| W9 | No experiment tracking / config management | Already caused one silent stale-cache bug; results not reproducible as experiments multiply |

---

## 4. Research-Backed Recommendations

### 4.1 Task formulation: hit events + windows (replaces W1, W4)

Every serious badminton dataset and system is stroke-centric, not frame-centric:

- **ShuttleSet** ([KDD 2023](https://dl.acm.org/doi/10.1145/3580305.3599906)), the largest
  public stroke-level badminton dataset (~36k strokes), annotates the **hit instant + one of
  18 shot types + positions** using a computer-aided tool — not frame ranges.
- **BST: Badminton Stroke-type Transformer**
  ([arXiv:2502.21085](https://arxiv.org/html/2502.21085v2)) classifies fixed-length
  windows around detected hits.
- Monocular badminton analysis systems
  ([MDPI Sensors 24(13):4372](https://www.mdpi.com/1424-8220/24/13/4372)) detect hit
  events first, then refine shot type.

**Concrete change to `label_swings.py`:** replace `s`/`e` range marking with a single
keypress at the contact frame, followed by one type key
(`1`=clear, `2`=smash, `3`=drop, `4`=drive, `5`=net, `6`=lift, `7`=serve, `0`=other —
a collapsed version of ShuttleSet's 18 classes; you can refine within classes later, but
you cannot recover types you never recorded). Store `video, player, hit_frame, stroke_type`.

**Windowing:** ±0.5 s around the hit (≈±12–15 frames at your frame rates) is the
literature-typical stroke duration. Classify the window, not the frame. Negative windows:
sample away from hits *within reviewed regions only* (your existing negative-accounting
logic carries over).

**Migration:** existing range labels convert automatically — midpoint of each range is a
good contact estimate (your ranges average ~8 frames long, so worst-case error ≈4 frames,
inside any reasonable window). Dedupe by merging labels whose ranges overlap (W7).

### 4.2 Feature engineering (fixes W3, W6; informed by your own ablation)

What the sports-classification literature consistently finds valuable, mapped to your set:

| Feature family | Verdict | Notes |
|---|---|---|
| Joint angles (elbow, shoulder, wrist, knees) | **Keep** | Standard; you have them |
| Angular *velocity* of those joints | **Add — high value** | A swing is a rate signal; you observed single-frame angles carry almost no cross-video signal. First differences × fps |
| Wrist/limb velocity, normalized | **Keep, fix units** | Multiply by fps → body-lengths/second (W3). Change reference from shoulder width to **torso length** (mid-shoulder→mid-hip): rotation-invariant, unlike shoulder width (W6) |
| Relative joint positions (root-centered) | **Add** | Center on mid-hip, scale by torso length — the standard normalization in pose-classification systems (e.g. MediaPipe's own pose-classification recipe, NVIDIA's [PoseClassificationNet](https://docs.nvidia.com/tao/archive/5.3.0/text/pose_classification/pose_classification.html) preprocessing) |
| Temporal window statistics (mean/std/max/min over window) | **Add** | This is how classical models consume sequences; replaces ad-hoc rolling features |
| Trunk rotation, torso lean, contact height | **Keep** | Badminton-specific, cheap |
| Raw `wrist_x`/`wrist_y` | **Drop from model** (already done) | Camera-position proxies |
| Handedness | **Mirror** left-handed players' keypoints horizontally at extraction (W8) |
| Center of mass, symmetry features | **Skip for now** | Marginal in stroke classification studies; revisit for footwork analysis |
| Pose embeddings / learned features | **Later** | Only competitive with large data |

### 4.3 Pose estimation (fixes W2, W5)

MediaPipe/BlazePose is built for near-field, single-person, front-facing use (fitness apps).
Match footage violates all three assumptions. Known weaknesses you've already measured:
motion blur destroys hitting-arm confidence exactly during swings; small subjects produce
out-of-frame landmark garbage; and the top-down single-person design degrades in
multi-person scenes ([LearnOpenCV comparison](https://learnopencv.com/yolov7-pose-vs-mediapipe-in-human-pose-estimation/),
[MediaPipe vs YOLOv8 evaluation](https://link.springer.com/chapter/10.1007/978-3-032-00232-7_13)).

**Recommendation (in order of effort):**

1. **Minimum viable fix (do before more labeling):** person detection → pick the target
   player's box (court-side heuristic: near court = bottom half) → crop → pose on the crop.
   `ultralytics` is already in your dependencies and already imported (unused) in
   `pose_tracker.py`. This fixes both subject scale (W5) and player identity (W2) in one
   step, and records *which* player each label belongs to.
2. **Better:** replace MediaPipe with a multi-person pose model —
   **YOLO-pose** (one line via ultralytics, 17 COCO keypoints, constant cost regardless of
   person count) or **RTMPose** ([arXiv:2303.07399](https://arxiv.org/pdf/2303.07399),
   via MMPose) which leads accuracy/speed benchmarks. Note COCO models lack MediaPipe's
   hand/finger points — you lose `wrist_angle`'s INDEX landmark; elbow→wrist direction is
   an acceptable substitute, and the sport literature rarely uses finger points.
3. **Evaluate, don't assume:** run both on ~200 frames of your footage and compare wrist
   trajectory smoothness during labeled swings. (A sport-climbing study found
   [ViTPose > MediaPipe > YOLOv8-pose](https://arxiv.org/pdf/2505.12854) for accuracy —
   rankings are domain-dependent.)

2D vs 3D: stick with 2D image coordinates. MediaPipe's z is unreliable at distance, and the
skeleton literature (below) performs well from 2D keypoints.

### 4.4 Temporal modeling — staged by dataset size

The literature is unambiguous that deep graph models need thousands of samples;
[handcrafted features remain more suitable for small datasets](https://arxiv.org/pdf/2012.02970)
(also the consistent finding across GCN surveys). Staged plan:

| Labeled strokes | Model | Rationale |
|---|---|---|
| Now (~30–100) | **Gradient-boosted trees (LightGBM/XGBoost) on window statistics** | Captures feature interactions your logistic regression structurally cannot; robust at tiny n; interpretable feature importances tell you *what to label next* |
| ~300–1000 | **1D temporal CNN or small BiGRU** on per-frame feature sequences | Learns temporal shape directly; still trainable on hundreds of samples with augmentation (mirroring, temporal jitter, speed perturbation, keypoint noise) |
| ~1000+ | **ST-GCN++ / PoseC3D via MMAction2, fine-tuned from NTU-pretrained weights** ([model zoo](https://mmaction2.readthedocs.io/en/latest/model_zoo/skeleton.html)) | Transfer learning sidesteps the data requirement; [PoseC3D is specifically more robust to pose noise and generalizes better cross-dataset than GCNs](https://pubmed.ncbi.nlm.nih.gov/39686219/) |
| Research-grade | BST-style skeleton+shuttlecock transformer | Only with shuttle tracking (§6) |

Skip: SVM/MLP (dominated by GBM at this scale), LSTM-from-scratch (needs more data than
GRU/TCN for no gain), plain ST-GCN from scratch (data-starved).

### 4.5 Dataset design

- **Splits:** by video *and* player, never random frames or random windows — temporal
  neighbors are near-duplicates and random splits leak. (You already do video-level; keep.)
- **Windows:** fixed length ≥1 s context; the common mistake is windows too short to span
  backswing→contact→follow-through.
- **Overlap:** for negatives, non-overlapping or 50% max; never let a window straddle a
  camera cut.
- **Class imbalance:** event-centric sampling mostly solves it (you choose the negative:positive
  ratio at window-sampling time rather than inheriting ~10:1 frame imbalance).
- **Label QA:** second-pass review of a 10% sample; measure self-agreement. Your duplicate
  labels (274–282 vs 276–281) are actually useful — they estimate your own boundary noise
  (±2 frames), which is fine for event labels and terrible for range-boundary learning.
- **Annotation metadata:** record per video: player handedness, camera position, fps.
  Costs nothing now; enables later slicing ("does it fail on left-handers?").

### 4.6 Labeling strategy — maximize strokes per hour

Your candidate-review loop is the right skeleton. Three upgrades, in order of value:

1. **Audio hit detection as the candidate generator.** A racket impact is a sharp broadband
   transient; onset detection finds it with high precision — racket-sport studies report
   [>90–95% hit-detection accuracy from audio](https://www.researchgate.net/publication/224649455_Ball_Hit_Detection_in_Table_Tennis_Games_Based_on_Audio_Analysis)
   ([tennis: audio+visual fusion](https://www.researchgate.net/publication/261119682_Detection_of_ball_hits_in_a_tennis_game_using_audio_and_visual_information)).
   `librosa` is *already in your dependencies*. Audio candidates are far more precise than
   wrist-motion peaks (your current generator fires on footwork), and they localize the
   contact instant — which is now exactly your labeling unit. Caveat: needs decent audio;
   test on 1 minute of your footage first. Also naturally finds the *opponent's* hits —
   which become labeled strokes too once player identity exists (§4.3), roughly doubling
   yield per video.
2. **Uncertainty-ordered review (active learning).** Once a GBM exists, present candidates
   sorted by model uncertainty instead of chronologically —
   [studies report ~50% annotation-time reduction](https://arxiv.org/abs/2402.06560)
   ([semi-supervised AL for video action detection](https://arxiv.org/html/2312.07169)).
   Cheap to add: score candidates, sort the CSV.
3. **Review logging.** Log every candidate *visited* (not just labeled) in
   `label_swings.py`, so confirmed-negative windows accumulate automatically instead of
   being inferred from label positions. Removes the current approximation in
   `build_dataset.py` entirely.

---

## 5. Must Fix Before Labeling More Data

Ordered; items 1–4 change what a "label" means, so doing them after labeling means relabeling.

1. **Switch to hit-event + stroke-type labels** (§4.1). Migrate existing ranges
   (midpoint → event; merge overlapping duplicates). *Effort: ~2–3 h.*
2. **Player identity: detect-and-crop pipeline + `player` field in labels** (§4.3.1).
   *Effort: ~half a day including box-selection heuristic and re-extraction.*
3. **fps-normalize all motion features** (multiply per-frame deltas by fps) and switch the
   scale reference to torso length (§4.2). *Effort: <1 h + feature re-extraction.*
4. **Add review logging to the labeling tool** (§4.6.3). *Effort: ~1 h.*
5. **Audio hit-detection prototype** — validates the biggest labeling accelerant before you
   commit to motion-peak review. *Effort: ~2–3 h to prototype on existing footage.*
6. **Window-based dataset builder + GBM baseline** replacing per-frame logistic regression —
   otherwise you can't measure whether 1–5 helped. *Effort: ~half a day.*

Total: roughly 2–3 focused days, after which every labeled stroke is durable.

## 6. Nice to Improve Later

- **Pose model swap to RTMPose/YOLO-pose** after benchmarking on your footage (§4.3.2) —
  labels survive; only re-extraction needed.
- **Temporal deep model** at ~300+ strokes (§4.4).
- **Shuttlecock tracking** ([TrackNetV3](https://dl.acm.org/doi/10.1145/3595916.3626370))
  — shuttle trajectory + hit direction is the strongest stroke-type signal in recent work
  (BST), and enables rally segmentation. Meaningful engineering; defer.
- **Data augmentation** for the deep-model stage: horizontal mirroring, temporal jitter
  (±2 frames), speed perturbation (0.9–1.1×), Gaussian keypoint noise.
- **Experiment tracking & config:** a `config.yaml` (paths, window size, fps policy,
  thresholds), a `runs.csv` appended by every training run (git hash, config hash, metrics),
  and feature-cache invalidation keyed on extraction-config hash — you have been bitten by
  silent stale caches once already. MLflow/DVC only if collaboration grows.
- **Refine stroke taxonomy** toward ShuttleSet's 18 classes as counts grow.

## 7. Effort vs. Expected Gain

| Change | Effort | Expected gain | Confidence |
|---|---|---|---|
| Event+type labels (1) | 2–3 h | Labeling speed ~2×; removes boundary noise; enables everything below | High — literature-universal formulation |
| Player crop + identity (2) | 0.5 d | Removes silent label corruption; pose quality on small subjects; doubles usable strokes via opponent | High |
| fps + torso normalization (3) | 1 h | Fixes systematic cross-video feature skew | High — arithmetic, not hypothesis |
| Audio hit candidates (5) | 2–3 h | Candidate precision ≫ motion peaks; labeling ~2–5× faster if audio is usable | Medium — depends on your footage's audio |
| Window features + GBM (6) | 0.5 d | First formulation that *can* exceed the current ceiling; interactions captured | High |
| Uncertainty-ordered review | 2 h | Further ~30–50% labeling-time cut | Medium |
| RTMPose/YOLO-pose swap | 1 d | Better keypoints during blur/distance | Medium — benchmark first |
| ST-GCN++/PoseC3D transfer | 2–3 d | SOTA-class accuracy — *only pays off at ~1000+ strokes* | Medium |
| Shuttle tracking | 1–2 wk | Strongest stroke-type feature; rally structure | Medium |

## 8. Key References

**Badminton-specific:**
[ShuttleSet (KDD 2023)](https://dl.acm.org/doi/10.1145/3580305.3599906) ·
[ShuttleSet22 benchmark](https://arxiv.org/html/2306.15664v1) ·
[CoachAI Projects (GitHub)](https://github.com/wywyWang/CoachAI-Projects) ·
[BST: Badminton Stroke-type Transformer](https://arxiv.org/html/2502.21085v2) ·
[Shuttlecock tracking + hit detection fusion (Sensors 2024)](https://www.mdpi.com/1424-8220/24/13/4372) ·
[Shuttlecock hitting event detection](https://arxiv.org/html/2306.10293) ·
[TrackNetV3](https://dl.acm.org/doi/10.1145/3595916.3626370) ·
[Hybrid RGB–skeleton ensemble stroke recognition](https://www.etasr.com/index.php/ETASR/article/view/15586) ·
[LSTM landmark badminton pose classification](https://jeeemi.org/index.php/jeeemi/article/view/488)

**Pose estimation:**
[RTMPose](https://arxiv.org/pdf/2303.07399) ·
[YOLOv7-pose vs MediaPipe](https://learnopencv.com/yolov7-pose-vs-mediapipe-in-human-pose-estimation/) ·
[MediaPipe vs YOLOv8 evaluation](https://link.springer.com/chapter/10.1007/978-3-032-00232-7_13) ·
[Pose model landscape (Roboflow)](https://blog.roboflow.com/best-pose-estimation-models/) ·
[Climbing study: ViTPose > MediaPipe > YOLOv8-pose](https://arxiv.org/pdf/2505.12854)

**Skeleton action recognition:**
[MMAction2 skeleton model zoo (ST-GCN++, PoseC3D, NTU-pretrained)](https://mmaction2.readthedocs.io/en/latest/model_zoo/skeleton.html) ·
[PoseC3D robustness to pose noise](https://pubmed.ncbi.nlm.nih.gov/39686219/) ·
[Handcrafted features for small datasets](https://arxiv.org/pdf/2012.02970) ·
[NVIDIA PoseClassificationNet preprocessing](https://docs.nvidia.com/tao/archive/5.3.0/text/pose_classification/pose_classification.html)

**Audio hit detection:**
[Table tennis ball-hit detection from audio](https://www.researchgate.net/publication/224649455_Ball_Hit_Detection_in_Table_Tennis_Games_Based_on_Audio_Analysis) ·
[Tennis hit detection, audio+visual](https://www.researchgate.net/publication/261119682_Detection_of_ball_hits_in_a_tennis_game_using_audio_and_visual_information)

**Labeling efficiency:**
[Video Annotator: active learning for video classifiers](https://arxiv.org/abs/2402.06560) ·
[Semi-supervised active learning for video action detection](https://arxiv.org/html/2312.07169)
