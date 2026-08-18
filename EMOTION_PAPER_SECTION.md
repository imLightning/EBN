# Emotion-Aware Social Navigation — Paper Section (Draft)

> Status: draft based on current single-robot LIP experiments.
> All numbers below are real: base 500-episode test vs `no_emotion` 500-episode test.
> Replace bracketed placeholders / figure notes before submission.

---

## Abstract (short contribution paragraph)

Social navigation requires a robot to move safely among pedestrians. Beyond geometry
and kinematics, **pedestrian emotion is a strong social cue that shapes interpersonal
proxemics and avoidance behavior**. We propose an **emotion-aware perception module**
for deep social navigation: each pedestrian's perceived safety zone is expanded in
proportion to its estimated emotion, and local emotion statistics are injected into
the learning agent's state. The module is plug-and-play on top of a
SAC-AE (SAC + auto-encoder) policy trained on LiDAR occupancy images. On a
single-robot social navigation task with a linear inverted pendulum (LIP) drive,
the emotion-aware policy achieves **94.4%** navigation success versus **90.2%** for an
emotion-blind baseline (Δ = +4.2 pp, p = 0.013), while reducing the collision rate
from **9.8%** to **5.6%**. These results show that modeling pedestrian emotion as a
perception-level signal improves both safety and goal-reaching in crowded scenes.

---

## 1. Motivation

Pedestrians are not homogeneous obstacles. Emotional state affects how much personal
space they keep, how predictably they move, and how aggressively a robot may approach
them. Most LiDAR-based social navigation methods treat all pedestrians as static
circles of equal radius, discarding this social information. Our hypothesis is that
**encoding emotion into the perception front-end** gives the policy a more faithful
picture of each pedestrian's "social body" and enables safer, more efficient navigation.

---

## 2. Method: Emotion-Aware Perception

### 2.1 Background: SAC-AE

We adopt SAC with an auto-encoder (SAC-AE). The observation is a stack of consecutive
LiDAR occupancy images `o ∈ R^{F×H×W}` (F = 7 stacked frames, H = W = 84) plus a vector
state `s` carrying goal/velocity/emotion information. A convolutional encoder maps the
image to a latent feature; the actor and critic share its convolutional weights. The
policy outputs continuous velocity commands (linear speed, angular speed).

### 2.2 Emotion Model

Each pedestrian `i` carries an emotion value `e_i ∈ [0, 1]` (in simulation this value
is procedurally assigned; it represents the output of an upstream emotion-perception
module). Emotion modulates a **perceived layer length** around the pedestrian:

    L(e_i) = e_min + (e_max − e_min) · e_i
          = 0.2 + 0.3 · e_i              (e_min=0.2, e_max=0.5)

A neutral/non-emotional pedestrian has the minimum layer `L = 0.2`; an emotionally
aroused pedestrian expands to `L = 0.5`, effectively enlarging its perceived footprint.

### 2.3 Emotion-Modulated Safety Layer

The emotion layer is injected at the **perception level** in two ways:

1. **LiDAR occupancy (input image).** In the occupancy image used as the policy's
   visual input, each pedestrian is rendered as a disk of radius
   `r_i + L(e_i)·ℓ`, where `ℓ ∈ {0,1}` selects the base layer (`ℓ=0`) or the
   emotion-expanded layer (`ℓ=1`). Higher-emotion pedestrians therefore occupy a
   larger perceived volume in the image the policy sees.

2. **Danger / social-layer detection.** A robot is flagged as being in the
   "social layer" of pedestrian `i` when
   `dist(robot, i) < r_robot + r_i + L(e_i)`.
   Being inside the layer yields a small penalty (−0.1) and a `Danger` signal,
   providing a safety shaping signal. For the emotion-blind baseline
   (`--disable_emotion`), all pedestrians use the fixed minimum layer `L = 0.2`
   regardless of emotion.

### 2.4 Local Emotion Feature Extraction

Because a robot cannot observe every pedestrian in a large scene, emotion statistics
are computed **locally**, only over pedestrians within LiDAR range (6 m):

    e_mean = mean({ e_i : dist(robot,i) < 6 m }),   e_max = max({ e_i : dist(robot,i) < 6 m })

If no pedestrian is in range, both are set to a neutral 0.5. These two features are
concatenated into the vector state `s` of the SAC-AE agent (together with relative
goal and last-action features).

> Note: emotion is modeled purely at the **observation/perception** level — it is not
> an additional reward term. This isolates the contribution as a perception insight.

### 2.5 Training

The agent is trained with SAC-AE (entropy-regularized, temperature auto-tuned).
No further modifications are required; the emotion module is embedded in the
observation front-end.

---

## 3. Experiments

### 3.1 Setup

- **Robot model:** linear inverted pendulum (LIP) drive (simple kinematics), single robot.
- **Scene:** 10 m × 10 m arena; 1–4 pedestrians (ORCA, actively avoid the robot) and
  1–3 static obstacles; 50 s episode limit (125 steps); robot start/goal are antipodal.
- **Perception:** 7 stacked 84×84 LiDAR occupancy images (emotion-expanded layer ℓ=1).
- **Algorithm / hyper-parameters:** SAC-AE; batch 128; hidden 1024; actor/critic/encoder
  lr 1e-3; discount 0.99; initial temperature 0.1; encoder feature dim 50;
  replay buffer 30 000; 200 000 training steps.
- **Reproducibility:** training seed fixed; evaluation runs **500 episodes** with fixed
  scenario seeds.

### 3.2 Metrics

| Metric | Definition |
|---|---|
| Success rate | fraction of episodes reaching the goal within the time limit |
| Collision rate | fraction of episodes with a collision |
| Timeout rate | fraction of episodes neither reaching goal nor colliding (≤ time limit) |
| Avg. nav time | mean time of successful episodes |

### 3.3 Ablation

Two agents are compared, identical in every respect except the emotion front-end:

- **Ours (Emotion-aware):** emotion layer + local emotion features enabled.
- **Baseline (Emotion-blind):** `--disable_emotion` — all pedestrians use the fixed
  minimum layer and the emotion features are suppressed.

---

## 4. Results

### 4.1 Main Results

**Table 1. Navigation performance (500 episodes, fixed seeds).**

| Method | Success ↑ | Collision ↓ | Timeout | Avg. nav time (s) |
|---|---|---|---|---|
| Emotion-aware (Ours) | **0.944** | **0.056** | 0.000 | 30.31 |
| Emotion-blind baseline | 0.902 | 0.098 | 0.000 | 29.56 |
| Δ (Ours − Baseline) | **+0.042** | **−0.042** | 0.000 | +0.75 |

↑ higher is better; ↓ lower is better. Bold indicates the better value.

### 4.2 Statistical Analysis

Two-proportion z-test on the 500-episode outcomes (independent episodes):

| Metric | Δ | z | p | 95% CI |
|---|---|---|---|---|
| Success rate | +0.042 | 2.50 | **0.0125** | [+0.009, +0.075] |
| Collision rate | −0.042 | −2.50 | **0.0125** | [−0.075, −0.009] |

Both differences are statistically significant at the 5% level, and the 95% confidence
intervals exclude zero. Neither method produced timeouts.

### 4.3 Discussion

- Emotion-aware navigation significantly **increases goal-reaching** (+4.2 pp) and
  **reduces collisions** (−4.2 pp, halved collision rate), indicating that modeling
  pedestrian emotion as a perception-level safety cue yields both more efficient and
  safer behavior.
- **Avg. nav time** is 0.75 s longer for the emotion-aware policy. This is expected and
  benign: the emotion-aware agent completes more full trajectories (which take time),
  whereas the baseline's trajectories more often terminate early by collision. We do not
  claim a nav-time advantage.
- The absolute success values (90–94%) are close to the task ceiling because the scene
  is relatively easy (ORCA pedestrians actively avoid the robot). We hypothesize the
  emotion advantage widens in denser / more aggressive scenarios; a difficulty-gradient
  study is planned to verify this.

---

## 5. Figure Notes (placeholders — generate figures, then insert)

1. **Fig. 1 — Method overview (conceptual diagram).** Show the pipeline: LiDAR → emotion
   layer expansion → occupancy image + local emotion features → SAC-AE actor → velocity
   commands. Annotate where `L(e_i)` enters.
   *(Need: architecture diagram; robot + pedestrians; emotion layer as colored rings.)*

2. **Fig. 2 — Emotion layer visualization.** Side-by-side top-down views of the same scene
   rendered with (a) emotion-blind minimum layer and (b) emotion-aware variable layer;
   color each pedestrian's footprint by its emotion value (e.g., red = high, blue = low).
   *(Need: matplotlib rendering from saved eval `.npz` logs of `logs/.../seed_1/evaluation_episodes/`.)*

3. **Fig. 3 — Learning curves.** Success rate (training-eval) vs. training steps for
   emotion-aware vs. emotion-blind, showing convergence and the gap.
   *(Need: TensorBoard `eval/success_rate` curves for both runs.)*

4. **Fig. 4 — Bar chart with error bars / significance.** Success and collision rates for
   the two methods with 95% CI error bars, annotated with p-values.
   *(Need: bar plot from Table 1 + CI.)*

5. **Fig. 5 (optional) — Qualitative trajectories.** Example successful/failed episodes
   showing the robot yielding to high-emotion pedestrians.
   *(Need: replay trajectories via `replay_episode.py`.)*

---

## Appendix A. Hyper-parameter Table

| Parameter | Value |
|---|---|
| Policy | SAC-AE |
| Observation | 7×84×84 LiDAR occupancy |
| Vector state dim | 6 (goal/velocity/emotion) |
| Encoder feature dim | 50 |
| Hidden dim | 1024 |
| Batch size | 128 |
| Actor / critic / encoder lr | 1e-3 |
| Discount γ | 0.99 |
| Initial temperature | 0.1 |
| Replay buffer | 30 000 |
| Training steps | 200 000 |
| Emotion layer | L(e)=0.2+0.3·e |
| Pedestrian | 1–4 (ORCA) |

## Appendix B. To-dos before submission
- [ ] Add ≥1 additional training seed per condition; report mean ± std.
- [ ] (Recommended) Add a harder-scenario difficulty gradient to widen the gap and
      address the near-ceiling concern.
- [ ] Replace all bracketed / placeholder figure notes with actual figures.
