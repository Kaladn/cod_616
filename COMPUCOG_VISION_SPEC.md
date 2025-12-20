# CompuCog Vision - Mathematical Specification

**Version:** 1.0  
**Date:** 2025-11-26  
**Status:** Design Phase - Ready for Implementation

## Philosophy

No ML bullshit. Pure signal processing. Frame-to-frame temporal analysis of screen patterns, input timing, and network behavior. The fusion engine works on **motion**, not classification. COD is not a real-world dataset. COCO is not a gaming dataset.

YOLO was two things: a distraction and a bottleneck waiting to happen. This system uses pure math to reveal manipulation patterns through tri-modal correlation: screen + gamepad + network.

---

## 1. Per-Frame Visual Features (ScreenResonanceState)

### Input State per Frame

- `grid_t` - Normalized 10×10 intensity grid (0-1) from current frame
- `prev_grid` - Previous frame's grid
- `ema_fast` - Fast exponential moving average of grid (α=0.3)
- `ema_slow` - Slow exponential moving average of grid (α=0.1)

### The 20 Visual Features

#### A. Core Motion & Energy (5 features)

**1. vis_energy_total**
- **Formula:** `sum(|grid_t - prev_grid|)` over all 100 cells
- **Measures:** Overall motion energy in the frame
- **Manipulation Signature:** Spikes during fights, recoil, camera moves. Unnaturally low during tracking = aim-assist smoothing.

**2. vis_energy_mean**
- **Formula:** `mean(|grid_t - prev_grid|)`
- **Measures:** Normalized motion level per cell
- **Manipulation Signature:** Used for cross-match comparison. Consistently lower than input intensity suggests assistance.

**3. vis_energy_std**
- **Formula:** `std(|grid_t - prev_grid|)` across cells
- **Measures:** Distinguishes uniform motion (sweeps) from localized motion (aim, recoil)
- **Manipulation Signature:** Too low = robotic smoothness, too high with low input = chaos compensation.

**4. vis_motion_concentration_50**
- **Formula:** Fraction of cells (sorted by delta) needed to reach 50% of total motion
- **Measures:** Spatial concentration of motion
- **Manipulation Signature:** High concentration (0.1-0.2) = focused aim region. Locked high value with low input = aim-lock.

**5. vis_static_ratio**
- **Formula:** `count(cells where delta < 0.01) / 100`
- **Measures:** Fraction of screen that is "dead"
- **Manipulation Signature:** Aim-assist creates weird patterns - quiet edges with over-smooth center.

#### B. Spatial Structure (5 features)

**6. vis_center_energy_ratio**
- **Formula:** `sum(deltas in central 4×4) / sum(deltas in full 10×10)`
- **Measures:** Motion concentration in screen center
- **Manipulation Signature:** High during ADS and tracking. High with low gamepad input = aim-assist suspect.

**7. vis_edge_energy_ratio**
- **Formula:** `sum(deltas in border cells) / total_energy`
- **Measures:** Peripheral motion vs central
- **Manipulation Signature:** Camera sweeps, environmental chaos. Low during combat is normal, but weird asymmetry suggests FOV manipulation.

**8. vis_horizontal_vs_vertical_ratio**
- **Formula:** `|horizontal gradient of delta| / |vertical gradient of delta|`
- **Measures:** Directional bias of motion
- **Manipulation Signature:** Real recoil has vertical bias. Unnatural stabilization (ratio near 1.0) during firing = recoil compensation.

**9. vis_recoil_vertical_bias**
- **Formula:** `(delta_energy in upper half) - (delta_energy in lower half)` during fire windows
- **Measures:** Vertical recoil pattern
- **Manipulation Signature:** Should be negative (upward recoil pushes screen up). Near zero during sustained fire = recoil comp.

**10. vis_scope_tunnel_index**
- **Formula:** `(edge_variance - center_variance) / (edge_variance + center_variance)` using raw intensity
- **Measures:** ADS tunnel vision effect
- **Manipulation Signature:** Sudden shifts without ADS input = forced visibility changes, EOMM environmental sabotage.

#### C. Temporal & Frequency Analysis (5 features)

Define frequency bands:
- `high_freq = |grid_t - ema_slow|` (fast changes)
- `low_freq = |ema_fast - ema_slow|` (slow sweeps)

**11. vis_highfreq_energy**
- **Formula:** `sum(high_freq)`
- **Measures:** Recoil, micro jitter, muzzle flash
- **Manipulation Signature:** Too low during firing = recoil compensation. Spikes with low input = forced visual chaos.

**12. vis_lowfreq_energy**
- **Formula:** `sum(low_freq)`
- **Measures:** Slow sweeps, tracking, camera motion
- **Manipulation Signature:** Dominates during aim-assist (smooth tracking with minimal high-freq noise).

**13. vis_high_to_low_ratio**
- **Formula:** `vis_highfreq_energy / (vis_lowfreq_energy + ε)`
- **Measures:** Ratio of twitchy to smooth motion
- **Manipulation Signature:** High = panic/recoil. Low = unnaturally smooth tracking. Look for mismatch with input intensity.

**14. vis_smoothness_index**
- **Formula:** `1 / (1 + vis_energy_std)` or inverse of temporal variance over N frames
- **Measures:** Temporal smoothness of motion
- **Manipulation Signature:** Unrealistically high smoothness = EOMM/aim-assist smoothing zone.

**15. vis_stutter_score**
- **Formula:** `|vis_energy_total_t - vis_energy_total_{t-1}| / mean_energy`
- **Measures:** Frame-to-frame energy swings
- **Manipulation Signature:** Large swings = frame pacing problems, invisible lag, server tick weirdness. EOMM lag signature.

#### D. Event-Focused Features (5 features)

**16. vis_flash_intensity**
- **Formula:** `max(cell delta)` in frame
- **Measures:** Explosions, muzzle flashes
- **Manipulation Signature:** Helps segment firefights. Weirdly low during own firing = muzzle flash suppression.

**17. vis_firefight_focus_ratio**
- **Formula:** `vis_center_energy_ratio` when `vis_flash_intensity > threshold`
- **Measures:** Where firefight action occurs on screen
- **Manipulation Signature:** Always center-focused with low input variance = aim-lock during combat.

**18. vis_jitter_band_energy**
- **Formula:** Mid-band metric: `highfreq_energy - raw_delta` for central 4×4 region
- **Measures:** Micro-corrections vs hand jitter
- **Manipulation Signature:** Inhuman micro-corrections (too regular or absent) vs normal human jitter patterns.

**19. vis_contrast_shift_score**
- **Formula:** `|variance(grid_t) - variance(ema_slow)|`
- **Measures:** Sudden visibility/gamma shifts
- **Manipulation Signature:** "You suddenly can't see shit" EOMM tricks. Correlated with enemy encounters = forced disadvantage.

**20. vis_aim_lock_score**
- **Formula:** Composite: `high(vis_center_energy_ratio) * high(vis_smoothness_index) * low(vis_highfreq_energy)`
- **Measures:** Visual signature of locked reticle
- **Manipulation Signature:** Sustained high score with low input correlation = aim-assist territory.

---

## 2. Match Fingerprint Vector (365 dimensions)

### Block Layout

| Block | Dims | Size | Content |
|-------|------|------|---------|
| A - Visual Summary | 0-127 | 128 | Screen pattern statistics |
| B - Gamepad Summary | 128-223 | 96 | Input biomechanics |
| C - Network Summary | 224-255 | 32 | Packet behavior |
| D - Cross-Modal Correlations | 256-335 | 80 | Fusion metrics |
| E - Meta & Anomaly | 336-364 | 29 | Match-level flags |

---

### Block A: Visual Summary (Dims 0-127)

#### Basic Statistics (Dims 0-79)
For each of 20 visual features: mean, std, max, p90

| Feature | Mean | Std | Max | P90 |
|---------|------|-----|-----|-----|
| vis_energy_total | 0 | 1 | 2 | 3 |
| vis_energy_mean | 4 | 5 | 6 | 7 |
| vis_energy_std | 8 | 9 | 10 | 11 |
| vis_motion_concentration_50 | 12 | 13 | 14 | 15 |
| vis_static_ratio | 16 | 17 | 18 | 19 |
| vis_center_energy_ratio | 20 | 21 | 22 | 23 |
| vis_edge_energy_ratio | 24 | 25 | 26 | 27 |
| vis_horizontal_vs_vertical_ratio | 28 | 29 | 30 | 31 |
| vis_recoil_vertical_bias | 32 | 33 | 34 | 35 |
| vis_scope_tunnel_index | 36 | 37 | 38 | 39 |
| vis_highfreq_energy | 40 | 41 | 42 | 43 |
| vis_lowfreq_energy | 44 | 45 | 46 | 47 |
| vis_high_to_low_ratio | 48 | 49 | 50 | 51 |
| vis_smoothness_index | 52 | 53 | 54 | 55 |
| vis_stutter_score | 56 | 57 | 58 | 59 |
| vis_flash_intensity | 60 | 61 | 62 | 63 |
| vis_firefight_focus_ratio | 64 | 65 | 66 | 67 |
| vis_jitter_band_energy | 68 | 69 | 70 | 71 |
| vis_contrast_shift_score | 72 | 73 | 74 | 75 |
| vis_aim_lock_score | 76 | 77 | 78 | 79 |

#### Firefight Segmentation (Dims 80-89)
- 80: `number_of_firefight_segments` (count)
- 81: `avg_firefight_duration_frames`
- 82: `max_firefight_duration_frames`
- 83: `fraction_of_match_in_firefight` (0-1)
- 84: `avg_vis_energy_total_during_firefights`
- 85: `avg_vis_highfreq_energy_during_firefights`
- 86: `avg_vis_smoothness_index_outside_firefights`
- 87: `fraction_time_high_flash` (flash_intensity > threshold)
- 88: `fraction_time_scope_tunnel_high`
- 89: `fraction_time_center_energy_ratio_high`

#### ADS-Like State Approximation (Dims 90-99)
If true ADS flag unavailable, use visual + input heuristics.

- 90: `number_of_ads_segments` (estimated)
- 91: `avg_ads_duration_frames`
- 92: `max_ads_duration_frames`
- 93: `fraction_of_match_in_ads`
- 94: `avg_vis_center_energy_ratio_during_ads`
- 95: `delta_energy_mean_ads_vs_hip` (ADS - non-ADS)
- 96: `delta_smoothness_index_ads_vs_hip`
- 97: `delta_highfreq_energy_ads_vs_hip`
- 98: `delta_center_ratio_ads_vs_hip`
- 99: `delta_static_ratio_ads_vs_hip`

#### Visual Anomaly Distribution (Dims 100-127)

**Anomaly Fraction Flags (100-103):**
- 100: `fraction_frames_high_aim_lock_score`
- 101: `fraction_frames_high_stutter_score`
- 102: `fraction_frames_high_contrast_shift_score`
- 103: `fraction_frames_extreme_high_to_low_ratio`

**Anomaly Zone Averages (104-111):**
When feature exceeds dynamic threshold (e.g., mean + 2*std):
- 104: `avg_aim_lock_score_in_high_zone`
- 105: `avg_stutter_score_in_high_zone`
- 106: `avg_contrast_shift_score_in_high_zone`
- 107: `avg_high_to_low_ratio_in_extreme_zone`
- 108-111: (spare for additional anomaly stats)

**Rolling Anomaly Distribution (112-119):**
8-bin histogram of anomaly counts per minute:
- 112: `anomaly_count_minute_0`
- 113: `anomaly_count_minute_1`
- ... (up to 8 minutes or normalized bins)

**Energy Distribution Histogram (120-127):**
8-bin histogram of `vis_energy_total` (low → extreme high):
- 120: `bin_0_energy_very_low`
- 121: `bin_1_energy_low`
- 122: `bin_2_energy_below_avg`
- 123: `bin_3_energy_avg`
- 124: `bin_4_energy_above_avg`
- 125: `bin_5_energy_high`
- 126: `bin_6_energy_very_high`
- 127: `bin_7_energy_extreme`

---

### Block B: Gamepad Summary (Dims 128-223)

#### 16 Core Gamepad Features
- `gp_input_intensity` - Magnitude of right stick aim
- `gp_left_stick_intensity` - Movement magnitude
- `gp_fire_rate` - Shots per second (smoothed)
- `gp_ads_state` - ADS toggle (0/1)
- `gp_micro_adjust_rate` - Small aim flicks per second
- `gp_big_flick_rate` - Large aim movements per second
- `gp_idle_aim_ratio` - Fraction of time with no aim input
- `gp_button_press_rate` - All buttons per second
- `gp_strafe_intensity` - Left stick lateral movement
- `gp_sprint_ratio` - Fraction of time sprinting
- `gp_crouch_toggle_rate` - Crouch events per second
- `gp_jump_rate` - Jump events per second
- `gp_reload_rate` - Reload events per second
- `gp_weapon_switch_rate` - Weapon change per second
- `gp_aim_acceleration` - Rate of change of aim velocity
- `gp_input_entropy` - Shannon entropy of button timings

#### Basic Statistics (Dims 128-191)
For each of 16 features: mean, std, max, p90
- 16 features × 4 stats = 64 dims → **128-191**

#### Event Counts (Dims 192-201)
- 192: `total_shots_fired`
- 193: `total_ads_toggles`
- 194: `total_big_flicks`
- 195: `total_micro_adjusts`
- 196: `frames_with_aim_input`
- 197: `frames_with_fire_without_aim` (panic fire)
- 198: `frames_with_aim_without_fire` (tracking only)
- 199: `estimated_time_sprinting` (seconds)
- 200: `estimated_time_strafing` (seconds)
- 201: `estimated_time_idle` (seconds)

#### Pattern / Fatigue Analysis (Dims 202-211)
Early vs late match comparison for fatigue/consistency:
- 202: `input_intensity_mean_early` (first 33%)
- 203: `input_intensity_mean_late` (last 33%)
- 204: `micro_adjust_rate_early`
- 205: `micro_adjust_rate_late`
- 206: `aim_intensity_first_third`
- 207: `aim_intensity_middle_third`
- 208: `aim_intensity_last_third`
- 209: `fire_rate_first_third`
- 210: `fire_rate_middle_third`
- 211: `fire_rate_last_third`

#### Skill vs Odd Behavior (Dims 212-223)
- 212-215: `micro_to_big_flick_ratio` in 4 match quarters
- 216: `clutch_segment_avg_fire_rate` (high fire + stable ping)
- 217: `clutch_segment_avg_aim_intensity`
- 218: `clutch_segment_count`
- 219: `clutch_segment_fraction_of_match`
- 220: `input_regularity_score` (low entropy = robotic)
- 221: `input_chaos_score` (high entropy = panic/erratic)
- 222: `fraction_frames_highly_regular_input`
- 223: `fraction_frames_chaotic_input`

---

### Block C: Network Summary (Dims 224-255)

#### 8 Network Features
- `net_ping_ms` - Latency to server
- `net_jitter_ms` - Short-term variance in ping
- `net_spike_flag` - Binary flag (1 if ping > threshold)
- `net_packet_loss_proxy` - Inferred packet loss indicator
- `net_rtt_change_rate` - Rate of ping change
- `net_low_ping_flag` - Binary flag (1 if ping < low threshold)
- `net_high_ping_flag` - Binary flag (1 if ping > high threshold)
- `net_stable_window_flag` - Binary flag (1 if jitter low)

#### Basic Statistics (Dims 224-255)
For each of 8 features: mean, std, max, p90
- 8 features × 4 stats = 32 dims → **224-255**

This is where **EOMM lag patterns** will show:
- Spikes correlated with enemy encounters (high visual energy)
- Ping increases during disadvantageous moments
- Asymmetric network treatment

---

### Block D: Cross-Modal Correlations (Dims 256-335)

#### 20 Key Relationships (4 stats each = 80 dims)

For each relationship, store:
1. Pearson correlation coefficient (r)
2. Mean of X given Y is high (threshold-based)
3. Mean of Y given X is high
4. Fraction of frames where both are "active" (above thresholds)

| # | Relationship | Dims | Manipulation Signature |
|---|-------------|------|----------------------|
| 1 | corr(vis_energy_total, gp_input_intensity) | 256-259 | Low = aim-assist (visual motion without input) |
| 2 | corr(vis_center_energy_ratio, gp_input_intensity) | 260-263 | Negative = center locked despite no aim input |
| 3 | corr(vis_highfreq_energy, gp_fire_rate) | 264-267 | Low = recoil compensation |
| 4 | corr(vis_smoothness_index, gp_input_intensity) | 268-271 | High smoothness with low input = assistance |
| 5 | corr(vis_aim_lock_score, gp_input_intensity) | 272-275 | Negative or zero = aim-lock suspect |
| 6 | corr(vis_stutter_score, net_ping_ms) | 276-279 | High = EOMM lag causing visual stutters |
| 7 | corr(vis_stutter_score, net_spike_flag) | 280-283 | High = network sabotage during action |
| 8 | corr(vis_contrast_shift_score, net_ping_ms) | 284-287 | Visibility sabotage via lag |
| 9 | corr(vis_high_to_low_ratio, gp_micro_adjust_rate) | 288-291 | Mismatch = forced chaos or damping |
| 10 | corr(vis_highfreq_energy, net_ping_ms) | 292-295 | Lag during recoil moments |
| 11 | corr(vis_energy_total, net_ping_ms) | 296-299 | High action = network sabotage |
| 12 | corr(gp_fire_rate, net_ping_ms) | 300-303 | Aggression punished by lag |
| 13 | corr(gp_input_intensity, net_jitter_ms) | 304-307 | Hard input = network instability |
| 14 | corr(vis_center_energy_ratio, net_spike_flag) | 308-311 | Target tracking during spikes |
| 15 | corr(vis_aim_lock_score, net_spike_flag) | 312-315 | Aim-assist during lag windows |
| 16 | corr(vis_highfreq_energy, gp_big_flick_rate) | 316-319 | Should be high; low = damping |
| 17 | corr(vis_lowfreq_energy, gp_input_intensity) | 320-323 | Smooth tracking correlation |
| 18 | corr(vis_stutter_score, gp_input_intensity) | 324-327 | Stutters during high input = sabotage |
| 19 | corr(vis_contrast_shift_score, gp_fire_rate) | 328-331 | Visibility drops during firing |
| 20 | corr(vis_energy_total, gp_fire_rate) | 332-335 | Combat energy correlation |

---

### Block E: Meta & Anomaly Stats (Dims 336-364)

#### Basic Meta (Dims 336-340)
- 336: `total_frames`
- 337: `match_duration_seconds`
- 338: `avg_fps_estimate`
- 339: `profile_id` (0=play_nice, 1=forensic, 2=overwatch)
- 340: `user_label` (0=unknown, 1="felt fair", 2="felt rigged")

#### Global Anomaly Indicators (Dims 341-352)
- 341: `fraction_of_frames_flagged_visual_anomaly`
- 342: `fraction_of_frames_flagged_input_anomaly`
- 343: `fraction_of_frames_flagged_network_anomaly`
- 344: `avg_per_frame_anomaly_score` (combined tri-modal)
- 345: `max_frame_anomaly_score`
- 346: `anomaly_score_top_1` (highest)
- 347: `anomaly_score_top_2`
- 348: `anomaly_score_top_3`
- 349: `anomaly_score_top_4`
- 350: `anomaly_count_early_third`
- 351: `anomaly_count_middle_third`
- 352: `anomaly_count_late_third`

#### Match-Level Pattern Flags (Dims 353-364)
Derived composite scores (0-1 range):
- 353: `aim_assist_suspect_score`
- 354: `recoil_compensation_suspect_score`
- 355: `eomm_lag_suspect_score`
- 356: `network_priority_bias_score`

Rule-based triggers (binary or soft flags):
- 357: `sustained_low_input_high_smoothness_flag`
- 358: `center_locked_zero_aim_input_flag`
- 359: `recoil_damping_during_fire_flag`
- 360: `ping_spike_during_combat_flag`

Future expansion slots:
- 361: `reserved_map_id` (0 = not set)
- 362: `reserved_lobby_type` (0 = not set)
- 363: `reserved_custom_flag_1` (0 = not set)
- 364: `reserved_custom_flag_2` (0 = not set)

---

## 3. Priority Ordering - What Hits What

### 🎯 Tier 1: Aim Assist Intervention

**Visual Features:**
- `vis_aim_lock_score` ⭐⭐⭐ (composite of center ratio + smoothness + low highfreq)
- `vis_smoothness_index` ⭐⭐⭐ (too smooth for input intensity)
- `vis_center_energy_ratio` ⭐⭐ (locked center with low input)

**Cross-Modal:**
- `corr(vis_aim_lock_score, gp_input_intensity)` ⭐⭐⭐ - **Low or negative = red flag**
- `corr(vis_center_energy_ratio, gp_input_intensity)` ⭐⭐⭐ - **Should be positive; near zero = suspect**
- `corr(vis_smoothness_index, gp_input_intensity)` ⭐⭐ - **High smoothness with low input**

**Fingerprint Flags:**
- `aim_assist_suspect_score` (dim 353)
- `sustained_low_input_high_smoothness_flag` (dim 357)
- `center_locked_zero_aim_input_flag` (dim 358)
- `fraction_frames_high_aim_lock_score` (dim 100)

**Detection Logic:**
```
IF vis_aim_lock_score > threshold
AND gp_input_intensity < low_threshold
AND corr(vis_aim_lock_score, gp_input_intensity) < 0.2
THEN aim_assist_suspect = HIGH
```

---

### 🎯 Tier 2: Recoil Compensation Artifacts

**Visual Features:**
- `vis_highfreq_energy` ⭐⭐⭐ (should spike during fire; low = damping)
- `vis_recoil_vertical_bias` ⭐⭐⭐ (should be negative; near zero = stabilization)
- `vis_high_to_low_ratio` ⭐⭐ (too low = over-smoothed)

**Cross-Modal:**
- `corr(vis_highfreq_energy, gp_fire_rate)` ⭐⭐⭐ - **Should be high; low = compensation**
- `corr(vis_recoil_vertical_bias, gp_fire_rate)` ⭐⭐ - **Should be negative; flat = suspect**
- `corr(vis_highfreq_energy, gp_big_flick_rate)` ⭐⭐ - **Low = motion damping**

**Fingerprint Flags:**
- `recoil_compensation_suspect_score` (dim 354)
- `recoil_damping_during_fire_flag` (dim 359)
- `avg_vis_highfreq_energy_during_firefights` (dim 85)

**Detection Logic:**
```
IF gp_fire_rate > threshold
AND vis_highfreq_energy < expected_recoil_energy
AND vis_recoil_vertical_bias > -0.1  # Too stable
THEN recoil_compensation_suspect = HIGH
```

---

### 🎯 Tier 3: EOMM Lag Manipulation

**Visual Features:**
- `vis_stutter_score` ⭐⭐⭐ (frame energy swings = lag)
- `vis_contrast_shift_score` ⭐⭐⭐ ("can't see shit" moments)
- `vis_energy_total` ⭐ (context for when stutters happen)

**Network Features:**
- `net_ping_ms` (mean/std/max) ⭐⭐⭐
- `net_spike_flag` ⭐⭐⭐
- `net_jitter_ms` ⭐⭐

**Cross-Modal:**
- `corr(vis_stutter_score, net_ping_ms)` ⭐⭐⭐ - **High = lag causing stutters**
- `corr(vis_stutter_score, net_spike_flag)` ⭐⭐⭐ - **Spikes during action**
- `corr(vis_contrast_shift_score, net_ping_ms)` ⭐⭐ - **Visibility sabotage**
- `corr(vis_energy_total, net_ping_ms)` ⭐⭐ - **High action = lag spike**

**Fingerprint Flags:**
- `eomm_lag_suspect_score` (dim 355)
- `ping_spike_during_combat_flag` (dim 360)
- `fraction_frames_high_stutter_score` (dim 101)
- `anomaly_count_early_third` vs `anomaly_count_late_third` (dims 350, 352) - **Asymmetric distribution**

**Detection Logic:**
```
IF vis_stutter_score > threshold
AND net_spike_flag == 1
AND vis_energy_total > combat_threshold  # During action
AND corr(vis_stutter_score, net_ping_ms) > 0.6
THEN eomm_lag_suspect = HIGH
```

---

### 🎯 Tier 4: Network Priority Bias

**Network Features:**
- `net_ping_ms` (mean vs baseline) ⭐⭐⭐
- `net_jitter_ms` ⭐⭐
- `net_stable_window_flag` vs `net_spike_flag` ratio ⭐⭐

**Cross-Modal:**
- `corr(vis_energy_total, net_ping_ms)` ⭐⭐⭐ - **Positive = action punished**
- `corr(gp_input_intensity, net_ping_ms)` ⭐⭐⭐ - **You push harder = lag**
- `corr(gp_fire_rate, net_ping_ms)` ⭐⭐ - **Aggression = sabotage**
- `corr(gp_input_intensity, net_jitter_ms)` ⭐⭐ - **Input causes instability**

**Fingerprint Flags:**
- `network_priority_bias_score` (dim 356)
- Ratio of good vs bad network windows during clutch segments:
  - `clutch_segment_count` (dim 218) with high `net_ping_ms` during those segments

**Detection Logic:**
```
IF mean(net_ping_ms) > user_baseline + 20ms
AND corr(gp_input_intensity, net_ping_ms) > 0.4
AND corr(vis_energy_total, net_ping_ms) > 0.3
THEN network_priority_bias = HIGH
```

---

## 4. Implementation Roadmap

### Phase 1: Per-Frame Feature Extraction (Week 1)

**1.1 ScreenResonanceState Class**
```python
class ScreenResonanceState:
    def __init__(self, ema_alpha_fast=0.3, ema_alpha_slow=0.1):
        self.ema_fast = None
        self.ema_slow = None
        self.prev_grid = None
        
    def update(self, grid_t: np.ndarray) -> dict:
        """Returns dict with all 20 visual features"""
        pass
```

**1.2 Update screen_grid_mapper.py**
- Instantiate `ScreenResonanceState`
- Call `update()` after grid extraction
- Return 20 visual features in telemetry dict

**1.3 Gamepad Feature Extraction**
- Extract 16 gamepad features per frame
- Requires parsing gamepad telemetry into per-frame metrics
- May need small moving windows for rates (fire_rate, micro_adjust_rate)

**1.4 Network Feature Extraction**
- Extract 8 network features per frame
- Requires timestamp alignment with screen frames
- Use nearest network sample or interpolate

**Target Output:**
```python
frame_features = {
    'timestamp': 123.45,
    'visual': {
        'vis_energy_total': 0.234,
        # ... 19 more
    },
    'gamepad': {
        'gp_input_intensity': 0.67,
        # ... 15 more
    },
    'network': {
        'net_ping_ms': 45.2,
        # ... 7 more
    }
}
```

---

### Phase 2: Match Fingerprint Builder (Week 2)

**2.1 MatchFingerprintBuilder Class**
```python
class MatchFingerprintBuilder:
    def __init__(self):
        self.accumulators = {}  # Running stats for each feature
        self.frame_count = 0
        
    def update(self, frame_features: dict):
        """Accumulate per-frame stats"""
        pass
        
    def build(self) -> np.ndarray:
        """Returns 365-dim vector"""
        pass
```

**2.2 Accumulation Logic**
- Track sums, sums_sq, mins, maxs for mean/std/max/p90
- Use heapq or running percentile estimator for p90
- Store histograms for energy distribution (bins 120-127)
- Track event counts (firefights, ADS segments, anomalies)

**2.3 Cross-Modal Correlation**
- Store paired (X, Y) samples for each correlation
- Compute Pearson r at build() time
- Compute conditional means (E[X|Y high])
- Count co-occurrence fraction

**2.4 Integration**
- Update `fusion_616_engine.py` to instantiate builder
- Call `builder.update()` per frame
- Call `builder.build()` at match end
- Save 365-dim vector to disk

---

### Phase 3: Baseline Capture & Analysis (Week 3)

**3.1 Capture Fair Matches**
- Play 5+ matches in "normal" conditions
- User labels as `user_label=1` ("felt fair")
- Save fingerprints to `baseline_fingerprints.npy`

**3.2 Capture Live Matches**
- Play 5+ matches (may include manipulation)
- User labels: 0=unknown, 2="felt rigged"
- Save to `live_fingerprints.npy`

**3.3 Distance Metrics**
```python
def fingerprint_distance(fp1, fp2):
    # Weighted Euclidean or Mahalanobis
    # Weight Tier 1 features higher
    pass

def cluster_fingerprints(fps):
    # K-means or DBSCAN
    # Check if live matches cluster away from baseline
    pass
```

**3.4 Anomaly Detection**
- Compute baseline centroid + covariance
- Flag live matches with Mahalanobis distance > threshold
- Identify which dimensions contribute most to distance

---

### Phase 4: Reporting & Visualization (Week 4)

**4.1 Fingerprint Comparison Report**
```
Match A vs Baseline:
  Distance: 12.34 (threshold: 8.0) ⚠️ SUSPECT

Top Contributing Features:
  1. vis_aim_lock_score (mean): baseline=0.12, match=0.78 (+550%)
  2. corr(vis_center_ratio, gp_input): baseline=0.65, match=0.02 (-97%)
  3. eomm_lag_suspect_score: baseline=0.05, match=0.89 (+1680%)

Tier 1 Flags: AIM_ASSIST_SUSPECT ⚠️
Tier 3 Flags: EOMM_LAG_SUSPECT ⚠️
```

**4.2 Time-Series Visualization**
- Plot per-frame features over match duration
- Highlight anomaly zones (shaded regions)
- Overlay gamepad + network events

**4.3 Correlation Heatmap**
- 20 visual × 16 gamepad × 8 network = 2,560 possible pairs
- Display top 40 correlations
- Compare baseline vs suspect matches

**4.4 Web Dashboard (Optional)**
- Upload telemetry JSONs
- Auto-compute fingerprint
- Display distance from baseline + flags
- Show evidence frames (high anomaly scores)

---

## 5. Validation Strategy

### Baseline Establishment
1. Capture 5-10 fair matches (user confirms no manipulation felt)
2. Compute mean + std of 365-dim vectors
3. Define "normal range" as [mean - 2*std, mean + 2*std]
4. Save baseline statistics

### Live Match Testing
1. Capture match with suspected manipulation
2. Compute fingerprint
3. Calculate distance from baseline centroid
4. Flag outliers (distance > 2.5σ)
5. Identify contributing features

### Ground Truth Validation
Since we can't know true manipulation state:
- **User labeling:** Subjective "felt rigged" (label=2)
- **Consistency check:** Do multiple "rigged" matches cluster together?
- **Feature inspection:** Do flagged matches show expected patterns (low input + high smoothness)?

### Success Criteria
- Baseline matches cluster tightly (low intra-cluster variance)
- Labeled "rigged" matches have distance > threshold
- Top contributing features match manipulation signatures
- System can flag specific frames with evidence (not just match-level)

---

## 6. Expected Manipulation Signatures Summary

| Manipulation Type | Key Indicators | Cross-Modal Flags | Tier |
|------------------|---------------|------------------|------|
| **Aim Assist** | High `vis_aim_lock_score`, low `vis_highfreq_energy`, high `vis_smoothness_index` | Low `corr(vis_aim_lock, gp_input)`, negative `corr(vis_center_ratio, gp_input)` | 1 |
| **Recoil Compensation** | Low `vis_highfreq_energy` during fire, flat `vis_recoil_vertical_bias` | Low `corr(vis_highfreq, gp_fire_rate)`, low `vis_high_to_low_ratio` | 2 |
| **EOMM Lag** | High `vis_stutter_score`, high `vis_contrast_shift_score`, `net_spike_flag` | High `corr(vis_stutter, net_ping)`, high `corr(vis_contrast, net_ping)` | 3 |
| **Network Bias** | High `net_ping_ms` during action, asymmetric ping patterns | High `corr(vis_energy, net_ping)`, high `corr(gp_input, net_ping)` | 4 |

---

## 7. Appendix: Per-Frame Feature Dictionary Schema

```python
frame_features = {
    'timestamp': float,  # Seconds since capture start
    'frame_number': int,
    
    'visual': {
        'vis_energy_total': float,
        'vis_energy_mean': float,
        'vis_energy_std': float,
        'vis_motion_concentration_50': float,
        'vis_static_ratio': float,
        'vis_center_energy_ratio': float,
        'vis_edge_energy_ratio': float,
        'vis_horizontal_vs_vertical_ratio': float,
        'vis_recoil_vertical_bias': float,
        'vis_scope_tunnel_index': float,
        'vis_highfreq_energy': float,
        'vis_lowfreq_energy': float,
        'vis_high_to_low_ratio': float,
        'vis_smoothness_index': float,
        'vis_stutter_score': float,
        'vis_flash_intensity': float,
        'vis_firefight_focus_ratio': float,
        'vis_jitter_band_energy': float,
        'vis_contrast_shift_score': float,
        'vis_aim_lock_score': float,
    },
    
    'gamepad': {
        'gp_input_intensity': float,
        'gp_left_stick_intensity': float,
        'gp_fire_rate': float,
        'gp_ads_state': float,
        'gp_micro_adjust_rate': float,
        'gp_big_flick_rate': float,
        'gp_idle_aim_ratio': float,
        'gp_button_press_rate': float,
        'gp_strafe_intensity': float,
        'gp_sprint_ratio': float,
        'gp_crouch_toggle_rate': float,
        'gp_jump_rate': float,
        'gp_reload_rate': float,
        'gp_weapon_switch_rate': float,
        'gp_aim_acceleration': float,
        'gp_input_entropy': float,
    },
    
    'network': {
        'net_ping_ms': float,
        'net_jitter_ms': float,
        'net_spike_flag': float,  # 0 or 1
        'net_packet_loss_proxy': float,
        'net_rtt_change_rate': float,
        'net_low_ping_flag': float,
        'net_high_ping_flag': float,
        'net_stable_window_flag': float,
    }
}
```

---

## 8. Code Skeleton - MatchFingerprintBuilder

```python
import numpy as np
from collections import defaultdict
from scipy.stats import pearsonr

class MatchFingerprintBuilder:
    def __init__(self):
        self.frame_count = 0
        self.visual_accum = defaultdict(list)
        self.gamepad_accum = defaultdict(list)
        self.network_accum = defaultdict(list)
        self.correlation_pairs = defaultdict(lambda: {'X': [], 'Y': []})
        
    def update(self, frame_features: dict):
        """Accumulate per-frame statistics"""
        self.frame_count += 1
        
        # Store all visual features
        for key, val in frame_features['visual'].items():
            self.visual_accum[key].append(val)
        
        # Store all gamepad features
        for key, val in frame_features['gamepad'].items():
            self.gamepad_accum[key].append(val)
        
        # Store all network features
        for key, val in frame_features['network'].items():
            self.network_accum[key].append(val)
        
        # Store correlation pairs (e.g., 20 key cross-modal relationships)
        self._accumulate_correlations(frame_features)
    
    def _accumulate_correlations(self, frame_features: dict):
        """Store paired (X, Y) samples for cross-modal correlations"""
        vis = frame_features['visual']
        gp = frame_features['gamepad']
        net = frame_features['network']
        
        # Example: store pairs for corr(vis_energy_total, gp_input_intensity)
        self.correlation_pairs['vis_energy_total_vs_gp_input_intensity']['X'].append(vis['vis_energy_total'])
        self.correlation_pairs['vis_energy_total_vs_gp_input_intensity']['Y'].append(gp['gp_input_intensity'])
        
        # ... repeat for all 20 correlation pairs
    
    def build(self) -> np.ndarray:
        """Construct 365-dim fingerprint vector"""
        fp = np.zeros(365, dtype=np.float32)
        
        # Block A: Visual Summary (0-127)
        self._fill_visual_block(fp)
        
        # Block B: Gamepad Summary (128-223)
        self._fill_gamepad_block(fp)
        
        # Block C: Network Summary (224-255)
        self._fill_network_block(fp)
        
        # Block D: Cross-Modal Correlations (256-335)
        self._fill_correlation_block(fp)
        
        # Block E: Meta & Anomaly (336-364)
        self._fill_meta_block(fp)
        
        return fp
    
    def _fill_visual_block(self, fp: np.ndarray):
        """Dims 0-127"""
        idx = 0
        # For each of 20 visual features: mean, std, max, p90
        for key in sorted(self.visual_accum.keys()):
            vals = self.visual_accum[key]
            fp[idx] = np.mean(vals)
            fp[idx+1] = np.std(vals)
            fp[idx+2] = np.max(vals)
            fp[idx+3] = np.percentile(vals, 90)
            idx += 4
        
        # Dims 80-127: firefight, ADS, anomaly stats
        # ... implementation details
    
    def _fill_gamepad_block(self, fp: np.ndarray):
        """Dims 128-223"""
        # Similar to visual block
        pass
    
    def _fill_network_block(self, fp: np.ndarray):
        """Dims 224-255"""
        # Similar to visual block
        pass
    
    def _fill_correlation_block(self, fp: np.ndarray):
        """Dims 256-335"""
        idx = 256
        for pair_name, data in sorted(self.correlation_pairs.items()):
            X = np.array(data['X'])
            Y = np.array(data['Y'])
            
            # 1. Pearson r
            r, _ = pearsonr(X, Y)
            fp[idx] = r
            
            # 2. E[X | Y high]
            high_Y_mask = Y > np.percentile(Y, 75)
            fp[idx+1] = np.mean(X[high_Y_mask]) if np.any(high_Y_mask) else 0
            
            # 3. E[Y | X high]
            high_X_mask = X > np.percentile(X, 75)
            fp[idx+2] = np.mean(Y[high_X_mask]) if np.any(high_X_mask) else 0
            
            # 4. Fraction both active
            fp[idx+3] = np.mean(high_X_mask & high_Y_mask)
            
            idx += 4
    
    def _fill_meta_block(self, fp: np.ndarray):
        """Dims 336-364"""
        fp[336] = self.frame_count
        fp[337] = self.frame_count / 20  # Assuming 20 FPS
        # ... other meta stats
```

---

**End of Specification v1.0**

**Status:** ✅ Ready for implementation  
**Next Step:** Implement `ScreenResonanceState` class with 20 visual feature extraction methods.
