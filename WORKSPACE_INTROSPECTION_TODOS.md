# Workspace Introspection TODOs

Generated from `WORKSPACE INTROSPECTION REPORT-1.md` (Dec 14, 2025)

> Status: Task 1 is marked *in-progress*. Work can begin on it immediately.

## Todos

1. **Limit fingerprint builder memory** (completed)
   - Problem: `MatchFingerprintBuilder` previously accumulated unbounded per-frame lists and could OOM on long captures.
   - Change: Added `max_frames` and `sample_size` configuration; uses bounded `deque` sampling for percentiles and correlations to limit memory.
   - Tests: Added `test_match_fingerprint_builder.py` covering bounded sampling and build correctness.
   - Files: `match_fingerprint_builder.py`, `test_match_fingerprint_builder.py`

2. **Integrate audio capture** (completed)
   - Change: Added `modules/audio_capture.py` which uses `sounddevice` when available and falls back to a stub otherwise.
   - Integration: `COD616Runner` now initializes `AudioCapture` and `AudioResonanceState` (configurable per-profile) and includes `audio_resonance` in per-frame telemetry and into the fingerprint builder.
   - Tests: Added `test_audio_integration.py` with unit tests for capture stub and audio feature extraction.
   - Files: `modules/audio_capture.py`, `modules/audio_resonance_state.py`, `cod_live_runner.py`, `test_audio_integration.py`, `config_616.yaml`

3. **Add audio to fusion** (completed)
   - Change: `Fusion616Engine.fuse()` now accepts `audio_resonance` (20 dims) and includes it in the `fused_vector`.
   - Integration: `COD616Runner` now passes `audio_resonance` into `fuse()`; audio contributes a small amount to resonance amplitude calculations.
   - Tests: Added `test_fusion_audio.py` to validate fused vector shape and audio slice presence.
   - Files: `modules/fusion_616_engine.py`, `cod_live_runner.py`, `test_fusion_audio.py`

4. **Save YOLO visualizations** (not-started)
   - Problem: `YOLODetector.visualize_detections()` outputs images but COD616Runner never saves them.
   - Next step: Add an opt-in config to save visualization images per N frames and tests for disk I/O.
   - Files: `modules/yolo_detector.py`, `cod_live_runner.py`

5. **Call visualize_grid in runner** (not-started)
   - Problem: `ScreenGridMapper.visualize_grid()` is unused.
   - Next step: Invoke it in the capture loop when verbose or on-demand and add save option.
   - Files: `modules/screen_grid_mapper.py`, `cod_live_runner.py`

6. **Add fingerprint schema versioning** (not-started)
   - Problem: No schema or versioning for fingerprints; changes will break baselines.
   - Next step: Add `schema_version` to fingerprint JSON, migration helpers in `RecognitionField`, and tests.
   - Files: `match_fingerprint_builder.py`, `recognition_field.py`

7. **Recompute grid on resolution change** (not-started)
   - Problem: Grid assumes static resolution; monitor resolution changes can break cell sizing.
   - Next step: Detect resolution changes and recompute `cell_width`/`cell_height` dynamically.
   - Files: `modules/screen_grid_mapper.py`

8. **Support multiple gamepads** (not-started)
   - Problem: GamepadCapture hardcodes joystick index 0.
   - Next step: Enumerate joysticks, allow selection via config or auto-merge inputs.
   - Files: `modules/gamepad_capture.py`

9. **Handle mss capture failures** (not-started)
   - Problem: No explicit error handling for screen capture failures (mss), may crash.
   - Next step: Wrap capture in try/except, add retry/backoff and a clear error message with fallback.
   - Files: `modules/screen_grid_mapper.py`, `cod_live_runner.py`

10. **Add pipeline and baseline tests** (not-started)
    - Problem: No integration/unit tests for main capture/fusion/baseline pipeline.
    - Next step: Add tests under `tests/` or `arc_organ/test_*` to validate end-to-end behavior with synthetic inputs.
    - Files: `test_phase_7.py`, new tests

11. **Clarify README baseline paths** (not-started)
    - Problem: README references `data/bot_match_baseline/` while code expects `baseline/` or `recognition/profiles/`.
    - Next step: Update README and CLI docs to reference canonical paths and add examples.
    - Files: `README.md`, `COMPUCOG_VISION_SPEC.md`, CLI help

12. **Add or remove reflex_telemetry stub** (not-started)
    - Problem: `reflex_telemetry.py` referenced in README but missing from repo.
    - Next step: Create a minimal stub (or remove the reference) and add a test or example.
    - Files: `reflex_telemetry.py`, `README.md`

---

If you'd like, I can start working on the in-progress task (#1) now and open a PR with the changes. Otherwise I can switch the in-progress item to another task you prefer.
