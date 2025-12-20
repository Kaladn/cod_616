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

4. **Save YOLO visualizations** (removed - YOLO stub)
   - Change: YOLO removed from active inference; `modules/yolo_detector.py` now provides a non-inferential stub that preserves API and outputs.
   - Impact: Visualization function exists but is a no-op; there are no YOLO artifacts written to disk.
   - Files affected: `modules/yolo_detector.py`, `cod_live_runner.py`, `test_yolo_performance.py`

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

13. **NetworkLogger integration & tests** (completed)
    - Change: Added `loggers/network_service.py` and `loggers/network_integration.py`, added unit and smoke tests in `gaming/tests/` (legacy `CompuCog_Visual_v2/tests/` archived), and ensured `CognitiveHarness._shutdown()` stops network and process services.
    - Tests: `test_network_integration_unit.py`, `test_network_integration.py` (both pass locally).
    - Branch: `feat/network-logger` (pushed to origin)

14. **Vision integration: worker, configs & harness tests** (not-started)
    - Goal: Add a first-class Vision worker and integration tests so TrueVision capture can run as a managed service under the harness and be exercised in CI.
    - Note: The legacy package `CompuCog_Visual_v2` has been archived (backup branch `backup/pre-removal-compucog_visual_v2`). Implement the worker and helpers under `gaming/`.
    - Next steps:
      - Add worker: `gaming/truevision_worker.py` (background capture loop with start/stop API).
      - Add integration helper: `gaming/integrate_vision.py` to start worker and register `vision` source in EventManager for testing.
      - Add configs: `gaming/config/vision.yaml` and add `gaming/config/loggers.yaml` to centralize logger config.
      - Add tests: harness-level smoke test that instantiates `CognitiveHarness` with a minimal `vision` config and mocks capture to run quickly; unit tests for the worker start/stop and for helper registration.
      - Add CI stubs to avoid heavy deps (mss/cv2) in CI environment.
    - Supporting files already in repo (useful foundations):
      - `CompuCog_Visual_v2/gaming/truevision_event_live.py` (reference capture & processing loop)
      - `CompuCog_Visual_v2/compositor/eomm_compositor.py` (EOMM composition)
      - Operators: `operators/crosshair_lock.py`, `operators/hit_registration.py`, `operators/death_event.py`, `operators/edge_entry.py` (detection algorithms)
      - Frame utilities: `core/frame_to_grid.py`, `core/frame_capture.py` (grid conversion & capture)
      - Session baseline: `gaming/session_baseline.py`
      - Fusion & recognition: `modules/fusion_616_engine.py`, `recognition/recognition_field.py`
      - Event system: `event_system/event_manager.py`, `event_system/chronos_manager.py` (EventManager + Chronos foundations)
      - Existing logger adapters & helpers: `loggers/*_integration.py` (pattern to follow)
    - Deliverables:
      - `truevision_worker.py`, `integrate_vision.py`, config files, unit + smoke tests, and a PR with CI-friendly test stubs.

---

**Screen Vector Engine (SVE)** — status: **core implemented, vectors & anomaly metrics implemented, unit tests passing**

Remaining SVE tasks (not-started):
- **Operator refactor**: refactor operators to consume `ScreenVectorState` fields (start with `crosshair_lock.py`)
- **TrueVision worker**: add `CompuCog_Visual_v2/gaming/truevision_worker.py` and `integrate_vision.py` helper
- **Forge schema & capsule buffer**: update Forge schema to support `sve_state` and change 6-1-6 buffer to store `ScreenVectorState` instances
- **Harness smoke tests & CI stubs**: add tests that instantiate `CognitiveHarness` with a minimal vision config and stub heavy deps (mss, cv2)

If you'd like, I can begin any of the remaining SVE tasks when you say you're ready; I will not prioritize them until you explicitly ask me to.

If you'd like, I can start working on the in-progress task (#1) now and open a PR with the changes. Otherwise I can switch the in-progress item to another task you prefer.

---

**Next start point (waiting):** `SVE: Operator refactor to use ScreenVectorState` — *I will not start until you say* **"go"**. When you say "go" I'll: 
- mark that todo `in-progress` in the TODO tracker, 
- start refactoring `crosshair_lock.py` to consume `ScreenVectorState`, and
- add unit tests and a small smoke harness test.

If you'd like me to open a draft PR for the SVE work now, say "open PR" (GH auth required) or push and I can open it for you.

---

### New housekeeping tasks (added as next actions)

32. **Workspace introspection & pruning (keep-list validation)** (not-started)
    - Goal: Run a full workspace introspection, compare files to the keep-list, and produce a prune plan (archive/delete candidates) with roll-back safety.
    - Deliverables: `WORKSPACE_INTROSPECTION_PRUNE_PLAN.md` listing files to archive, move, or delete; a small script `tools/prune_workspace.py` to perform safe moves to `archive/`.

33. **Prune/archive `arc_organ` & ARC artifacts if not on keep list** (not-started)
    - Goal: If `arc_organ` and related artifacts are not marked **K**, move them to `archive/arc_organ-YYYYMMDD/` and update README to record the archive location.
    - Safety: All moves are reversible; files are not deleted permanently until you confirm.

When you say **"run introspection"**, I'll mark task 32 `in-progress`, run the analysis, and produce the prune plan for your review.
