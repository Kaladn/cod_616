# 6-1-6 Temporal Fusion Block Builder

**Status**: IMPLEMENTED (January 3, 2026)

## Purpose

Stand-alone module for ML training data generation. NOT part of real-time detection.

Reads from Forge Memory JSONL logs, builds 13-frame multi-modal aligned windows
centered on detected anchor events (kills, deaths, crosshair locks, EOMM peaks).

## The Two 616 Concepts

This system **unifies two 616 architectures**:

### Temporal 616 (Primary)
- **6 frames** - Precursor context (what happened before)
- **1 frame** - Anchor event (the detection)  
- **6 frames** - Consequence tracking (what happened after)
- **Total**: 13-frame causality window

### Resonance 616 (Secondary)
- From old `fusion_616_engine.py` (Nov 2025)
- **6Hz-1Hz-6Hz** frequency-domain coherence analysis
- Now integrated as `resonance_coherence` metric

## Usage

```powershell
# Build fusion blocks from session
python fusion.py --session D:\sessions\2025-01-03_game1

# Custom output path
python fusion.py --session D:\sessions\game1 --output training_data.jsonl

# Different frame rate
python fusion.py --session D:\sessions\game1 --fps 120
```

## Output

Produces `fusion_blocks.jsonl` with FusionBlock records:

```json
{
  "block_id": "block_000001_crosshair_lock",
  "anchor_timestamp": 12345.678,
  "anchor_event_type": "crosshair_lock",
  "precursor_frames": [...6 ModalitySlice objects...],
  "anchor_frame": {...ModalitySlice...},
  "consequence_frames": [...6 ModalitySlice objects...],
  "alignment_score": 0.85,
  "resonance_coherence": 0.72,
  "labels": {"source_event": "crosshair_lock"}
}
```

## Architecture Lineage

```
CompuCogVision Phase 1 (Nov 26)  ─┐
                                  ├──> 616 Temporal Fusion
CompuCog_Visual_v2 (TrueVision) ──┘
```

## Contracts

- [fusion_block.schema.json](contracts/fusion_block.schema.json) - FusionBlock structure
- [fusion_input.schema.json](contracts/fusion_input.schema.json) - Input specification
