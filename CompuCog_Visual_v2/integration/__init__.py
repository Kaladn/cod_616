"""
CompuCog Integration Layer

STATUS: DORMANT (DeepSeek awaiting activation)

This package provides read-only integration between your deterministic
616 fusion system and the DeepSeek LLM for expert-level explanation.

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │  Your System (Deterministic Truth)                          │
    │  ┌────────────┐  ┌────────────┐  ┌────────────┐             │
    │  │ TrueVision │  │  Network   │  │   Input    │  + more     │
    │  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘             │
    │        │               │               │                     │
    │        └───────────────┼───────────────┘                     │
    │                        ▼                                     │
    │              ┌─────────────────┐                             │
    │              │  616 Fusion     │                             │
    │              │  (6-1-6 blocks) │                             │
    │              └────────┬────────┘                             │
    └───────────────────────┼─────────────────────────────────────┘
                            │
                            │ READ-ONLY
                            ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  Integration Layer (This Package)                           │
    │  ┌────────────────┐     ┌────────────────┐                  │
    │  │ EvidencePackager│────▶│ DeepSeekClient │ ◀── DORMANT    │
    │  └────────────────┘     └────────────────┘                  │
    │          │                      │                            │
    │          │                      ▼                            │
    │          │              ┌────────────────┐                  │
    │          └─────────────▶│ForensicExplainer│                  │
    │                         └────────────────┘                  │
    └─────────────────────────────────────────────────────────────┘
                            │
                            ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  Output (Future)                                            │
    │  ┌────────────┐  ┌────────────┐  ┌────────────┐             │
    │  │ JSON API   │  │  UI/Web    │  │  Reports   │             │
    │  └────────────┘  └────────────┘  └────────────┘             │
    └─────────────────────────────────────────────────────────────┘

Usage:
    from integration import ForensicExplainer
    
    explainer = ForensicExplainer()
    print(explainer.status)  # "DORMANT"
    
    # Get explanation (mock if dormant, real if activated)
    result = explainer.explain_session("my_session_id")
    
Activation:
    Set environment variables:
        DEEPSEEK_ACTIVE=true
        DEEPSEEK_API_KEY=your-api-key
    
    Or run: python deepseek_integration.py --activate
"""

from .deepseek_integration import (
    DeepSeekConfig,
    DeepSeekClient,
    EvidencePackage,
    EvidencePackager,
    ForensicExplainer,
    DEEPSEEK_ACTIVE,
)

__all__ = [
    "DeepSeekConfig",
    "DeepSeekClient", 
    "EvidencePackage",
    "EvidencePackager",
    "ForensicExplainer",
    "DEEPSEEK_ACTIVE",
]

__version__ = "1.0.0"
__status__ = "DORMANT"
