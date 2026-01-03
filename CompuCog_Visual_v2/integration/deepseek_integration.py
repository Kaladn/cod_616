"""
DeepSeek Integration Layer

STATUS: DORMANT (awaiting activation)

This module provides the integration with DeepSeek API for expert-level
forensic explanation of 616 fusion block findings. The LLM serves as the
"system helm" - interpreting deterministic findings for human understanding.

Architecture:
    Your System (Deterministic Truth) → DeepSeek (Expert Explanation) → Human/UI
    
The system NEVER modifies data. It only reads fusion blocks and explains them.

Activation:
    Set DEEPSEEK_ACTIVE=True in config or environment variable
    Provide DEEPSEEK_API_KEY
"""

import os
import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime


# ═══════════════════════════════════════════════════════════════════════════════
# DORMANT FLAG - Set to True when ready to activate
# ═══════════════════════════════════════════════════════════════════════════════
DEEPSEEK_ACTIVE = False
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"


@dataclass
class DeepSeekConfig:
    """Configuration for DeepSeek integration."""
    active: bool = False
    api_key: str = ""
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-chat"
    max_tokens: int = 4096
    temperature: float = 0.3  # Low temp for forensic precision
    
    # Rate limiting
    requests_per_minute: int = 30
    retry_attempts: int = 3
    retry_delay_seconds: float = 2.0
    
    @classmethod
    def from_env(cls) -> "DeepSeekConfig":
        """Load config from environment variables."""
        return cls(
            active=os.environ.get("DEEPSEEK_ACTIVE", "false").lower() == "true",
            api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        )


@dataclass
class EvidencePackage:
    """
    Read-only evidence package for LLM explanation.
    
    Contains summarized findings from fusion blocks - NO raw data,
    only the deterministic conclusions for explanation.
    """
    package_id: str
    session_id: str
    created_at: str
    
    # High-level findings
    manipulation_detected: bool = False
    confidence_score: float = 0.0
    pattern_types: List[str] = field(default_factory=list)
    
    # Key evidence points (summarized, not raw)
    findings: List[Dict[str, Any]] = field(default_factory=list)
    
    # Temporal pattern summary
    temporal_summary: Dict[str, Any] = field(default_factory=dict)
    
    # Statistics
    total_blocks: int = 0
    high_eomm_blocks: int = 0
    causality_violations: int = 0
    
    def to_prompt_context(self) -> str:
        """Convert to context string for LLM prompt."""
        lines = [
            f"Session: {self.session_id}",
            f"Manipulation Detected: {'YES' if self.manipulation_detected else 'NO'}",
            f"Confidence: {self.confidence_score:.1%}",
            f"Pattern Types: {', '.join(self.pattern_types) if self.pattern_types else 'None'}",
            f"",
            f"Statistics:",
            f"  Total 616 Blocks: {self.total_blocks}",
            f"  High EOMM Blocks: {self.high_eomm_blocks}",
            f"  Causality Violations: {self.causality_violations}",
            f"",
            f"Key Findings:"
        ]
        
        for i, finding in enumerate(self.findings[:10], 1):  # Limit to top 10
            lines.append(f"  {i}. [{finding.get('modality', 'unknown')}] {finding.get('description', '')}")
            lines.append(f"     Confidence: {finding.get('confidence', 0):.1%}, Offset: {finding.get('offset_ms', 0):.1f}ms")
        
        if self.temporal_summary:
            lines.append(f"")
            lines.append(f"Temporal Pattern: {self.temporal_summary.get('type', 'unknown')}")
            lines.append(f"  {self.temporal_summary.get('description', '')}")
        
        return "\n".join(lines)


class EvidencePackager:
    """
    Packages fusion block findings into read-only evidence for LLM.
    
    READ-ONLY: Only reads fusion blocks, never modifies them.
    """
    
    def __init__(self, logs_dir: Path = None):
        self.logs_dir = logs_dir or Path(__file__).parent.parent / "logs"
    
    def package_session(self, session_id: str) -> Optional[EvidencePackage]:
        """
        Package a session's findings for LLM explanation.
        
        Args:
            session_id: Session directory name
            
        Returns:
            EvidencePackage with summarized findings (read-only)
        """
        session_dir = self.logs_dir / session_id
        
        if not session_dir.exists():
            return None
        
        # Find fusion blocks file
        fusion_file = None
        for pattern in ["fusion_blocks.jsonl", "truevision/*.jsonl"]:
            matches = list(session_dir.glob(pattern))
            if matches:
                fusion_file = matches[0]
                break
        
        if not fusion_file or not fusion_file.exists():
            # Try truevision subfolder
            tv_dir = session_dir / "truevision"
            if tv_dir.exists():
                jsonl_files = list(tv_dir.glob("*.jsonl"))
                if jsonl_files:
                    fusion_file = jsonl_files[0]
        
        # Load and analyze blocks
        blocks = []
        if fusion_file and fusion_file.exists():
            with open(fusion_file, 'r', encoding='utf-8-sig') as f:
                for line in f:
                    if line.strip():
                        try:
                            blocks.append(json.loads(line))
                        except:
                            pass
        
        # Extract findings
        findings = []
        high_eomm_count = 0
        pattern_types = set()
        total_confidence = 0.0
        
        for block in blocks:
            eomm = block.get("eomm_composite_score", 0)
            flags = block.get("eomm_flags", [])
            offset = block.get("event_offset_ms", 0)
            
            if eomm > 0.5:
                high_eomm_count += 1
                total_confidence += eomm
                
                for flag in flags:
                    pattern_types.add(flag)
                    findings.append({
                        "modality": "truevision",
                        "type": flag,
                        "description": f"{flag} detected with {eomm:.1%} confidence",
                        "confidence": eomm,
                        "offset_ms": offset
                    })
        
        # Build package
        avg_confidence = total_confidence / high_eomm_count if high_eomm_count > 0 else 0
        
        return EvidencePackage(
            package_id=f"pkg_{session_id}_{int(time.time())}",
            session_id=session_id,
            created_at=datetime.now().isoformat(),
            manipulation_detected=high_eomm_count > 0,
            confidence_score=avg_confidence,
            pattern_types=list(pattern_types),
            findings=findings[:20],  # Limit to top 20
            total_blocks=len(blocks),
            high_eomm_blocks=high_eomm_count,
            temporal_summary={
                "type": "6-1-6 temporal fusion",
                "description": f"{len(blocks)} anchor events analyzed across {len(pattern_types)} pattern types"
            }
        )


class DeepSeekClient:
    """
    DeepSeek API client for forensic explanation.
    
    STATUS: DORMANT until DEEPSEEK_ACTIVE=True
    
    READ-ONLY: Only sends evidence summaries, receives explanations.
    Never modifies system data.
    """
    
    def __init__(self, config: DeepSeekConfig = None):
        self.config = config or DeepSeekConfig.from_env()
        self._last_request_time = 0
        
    @property
    def is_active(self) -> bool:
        """Check if DeepSeek integration is active."""
        return self.config.active and bool(self.config.api_key)
    
    def _check_dormant(self) -> bool:
        """Check if system is dormant. Returns True if should skip."""
        if not self.is_active:
            print("[DeepSeek] DORMANT - Integration not activated")
            print("           Set DEEPSEEK_ACTIVE=true and provide DEEPSEEK_API_KEY to activate")
            return True
        return False
    
    def explain_evidence(self, evidence: EvidencePackage) -> Optional[Dict[str, Any]]:
        """
        Send evidence to DeepSeek for expert explanation.
        
        Args:
            evidence: Read-only evidence package
            
        Returns:
            Structured explanation or None if dormant
        """
        if self._check_dormant():
            return self._mock_explanation(evidence)
        
        # Build prompt
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(evidence)
        
        # Make API call
        response = self._call_api(system_prompt, user_prompt)
        
        if response:
            return self._parse_response(response, evidence)
        
        return None
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for forensic expert role."""
        return """You are a forensic gaming analyst expert. Your role is to explain 
evidence of potential matchmaking manipulation (EOMM - Engagement Optimized Matchmaking) 
detected by an automated analysis system.

The system uses a 6-1-6 temporal fusion architecture:
- 6 frames of precursor data (what happened before)
- 1 anchor frame (the detection event)
- 6 frames of consequence data (what happened after)

This allows detection of causality violations where effects precede causes,
indicating server-side manipulation rather than legitimate gameplay.

Your explanations should be:
1. TECHNICAL - For developers and engineers
2. PRACTICAL - For affected players
3. FORENSIC - For legal/evidentiary purposes

Base your analysis ONLY on the evidence provided. Do not speculate beyond the data.
Be precise, factual, and cite specific findings from the evidence."""
    
    def _build_user_prompt(self, evidence: EvidencePackage) -> str:
        """Build the user prompt with evidence context."""
        return f"""Analyze the following forensic evidence and provide a structured explanation:

{evidence.to_prompt_context()}

Provide your analysis in the following format:

## Technical Analysis
[Detailed technical explanation of what was detected]

## Player Impact
[What this means for the affected player in practical terms]

## Forensic Summary
[Summary suitable for evidentiary purposes]

## Recommendations
[Specific next steps based on the findings]"""
    
    def _call_api(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Make API call to DeepSeek."""
        try:
            import requests
            
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.config.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature
            }
            
            response = requests.post(
                f"{self.config.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            else:
                print(f"[DeepSeek] API error: {response.status_code} - {response.text}")
                return None
                
        except ImportError:
            print("[DeepSeek] requests library not installed. Run: pip install requests")
            return None
        except Exception as e:
            print(f"[DeepSeek] Error: {e}")
            return None
    
    def _parse_response(self, response: str, evidence: EvidencePackage) -> Dict[str, Any]:
        """Parse LLM response into structured format."""
        return {
            "explanation_id": f"exp_{evidence.package_id}",
            "evidence_id": evidence.package_id,
            "session_id": evidence.session_id,
            "timestamp": datetime.now().isoformat(),
            "raw_explanation": response,
            "evidence_summary": {
                "manipulation_detected": evidence.manipulation_detected,
                "confidence": evidence.confidence_score,
                "patterns": evidence.pattern_types,
                "block_count": evidence.total_blocks
            }
        }
    
    def _mock_explanation(self, evidence: EvidencePackage) -> Dict[str, Any]:
        """Return mock explanation when dormant (for testing)."""
        return {
            "explanation_id": f"mock_{evidence.package_id}",
            "evidence_id": evidence.package_id,
            "session_id": evidence.session_id,
            "timestamp": datetime.now().isoformat(),
            "status": "DORMANT",
            "message": "DeepSeek integration is dormant. Activate to get real explanations.",
            "evidence_summary": {
                "manipulation_detected": evidence.manipulation_detected,
                "confidence": evidence.confidence_score,
                "patterns": evidence.pattern_types,
                "block_count": evidence.total_blocks
            },
            "mock_explanation": {
                "technical": f"[DORMANT] Would analyze {evidence.total_blocks} fusion blocks with {len(evidence.pattern_types)} pattern types.",
                "practical": f"[DORMANT] Would explain impact of {evidence.high_eomm_blocks} high-EOMM events.",
                "forensic": f"[DORMANT] Would provide evidentiary summary with {evidence.confidence_score:.1%} average confidence."
            }
        }


class ForensicExplainer:
    """
    Main interface for forensic explanation.
    
    Orchestrates: Evidence Packaging → DeepSeek API → Structured Output
    
    STATUS: DORMANT until activated
    READ-ONLY: Never modifies source data
    """
    
    def __init__(self, logs_dir: Path = None, config: DeepSeekConfig = None):
        self.packager = EvidencePackager(logs_dir)
        self.client = DeepSeekClient(config)
    
    @property
    def is_active(self) -> bool:
        """Check if the explainer is active."""
        return self.client.is_active
    
    @property
    def status(self) -> str:
        """Get current status."""
        if self.is_active:
            return "ACTIVE"
        return "DORMANT"
    
    def explain_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Generate expert explanation for a session's findings.
        
        Args:
            session_id: Session to explain
            
        Returns:
            Structured explanation with technical/practical/forensic views
        """
        print(f"[ForensicExplainer] Status: {self.status}")
        print(f"[ForensicExplainer] Packaging evidence for session: {session_id}")
        
        # Package evidence (read-only)
        evidence = self.packager.package_session(session_id)
        
        if not evidence:
            print(f"[ForensicExplainer] No evidence found for session: {session_id}")
            return None
        
        print(f"[ForensicExplainer] Packaged {evidence.total_blocks} blocks, {evidence.high_eomm_blocks} high-EOMM")
        
        # Get explanation (or mock if dormant)
        explanation = self.client.explain_evidence(evidence)
        
        return explanation
    
    def list_sessions(self) -> List[str]:
        """List available sessions for explanation."""
        if not self.packager.logs_dir.exists():
            return []
        
        sessions = []
        for d in self.packager.logs_dir.iterdir():
            if d.is_dir() and not d.name.startswith('.'):
                sessions.append(d.name)
        
        return sorted(sessions)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """CLI for DeepSeek integration (DORMANT by default)."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="DeepSeek Forensic Explanation System (DORMANT)"
    )
    parser.add_argument("--session", "-s", type=str, help="Session ID to explain")
    parser.add_argument("--list", "-l", action="store_true", help="List available sessions")
    parser.add_argument("--status", action="store_true", help="Show integration status")
    parser.add_argument("--activate", action="store_true", help="Show activation instructions")
    
    args = parser.parse_args()
    
    explainer = ForensicExplainer()
    
    if args.status:
        print(f"\n{'='*60}")
        print("DeepSeek Integration Status")
        print(f"{'='*60}")
        print(f"  Status: {explainer.status}")
        print(f"  Active: {explainer.is_active}")
        print(f"  API Key Set: {'Yes' if DEEPSEEK_API_KEY else 'No'}")
        print(f"{'='*60}\n")
        return
    
    if args.activate:
        print(f"\n{'='*60}")
        print("DeepSeek Activation Instructions")
        print(f"{'='*60}")
        print("""
To activate DeepSeek integration:

1. Get API key from: https://platform.deepseek.com

2. Set environment variables:
   Windows (PowerShell):
     $env:DEEPSEEK_ACTIVE = "true"
     $env:DEEPSEEK_API_KEY = "your-api-key-here"
   
   Windows (CMD):
     set DEEPSEEK_ACTIVE=true
     set DEEPSEEK_API_KEY=your-api-key-here
   
   Linux/Mac:
     export DEEPSEEK_ACTIVE=true
     export DEEPSEEK_API_KEY=your-api-key-here

3. Or modify this file:
   Set DEEPSEEK_ACTIVE = True
   Set DEEPSEEK_API_KEY = "your-api-key-here"

4. Run: python deepseek_integration.py --session <session_id>
""")
        print(f"{'='*60}\n")
        return
    
    if args.list:
        sessions = explainer.list_sessions()
        print(f"\nAvailable sessions ({len(sessions)}):")
        for s in sessions:
            print(f"  - {s}")
        print()
        return
    
    if args.session:
        print(f"\n[DeepSeek] Explaining session: {args.session}")
        print(f"[DeepSeek] Status: {explainer.status}")
        
        result = explainer.explain_session(args.session)
        
        if result:
            print(f"\n{'='*60}")
            print("Explanation Result")
            print(f"{'='*60}")
            print(json.dumps(result, indent=2, default=str))
        else:
            print("No explanation generated.")
        return
    
    # Default: show status
    print(f"\n{'='*60}")
    print("DeepSeek Forensic Expert - System Helm")
    print(f"{'='*60}")
    print(f"  Status: {explainer.status}")
    print(f"  Sessions Available: {len(explainer.list_sessions())}")
    print()
    print("Commands:")
    print("  --status     Show integration status")
    print("  --activate   Show activation instructions")
    print("  --list       List available sessions")
    print("  --session X  Explain session X")
    print()
    if not explainer.is_active:
        print("⚠️  DORMANT: Run --activate for setup instructions")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
