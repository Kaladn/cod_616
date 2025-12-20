"""
Recognition CLI - Phase 7 Implementation
CompuCog 616 COD Telemetry Engine

Command-line interface for Recognition Field analysis.

Modes:
- index-baseline: Build baseline index from fair matches
- analyze: Analyze a single match fingerprint
- compare: Compare two fingerprints directly

Built: November 26, 2025
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List

from recognition_field import RecognitionField, BaselineIndex


class RecognitionCLI:
    """CLI for Recognition Field operations."""
    
    def __init__(self):
        self.recognition_dir = Path(__file__).parent
        self.baseline_path = self.recognition_dir / 'profiles' / 'baseline_index.json'
        self.reports_dir = self.recognition_dir / 'reports'
        self.reports_dir.mkdir(exist_ok=True)
    
    def index_baseline(self, baseline_dir: str):
        """
        Build baseline index from all fingerprints in directory.
        
        Args:
            baseline_dir: Path to directory with baseline fingerprints
        """
        print("[Recognition CLI - INDEX BASELINE]")
        print(f"  Source: {baseline_dir}")
        print(f"  Target: {self.baseline_path}\n")
        
        # Find all fingerprint files
        baseline_path = Path(baseline_dir)
        fingerprint_files = sorted(baseline_path.glob('fingerprint_*.json'))
        
        if not fingerprint_files:
            print("  [ERROR] No fingerprint files found")
            return
        
        print(f"  Found {len(fingerprint_files)} fingerprints")
        
        # Build baseline
        rf = RecognitionField()
        baseline = rf.build_baseline([str(f) for f in fingerprint_files])
        
        # Save
        rf.save_baseline(baseline, str(self.baseline_path))
        
        print(f"\n✓ Baseline index built successfully")
        print(f"  Count: {baseline.count} matches")
        print(f"  Dims: {len(baseline.mean)}")
        print(f"  Mean range: [{baseline.mean.min():.3f}, {baseline.mean.max():.3f}]")
        print(f"  Std range: [{baseline.std.min():.3f}, {baseline.std.max():.3f}]")
    
    def analyze(self, fingerprint_path: str, save_report: bool = True):
        """
        Analyze a match fingerprint.
        
        Args:
            fingerprint_path: Path to fingerprint JSON
            save_report: Whether to save report to disk
        """
        print("[Recognition CLI - ANALYZE MATCH]")
        print(f"  Fingerprint: {fingerprint_path}")
        print(f"  Baseline: {self.baseline_path}\n")
        
        # Check baseline exists
        if not self.baseline_path.exists():
            print("  [ERROR] No baseline index found")
            print("  Run 'index-baseline' first")
            return
        
        # Load fingerprint
        with open(fingerprint_path, 'r') as f:
            fingerprint = json.load(f)
        
        # Load baseline and analyze
        rf = RecognitionField()
        rf.load_baseline(str(self.baseline_path))
        
        print(f"  Analyzing...")
        report = rf.analyze(fingerprint)
        
        # Print report
        self._print_report(report)
        
        # Save report
        if save_report:
            report_path = self._save_report(report)
            print(f"\n✓ Report saved: {report_path}")
    
    def compare(self, path_a: str, path_b: str):
        """
        Compare two fingerprints directly.
        
        Args:
            path_a: Path to first fingerprint
            path_b: Path to second fingerprint
        """
        print("[Recognition CLI - COMPARE MATCHES]")
        print(f"  Match A: {path_a}")
        print(f"  Match B: {path_b}\n")
        
        # Load fingerprints
        with open(path_a, 'r') as f:
            fp_a = json.load(f)
        
        with open(path_b, 'r') as f:
            fp_b = json.load(f)
        
        # Compare
        rf = RecognitionField()
        comparison = rf.compare(fp_a, fp_b)
        
        # Print comparison
        print("[COMPARISON METRICS]")
        print(f"  Euclidean distance: {comparison['euclidean_distance']:.3f}")
        print(f"  Cosine similarity: {comparison['cosine_similarity']:.3f}")
        print(f"\n[BLOCK DISTANCES]")
        print(f"  Visual: {comparison['visual_block_distance']:.3f}")
        print(f"  Gamepad: {comparison['gamepad_block_distance']:.3f}")
        print(f"  Network: {comparison['network_block_distance']:.3f}")
        print(f"  Cross-modal: {comparison['crossmodal_block_distance']:.3f}")
        
        # If baseline exists, analyze both
        if self.baseline_path.exists():
            print(f"\n[INDIVIDUAL ANALYSIS]")
            rf.load_baseline(str(self.baseline_path))
            
            print(f"\nMatch A:")
            report_a = rf.analyze(fp_a)
            print(f"  Verdict: {report_a.verdict} (confidence: {report_a.confidence:.2f})")
            print(f"  Global anomaly: {report_a.global_anomaly_score:.2f}")
            
            print(f"\nMatch B:")
            report_b = rf.analyze(fp_b)
            print(f"  Verdict: {report_b.verdict} (confidence: {report_b.confidence:.2f})")
            print(f"  Global anomaly: {report_b.global_anomaly_score:.2f}")
        
        print(f"\n✓ Comparison complete")
    
    def batch_analyze(self, fingerprints_dir: str):
        """
        Analyze all fingerprints in directory.
        
        Args:
            fingerprints_dir: Path to directory with fingerprints
        """
        print("[Recognition CLI - BATCH ANALYZE]")
        print(f"  Directory: {fingerprints_dir}")
        print(f"  Baseline: {self.baseline_path}\n")
        
        # Check baseline
        if not self.baseline_path.exists():
            print("  [ERROR] No baseline index found")
            print("  Run 'index-baseline' first")
            return
        
        # Find fingerprints
        fp_dir = Path(fingerprints_dir)
        fingerprint_files = sorted(fp_dir.glob('fingerprint_*.json'))
        
        if not fingerprint_files:
            print("  [ERROR] No fingerprints found")
            return
        
        print(f"  Found {len(fingerprint_files)} fingerprints\n")
        
        # Load baseline
        rf = RecognitionField()
        rf.load_baseline(str(self.baseline_path))
        
        # Analyze all
        verdicts = {'normal': 0, 'suspicious': 0, 'manipulated_likely': 0, 'manipulated_certain': 0}
        
        for i, fp_path in enumerate(fingerprint_files, 1):
            with open(fp_path, 'r') as f:
                fp = json.load(f)
            
            report = rf.analyze(fp)
            verdicts[report.verdict] += 1
            
            print(f"  [{i:>3}/{len(fingerprint_files)}] {fp_path.name}: "
                  f"{report.verdict} (anomaly: {report.global_anomaly_score:.2f})")
            
            # Save report
            self._save_report(report)
        
        # Summary
        print(f"\n[BATCH SUMMARY]")
        print(f"  Total analyzed: {len(fingerprint_files)}")
        print(f"  Normal: {verdicts['normal']}")
        print(f"  Suspicious: {verdicts['suspicious']}")
        print(f"  Manipulated (likely): {verdicts['manipulated_likely']}")
        print(f"  Manipulated (certain): {verdicts['manipulated_certain']}")
        
        manip_rate = (verdicts['manipulated_likely'] + verdicts['manipulated_certain']) / len(fingerprint_files) * 100
        print(f"  Manipulation rate: {manip_rate:.1f}%")
        
        print(f"\n✓ Batch analysis complete")
    
    def _print_report(self, report):
        """Pretty-print a recognition report."""
        print("[RECOGNITION REPORT]")
        print(f"  Match ID: {report.match_id}")
        print(f"  Profile: {report.profile}")
        print(f"  Duration: {report.duration_seconds:.1f}s ({report.frame_count} frames)")
        print(f"\n[BLOCK Z-SCORES]")
        print(f"  Global anomaly: {report.global_anomaly_score:.2f}")
        print(f"  Visual: {report.visual_block_z:.2f}")
        print(f"  Gamepad: {report.gamepad_block_z:.2f}")
        print(f"  Network: {report.network_block_z:.2f}")
        print(f"  Cross-modal: {report.crossmodal_block_z:.2f}")
        print(f"  Meta: {report.meta_block_z:.2f}")
        
        print(f"\n[SUSPECT CHANNELS]")
        print(f"  Aim assist: {report.aim_assist.level.upper()} ({report.aim_assist.score:.2f})")
        print(f"    Key dims: {report.aim_assist.contributing_dims}")
        print(f"  Recoil comp: {report.recoil_compensation.level.upper()} ({report.recoil_compensation.score:.2f})")
        print(f"    Key dims: {report.recoil_compensation.contributing_dims}")
        print(f"  EOMM lag: {report.eomm_lag.level.upper()} ({report.eomm_lag.score:.2f})")
        print(f"    Key dims: {report.eomm_lag.contributing_dims}")
        print(f"  Network bias: {report.network_priority_bias.level.upper()} ({report.network_priority_bias.score:.2f})")
        print(f"    Key dims: {report.network_priority_bias.contributing_dims}")
        
        print(f"\n[VERDICT]")
        print(f"  {report.verdict.upper().replace('_', ' ')}")
        print(f"  Confidence: {report.confidence:.1%}")
        
        print(f"\n[EXPLANATION]")
        for line in report.explanation:
            print(f"  • {line}")
    
    def _save_report(self, report) -> Path:
        """Save report to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.reports_dir / f"recognition_report_{timestamp}.json"
        
        with open(report_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        
        return report_path


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Recognition Field CLI - COD 616 Manipulation Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build baseline from fair matches
  python -m cod_616.recognition.recognition_cli index-baseline \\
      --baseline-dir data/fingerprints/baseline

  # Analyze single match
  python -m cod_616.recognition.recognition_cli analyze \\
      --path data/fingerprints/real/fingerprint_20251126_140000.json

  # Compare two matches
  python -m cod_616.recognition.recognition_cli compare \\
      --a data/fingerprints/baseline/fingerprint_good.json \\
      --b data/fingerprints/real/fingerprint_bad.json

  # Batch analyze all real matches
  python -m cod_616.recognition.recognition_cli batch \\
      --dir data/fingerprints/real
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # index-baseline
    parser_index = subparsers.add_parser('index-baseline', help='Build baseline index')
    parser_index.add_argument('--baseline-dir', type=str, default='data/fingerprints/baseline',
                             help='Directory with baseline fingerprints')
    
    # analyze
    parser_analyze = subparsers.add_parser('analyze', help='Analyze single match')
    parser_analyze.add_argument('--path', type=str, required=True,
                               help='Path to fingerprint JSON')
    parser_analyze.add_argument('--no-save', action='store_true',
                               help='Do not save report')
    
    # compare
    parser_compare = subparsers.add_parser('compare', help='Compare two matches')
    parser_compare.add_argument('--a', type=str, required=True,
                               help='Path to first fingerprint')
    parser_compare.add_argument('--b', type=str, required=True,
                               help='Path to second fingerprint')
    
    # batch
    parser_batch = subparsers.add_parser('batch', help='Batch analyze directory')
    parser_batch.add_argument('--dir', type=str, required=True,
                             help='Directory with fingerprints')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    cli = RecognitionCLI()
    
    if args.command == 'index-baseline':
        cli.index_baseline(args.baseline_dir)
    
    elif args.command == 'analyze':
        cli.analyze(args.path, save_report=not args.no_save)
    
    elif args.command == 'compare':
        cli.compare(args.a, args.b)
    
    elif args.command == 'batch':
        cli.batch_analyze(args.dir)


if __name__ == "__main__":
    main()
