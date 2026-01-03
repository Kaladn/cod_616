#!/usr/bin/env python3
"""
🔗 CHAIN REACTION SMOKE TEST RUNNER

Executes the complete smoke test suite:
1. Module isolation tests
2. Chain reaction tests
3. Full system test
4. Failure injection tests

Usage:
    python run_smoke_tests.py              # Run all tests
    python run_smoke_tests.py --quick      # Run quick subset
    python run_smoke_tests.py --chain-only # Only chain reaction tests
    python run_smoke_tests.py --failures   # Only failure injection tests
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def color(text: str, c: str) -> str:
    return f"{c}{text}{Colors.RESET}"

def banner(text: str):
    print("\n" + "=" * 80)
    print(color(text, Colors.BOLD + Colors.BLUE))
    print("=" * 80)

def run_pytest(test_file: str, verbose: bool = True) -> tuple:
    """
    Run pytest on a test file.
    Returns (passed, failed, duration)
    """
    cmd = [
        sys.executable, "-m", "pytest",
        test_file,
        "-v" if verbose else "-q",
        "--tb=short",
        "-x"  # Stop on first failure
    ]
    
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    duration = time.time() - start
    
    # Parse results (rough)
    output = result.stdout + result.stderr
    
    passed = output.count(" PASSED")
    failed = output.count(" FAILED") + output.count(" ERROR")
    
    return passed, failed, duration, result.returncode == 0, output


def main():
    parser = argparse.ArgumentParser(description="Run Chain Reaction Smoke Tests")
    parser.add_argument("--quick", action="store_true", help="Run quick subset of tests")
    parser.add_argument("--chain-only", action="store_true", help="Only chain reaction tests")
    parser.add_argument("--failures", action="store_true", help="Only failure injection tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--no-stop", action="store_true", help="Don't stop on failure")
    
    args = parser.parse_args()
    
    # Get test directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    banner("🔗 CHAIN REACTION SMOKE TEST SUITE")
    print(f"Start Time: {datetime.now().isoformat()}")
    print(f"Test Directory: {script_dir}")
    
    # Define test files
    all_tests = [
        ("TrueVision Source", "test_01_truevision_source.py"),
        ("Schema Map", "test_02_schema_map.py"),
        ("Pulse Writer", "test_03_pulse_writer.py"),
        ("WAL Writer", "test_04_wal_writer.py"),
        ("Binary Log", "test_05_binary_log.py"),
        ("Logger Pulse Writer", "test_06_logger_pulse_writer.py"),
        ("Immortal Loggers", "test_07_immortal_loggers.py"),
        ("Orchestrator", "test_08_orchestrator.py"),
        ("Chain Reaction", "test_chain_reaction.py"),
        ("Failure Injection", "test_failure_injection.py"),
    ]
    
    # Filter based on args
    if args.quick:
        tests = [t for t in all_tests if t[0] in ["Schema Map", "Chain Reaction"]]
    elif args.chain_only:
        tests = [t for t in all_tests if "Chain" in t[0]]
    elif args.failures:
        tests = [t for t in all_tests if "Failure" in t[0]]
    else:
        tests = all_tests
    
    # Run tests
    results = []
    total_passed = 0
    total_failed = 0
    total_duration = 0
    
    banner("PHASE 1: MODULE ISOLATION TESTS")
    
    isolation_tests = [t for t in tests if "Chain" not in t[0] and "Failure" not in t[0]]
    for name, test_file in isolation_tests:
        if not Path(test_file).exists():
            print(f"⚠️ {name}: Test file not found: {test_file}")
            continue
        
        print(f"\n[{len(results)+1}/{len(tests)}] Testing {name}...")
        passed, failed, duration, success, output = run_pytest(test_file, args.verbose)
        
        results.append((name, passed, failed, duration, success))
        total_passed += passed
        total_failed += failed
        total_duration += duration
        
        status = color("✅ PASS", Colors.GREEN) if success else color("❌ FAIL", Colors.RED)
        print(f"   {status}: {passed} passed, {failed} failed ({duration:.1f}s)")
        
        if not success and not args.no_stop:
            if args.verbose:
                print(output)
            print(color("\n⛔ Stopping due to test failure. Use --no-stop to continue.", Colors.RED))
            break
    
    # Chain reaction tests
    chain_tests = [t for t in tests if "Chain" in t[0]]
    if chain_tests and (not results or results[-1][4] or args.no_stop):
        banner("PHASE 2: CHAIN REACTION TESTS")
        
        for name, test_file in chain_tests:
            if not Path(test_file).exists():
                print(f"⚠️ {name}: Test file not found: {test_file}")
                continue
            
            print(f"\n[{len(results)+1}/{len(tests)}] Testing {name}...")
            passed, failed, duration, success, output = run_pytest(test_file, args.verbose)
            
            results.append((name, passed, failed, duration, success))
            total_passed += passed
            total_failed += failed
            total_duration += duration
            
            status = color("✅ PASS", Colors.GREEN) if success else color("❌ FAIL", Colors.RED)
            print(f"   {status}: {passed} passed, {failed} failed ({duration:.1f}s)")
    
    # Failure injection tests
    failure_tests = [t for t in tests if "Failure" in t[0]]
    if failure_tests and (not results or results[-1][4] or args.no_stop):
        banner("PHASE 3: FAILURE INJECTION TESTS")
        
        for name, test_file in failure_tests:
            if not Path(test_file).exists():
                print(f"⚠️ {name}: Test file not found: {test_file}")
                continue
            
            print(f"\n[{len(results)+1}/{len(tests)}] Testing {name}...")
            passed, failed, duration, success, output = run_pytest(test_file, args.verbose)
            
            results.append((name, passed, failed, duration, success))
            total_passed += passed
            total_failed += failed
            total_duration += duration
            
            status = color("✅ PASS", Colors.GREEN) if success else color("❌ FAIL", Colors.RED)
            print(f"   {status}: {passed} passed, {failed} failed ({duration:.1f}s)")
    
    # Final report
    banner("🔗 CHAIN REACTION SMOKE TEST REPORT")
    print(f"Test End: {datetime.now().isoformat()}")
    print(f"Duration: {total_duration:.1f} seconds")
    print()
    
    print("MODULE RESULTS:")
    for name, passed, failed, duration, success in results:
        status = color("✅", Colors.GREEN) if success else color("❌", Colors.RED)
        print(f"  {status} {name}: {passed}/{passed+failed} passed ({duration:.1f}s)")
    
    print()
    print("SUMMARY:")
    print(f"  Total Tests: {total_passed + total_failed}")
    print(f"  Passed: {color(str(total_passed), Colors.GREEN)}")
    print(f"  Failed: {color(str(total_failed), Colors.RED) if total_failed else '0'}")
    print(f"  Duration: {total_duration:.1f}s")
    
    # Final verdict
    all_passed = all(r[4] for r in results)
    print()
    if all_passed:
        print(color("✅ ALL CHAINS INTACT - SYSTEM STABLE", Colors.GREEN + Colors.BOLD))
        return 0
    else:
        print(color("❌ CHAIN BREAKS DETECTED - FIX REQUIRED", Colors.RED + Colors.BOLD))
        return 1


if __name__ == "__main__":
    sys.exit(main())
