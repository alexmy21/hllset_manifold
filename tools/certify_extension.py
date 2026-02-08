#!/usr/bin/env python3
"""
Extension Certification Tool

Validates and certifies ManifoldOS extensions for:
  - Immutability (frozen configuration)
  - Idempotence (same input → same output)
  - Content-addressability (stable hashing)
  - Statelessness (no accumulated state)

Usage:
    python tools/certify_extension.py <extension_module> <extension_class>
    
Example:
    python tools/certify_extension.py core.extensions.storage DuckDBStorageExtension
"""

import sys
import os
import importlib
from datetime import datetime

sys.path.insert(0, os.path.abspath('.'))

from core.extensions.stateless_validator import (
    StatelessnessValidator,
    validate_extension_statelessness
)


def certify_extension(module_name: str, class_name: str) -> bool:
    """
    Certify an extension for ManifoldOS.
    
    Runs all validation checks and generates certification report.
    
    Returns:
        True if certified, False otherwise
    """
    print("=" * 80)
    print("ManifoldOS Extension Certification Tool")
    print("=" * 80)
    print(f"\nExtension: {module_name}.{class_name}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Import extension
    try:
        module = importlib.import_module(module_name)
        extension_class = getattr(module, class_name)
        print(f"✓ Extension class loaded: {extension_class.__name__}")
    except Exception as e:
        print(f"✗ Failed to load extension: {e}")
        return False
    
    # Run validation
    print("\n" + "-" * 80)
    print("Running Statelessness Validation...")
    print("-" * 80)
    
    validator = StatelessnessValidator()
    violations = validator.validate_extension(extension_class, class_name)
    
    # Generate report
    report = validator.generate_report(violations)
    print(report)
    
    # Check certification status
    errors = [v for v in violations if v.severity == 'error']
    warnings = [v for v in violations if v.severity == 'warning']
    
    print("\n" + "=" * 80)
    print("Certification Results")
    print("=" * 80)
    
    if errors:
        print(f"\n✗ CERTIFICATION FAILED")
        print(f"  {len(errors)} error(s) must be fixed")
        print(f"  {len(warnings)} warning(s) should be reviewed")
        print("\nExtension is NOT certified for ManifoldOS")
        return False
    elif warnings:
        print(f"\n⚠ CERTIFICATION WITH WARNINGS")
        print(f"  {len(warnings)} warning(s) found")
        print("  Review warnings before production use")
        print("\n✓ Extension meets minimum requirements")
        return True
    else:
        print(f"\n✓ CERTIFICATION PASSED")
        print("  No violations detected")
        print("  Extension is fully certified")
        print("\n✓ Extension certified STATELESS and IMMUTABLE")
        return True


def generate_badge(class_name: str, status: str) -> str:
    """Generate certification badge for documentation."""
    date = datetime.now().strftime('%Y-%m-%d')
    
    if status == 'certified':
        return f"""
    \"\"\"
    {class_name}
    
    ✓ CERTIFIED STATELESS - Validated {date}
    
    Implements ManifoldOS core principles:
      - Immutability: Configuration frozen after init
      - Idempotence: Same input → same output
      - Content-addressability: Stable hashing
      - Statelessness: No accumulated state
    
    Validated with StatelessnessValidator:
      ✓ No mutable class variables
      ✓ No accumulated instance state
      ✓ All operations idempotent
      ✓ Multiple instances independent
    \"\"\"
"""
    else:
        return f"""
    \"\"\"
    {class_name}
    
    ⚠ NOT CERTIFIED - Validation failed {date}
    
    Review violations and re-certify before production use.
    \"\"\"
"""


def main():
    """Main entry point."""
    if len(sys.argv) != 3:
        print("Usage: python tools/certify_extension.py <module> <class>")
        print("Example: python tools/certify_extension.py core.extensions.storage DuckDBStorageExtension")
        sys.exit(1)
    
    module_name = sys.argv[1]
    class_name = sys.argv[2]
    
    # Run certification
    certified = certify_extension(module_name, class_name)
    
    # Generate badge
    print("\n" + "=" * 80)
    print("Suggested Documentation Badge")
    print("=" * 80)
    
    status = 'certified' if certified else 'not_certified'
    badge = generate_badge(class_name, status)
    print(badge)
    
    # Exit with status
    sys.exit(0 if certified else 1)


if __name__ == "__main__":
    main()
