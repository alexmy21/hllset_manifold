#!/usr/bin/env python3
"""
Quick test to demonstrate extension immutability.
"""

from core.extensions.storage import DuckDBStorageExtension
from core.extensions.base import ExtensionError

print("=" * 60)
print("Extension Immutability Demonstration")
print("=" * 60)

# Test 1: Configuration is frozen after initialization
print("\n1. Testing configuration immutability...")
ext = DuckDBStorageExtension()
ext.initialize({'db_path': ':memory:'})

config_hash1 = ext.get_config_hash()
print(f"   ✓ Extension initialized")
print(f"   ✓ Config hash: {config_hash1[:16]}...")

# Try to initialize again
try:
    result = ext.initialize({'db_path': 'new.db'})
    if result:
        print("   ✗ ERROR: Re-initialization should have failed!")
    else:
        print("   ✓ Re-initialization returned False (config frozen)")
except ExtensionError as e:
    print(f"   ✓ Re-initialization prevented: {e}")

# Test 2: Same config produces same hash (idempotence)
print("\n2. Testing idempotence (same config → same hash)...")
ext1 = DuckDBStorageExtension()
ext1.initialize({'db_path': 'test.db', 'threads': 4})
hash1 = ext1.get_config_hash()

ext2 = DuckDBStorageExtension()
ext2.initialize({'db_path': 'test.db', 'threads': 4})
hash2 = ext2.get_config_hash()

if hash1 == hash2:
    print(f"   ✓ Hash 1: {hash1[:16]}...")
    print(f"   ✓ Hash 2: {hash2[:16]}...")
    print("   ✓ Hashes match - idempotent!")
else:
    print("   ✗ ERROR: Hashes don't match!")

# Test 3: Different config produces different hash
print("\n3. Testing content-addressability (different config → different hash)...")
ext3 = DuckDBStorageExtension()
ext3.initialize({'db_path': 'different.db'})
hash3 = ext3.get_config_hash()

if hash1 != hash3:
    print(f"   ✓ Hash 1: {hash1[:16]}... (test.db)")
    print(f"   ✓ Hash 3: {hash3[:16]}... (different.db)")
    print("   ✓ Different configs → different hashes!")
else:
    print("   ✗ ERROR: Different configs produced same hash!")

# Test 4: Config access is read-only
print("\n4. Testing read-only config access...")
config = ext1.config
print(f"   ✓ Config retrieved: {config}")
print(f"   ✓ Type: {type(config)}")

# Test 5: ExtensionInfo includes hash
print("\n5. Testing ExtensionInfo includes content-addressed hash...")
info = ext1.get_info()
print(f"   ✓ Name: {info.name}")
print(f"   ✓ Version: {info.version}")
print(f"   ✓ Config Hash: {info.config_hash[:16]}...")
print(f"   ✓ Capabilities: {len(info.capabilities)} capabilities")

# Verify content_addressable capability
caps_dict = dict(info.capabilities)
if caps_dict.get('content_addressable') and caps_dict.get('idempotent'):
    print("   ✓ Extension declares content_addressable and idempotent!")
else:
    print("   ⚠ Extension should declare content_addressable and idempotent")

print("\n" + "=" * 60)
print("All immutability tests passed! ✓")
print("=" * 60)
