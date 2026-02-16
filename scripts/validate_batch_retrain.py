"""
Quick validation of batch retrain implementation.
Checks that all components are properly integrated.

Usage:
    python scripts/validate_batch_retrain.py
"""

import sys
from pathlib import Path

def validate_imports():
    """Check that all required modules can be imported."""
    print("Validating imports...")
    
    try:
        from scripts.batch_retrain import (
            fetch_recent_swipes,
            build_updated_training_matrix,
            retrain_cf_model,
            save_artifacts,
            update_last_training_date,
            get_last_training_date,
            main
        )
        print("✓ batch_retrain module imports successful")
    except ImportError as e:
        print(f"✗ Failed to import batch_retrain: {e}")
        return False
    
    try:
        from db.connection import get_db
        print("✓ Database connection imports successful")
    except ImportError as e:
        print(f"✗ Failed to import database: {e}")
        return False
    
    try:
        from db import Interactions
        print("✓ Interactions imports successful")
    except ImportError as e:
        print(f"✗ Failed to import Interactions: {e}")
        return False
    
    return True


def validate_database_schema():
    """Check that metadata table exists in database."""
    print("\nValidating database schema...")
    
    try:
        from db.connection import get_db
        db = get_db()
        
        if not db.table_exists("metadata"):
            print("✗ metadata table does not exist")
            return False
        print("✓ metadata table exists")
        
        if not db.table_exists("interactions"):
            print("✗ interactions table does not exist")
            return False
        print("✓ interactions table exists")
        
        return True
    except Exception as e:
        print(f"✗ Database schema validation failed: {e}")
        return False


def validate_metadata_initialization():
    """Check that metadata is initialized."""
    print("\nValidating metadata initialization...")
    
    try:
        from db.connection import get_db
        db = get_db()
        conn = db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT value FROM metadata WHERE key = ?", ("last_training_date",))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            print(f"✓ Metadata initialized: last_training_date={result[0]}")
            return True
        else:
            print("✗ Metadata not initialized")
            return False
    except Exception as e:
        print(f"✗ Metadata validation failed: {e}")
        return False


def validate_model_artifacts():
    """Check that model artifacts exist."""
    print("\nValidating model artifacts...")
    
    from common.constants import PATHS
    
    artifacts = {
        "user_factors": PATHS["user_factors"],
        "book_factors": PATHS["book_factors"],
        "user_idx_pkl": PATHS["user_idx_pkl"],
        "book_idx_pkl": PATHS["book_idx_pkl"],
        "clean_ratings": PATHS["clean_ratings"],
    }
    
    all_exist = True
    for name, path in artifacts.items():
        if Path(path).exists():
            size = Path(path).stat().st_size
            print(f"✓ {name}: {path} ({size:,} bytes)")
        else:
            print(f"✗ {name}: {path} NOT FOUND")
            all_exist = False
    
    return all_exist


def validate_server_endpoint():
    """Check that /retrain/trigger endpoint is registered."""
    print("\nValidating server endpoint...")
    
    try:
        with open("server/main.py", "r") as f:
            content = f.read()
            if "/retrain/trigger" in content:
                print("✓ /retrain/trigger endpoint is registered")
                return True
            else:
                print("✗ /retrain/trigger endpoint not found")
                return False
    except Exception as e:
        print(f"✗ Failed to check endpoint: {e}")
        return False


def main():
    """Run all validations."""
    print("=" * 70)
    print("BATCH RETRAIN IMPLEMENTATION VALIDATION")
    print("=" * 70)
    
    checks = [
        ("Imports", validate_imports),
        ("Database Schema", validate_database_schema),
        ("Metadata Initialization", validate_metadata_initialization),
        ("Model Artifacts", validate_model_artifacts),
        ("Server Endpoint", validate_server_endpoint),
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"✗ {check_name} validation crashed: {e}")
            results.append((check_name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for check_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {check_name}")
    
    print(f"\nTotal: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n✓ ALL VALIDATIONS PASSED - Ready for testing!")
        return True
    else:
        print(f"\n✗ {total - passed} VALIDATION(S) FAILED")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
