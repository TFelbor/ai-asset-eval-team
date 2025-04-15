#!/usr/bin/env python
"""
Script to help with the transition from FastAPI to Streamlit.

This script checks for any remaining FastAPI dependencies and provides
guidance on how to remove them.
"""
import os
import sys
import re
import argparse
from pathlib import Path

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Transition from FastAPI to Streamlit")
    parser.add_argument(
        "--check",
        action="store_true",
        default=False,
        help="Check for FastAPI dependencies"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        default=False,
        help="Remove FastAPI dependencies (use with caution)"
    )
    return parser.parse_args()

def find_fastapi_imports(file_path):
    """Find FastAPI imports in a file."""
    fastapi_imports = []
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        # Look for FastAPI imports
        if re.search(r"import\s+fastapi|from\s+fastapi", content):
            fastapi_imports.append("FastAPI import")
        # Look for uvicorn imports
        if re.search(r"import\s+uvicorn|from\s+uvicorn", content):
            fastapi_imports.append("uvicorn import")
        # Look for Jinja2Templates imports
        if re.search(r"import\s+Jinja2Templates|from\s+fastapi\.templating\s+import\s+Jinja2Templates", content):
            fastapi_imports.append("Jinja2Templates import")
        # Look for StaticFiles imports
        if re.search(r"import\s+StaticFiles|from\s+fastapi\.staticfiles\s+import\s+StaticFiles", content):
            fastapi_imports.append("StaticFiles import")
        # Look for CORSMiddleware imports
        if re.search(r"import\s+CORSMiddleware|from\s+fastapi\.middleware\.cors\s+import\s+CORSMiddleware", content):
            fastapi_imports.append("CORSMiddleware import")
    return fastapi_imports

def find_fastapi_usage(file_path):
    """Find FastAPI usage in a file."""
    fastapi_usage = []
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        # Look for FastAPI initialization
        if re.search(r"app\s*=\s*FastAPI\(", content):
            fastapi_usage.append("FastAPI initialization")
        # Look for app.include_router
        if re.search(r"app\.include_router\(", content):
            fastapi_usage.append("Router inclusion")
        # Look for app.add_middleware
        if re.search(r"app\.add_middleware\(", content):
            fastapi_usage.append("Middleware addition")
        # Look for app.mount
        if re.search(r"app\.mount\(", content):
            fastapi_usage.append("Static files mounting")
        # Look for @app.get, @app.post, etc.
        if re.search(r"@app\.(get|post|put|delete|patch)\(", content):
            fastapi_usage.append("Route decoration")
        # Look for uvicorn.run
        if re.search(r"uvicorn\.run\(", content):
            fastapi_usage.append("uvicorn.run call")
    return fastapi_usage

def check_fastapi_dependencies():
    """Check for FastAPI dependencies."""
    print("Checking for FastAPI dependencies...")
    
    # Find all Python files
    python_files = []
    for root, _, files in os.walk("."):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    
    # Check each file for FastAPI dependencies
    fastapi_files = []
    for file_path in python_files:
        imports = find_fastapi_imports(file_path)
        usage = find_fastapi_usage(file_path)
        if imports or usage:
            fastapi_files.append((file_path, imports, usage))
    
    # Print results
    if fastapi_files:
        print(f"Found {len(fastapi_files)} files with FastAPI dependencies:")
        for file_path, imports, usage in fastapi_files:
            print(f"\n{file_path}:")
            if imports:
                print("  Imports:")
                for imp in imports:
                    print(f"    - {imp}")
            if usage:
                print("  Usage:")
                for use in usage:
                    print(f"    - {use}")
        
        print("\nRecommendations:")
        print("1. Update these files to use Streamlit instead of FastAPI")
        print("2. If the file is part of the legacy FastAPI implementation, mark it as deprecated")
        print("3. Consider creating a Streamlit alternative for each FastAPI endpoint")
    else:
        print("No FastAPI dependencies found. The transition to Streamlit is complete!")

def clean_fastapi_dependencies():
    """Remove FastAPI dependencies (use with caution)."""
    print("This feature is not yet implemented.")
    print("Please manually remove FastAPI dependencies for now.")

def main():
    """Main entry point."""
    args = parse_args()
    
    if args.check:
        check_fastapi_dependencies()
    elif args.clean:
        clean_fastapi_dependencies()
    else:
        print("Please specify an action: --check or --clean")
        print("Example: python transition_to_streamlit.py --check")

if __name__ == "__main__":
    main()
