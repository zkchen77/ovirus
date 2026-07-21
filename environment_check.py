# -*- coding: utf-8 -*-
"""
@author: zk
"""


"""
Environment Package Version Checker
This script checks and displays all required package versions for the drug risk prediction pipeline.
Run this script before executing the main program to verify your environment.
"""

import sys
import platform
import importlib.metadata

def get_package_version(package_name):
    """
    Get the installed version of a package.
    
    Parameters:
        package_name (str): Name of the package
    
    Returns:
        str: Version string or 'Not installed' if package is not found
    """
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "Not installed"

def check_environment():
    """
    Check and display system information and package versions.
    """
    print("=" * 80)
    print("ENVIRONMENT PACKAGE VERSION CHECKER")
    print("=" * 80)
    
    # System information
    print("\n【System Information】")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.architecture()[0]}")
    
    # Required packages
    packages = [
        "numpy",
        "pandas", 
        "matplotlib",
        "scikit-learn",
        "rdkit",
        "xgboost",
        "catboost"
    ]
    
    print("\n【Required Package Versions】")
    print("-" * 40)
    
    # Check each package
    for package in packages:
        version = get_package_version(package)
        status = "✓" if version != "Not installed" else "✗"
        print(f"{status} {package:20} version: {version}")
    
    print("-" * 40)
    
    # Check optional but recommended packages
    optional_packages = [
        "torch",
        "tensorflow",
        "jupyter",
        "ipython"
    ]
    
    print("\n【Optional Package Versions (not required)】")
    print("-" * 40)
    for package in optional_packages:
        version = get_package_version(package)
        if version != "Not installed":
            print(f"✓ {package:20} version: {version}")
        else:
            print(f"✗ {package:20} not installed")
    
    print("-" * 40)
    print("=" * 80)
    
    # Compatibility check
    print("\n【Compatibility Check】")
    print("-" * 40)
    
    compatibility_issues = []
    
    # Check RDKit
    rdkit_version = get_package_version("rdkit")
    if rdkit_version != "Not installed":
        print(f"✓ RDKit version {rdkit_version} is installed")
    else:
        compatibility_issues.append("RDKit is required for molecular processing")
    
    # Check scikit-learn
    sklearn_version = get_package_version("scikit-learn")
    if sklearn_version != "Not installed":
        print(f"✓ scikit-learn version {sklearn_version} is installed")
        # Check if version is compatible (requires >= 0.24)
        if sklearn_version.split('.')[0] == '0' and int(sklearn_version.split('.')[1]) < 24:
            compatibility_issues.append("scikit-learn version should be >= 0.24")
    else:
        compatibility_issues.append("scikit-learn is required for machine learning models")
    
    # Check XGBoost
    xgboost_version = get_package_version("xgboost")
    if xgboost_version != "Not installed":
        print(f"✓ XGBoost version {xgboost_version} is installed")
    else:
        compatibility_issues.append("XGBoost is optional but recommended")
    
    # Check CatBoost
    catboost_version = get_package_version("catboost")
    if catboost_version != "Not installed":
        print(f"✓ CatBoost version {catboost_version} is installed")
    else:
        compatibility_issues.append("CatBoost is optional but recommended")
    
    # Check NumPy
    numpy_version = get_package_version("numpy")
    if numpy_version != "Not installed":
        print(f"✓ NumPy version {numpy_version} is installed")
        # Check if version is compatible (requires >= 1.19)
        if int(numpy_version.split('.')[0]) < 1 or (int(numpy_version.split('.')[0]) == 1 and int(numpy_version.split('.')[1]) < 19):
            compatibility_issues.append("NumPy version should be >= 1.19")
    else:
        compatibility_issues.append("NumPy is required for numerical operations")
    
    # Check Pandas
    pandas_version = get_package_version("pandas")
    if pandas_version != "Not installed":
        print(f"✓ Pandas version {pandas_version} is installed")
    else:
        compatibility_issues.append("Pandas is required for CSV file handling")
    
    # Check Matplotlib
    matplotlib_version = get_package_version("matplotlib")
    if matplotlib_version != "Not installed":
        print(f"✓ Matplotlib version {matplotlib_version} is installed")
    else:
        compatibility_issues.append("Matplotlib is required for plotting ROC curves")
    
    print("-" * 40)
    
    # Summary
    print("\n【Summary】")
    print("-" * 40)
    if compatibility_issues:
        print("⚠️  Compatibility issues found:")
        for issue in compatibility_issues:
            print(f"   - {issue}")
        print("\nRecommended actions:")
        print("   1. Install missing packages: pip install <package_name>")
        print("   2. Update outdated packages: pip install --upgrade <package_name>")
        print("   3. See requirements.txt for exact package versions")
    else:
        print("✅ All required packages are installed and compatible!")
    
    print("-" * 40)
    print("=" * 80)

def generate_requirements_file():
    """
    Generate a requirements.txt file with current package versions.
    """
    packages = [
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "rdkit",
        "xgboost",
        "catboost"
    ]
    
    requirements = []
    for package in packages:
        version = get_package_version(package)
        if version != "Not installed":
            requirements.append(f"{package}=={version}")
        else:
            requirements.append(f"{package}  # Not installed - please install")
    
    with open("requirements_current.txt", "w") as f:
        f.write("# Current environment package versions\n")
        f.write("# Generated by environment_check.py\n\n")
        f.write("\n".join(requirements))
    
    print(f"\n✅ Generated 'requirements_current.txt' with current package versions")

if __name__ == "__main__":
    # Run environment check
    check_environment()
    
    # Ask if user wants to generate requirements file
    print("\nDo you want to generate a requirements file with current versions?")
    print("(This will create 'requirements_current.txt')")
    response = input("Generate? (y/n): ").strip().lower()
    
    if response == 'y' or response == 'yes':
        generate_requirements_file()
    else:
        print("Skipping requirements file generation.")