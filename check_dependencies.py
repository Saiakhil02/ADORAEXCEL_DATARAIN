#!/usr/bin/env python3
"""
Script to check and install Pinecone dependencies
"""
import subprocess
import sys
import importlib
import pkg_resources

def check_and_install_package(package_name, import_name=None, version=None):
    """
    Check if a package is installed and install it if not.
    
    Args:
        package_name: Name of the package to install (e.g., 'pinecone-client')
        import_name: Name to use for import (e.g., 'pinecone'), defaults to package_name
        version: Minimum version required
    """
    if import_name is None:
        import_name = package_name.replace('-', '_')
    
    try:
        # Try to import the package
        importlib.import_module(import_name)
        
        # Check version if specified
        if version:
            try:
                installed_version = pkg_resources.get_distribution(package_name).version
                print(f"✅ {package_name} is installed (version {installed_version})")
                return True
            except pkg_resources.DistributionNotFound:
                print(f"⚠️  {package_name} is imported but version info not found")
                return True
        else:
            print(f"✅ {import_name} is available for import")
            return True
            
    except ImportError:
        print(f"❌ {import_name} not found, installing {package_name}...")
        
        # Install the package
        try:
            if version:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', f'{package_name}>={version}'])
            else:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name])
            print(f"✅ Successfully installed {package_name}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package_name}: {e}")
            return False

def main():
    """Main function to check and install all required packages."""
    print("🔧 Checking Python dependencies for Pinecone integration...\n")
    
    # List of required packages
    packages = [
        ('pinecone-client', 'pinecone', '4.0.0'),
        ('openai', 'openai', '1.14.2'),
        ('python-dotenv', 'dotenv'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('langchain', 'langchain'),
        ('langchain-openai', 'langchain_openai'),
        ('langchain-community', 'langchain_community'),
    ]
    
    all_success = True
    
    for package_info in packages:
        if len(package_info) == 3:
            package_name, import_name, version = package_info
        else:
            package_name, import_name = package_info
            version = None
            
        success = check_and_install_package(package_name, import_name, version)
        if not success:
            all_success = False
        print()  # Empty line for readability
    
    if all_success:
        print("🎉 All dependencies are installed and ready!")
        print("\n🔧 Next steps:")
        print("1. Ensure your .env file contains PINECONE_API_KEY and OPENAI_API_KEY")
        print("2. Run: python test_integration.py")
    else:
        print("⚠️  Some dependencies failed to install. Please check the errors above.")
        return 1
    
    # Test imports
    print("\n🧪 Testing imports...")
    try:
        from pinecone import Pinecone, ServerlessSpec
        print("✅ Pinecone imports successful")
        
        from openai import OpenAI
        print("✅ OpenAI imports successful")
        
        from langchain_core.documents import Document
        print("✅ LangChain imports successful")
        
        print("\n🎉 All imports working correctly!")
        
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
