#!/usr/bin/env python3
"""
Test script for GitHub Copilot API functionality
"""

import sys
from pathlib import Path


def test_copilot_api():
    """Test the GitHub Copilot API with a simple example"""
    try:
        from copilot_api import CopilotAPI

        print("🧪 Testing GitHub Copilot API...")

        # Initialize API
        copilot = CopilotAPI()

        # Test with a simple PL/SQL snippet
        test_plsql = """
        PROCEDURE Update_Customer_Status (
           customer_id_ IN NUMBER,
           new_status_ IN VARCHAR2 )
        IS
        BEGIN
           UPDATE customer_tab 
           SET status = new_status_,
               last_modified = SYSDATE
           WHERE customer_id = customer_id_;
               
           Customer_History_API.Log_Status_Change(customer_id_, new_status_);
        END Update_Customer_Status;
        
        FUNCTION Get_Customer_Balance (
           customer_id_ IN NUMBER ) RETURN NUMBER
        IS
           balance_ NUMBER;
        BEGIN
           SELECT SUM(amount) 
           INTO balance_
           FROM invoice_tab
           WHERE customer_id = customer_id_
           AND status = 'OPEN';
           
           RETURN NVL(balance_, 0);
        END Get_Customer_Balance;
        """

        prompt = f"""
        Analyze this PL/SQL code and extract procedure summaries:

        INSTRUCTIONS:
        For each procedure/function, provide:
        1. The procedure/function name and parameters  
        2. A 2-3 sentence summary of what it conceptually does

        Format as:
        ## PROCEDURE_NAME(parameters)
        Summary text here.

        PL/SQL CODE:
        {test_plsql}
        """

        print("🔄 Making API call...")
        response = copilot.copilot_completion(prompt, max_tokens=500, temperature=0.3)

        if response:
            print("✅ API call successful!")
            print("📝 Response:")
            print("-" * 50)
            print(response)
            print("-" * 50)
            return True
        else:
            print("❌ No response received")
            return False

    except Exception as e:
        print(f"❌ Error testing Copilot API: {e}")
        return False


def test_with_real_file():
    """Test with a real PLSQL file from top_10"""
    top_10_dir = Path("top_10")

    if not top_10_dir.exists():
        print("❌ top_10 directory not found")
        return False

    # Find the first PLSQL file
    plsql_file = None
    for module_dir in top_10_dir.iterdir():
        if module_dir.is_dir():
            plsql_files = list(module_dir.glob("*.plsql"))
            if plsql_files:
                plsql_file = plsql_files[0]
                break

    if not plsql_file:
        print("❌ No PLSQL files found in top_10")
        return False

    print(f"🧪 Testing with real file: {plsql_file}")

    try:
        # Read first 10000 chars of the file
        with open(plsql_file, "r", encoding="utf-8") as f:
            content = f.read()[:10000]

        from copilot_api import CopilotAPI

        copilot = CopilotAPI()

        prompt = f"""
        Analyze this PL/SQL file and identify 5 procedures/functions:

        FORMAT:
        ## PROCEDURE_NAME(parameters)
        Brief summary.

        FILE: {plsql_file.name}
        CONTENT:
        {content}
        """

        print("🔄 Analyzing real PLSQL file...")
        response = copilot.copilot_completion(prompt, max_tokens=1000, temperature=0.2)

        if response:
            print("✅ Real file analysis successful!")
            print("📝 Response preview:")
            print("-" * 50)
            print(response[:500] + "..." if len(response) > 500 else response)
            print("-" * 50)
            return True
        else:
            print("❌ No response for real file")
            return False

    except Exception as e:
        print(f"❌ Error testing with real file: {e}")
        return False


def main():
    print("🚀 GitHub Copilot API Test Suite")
    print("=" * 50)

    # Test basic functionality
    if not test_copilot_api():
        print("❌ Basic API test failed")
        sys.exit(1)

    print("\n" + "=" * 50)

    # Test with real file
    if not test_with_real_file():
        print("❌ Real file test failed")
        sys.exit(1)

    print("\n🎉 All tests passed! GitHub Copilot API is working.")
    print("\n📋 Next steps:")
    print("   1. Run: python run_plsql_analysis.py --api copilot")
    print("   2. Or run: python analyze_plsql_procedures.py")


if __name__ == "__main__":
    main()
