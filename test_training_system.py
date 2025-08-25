#!/usr/bin/env python3
"""
Test script for the supervised training loop.
This script tests the basic functionality without requiring the full IFS codebase.
"""

import sys
import json
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def create_test_procedures():
    """Create test procedure data for demonstration."""
    test_procedures = [
        {
            "name": "Calculate_Customer_Discount___",
            "parameters": ["customer_id_", "order_amount_", "discount_type_"],
            "body": """BEGIN
   IF order_amount_ > 10000 THEN
      discount_rate_ := 0.15;
   ELSIF order_amount_ > 5000 THEN
      discount_rate_ := 0.10;
   ELSE
      discount_rate_ := 0.05;
   END IF;
   
   RETURN discount_rate_ * order_amount_;
END;""",
            "file_path": "/test/sales/customer_order.pls",
            "module_name": "SALES",
            "file_header": "-- Customer order management procedures\n-- Handles discount calculations",
            "line_number": 145,
            "ast_info": {
                "control_structures": [
                    {"type": "IF", "condition": "order_amount_ > 10000"},
                    {"type": "ELSIF", "condition": "order_amount_ > 5000"},
                ]
            },
        },
        {
            "name": "Validate_Inventory_Level___",
            "parameters": ["part_no_", "site_", "required_qty_"],
            "body": """BEGIN
   SELECT qty_onhand INTO available_qty_
   FROM inventory_part_tab
   WHERE part_no = part_no_ AND contract = site_;
   
   IF available_qty_ < required_qty_ THEN
      Error_SYS.Record_General('InsufficientInventory', 'Not enough inventory');
   END IF;
END;""",
            "file_path": "/test/inventory/inventory_part.pls",
            "module_name": "INVENTORY",
            "file_header": "-- Inventory management procedures\n-- Validates stock levels",
            "line_number": 89,
            "ast_info": {
                "control_structures": [
                    {
                        "type": "SELECT",
                        "condition": "qty_onhand FROM inventory_part_tab",
                    },
                    {"type": "IF", "condition": "available_qty_ < required_qty_"},
                ]
            },
        },
        {
            "name": "Process_Work_Order___",
            "parameters": ["wo_no_", "operation_", "employee_id_"],
            "body": """BEGIN
   FOR rec IN (SELECT * FROM work_order_operation WHERE wo_no = wo_no_) LOOP
      IF rec.status = 'RELEASED' THEN
         Update_Operation_Status___(wo_no_, rec.operation_no, 'IN_PROGRESS');
         Log_Work_Progress___(employee_id_, wo_no_, rec.operation_no);
      END IF;
   END LOOP;
END;""",
            "file_path": "/test/manufacturing/work_order.pls",
            "module_name": "MANUFACTURING",
            "file_header": "-- Manufacturing work order procedures\n-- Processes work order operations",
            "line_number": 234,
            "ast_info": {
                "control_structures": [
                    {
                        "type": "FOR",
                        "condition": "rec IN (SELECT * FROM work_order_operation",
                    },
                    {"type": "IF", "condition": "rec.status = 'RELEASED'"},
                ]
            },
        },
    ]

    # Add business code extraction for each procedure
    for proc in test_procedures:
        business_parts = []

        # Add control structures
        if "control_structures" in proc.get("ast_info", {}):
            for structure in proc["ast_info"]["control_structures"]:
                business_parts.append(
                    f"-- {structure['type']}: {structure.get('condition', '')}"
                )

        # Add simplified body
        lines = proc["body"].split("\n")
        code_lines = [
            line for line in lines[1:6] if line.strip()
        ]  # Skip BEGIN, take next 5 lines
        business_parts.extend(code_lines)
        business_parts.append("  -- ... (more code below)")

        proc["business_code"] = "\n".join(business_parts)

    return test_procedures


def test_gui_only():
    """Test just the GUI component with mock data."""
    print("🧪 Testing GUI component...")

    try:
        import tkinter as tk
        from supervised_training_loop import SummaryReviewGUI

        # Create test data
        test_data = create_test_procedures()

        # Add generated summaries
        test_data[0][
            "generated_summary"
        ] = "Calculates customer discount rates based on order amount with tiered discount structure"
        test_data[0]["human_summary"] = test_data[0]["generated_summary"]
        test_data[0]["status"] = "pending"

        test_data[1][
            "generated_summary"
        ] = "Validates inventory levels for parts at specific sites against required quantities"
        test_data[1]["human_summary"] = test_data[1]["generated_summary"]
        test_data[1]["status"] = "pending"

        test_data[2][
            "generated_summary"
        ] = "Processes work order operations by updating status and logging employee progress"
        test_data[2]["human_summary"] = test_data[2]["generated_summary"]
        test_data[2]["status"] = "pending"

        # Launch GUI
        root = tk.Tk()
        gui = SummaryReviewGUI(root, test_data)

        print("✅ GUI launched successfully!")
        print("📋 Review the test procedures and use keyboard shortcuts:")
        print("   • Enter: Accept summary")
        print("   • S: Skip procedure")
        print("   • E: Edit summary")
        print("   • ←/→: Navigate")
        print("   • Escape: Exit")

        gui.run()

        # Show results
        accepted = [p for p in test_data if p["status"] in ["accepted", "edited"]]
        skipped = [p for p in test_data if p["status"] == "skipped"]

        print(f"\n📊 Results:")
        print(f"   • Accepted: {len(accepted)}")
        print(f"   • Edited: {len([p for p in accepted if p['status'] == 'edited'])}")
        print(f"   • Skipped: {len(skipped)}")

        if accepted:
            print("\n✏️  Accepted summaries:")
            for proc in accepted:
                print(f"   • {proc['name']}: {proc['human_summary'][:60]}...")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing GUI: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_validation_system():
    """Test the validation and overfitting detection system."""
    print("\n🧪 Testing validation system...")

    try:
        from training_validator import TrainingValidator

        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            validator = TrainingValidator(save_dir=temp_dir)

            # Create mock training data
            mock_summaries = [
                {
                    "name": f"Test_Procedure_{i}___",
                    "human_summary": f"This procedure handles test operation {i} for business process validation",
                    "parameters": [f"param_{j}_" for j in range(3)],
                    "business_code": f"BEGIN\n  -- Test code {i}\nEND;",
                    "module_name": "TEST_MODULE",
                }
                for i in range(15)
            ]

            # Test validation split
            train_data, val_data = validator.create_validation_set([], mock_summaries)
            print(f"   ✅ Created train/val split: {len(train_data)}/{len(val_data)}")

            # Test overfitting detection
            validator.training_history["val_loss"] = [
                1.5,
                1.2,
                1.0,
                1.1,
                1.2,
                1.3,
            ]  # Simulate overfitting
            validator.training_history["iterations"] = list(range(6))

            should_continue, reason = validator.should_continue_training(6, 1.4)
            print(f"   ✅ Overfitting detection: {reason}")

            # Test recommendation
            recommendation = validator.get_overfitting_recommendation()
            print(f"   ✅ Recommendation: {recommendation}")

            # Test report generation
            report = validator.create_training_report()
            print("   ✅ Training report generated")
            print(f"   📊 Report preview:\n{report[:200]}...")

        return True

    except Exception as e:
        print(f"❌ Error testing validation: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🚀 Testing Supervised Training Loop Components")
    print("=" * 50)

    # Test 1: GUI Component
    gui_success = test_gui_only()

    # Test 2: Validation System
    val_success = test_validation_system()

    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY:")
    print(f"   • GUI Component: {'✅ PASSED' if gui_success else '❌ FAILED'}")
    print(f"   • Validation System: {'✅ PASSED' if val_success else '❌ FAILED'}")

    if gui_success and val_success:
        print("\n🎉 All tests passed! The system is ready for use.")
        print("\nTo start the full training loop, run:")
        print("   python launch_training.py")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")

    print(
        "\n💡 Note: Full model testing requires the actual IFS codebase and significant GPU memory."
    )


if __name__ == "__main__":
    main()
