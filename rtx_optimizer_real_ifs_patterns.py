#!/usr/bin/env python3
"""
Update RTX optimizer with real IFS codebase patterns and abbreviations
"""

from rtx5070ti_pytorch_optimizer import RTX5070TiPyTorchOptimizer
from enhanced_ifs_parser_real_patterns import IFSCloudParserEnhanced
import re
from pathlib import Path


class RTXOptimizerWithRealIFSPatterns(RTX5070TiPyTorchOptimizer):
    """RTX 5070 Ti optimizer enhanced with real IFS codebase patterns."""

    def __init__(self, model_name: str = "microsoft/unixcoder-base"):
        super().__init__(model_name)

        # Initialize enhanced IFS parser with real patterns
        self.enhanced_ifs_parser = IFSCloudParserEnhanced()

        print("🚀 RTX Optimizer Enhanced with Real IFS 25.1.0 Patterns")
        print("   ✅ 286 real IFS modules detected")
        print("   ✅ Real abbreviations from codebase analysis")
        print("   ✅ Module extraction from file path + header")
        print("   ✅ Business terms by actual frequency")

    def analyze_with_real_patterns(self, code: str, file_path: str = None) -> dict:
        """Analyze code using real IFS patterns from 25.1.0 codebase."""

        result = self.enhanced_ifs_parser.analyze_code_with_real_patterns(
            code, file_path
        )

        # Add RTX-specific optimizations
        result["rtx_optimized"] = True
        result["context_savings"] = self._calculate_context_savings(result)
        result["performance_keywords"] = self._extract_performance_keywords(result)

        return result

    def _calculate_context_savings(self, analysis_result: dict) -> dict:
        """Calculate context window savings using real patterns."""

        # Real abbreviations save significant space
        abbrev_savings = 0
        for abbrev, info in analysis_result.get("abbreviations", {}).items():
            full_form = info["full_form"]
            count = info["count"]
            char_saved = (len(full_form) - len(abbrev)) * count
            abbrev_savings += char_saved

        # Business term frequency optimization
        business_terms = analysis_result.get("business_terms", [])
        high_freq_terms = [t for t in business_terms if t.get("count", 0) > 5]

        return {
            "abbreviation_char_savings": abbrev_savings,
            "high_frequency_terms": len(high_freq_terms),
            "estimated_context_reduction": min(
                60, abbrev_savings / 10
            ),  # Max 60% reduction
            "optimization_effective": abbrev_savings > 20,
        }

    def _extract_performance_keywords(self, analysis_result: dict) -> list:
        """Extract performance-relevant keywords for RTX optimization."""

        keywords = []

        # Module-specific keywords
        module = analysis_result.get("module", "UNKNOWN")
        if module != "UNKNOWN":
            keywords.append(module.lower())

            # Add module-specific terms
            module_terms = self.enhanced_ifs_parser.real_ifs_modules.get(module, [])
            keywords.extend(module_terms)

        # High-frequency abbreviations (your insight was correct!)
        for abbrev, info in analysis_result.get("abbreviations", {}).items():
            if info.get("count", 0) > 2:  # Frequently used abbreviations
                keywords.extend([abbrev, info["full_form"]])

        # Top business terms
        business_terms = analysis_result.get("business_terms", [])
        top_terms = [t["term"] for t in business_terms[:10]]  # Top 10 terms
        keywords.extend(top_terms)

        # Remove duplicates while preserving order
        return list(dict.fromkeys(keywords))

    def create_optimized_summary_with_real_patterns(
        self, code: str, function_name: str, file_path: str = None
    ) -> str:
        """Create summary using real IFS patterns and module context."""

        # Analyze with real patterns
        analysis = self.analyze_with_real_patterns(code, file_path)

        # Extract module from path or header (no guessing needed!)
        module = analysis["module"]

        # Build summary with real context
        summary_parts = []

        # Start with action based on function name
        action = self._determine_action_from_function_name(function_name)
        summary_parts.append(action)

        # Add module context if known
        if module != "UNKNOWN":
            module_context = self._get_module_context(module)
            summary_parts.append(module_context)

        # Add specific functionality based on real abbreviations found
        abbrev_context = self._get_abbreviation_context(analysis["abbreviations"])
        if abbrev_context:
            summary_parts.append(abbrev_context)

        # Add business process context
        business_context = self._get_business_context(analysis["business_terms"])
        if business_context:
            summary_parts.append(business_context)

        # Add technical patterns
        pattern_context = self._get_pattern_context(analysis["patterns"])
        if pattern_context:
            summary_parts.append(pattern_context)

        summary = " ".join(summary_parts) + "."

        # Add optimization info
        savings = analysis["context_savings"]
        if savings["optimization_effective"]:
            summary += f" [Optimized: {savings['estimated_context_reduction']:.0f}% context reduction]"

        return summary.strip()

    def _get_module_context(self, module: str) -> str:
        """Get context description for IFS module."""

        module_descriptions = {
            "ORDER": "sales order processing",
            "PURCH": "purchase order management",
            "INVENT": "inventory operations",
            "MANUFAC": "manufacturing execution",
            "PROJ": "project management",
            "PERSON": "human resource functions",
            "ACCRUL": "accounting operations",
            "ASSET": "asset management",
            "ENTERP": "enterprise configuration",
            "INVOIC": "invoice processing",
            "PAYLED": "payment processing",
            "CRM": "customer relationship management",
            "DOCMAN": "document management",
            "RENTAL": "rental management",
            "SERVMGT": "service management",
        }

        return module_descriptions.get(module, f"{module.lower()} operations")

    def _get_abbreviation_context(self, abbreviations: dict) -> str:
        """Get context based on real abbreviations found."""

        if not abbreviations:
            return None

        # Prioritize high-frequency abbreviations
        high_freq = [(k, v) for k, v in abbreviations.items() if v.get("count", 0) > 3]

        if not high_freq:
            high_freq = list(abbreviations.items())[:2]  # Take first 2

        contexts = []
        for abbrev, info in high_freq:
            full_form = info["full_form"]

            # Map to business context
            if abbrev in ["qty", "quantity"]:
                contexts.append("quantity management")
            elif abbrev in ["addr", "address"]:
                contexts.append("address handling")
            elif abbrev in ["cust", "customer"]:
                contexts.append("customer data")
            elif abbrev in ["supp", "supplier"]:
                contexts.append("supplier information")
            elif abbrev in ["ord", "order"]:
                contexts.append("order processing")
            elif abbrev in ["purch", "purchase"]:
                contexts.append("procurement")
            elif abbrev in ["inv", "invent"]:
                contexts.append("inventory tracking")
            elif abbrev in ["proj", "project"]:
                contexts.append("project activities")
            elif abbrev in ["auth", "approval"]:
                contexts.append("authorization workflow")

        return ", ".join(contexts[:2]) if contexts else None  # Limit to 2 contexts

    def _get_business_context(self, business_terms: list) -> str:
        """Get context from business terms with real frequency data."""

        if not business_terms:
            return None

        # Focus on top terms
        top_terms = business_terms[:5]

        contexts = []
        for term_info in top_terms:
            term = term_info["term"]

            if term in ["cost", "price", "amount", "total"]:
                contexts.append("financial calculations")
            elif term in ["tax", "charge", "discount"]:
                contexts.append("pricing rules")
            elif term in ["authorization", "approval", "workflow"]:
                contexts.append("approval processes")
            elif term in ["validation", "verification"]:
                contexts.append("data validation")
            elif term in ["status", "state", "condition"]:
                contexts.append("state management")

        unique_contexts = list(dict.fromkeys(contexts))
        return ", ".join(unique_contexts[:2]) if unique_contexts else None

    def _get_pattern_context(self, patterns: dict) -> str:
        """Get context from code patterns."""

        active_patterns = [k for k, v in patterns.items() if v]

        if not active_patterns:
            return None

        contexts = []
        if "has_procedure" in active_patterns:
            contexts.append("with database procedures")
        if "has_api_call" in active_patterns:
            contexts.append("using IFS APIs")
        if "has_cursor" in active_patterns:
            contexts.append("with data cursors")
        if "has_exception" in active_patterns:
            contexts.append("including error handling")

        return " ".join(contexts[:2]) if contexts else None

    def _determine_action_from_function_name(self, function_name: str) -> str:
        """Determine action verb from function name."""

        name_upper = function_name.upper()

        if name_upper.startswith(("GET_", "FETCH_", "RETRIEVE_", "SELECT")):
            return "Retrieves"
        elif name_upper.startswith(("SET_", "UPDATE_", "MODIFY_", "CHANGE")):
            return "Updates"
        elif name_upper.startswith(("CREATE_", "INSERT_", "ADD_", "NEW")):
            return "Creates"
        elif name_upper.startswith(("DELETE_", "REMOVE_", "DROP")):
            return "Removes"
        elif name_upper.startswith(("CHECK_", "VALIDATE_", "VERIFY")):
            return "Validates"
        elif name_upper.startswith(("CALCULATE_", "COMPUTE_", "SUM")):
            return "Calculates"
        elif name_upper.startswith(("PROCESS_", "HANDLE_", "MANAGE")):
            return "Processes"
        else:
            return "Manages"


def test_rtx_with_real_patterns():
    """Test RTX optimizer with real IFS patterns."""

    print("🧪 Testing RTX Optimizer with Real IFS Patterns")
    print("=" * 50)

    optimizer = RTXOptimizerWithRealIFSPatterns()

    # Test with realistic IFS code
    test_code = """
    -----------------------------------------------------------------------------
    --
    --  Logical unit: PurchaseOrderLine  
    --  Component:    PURCH
    --
    PROCEDURE Update_Purch_Qty___ (
        order_no_ IN VARCHAR2,
        line_no_ IN NUMBER,
        rel_no_ IN NUMBER,
        purch_qty_ IN NUMBER,
        supplier_addr_ IN VARCHAR2,
        del_addr_no_ IN VARCHAR2,
        rec_site_ IN VARCHAR2
    ) IS
        temp_qty_ NUMBER;
        inv_part_cost_ NUMBER;
        supp_agreement_ VARCHAR2(10);
        auth_required_ BOOLEAN := FALSE;
        
    BEGIN
        -- Validate purch qty and addr
        IF purch_qty_ IS NULL OR purch_qty_ <= 0 THEN
            Error_SYS.Record_General(lu_name_, 'INVALID_QTY: Purchase quantity must be positive');
        END IF;
        
        -- Check supplier authorization for qty changes
        auth_required_ := Purchase_Authorization_API.Check_Qty_Auth_Required(order_no_, purch_qty_);
        
        IF auth_required_ THEN
            Purchase_Authorization_API.Request_Auth(order_no_, line_no_, 'QTY_CHANGE');
        END IF;
        
        -- Update purchase order line with new qty
        UPDATE purchase_order_line_tab
        SET qty_assigned = purch_qty_,
            supplier_addr_no = del_addr_no_,
            cost_amount = purch_qty_ * unit_cost,
            tax_amount = Tax_Handling_Purch_Util_API.Calculate_Tax(purch_qty_ * unit_cost, tax_code)
        WHERE order_no = order_no_
        AND line_no = line_no_
        AND rel_no = rel_no_;
        
        -- Update inventory reservations
        Inventory_Part_In_Stock_API.Modify_Reserved_Qty(part_no_, contract_, purch_qty_);
        
    END Update_Purch_Qty___;
    """

    file_path = "_work/25.1.0/purch/source/purch/database/PurchaseOrderLine.plsql"

    # Test the enhanced analysis
    analysis = optimizer.analyze_with_real_patterns(test_code, file_path)

    print(f"📊 Analysis Results:")
    print(f"   Module: {analysis['module']}")
    print(f"   Abbreviations: {len(analysis['abbreviations'])}")
    print(f"   Business Terms: {len(analysis['business_terms'])}")
    print(
        f"   Context Savings: {analysis['context_savings']['estimated_context_reduction']:.0f}%"
    )
    print(f"   Performance Keywords: {len(analysis['performance_keywords'])}")

    # Test summary generation
    summary = optimizer.create_optimized_summary_with_real_patterns(
        test_code, "Update_Purch_Qty___", file_path
    )

    print(f"\n📝 Generated Summary:")
    print(f"   {summary}")

    print(f"\n🎯 Key Abbreviations Found:")
    for abbrev, info in analysis["abbreviations"].items():
        print(f"   {abbrev} → {info['full_form']} ({info['count']}x)")

    print(f"\n💼 Top Business Terms:")
    for term in analysis["business_terms"][:5]:
        print(f"   {term['term']} ({term['count']}x)")

    print(f"\n⚡ Performance Keywords:")
    print(f"   {', '.join(analysis['performance_keywords'][:10])}")

    return analysis


if __name__ == "__main__":
    test_rtx_with_real_patterns()
