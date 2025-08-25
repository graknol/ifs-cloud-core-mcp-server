#!/usr/bin/env python3
"""
Comprehensive IFS Cloud Codebase Analysis - ALL 286 Modules
Scan every module to extract real keywords, abbreviations, and patterns
"""

import os
import re
from collections import Counter, defaultdict
from pathlib import Path
import json


class ComprehensiveIFSAnalyzer:
    """Analyze ALL IFS modules for real keywords and abbreviations."""

    def __init__(self):
        """Initialize comprehensive analyzer."""
        self.base_path = Path("_work/25.1.0")
        self.results = {
            "modules_analyzed": 0,
            "files_processed": 0,
            "total_abbreviations": Counter(),
            "module_specific_abbrevs": defaultdict(Counter),
            "business_terms_by_module": defaultdict(Counter),
            "api_patterns_by_module": defaultdict(Counter),
            "module_descriptions": {},
            "cross_module_patterns": Counter(),
            "abbreviation_contexts": defaultdict(list),
        }

        # Enhanced abbreviation patterns based on IFS conventions
        self.abbreviation_patterns = [
            # Core business abbreviations
            r"\b(qty|quantity)\b",
            r"\b(addr|address)\b",
            r"\b(cust|customer)\b",
            r"\b(supp|supplier)\b",
            r"\b(ord|order)\b",
            r"\b(inv|invent|inventory)\b",
            r"\b(pur|purch|purchase)\b",
            r"\b(req|requis|requisition)\b",
            r"\b(proj|project)\b",
            r"\b(del|deliv|delivery)\b",
            r"\b(rec|recv|receipt|receive)\b",
            r"\b(auth|author|authorization)\b",
            r"\b(appr|approval)\b",
            # Technical abbreviations
            r"\b(desc|description)\b",
            r"\b(ref|reference)\b",
            r"\b(id|identifier)\b",
            r"\b(no|num|number)\b",
            r"\b(amt|amount)\b",
            r"\b(val|value)\b",
            r"\b(calc|calculation)\b",
            r"\b(proc|process)\b",
            r"\b(temp|temporary)\b",
            r"\b(hist|history)\b",
            r"\b(stat|statistics)\b",
            r"\b(info|information)\b",
            r"\b(mgr|manager)\b",
            r"\b(ctrl|control)\b",
            r"\b(util|utility)\b",
            r"\b(cfg|config|configuration)\b",
            # IFS specific patterns
            r"\b(mfg|manuf|manufacturing)\b",
            r"\b(fin|financial)\b",
            r"\b(hr|human_resource)\b",
            r"\b(wm|warehouse)\b",
            r"\b(pm|project_mgmt)\b",
            r"\b(crm|customer_rel)\b",
            r"\b(scm|supply_chain)\b",
            r"\b(erp|enterprise)\b",
            r"\b(plm|product_lifecycle)\b",
            r"\b(eam|enterprise_asset)\b",
        ]

        # Business domain patterns
        self.business_patterns = [
            # Financial terms
            r"\b(cost|price|amount|total|discount|charge|tax|commission|rebate|budget|currency|payment|balance|invoice|billing)\b",
            # Operations terms
            r"\b(order|purchase|delivery|shipment|receipt|inventory|warehouse|manufacturing|production|quality|maintenance)\b",
            # Process terms
            r"\b(authorization|approval|workflow|validation|verification|processing|calculation|allocation|reservation|scheduling|planning)\b",
            # Status terms
            r"\b(status|state|condition|planned|released|confirmed|delivered|invoiced|closed|cancelled|approved|rejected)\b",
            # Entity terms
            r"\b(customer|supplier|vendor|employee|person|company|site|organization|product|part|item|project|activity|resource|equipment|facility)\b",
        ]

    def scan_all_modules(self):
        """Scan all 286 IFS modules comprehensively."""

        print("🚀 COMPREHENSIVE IFS CLOUD ANALYSIS - ALL 286 MODULES")
        print("=" * 65)

        if not self.base_path.exists():
            print(f"❌ Path not found: {self.base_path}")
            return

        # Get all module directories
        module_dirs = [d for d in self.base_path.iterdir() if d.is_dir()]
        total_modules = len(module_dirs)

        print(f"📁 Found {total_modules} modules to analyze")
        print(f"🔍 Starting comprehensive scan...")
        print()

        for i, module_dir in enumerate(sorted(module_dirs), 1):
            module_name = module_dir.name.upper()
            print(f"[{i:3}/{total_modules}] 📂 {module_name}")

            module_result = self.analyze_module(module_dir, module_name)
            self.results["modules_analyzed"] += 1

            # Show progress every 25 modules
            if i % 25 == 0:
                self.show_progress_summary(i, total_modules)

        print(f"\n✅ Analysis Complete!")
        self.generate_comprehensive_report()

    def analyze_module(self, module_path: Path, module_name: str) -> dict:
        """Analyze a single IFS module comprehensively."""

        module_files = []
        abbreviations = Counter()
        business_terms = Counter()
        api_patterns = Counter()

        # Find all PLSQL files in the module
        try:
            for plsql_file in module_path.rglob("*.plsql"):
                module_files.append(plsql_file)

                # Analyze each file
                try:
                    with open(plsql_file, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()

                        # Extract abbreviations
                        file_abbrevs = self.extract_abbreviations(content)
                        abbreviations.update(file_abbrevs)

                        # Extract business terms
                        file_business = self.extract_business_terms(content)
                        business_terms.update(file_business)

                        # Extract API patterns
                        file_apis = self.extract_api_patterns(content)
                        api_patterns.update(file_apis)

                        self.results["files_processed"] += 1

                except Exception as e:
                    continue  # Skip problematic files

        except Exception as e:
            print(f"    ⚠️  Error scanning {module_name}: {e}")
            return {}

        # Store module-specific results
        self.results["module_specific_abbrevs"][module_name] = abbreviations
        self.results["business_terms_by_module"][module_name] = business_terms
        self.results["api_patterns_by_module"][module_name] = api_patterns

        # Update global counters
        self.results["total_abbreviations"].update(abbreviations)
        self.results["cross_module_patterns"].update(business_terms)

        # Generate module description
        self.results["module_descriptions"][module_name] = (
            self.generate_module_description(
                module_name, abbreviations, business_terms, len(module_files)
            )
        )

        # Show key findings for this module
        if abbreviations or business_terms:
            top_abbrevs = abbreviations.most_common(3)
            top_business = business_terms.most_common(3)

            print(
                f"    📊 {len(module_files)} files, {len(abbreviations)} abbrevs, {len(business_terms)} business terms"
            )
            if top_abbrevs:
                abbrev_str = ", ".join([f"{k}({v})" for k, v in top_abbrevs])
                print(f"    🔤 Top: {abbrev_str}")

        return {
            "files": len(module_files),
            "abbreviations": dict(abbreviations),
            "business_terms": dict(business_terms),
            "apis": dict(api_patterns),
        }

    def extract_abbreviations(self, content: str) -> Counter:
        """Extract abbreviations from file content."""
        abbreviations = Counter()
        content_lower = content.lower()

        for pattern in self.abbreviation_patterns:
            matches = re.findall(pattern, content_lower, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    # Handle grouped patterns - take the matched group
                    for group in match:
                        if group:
                            abbreviations[group] += 1
                            break
                else:
                    abbreviations[match] += 1

        return abbreviations

    def extract_business_terms(self, content: str) -> Counter:
        """Extract business terms from file content."""
        business_terms = Counter()
        content_lower = content.lower()

        for pattern in self.business_patterns:
            matches = re.findall(pattern, content_lower, re.IGNORECASE)
            for match in matches:
                business_terms[match] += 1

        return business_terms

    def extract_api_patterns(self, content: str) -> Counter:
        """Extract API call patterns."""
        api_patterns = Counter()

        # Find API calls
        api_matches = re.findall(r"(\w+_API)\.(\w+)", content, re.IGNORECASE)
        for api_class, method in api_matches:
            api_patterns[f"{api_class}.{method}"] += 1

        return api_patterns

    def generate_module_description(
        self,
        module_name: str,
        abbreviations: Counter,
        business_terms: Counter,
        file_count: int,
    ) -> str:
        """Generate a description for the module based on its patterns."""

        # Predefined module descriptions based on IFS knowledge
        known_modules = {
            "ORDER": "Sales Order Management - customer orders, quotations, pricing",
            "PURCH": "Purchase Order Management - procurement, suppliers, requisitions",
            "INVENT": "Inventory Management - stock, warehousing, locations, receipts",
            "MANUFAC": "Manufacturing Execution - work orders, shop orders, resources",
            "PROJ": "Project Management - projects, activities, time reporting",
            "PERSON": "Human Resources - employees, payroll, competencies",
            "ACCRUL": "Accounting Rules - general ledger, vouchers, postings",
            "ASSET": "Asset Management - equipment, maintenance, facilities",
            "ENTERP": "Enterprise Setup - companies, sites, organizations",
            "INVOIC": "Invoice Management - customer/supplier invoicing",
            "PAYLED": "Payment Processing - payments, banking, cash management",
            "CRM": "Customer Relationship Management - opportunities, campaigns",
            "DOCMAN": "Document Management - documents, workflows, approvals",
            "RENTAL": "Rental Management - rental agreements, contracts",
            "SERVMGT": "Service Management - service requests, agreements",
        }

        if module_name in known_modules:
            base_desc = known_modules[module_name]
        else:
            # Generate description from patterns
            top_terms = business_terms.most_common(3)
            if top_terms:
                terms_str = ", ".join([term for term, count in top_terms])
                base_desc = f"Manages {terms_str} operations"
            else:
                base_desc = "IFS Cloud business module"

        return f"{base_desc} ({file_count} files)"

    def show_progress_summary(self, current: int, total: int):
        """Show progress summary every 25 modules."""
        progress_pct = (current / total) * 100

        print(f"\n📈 PROGRESS CHECKPOINT ({current}/{total} - {progress_pct:.1f}%)")
        print(f"   Files processed: {self.results['files_processed']}")
        print(
            f"   Total abbreviations found: {len(self.results['total_abbreviations'])}"
        )
        print(f"   Most common abbreviations:")

        top_abbrevs = self.results["total_abbreviations"].most_common(10)
        for abbrev, count in top_abbrevs:
            print(f"      {abbrev}: {count}")
        print()

    def generate_comprehensive_report(self):
        """Generate the final comprehensive report."""

        print(f"🎯 COMPREHENSIVE ANALYSIS RESULTS")
        print("=" * 50)

        print(f"📊 SUMMARY STATISTICS:")
        print(f"   Modules analyzed: {self.results['modules_analyzed']}")
        print(f"   Files processed: {self.results['files_processed']}")
        print(f"   Total abbreviations: {len(self.results['total_abbreviations'])}")
        print(f"   Unique business terms: {len(self.results['cross_module_patterns'])}")

        print(f"\n🔤 TOP ABBREVIATIONS (ALL MODULES):")
        for abbrev, count in self.results["total_abbreviations"].most_common(20):
            print(f"   {abbrev:15} {count:6} occurrences")

        print(f"\n💼 TOP BUSINESS TERMS (ALL MODULES):")
        for term, count in self.results["cross_module_patterns"].most_common(20):
            print(f"   {term:15} {count:6} occurrences")

        print(f"\n🏗️  MODULE BREAKDOWN (Top 20 by Activity):")
        module_activity = {}
        for module, abbrevs in self.results["module_specific_abbrevs"].items():
            activity_score = sum(abbrevs.values()) + sum(
                self.results["business_terms_by_module"][module].values()
            )
            module_activity[module] = activity_score

        sorted_modules = sorted(
            module_activity.items(), key=lambda x: x[1], reverse=True
        )
        for module, score in sorted_modules[:20]:
            desc = self.results["module_descriptions"].get(module, "Unknown module")
            print(f"   {module:12} {score:6} - {desc}")

        print(f"\n🔧 TOP API PATTERNS:")
        all_apis = Counter()
        for module_apis in self.results["api_patterns_by_module"].values():
            all_apis.update(module_apis)

        for api, count in all_apis.most_common(15):
            print(f"   {api:40} {count:4}")

        # Save detailed results
        self.save_detailed_results()

    def save_detailed_results(self):
        """Save detailed results to JSON file."""

        # Convert Counters to regular dicts for JSON serialization
        json_results = {
            "summary": {
                "modules_analyzed": self.results["modules_analyzed"],
                "files_processed": self.results["files_processed"],
                "total_unique_abbreviations": len(self.results["total_abbreviations"]),
                "total_unique_business_terms": len(
                    self.results["cross_module_patterns"]
                ),
            },
            "top_abbreviations": dict(
                self.results["total_abbreviations"].most_common(50)
            ),
            "top_business_terms": dict(
                self.results["cross_module_patterns"].most_common(50)
            ),
            "module_descriptions": self.results["module_descriptions"],
            "module_specific_data": {},
        }

        # Add module-specific data (top items only to keep file manageable)
        for module in self.results["module_specific_abbrevs"]:
            json_results["module_specific_data"][module] = {
                "abbreviations": dict(
                    self.results["module_specific_abbrevs"][module].most_common(20)
                ),
                "business_terms": dict(
                    self.results["business_terms_by_module"][module].most_common(20)
                ),
                "apis": dict(
                    self.results["api_patterns_by_module"][module].most_common(10)
                ),
            }

        # Save to file
        output_file = "comprehensive_ifs_analysis_all_modules.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)

        print(f"\n💾 DETAILED RESULTS SAVED:")
        print(f"   File: {output_file}")
        print(f"   Size: {os.path.getsize(output_file) / 1024:.1f} KB")

        # Create enhanced business keywords list
        self.create_enhanced_keywords_file()

    def create_enhanced_keywords_file(self):
        """Create enhanced keywords file for RTX optimizer."""

        enhanced_keywords = []

        # Add top abbreviations (both short and long forms where applicable)
        abbreviation_mappings = {
            "qty": "quantity",
            "addr": "address",
            "cust": "customer",
            "supp": "supplier",
            "ord": "order",
            "inv": "inventory",
            "invent": "inventory",
            "pur": "purchase",
            "purch": "purchase",
            "req": "requisition",
            "requis": "requisition",
            "proj": "project",
            "del": "delivery",
            "deliv": "delivery",
            "rec": "receipt",
            "recv": "receive",
            "auth": "authorization",
            "appr": "approval",
            "desc": "description",
            "ref": "reference",
            "info": "information",
            "mgr": "manager",
            "ctrl": "control",
            "util": "utility",
            "cfg": "configuration",
            "config": "configuration",
            "temp": "temporary",
            "hist": "history",
            "stat": "statistics",
        }

        # Add all abbreviations found
        for abbrev, count in self.results["total_abbreviations"].most_common(100):
            enhanced_keywords.append(abbrev)
            # Add full form if known
            if abbrev in abbreviation_mappings:
                enhanced_keywords.append(abbreviation_mappings[abbrev])

        # Add top business terms
        for term, count in self.results["cross_module_patterns"].most_common(100):
            enhanced_keywords.append(term)

        # Remove duplicates while preserving order
        enhanced_keywords = list(dict.fromkeys(enhanced_keywords))

        # Save enhanced keywords
        keywords_data = {
            "total_keywords": len(enhanced_keywords),
            "generated_from": f"{self.results['modules_analyzed']} IFS modules",
            "files_analyzed": self.results["files_processed"],
            "keywords": enhanced_keywords,
            "abbreviation_mappings": abbreviation_mappings,
            "module_count_by_type": {},
        }

        # Add module categorization
        for module in self.results["module_descriptions"]:
            category = self.categorize_module(module)
            if category not in keywords_data["module_count_by_type"]:
                keywords_data["module_count_by_type"][category] = 0
            keywords_data["module_count_by_type"][category] += 1

        keywords_file = "enhanced_ifs_keywords_all_modules.json"
        with open(keywords_file, "w", encoding="utf-8") as f:
            json.dump(keywords_data, f, indent=2, ensure_ascii=False)

        print(f"   Enhanced Keywords: {keywords_file}")
        print(f"   Total keywords: {len(enhanced_keywords)}")
        print(f"   Abbreviation mappings: {len(abbreviation_mappings)}")

    def categorize_module(self, module_name: str) -> str:
        """Categorize modules by business function."""

        categories = {
            "Core Business": ["ORDER", "PURCH", "INVENT", "INVOIC", "PAYLED"],
            "Manufacturing": ["MANUFAC", "MRO", "QUALIT", "PRODUC", "SCHEDUL"],
            "Project Management": ["PROJ", "PROJBF", "PROJMSP", "PRJDEL", "PRJREP"],
            "Human Resources": ["PERSON", "PAYROL", "EMPSER", "COMPET"],
            "Financial": ["ACCRUL", "FINCON", "BUDGET", "COST", "FIXASS"],
            "Asset Management": ["ASSET", "MAINT", "EQUIP", "FACILI"],
            "CRM & Sales": ["CRM", "SALES", "MARKET", "CAMPAI"],
            "Document & Workflow": ["DOCMAN", "WORKFL", "APPROV"],
            "Enterprise Setup": ["ENTERP", "COMPAN", "SITE", "ORGANI"],
            "Service Management": ["SERVMGT", "RENTAL", "CONTRA", "AGREEM"],
        }

        for category, modules in categories.items():
            if any(mod in module_name for mod in modules):
                return category

        return "Other"


def main():
    """Run comprehensive analysis of all IFS modules."""

    print("🚀 COMPREHENSIVE IFS CLOUD CODEBASE ANALYSIS")
    print("🎯 Scanning ALL 286 modules for real patterns...")
    print()

    analyzer = ComprehensiveIFSAnalyzer()
    analyzer.scan_all_modules()

    print(f"\n🎉 COMPREHENSIVE ANALYSIS COMPLETE!")
    print(f"   🔍 Every IFS module analyzed")
    print(f"   📊 Real abbreviation frequencies extracted")
    print(f"   💼 Business terms ranked by actual usage")
    print(f"   🔧 API patterns identified across modules")
    print(f"   💾 Results saved for RTX optimizer integration")


if __name__ == "__main__":
    main()
