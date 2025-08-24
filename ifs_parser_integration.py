#!/usr/bin/env python3
"""
IFS Cloud Parser Integration with Fallback
"""


class IFSCloudParserIntegration:
    """Integration wrapper for IFS Cloud Parser with fallback"""

    def __init__(self):
        self.parser = None
        self.language = None
        self.parser_available = False
        self._init_parser()

    def _init_parser(self):
        """Initialize the parser if possible, fallback to heuristics"""
        try:
            import tree_sitter_ifs_cloud_parser
            from tree_sitter import Language, Parser

            # Try to create language and parser
            self.language = Language(tree_sitter_ifs_cloud_parser.language())
            self.parser = Parser(self.language)
            self.parser_available = True
            print("✅ IFS Cloud Parser initialized successfully with tree-sitter")

        except Exception as e:
            print(f"⚠️  IFS Cloud Parser not available: {e}")
            print("   Falling back to heuristic analysis")
            self.parser_available = False

    def parse_code(self, code: str) -> dict:
        """Parse IFS Cloud code and extract structure"""
        if self.parser_available:
            return self._parse_with_treesitter(code)
        else:
            return self._parse_with_heuristics(code)

    def _parse_with_treesitter(self, code: str) -> dict:
        """Parse using tree-sitter parser"""
        try:
            tree = self.parser.parse(code.encode("utf-8"))
            root = tree.root_node

            # Extract detailed structure information
            structure = self._extract_detailed_structure(root)

            return {
                "method": "tree-sitter",
                "root_type": root.type,
                "child_count": root.child_count,
                "has_error": root.has_error,
                "functions": structure.get("functions", []),
                "procedures": structure.get("procedures", []),
                "variables": structure.get("variables", []),
                "parameters": structure.get("parameters", []),
                "data_types": structure.get("data_types", []),
                "patterns": {
                    "has_procedure": len(structure.get("procedures", [])) > 0,
                    "has_function": len(structure.get("functions", [])) > 0,
                    "has_cursor": "cursor" in code.lower(),
                    "has_exception": "exception" in code.lower(),
                    "has_loop": any(
                        keyword in code.lower() for keyword in ["for", "while", "loop"]
                    ),
                    "has_conditional": any(
                        keyword in code.lower() for keyword in ["if", "case", "when"]
                    ),
                    "has_dml": any(
                        keyword in code.lower()
                        for keyword in ["select", "insert", "update", "delete"]
                    ),
                },
            }
        except Exception as e:
            print(f"Tree-sitter parsing failed: {e}")
            return self._parse_with_heuristics(code)

    def _parse_with_heuristics(self, code: str) -> dict:
        """Parse using heuristic analysis"""
        lines = code.split("\n")
        code_upper = code.upper()

        # Extract basic structure
        structure = {
            "method": "heuristics",
            "procedures": [],
            "functions": [],
            "variables": [],
            "parameters": [],
            "patterns": {
                "has_procedure": "PROCEDURE" in code_upper,
                "has_function": "FUNCTION" in code_upper,
                "has_cursor": "CURSOR" in code_upper,
                "has_exception": "EXCEPTION" in code_upper,
                "has_loop": any(
                    keyword in code_upper for keyword in ["FOR", "WHILE", "LOOP"]
                ),
                "has_conditional": any(
                    keyword in code_upper for keyword in ["IF", "CASE", "WHEN"]
                ),
                "has_dml": any(
                    keyword in code_upper
                    for keyword in ["SELECT", "INSERT", "UPDATE", "DELETE"]
                ),
            },
        }

        # Extract procedures and functions with parameters
        current_function = None
        in_parameters = False
        param_buffer = ""

        for line in lines:
            line_upper = line.strip().upper()

            # Check for function/procedure start
            if line_upper.startswith("PROCEDURE") or line_upper.startswith("FUNCTION"):
                name = self._extract_name_from_declaration(line)
                if line_upper.startswith("PROCEDURE"):
                    structure["procedures"].append(name)
                else:
                    structure["functions"].append(name)
                current_function = name

                # Check if parameters start on same line
                if "(" in line:
                    in_parameters = True
                    param_buffer = line

            # Continue collecting parameter lines
            elif in_parameters:
                param_buffer += " " + line.strip()

            # Check for end of parameters and function/procedure declaration
            if in_parameters and (
                "IS" in param_buffer.upper()
                or "AS" in param_buffer.upper()
                or "RETURN" in param_buffer.upper()
            ):
                # Extract parameters from buffer
                params = self._extract_parameters_from_text(param_buffer)
                structure["parameters"].extend(params)
                in_parameters = False
                param_buffer = ""
                current_function = None

        return structure

    def _extract_name_from_declaration(self, line: str) -> str:
        """Extract procedure/function name from declaration"""
        try:
            # Remove PROCEDURE/FUNCTION keyword
            line = line.strip()
            if line.upper().startswith("PROCEDURE"):
                line = line[9:].strip()
            elif line.upper().startswith("FUNCTION"):
                line = line[8:].strip()

            # Get name before parameters or IS/AS
            name = line.split("(")[0].split()[0].split("_")[0]
            return name.strip()
        except:
            return "unknown"

    def _extract_parameters_from_text(self, param_text: str) -> list:
        """Extract parameter names from parameter text"""
        parameters = []
        try:
            # Find the parameter section between ( and )
            start = param_text.find("(")
            end = param_text.rfind(")")
            if start != -1 and end != -1:
                param_section = param_text[start + 1 : end]

                # Split by comma, but be careful with nested parentheses
                param_parts = []
                current_part = ""
                paren_depth = 0

                for char in param_section:
                    if char == "(":
                        paren_depth += 1
                    elif char == ")":
                        paren_depth -= 1
                    elif char == "," and paren_depth == 0:
                        param_parts.append(current_part.strip())
                        current_part = ""
                        continue
                    current_part += char

                if current_part.strip():
                    param_parts.append(current_part.strip())

                # Extract parameter names
                for part in param_parts:
                    # Look for parameter name (first word before IN/OUT/IN OUT)
                    words = part.strip().split()
                    if words:
                        param_name = words[0]
                        # Clean up parameter name
                        if param_name.endswith("_"):
                            parameters.append(param_name)
        except Exception as e:
            print(f"Parameter extraction failed: {e}")

        return parameters

    def _extract_detailed_structure(self, node) -> dict:
        """Extract detailed structural information from tree-sitter node"""
        structure = {
            "functions": [],
            "procedures": [],
            "variables": [],
            "parameters": [],
            "data_types": [],
        }

        def traverse_node(n):
            """Recursively traverse tree to extract information"""
            # Extract function declarations
            if n.type == "function_declaration":
                func_name = self._get_identifier_from_node(n)
                if func_name:
                    structure["functions"].append(func_name)

            # Extract procedure declarations
            elif n.type == "procedure_declaration":
                proc_name = self._get_identifier_from_node(n)
                if proc_name:
                    structure["procedures"].append(proc_name)

            # Extract variable declarations
            elif n.type == "variable_declaration":
                var_name = self._get_identifier_from_node(n)
                if var_name:
                    structure["variables"].append(var_name)

            # Extract parameter declarations
            elif n.type == "parameter_declaration":
                param_name = self._get_identifier_from_node(n)
                if param_name:
                    structure["parameters"].append(param_name)

            # Extract data types
            elif n.type == "data_type":
                type_name = n.text.decode("utf-8") if n.text else None
                if type_name and type_name not in structure["data_types"]:
                    structure["data_types"].append(type_name)

            # Recursively process children
            for child in n.children:
                traverse_node(child)

        traverse_node(node)
        return structure

    def _get_identifier_from_node(self, node) -> str:
        """Extract identifier name from a node"""
        try:
            # Look for identifier child node
            for child in node.children:
                if child.type == "identifier":
                    return child.text.decode("utf-8")
            # If no identifier child, return the first text content
            if node.text:
                return (
                    node.text.decode("utf-8").split()[1]
                    if len(node.text.decode("utf-8").split()) > 1
                    else None
                )
        except:
            pass
        return None

    def analyze_complexity(self, parsed_data: dict) -> dict:
        """Analyze code complexity based on parsed data"""
        if parsed_data["method"] == "tree-sitter":
            # Use tree structure for complexity
            complexity = parsed_data["child_count"]
        else:
            # Use heuristic patterns
            patterns = parsed_data.get("patterns", {})
            complexity = sum(
                [
                    2 if patterns.get("has_loop", False) else 0,
                    1 if patterns.get("has_conditional", False) else 0,
                    1 if patterns.get("has_exception", False) else 0,
                    1 if patterns.get("has_dml", False) else 0,
                ]
            )

        return {
            "complexity_score": complexity,
            "complexity_level": (
                "high" if complexity > 5 else "medium" if complexity > 2 else "low"
            ),
        }


# Test the integration
if __name__ == "__main__":
    parser = IFSCloudParserIntegration()

    # Test with sample IFS Cloud code
    sample_code = """
    PROCEDURE Test_Processing___ IS
        cursor_ NUMBER;
        result_ VARCHAR2(100);
    BEGIN
        IF condition_ THEN
            FOR rec_ IN cursor_ LOOP
                UPDATE table_name SET value = rec_.value;
            END LOOP;
        END IF;
    EXCEPTION
        WHEN OTHERS THEN
            Error_SYS.Record_General('ERROR', 'Processing failed');
    END Test_Processing___;
    """

    print("Testing parser with sample code:")
    result = parser.parse_code(sample_code)
    print(f"Parse result: {result}")

    complexity = parser.analyze_complexity(result)
    print(f"Complexity analysis: {complexity}")
