#!/usr/bin/env python3
"""
Simple GUI test for the supervised training system.
This version doesn't require the full IFS analyzer components.
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from typing import Dict, List
from pathlib import Path


class SummaryReviewGUI:
    """GUI for reviewing and editing procedure summaries."""

    def __init__(self, root: tk.Tk, batch_data: List[Dict]):
        self.root = root
        self.batch_data = batch_data
        self.current_index = 0

        self.setup_gui()
        self.setup_bindings()
        self.display_current_procedure()

    def setup_gui(self):
        """Setup the GUI layout."""
        self.root.title("IFS Cloud Procedure Summary Review")
        self.root.geometry("1400x900")

        # Modern dark theme colors
        bg_dark = "#1e1e1e"
        bg_medium = "#2d2d30"
        bg_light = "#3e3e42"
        text_primary = "#ffffff"
        text_secondary = "#cccccc"
        accent_blue = "#0078d4"
        accent_green = "#16c60c"
        accent_yellow = "#ffb900"
        accent_red = "#d13438"

        self.root.configure(bg=bg_dark)

        # Configure modern style
        style = ttk.Style()
        style.theme_use("clam")

        # Configure custom styles
        style.configure(
            "Modern.TLabel",
            font=("Segoe UI", 11),
            background=bg_dark,
            foreground=text_primary,
        )

        style.configure(
            "Title.TLabel",
            font=("Segoe UI", 14, "bold"),
            background=bg_dark,
            foreground=accent_blue,
        )

        style.configure(
            "Info.TLabel",
            font=("Segoe UI", 9),
            background=bg_dark,
            foreground=text_secondary,
        )

        # Main container
        main_container = tk.Frame(self.root, bg=bg_dark)
        main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # Header with progress
        header_frame = tk.Frame(main_container, bg=bg_dark)
        header_frame.pack(fill=tk.X, pady=(0, 15))

        self.progress_label = ttk.Label(header_frame, text="", style="Title.TLabel")
        self.progress_label.pack(side=tk.LEFT)

        # Status indicator
        self.status_indicator = tk.Label(
            header_frame, text="●", font=("Segoe UI", 16), fg=accent_yellow, bg=bg_dark
        )
        self.status_indicator.pack(side=tk.RIGHT)

        # Main content area - THREE COLUMNS
        content_frame = tk.Frame(main_container, bg=bg_dark)
        content_frame.pack(fill=tk.BOTH, expand=True)

        # === LEFT COLUMN - Full File Contents ===
        left_panel = tk.Frame(content_frame, bg=bg_medium, relief="solid", bd=1)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))

        # File contents header
        file_header = tk.Frame(left_panel, bg=bg_medium, height=40)
        file_header.pack(fill=tk.X, padx=15, pady=(10, 5))
        file_header.pack_propagate(False)

        ttk.Label(file_header, text="📄 Full File Contents", style="Title.TLabel").pack(
            anchor=tk.W
        )

        # File contents text with line numbers
        file_container = tk.Frame(left_panel, bg=bg_medium)
        file_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))

        self.file_text = scrolledtext.ScrolledText(
            file_container,
            height=30,
            width=50,
            bg=bg_light,
            fg=text_primary,
            font=("Cascadia Code", 9),
            insertbackground=text_primary,
            selectbackground=accent_blue,
            selectforeground="white",
            state=tk.DISABLED,
            relief="flat",
            borderwidth=0,
            padx=10,
            pady=10,
        )
        self.file_text.pack(fill=tk.BOTH, expand=True)

        # === MIDDLE COLUMN - Context & Prompt ===
        middle_panel = tk.Frame(content_frame, bg=bg_medium, relief="solid", bd=1)
        middle_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)

        # Context header
        context_header = tk.Frame(middle_panel, bg=bg_medium, height=40)
        context_header.pack(fill=tk.X, padx=15, pady=(10, 5))
        context_header.pack_propagate(False)

        ttk.Label(
            context_header, text="📋 Context Information", style="Title.TLabel"
        ).pack(anchor=tk.W)

        # Context text
        context_container = tk.Frame(middle_panel, bg=bg_medium)
        context_container.pack(fill=tk.X, padx=15, pady=(0, 10))

        self.context_text = scrolledtext.ScrolledText(
            context_container,
            height=12,
            width=45,
            bg=bg_light,
            fg=text_primary,
            font=("Segoe UI", 10),
            insertbackground=text_primary,
            selectbackground=accent_blue,
            selectforeground="white",
            state=tk.DISABLED,
            relief="flat",
            borderwidth=0,
            padx=10,
            pady=8,
        )
        self.context_text.pack(fill=tk.X)

        # Prompt header
        prompt_header = tk.Frame(middle_panel, bg=bg_medium)
        prompt_header.pack(fill=tk.X, padx=15, pady=(10, 5))

        ttk.Label(prompt_header, text="🤖 Model Prompt", style="Title.TLabel").pack(
            anchor=tk.W
        )

        # Prompt text
        prompt_container = tk.Frame(middle_panel, bg=bg_medium)
        prompt_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))

        self.prompt_text = scrolledtext.ScrolledText(
            prompt_container,
            height=15,
            width=45,
            bg=bg_light,
            fg=text_secondary,
            font=("Cascadia Code", 9),
            insertbackground=text_secondary,
            selectbackground=accent_blue,
            selectforeground="white",
            state=tk.DISABLED,
            relief="flat",
            borderwidth=0,
            padx=10,
            pady=8,
        )
        self.prompt_text.pack(fill=tk.BOTH, expand=True)

        # === RIGHT COLUMN - Summary Editor ===
        right_panel = tk.Frame(content_frame, bg=bg_medium, relief="solid", bd=1)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(4, 0))

        # Summary header
        summary_header = tk.Frame(right_panel, bg=bg_medium, height=40)
        summary_header.pack(fill=tk.X, padx=15, pady=(10, 5))
        summary_header.pack_propagate(False)

        ttk.Label(
            summary_header, text="✏️ Generated Summary", style="Title.TLabel"
        ).pack(anchor=tk.W)

        # Summary text editor with clear editability
        summary_container = tk.Frame(right_panel, bg=bg_medium)
        summary_container.pack(fill=tk.X, padx=15, pady=(0, 10))

        # Make it clear this is editable
        edit_hint = ttk.Label(
            summary_container,
            text="💡 Click to edit, or press 'E' to focus",
            style="Info.TLabel",
        )
        edit_hint.pack(anchor=tk.W, pady=(0, 5))

        self.summary_text = scrolledtext.ScrolledText(
            summary_container,
            height=8,
            width=55,
            bg="#ffffff",  # White background to clearly show it's editable
            fg="#000000",  # Black text for contrast
            font=("Segoe UI", 11),
            insertbackground="#000000",  # Black cursor
            selectbackground=accent_blue,
            selectforeground="white",
            wrap=tk.WORD,
            relief="solid",
            borderwidth=2,
            padx=10,
            pady=8,
        )
        self.summary_text.pack(fill=tk.X)

        # Instructions with better styling
        instructions_container = tk.Frame(right_panel, bg=bg_medium)
        instructions_container.pack(fill=tk.X, padx=15, pady=(10, 0))

        ttk.Label(
            instructions_container, text="⌨️ Keyboard Shortcuts", style="Title.TLabel"
        ).pack(anchor=tk.W)

        instructions = """Ctrl+Enter → Accept current summary
Ctrl+E → Edit summary (focus text area)  
Ctrl+S → Skip this procedure
Ctrl+← / Ctrl+→ → Navigate procedures
Ctrl+Q → Save and continue
Escape → Remove focus from editor
Tab / Shift+Tab → Cycle between panels

Review the context and edit the summary as needed.
Your changes are automatically saved."""

        instruction_text = scrolledtext.ScrolledText(
            instructions_container,
            height=10,
            width=55,
            bg=bg_light,
            fg=text_secondary,
            font=("Segoe UI", 9),
            state=tk.DISABLED,
            wrap=tk.WORD,
            relief="flat",
            borderwidth=0,
            padx=10,
            pady=5,
        )
        instruction_text.pack(fill=tk.X, pady=(5, 10))
        instruction_text.config(state=tk.NORMAL)
        instruction_text.insert(tk.END, instructions)
        instruction_text.config(state=tk.DISABLED)

        # Action buttons with modern styling
        button_container = tk.Frame(right_panel, bg=bg_medium)
        button_container.pack(fill=tk.X, padx=15, pady=(0, 15))

        button_frame = tk.Frame(button_container, bg=bg_medium)
        button_frame.pack()

        # Custom button styling
        button_style = {
            "font": ("Segoe UI", 10, "bold"),
            "padx": 15,
            "pady": 8,
            "relief": "flat",
            "cursor": "hand2",
        }

        self.accept_btn = tk.Button(
            button_frame,
            text="✓ Accept (Enter)",
            command=self.accept_summary,
            bg=accent_green,
            fg="white",
            **button_style,
        )
        self.accept_btn.pack(side=tk.LEFT, padx=(0, 8))

        self.skip_btn = tk.Button(
            button_frame,
            text="⏭ Skip (S)",
            command=self.skip_summary,
            bg=accent_yellow,
            fg="black",
            **button_style,
        )
        self.skip_btn.pack(side=tk.LEFT, padx=(0, 8))

        self.prev_btn = tk.Button(
            button_frame,
            text="← Previous",
            command=self.previous_procedure,
            bg=bg_light,
            fg=text_primary,
            **button_style,
        )
        self.prev_btn.pack(side=tk.LEFT, padx=(0, 8))

        self.next_btn = tk.Button(
            button_frame,
            text="Next →",
            command=self.next_procedure,
            bg=accent_blue,
            fg="white",
            **button_style,
        )
        self.next_btn.pack(side=tk.LEFT)

        # Status bar
        status_frame = tk.Frame(right_panel, bg=bg_medium)
        status_frame.pack(fill=tk.X, padx=15, pady=(10, 15))

        self.status_label = ttk.Label(status_frame, text="", style="Info.TLabel")
        self.status_label.pack(anchor=tk.W)

    def setup_bindings(self):
        """Setup keyboard bindings that work globally."""
        # Global bindings that work regardless of focus
        self.root.bind_all("<Control-Return>", lambda e: self.accept_summary())
        self.root.bind_all("<Control-s>", lambda e: self.skip_summary())
        self.root.bind_all("<Control-e>", lambda e: self.focus_summary_editor())
        self.root.bind_all("<Control-Left>", lambda e: self.previous_procedure())
        self.root.bind_all("<Control-Right>", lambda e: self.next_procedure())
        self.root.bind_all("<Control-q>", lambda e: self.save_and_continue())

        # Escape should just remove focus from editor, not exit
        self.root.bind_all("<Escape>", lambda e: self.root.focus())

        # Allow Tab navigation between panels
        self.root.bind_all("<Tab>", lambda e: self.cycle_focus())
        self.root.bind_all("<Shift-Tab>", lambda e: self.cycle_focus(reverse=True))

        # Make sure clicking in text areas still allows editing
        self.summary_text.bind("<Button-1>", lambda e: self.summary_text.focus())
        if hasattr(self, "file_text"):
            self.file_text.bind("<Button-1>", lambda e: self.file_text.focus())
        if hasattr(self, "context_text"):
            self.context_text.bind("<Button-1>", lambda e: self.context_text.focus())
        if hasattr(self, "prompt_text"):
            self.prompt_text.bind("<Button-1>", lambda e: self.prompt_text.focus())

    def focus_summary_editor(self):
        """Focus the summary editor for editing."""
        self.summary_text.focus()
        # Place cursor at end of text
        self.summary_text.mark_set(tk.INSERT, tk.END)
        self.summary_text.see(tk.INSERT)
        return "break"  # Prevent default behavior

    def cycle_focus(self, reverse=False):
        """Cycle focus between the main text areas."""
        focused_widget = self.root.focus_get()

        # Define focus order based on available widgets
        focus_order = []
        if hasattr(self, "file_text"):
            focus_order.append(self.file_text)
        if hasattr(self, "context_text"):
            focus_order.append(self.context_text)
        if hasattr(self, "prompt_text"):
            focus_order.append(self.prompt_text)
        focus_order.append(self.summary_text)

        if reverse:
            focus_order.reverse()

        try:
            current_index = focus_order.index(focused_widget)
            next_index = (current_index + 1) % len(focus_order)
        except ValueError:
            # If current widget not in order, start with first
            next_index = 0

        focus_order[next_index].focus()
        return "break"

    def display_current_procedure(self):
        """Display the current procedure in the GUI."""
        if not self.batch_data or self.current_index >= len(self.batch_data):
            return

        proc = self.batch_data[self.current_index]

        # Update progress
        total = len(self.batch_data)
        self.progress_label.config(
            text=f"Procedure {self.current_index + 1} of {total}"
        )

        # Update status indicator color
        status = proc.get("status", "pending")
        status_colors = {
            "pending": "#ffb900",  # Yellow
            "accepted": "#16c60c",  # Green
            "edited": "#0078d4",  # Blue
            "skipped": "#d13438",  # Red
        }
        self.status_indicator.config(fg=status_colors.get(status, "#ffb900"))

        # === LEFT COLUMN - Full File Contents ===
        full_file_content = self.get_full_file_content(proc)
        self.file_text.config(state=tk.NORMAL)
        self.file_text.delete(1.0, tk.END)
        self.file_text.insert(tk.END, full_file_content)
        self.file_text.config(state=tk.DISABLED)

        # === MIDDLE COLUMN - Context Information ===
        context_info = f"""📁 MODULE: {proc.get('module_name', 'unknown').upper()}
📄 FILE: {Path(proc['file_path']).name}
⚙️ PROCEDURE: {proc['name']}
📝 PARAMETERS: {', '.join(proc.get('parameters', []))}
📍 LINE: {proc.get('line_number', '?')}
🔄 STATUS: {status.upper()}

📋 FILE HEADER:
{proc.get('file_header', '(no header)')}"""

        self.context_text.config(state=tk.NORMAL)
        self.context_text.delete(1.0, tk.END)
        self.context_text.insert(tk.END, context_info)
        self.context_text.config(state=tk.DISABLED)

        # === MIDDLE COLUMN - Model Prompt ===
        model_prompt = self.create_model_prompt(proc)
        self.prompt_text.config(state=tk.NORMAL)
        self.prompt_text.delete(1.0, tk.END)
        self.prompt_text.insert(tk.END, model_prompt)
        self.prompt_text.config(state=tk.DISABLED)

        # === RIGHT COLUMN - Summary Editor ===
        self.summary_text.config(state=tk.NORMAL)
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(tk.END, proc.get("human_summary", ""))
        # Keep it editable - don't disable!

        # Update status display
        self.status_label.config(
            text=f"Status: {status.upper()} • Use keyboard shortcuts to navigate"
        )

    def get_full_file_content(self, proc):
        """Generate full file content with line numbers."""
        # Simulate full file content (in real implementation, this would read the actual file)
        full_content = f"""-- {proc.get('file_header', '').split(chr(10))[0] if proc.get('file_header') else 'IFS Cloud Package'}
-- Module: {proc.get('module_name', 'UNKNOWN')}
-- File: {Path(proc['file_path']).name}

PACKAGE BODY {proc.get('module_name', 'Unknown')}_API IS

-- Constants and variables
lu_name_ CONSTANT VARCHAR2(30) := '{proc.get('module_name', 'Unknown')}_API';

{proc.get('body', 'PROCEDURE body not available')}

-- Other procedures in the same file...
PROCEDURE Another_Procedure___ IS
BEGIN
   NULL;
END Another_Procedure___;

END {proc.get('module_name', 'Unknown')}_API;"""

        # Add line numbers
        lines = full_content.split("\n")
        numbered_lines = [f"{i+1:3d}  {line}" for i, line in enumerate(lines)]
        return "\n".join(numbered_lines)

    def create_model_prompt(self, proc):
        """Create the exact prompt that would be sent to the model."""
        name = proc["name"]
        params = proc.get("parameters", [])
        business_code = proc.get("business_code", "")
        module = proc.get("module_name", "unknown")

        # Filter out common CRUD parameters
        filtered_params = [
            p
            for p in params
            if not any(
                crud in p.lower()
                for crud in ["info_", "objid_", "objversion_", "attr_", "action_"]
            )
        ]

        param_list = ", ".join(filtered_params) if filtered_params else "no parameters"

        prompt = f"""Analyze this IFS Cloud procedure and provide a concise business summary:

Module: {module}
Procedure: {name}
Parameters: {param_list}

Code Logic:
{business_code}

Provide a single sentence summary that describes the business purpose of this procedure."""

        return prompt

    def accept_summary(self):
        """Accept the current summary."""
        if not self.batch_data or self.current_index >= len(self.batch_data):
            return

        proc = self.batch_data[self.current_index]
        current_summary = self.summary_text.get(1.0, tk.END).strip()

        if current_summary != proc.get("generated_summary", ""):
            proc["status"] = "edited"
        else:
            proc["status"] = "accepted"

        proc["human_summary"] = current_summary

        self.next_procedure()

    def skip_summary(self):
        """Skip the current summary."""
        if not self.batch_data or self.current_index >= len(self.batch_data):
            return

        proc = self.batch_data[self.current_index]
        proc["status"] = "skipped"

        self.next_procedure()

    def previous_procedure(self):
        """Go to previous procedure."""
        if self.current_index > 0:
            self.current_index -= 1
            self.display_current_procedure()

    def next_procedure(self):
        """Go to next procedure."""
        if self.current_index < len(self.batch_data) - 1:
            self.current_index += 1
            self.display_current_procedure()
        else:
            # All procedures reviewed
            self.save_and_continue()

    def save_and_continue(self):
        """Save current work and continue."""
        # Update current procedure with any unsaved changes
        if self.batch_data and self.current_index < len(self.batch_data):
            proc = self.batch_data[self.current_index]
            current_summary = self.summary_text.get(1.0, tk.END).strip()

            if proc["status"] == "pending":
                if current_summary != proc.get("generated_summary", ""):
                    proc["status"] = "edited"
                    proc["human_summary"] = current_summary

        self.root.quit()

    def run(self):
        """Run the GUI main loop."""
        self.root.mainloop()


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
            "file_path": "/test/sales/customer_order.plsql",
            "module_name": "SALES",
            "file_header": "-- Customer order management procedures\n-- Handles discount calculations",
            "line_number": 145,
            "ast_info": {
                "control_structures": [
                    {"type": "IF", "condition": "order_amount_ > 10000"},
                    {"type": "ELSIF", "condition": "order_amount_ > 5000"},
                ]
            },
            "business_code": """-- IF: order_amount_ > 10000
-- ELSIF: order_amount_ > 5000
   IF order_amount_ > 10000 THEN
      discount_rate_ := 0.15;
   ELSIF order_amount_ > 5000 THEN
      discount_rate_ := 0.10;
   ELSE
      discount_rate_ := 0.05;
  -- ... (more code below)""",
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
            "file_path": "/test/inventory/inventory_part.plsql",
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
            "business_code": """-- SELECT: qty_onhand FROM inventory_part_tab
-- IF: available_qty_ < required_qty_
   SELECT qty_onhand INTO available_qty_
   FROM inventory_part_tab
   WHERE part_no = part_no_ AND contract = site_;
   
   IF available_qty_ < required_qty_ THEN
  -- ... (more code below)""",
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
            "file_path": "/test/manufacturing/work_order.plsql",
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
            "business_code": """-- FOR: rec IN (SELECT * FROM work_order_operation
-- IF: rec.status = 'RELEASED'
   FOR rec IN (SELECT * FROM work_order_operation WHERE wo_no = wo_no_) LOOP
      IF rec.status = 'RELEASED' THEN
         Update_Operation_Status___(wo_no_, rec.operation_no, 'IN_PROGRESS');
         Log_Work_Progress___(employee_id_, wo_no_, rec.operation_no);
  -- ... (more code below)""",
        },
    ]

    return test_procedures


def main():
    """Test the GUI with mock data."""
    print("🧪 Testing GUI component...")

    try:
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

        print("\n🎉 GUI test completed successfully!")

    except Exception as e:
        print(f"❌ Error testing GUI: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
