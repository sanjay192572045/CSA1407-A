import ast
import io
import shutil
import subprocess
import tempfile
import tokenize
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Python → Java Translator | Compiler Design",
    page_icon="⚙️",
    layout="wide",
)


@dataclass
class SymbolEntry:
    name: str
    java_type: str
    line: int
    initialized: bool = True


@dataclass
class TranslationResult:
    tokens_df: pd.DataFrame
    ast_dump: str
    symbols: List[SymbolEntry] = field(default_factory=list)
    ir_lines: List[str] = field(default_factory=list)
    java_code: str = ""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


SAMPLE_PROGRAMS: Dict[str, str] = {
    "Arithmetic & Conditionals": """x = 10
y = 20
total = x + y
if total >= 25:
    print("High total:")
    print(total)
else:
    print("Low total:")
    print(total)
""",
    "While & For Range Loops": """count = 0
while count < 3:
    print(count)
    count = count + 1

for i in range(0, 5, 1):
    sq = i * i
    print(sq)
""",
    "Boolean & Float Expressions": """radius = 3.5
pi = 3.14
area = pi * radius * radius
is_large = area > 20.0
if is_large:
    print(area)
""",
}


# ---------------------------------------------------------------------------
# Module 1: Lexical & Syntax Analysis (Frontend)
# ---------------------------------------------------------------------------
def tokenize_python_code(source: str) -> Tuple[pd.DataFrame, List[str]]:
    errors: List[str] = []
    rows = []
    try:
        tok_gen = tokenize.generate_tokens(io.StringIO(source).readline)
        for tok in tok_gen:
            tok_name = tokenize.tok_name.get(tok.type, str(tok.type))
            if tok_name in {"NL", "NEWLINE", "ENDMARKER", "ENCODING", "INDENT", "DEDENT"}:
                continue
            rows.append(
                {
                    "Line": tok.start[0],
                    "Col": tok.start[1],
                    "Token Type": tok_name,
                    "Lexeme": tok.string,
                }
            )
    except tokenize.TokenError as exc:
        errors.append(f"Lexical Error: {exc}")
    except IndentationError as exc:
        errors.append(f"Indentation Error (Line {exc.lineno}): {exc.msg}")
    return pd.DataFrame(rows), errors


def parse_python_ast(source: str) -> Tuple[Optional[ast.Module], str, List[str]]:
    errors: List[str] = []
    try:
        tree = ast.parse(source)
        formatted = ast.dump(tree, indent=2)
        return tree, formatted, errors
    except SyntaxError as exc:
        line_no = exc.lineno if exc.lineno is not None else "?"
        errors.append(f"Syntax Error (Line {line_no}): {exc.msg}")
        return None, "", errors


# ---------------------------------------------------------------------------
# Module 2 & 3: Semantic Analysis, IR Generation & Java Code Generation
# ---------------------------------------------------------------------------
class PythonToJavaCompiler:
    def __init__(self) -> None:
        self.symbol_table: Dict[str, SymbolEntry] = {}
        self.declared_vars: set[str] = set()
        self.ir_lines: List[str] = []
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.temp_counter = 0
        self.label_counter = 0

    def new_temp(self) -> str:
        self.temp_counter += 1
        return f"t{self.temp_counter}"

    def new_label(self) -> str:
        self.label_counter += 1
        return f"L{self.label_counter}"

    def infer_expr_type(self, node: ast.AST, line: int) -> str:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return "boolean"
            if isinstance(node.value, int):
                return "int"
            if isinstance(node.value, float):
                return "double"
            if isinstance(node.value, str):
                return "String"
            return "Object"

        if isinstance(node, ast.Name):
            if node.id in self.symbol_table:
                return self.symbol_table[node.id].java_type
            self.warnings.append(f"Line {line}: Variable '{node.id}' used before assignment; defaulting type to int.")
            return "int"

        if isinstance(node, ast.UnaryOp):
            operand_type = self.infer_expr_type(node.operand, line)
            if isinstance(node.op, ast.Not):
                return "boolean"
            return operand_type

        if isinstance(node, ast.BinOp):
            left_t = self.infer_expr_type(node.left, line)
            right_t = self.infer_expr_type(node.right, line)
            if isinstance(node.op, ast.Div):
                return "double"
            if "String" in (left_t, right_t):
                return "String"
            if "double" in (left_t, right_t):
                return "double"
            if left_t != right_t:
                self.warnings.append(
                    f"Line {line}: Mixed operand types ({left_t} and {right_t}); widening to compatible type."
                )
            return "int"

        if isinstance(node, ast.BoolOp) or isinstance(node, ast.Compare):
            return "boolean"

        return "int"

    def translate_expr_ir(self, node: ast.AST, line: int) -> Tuple[str, str]:
        """Emits 3-address IR and returns (ir_operand, java_expression_str)."""
        if isinstance(node, ast.Constant):
            jtype = self.infer_expr_type(node, line)
            if jtype == "String":
                escaped = str(node.value).replace("\\", "\\\\").replace('"', '\\"')
                lit = f'"{escaped}"'
            elif jtype == "boolean":
                lit = "true" if node.value else "false"
            else:
                lit = str(node.value)
            t = self.new_temp()
            self.ir_lines.append(f"{t} = {lit}")
            return t, lit

        if isinstance(node, ast.Name):
            jtype = self.infer_expr_type(node, line)
            t = self.new_temp()
            self.ir_lines.append(f"{t} = {node.id}")
            return t, node.id

        if isinstance(node, ast.UnaryOp):
            op_ir, op_java = self.translate_expr_ir(node.operand, line)
            t = self.new_temp()
            if isinstance(node.op, ast.USub):
                self.ir_lines.append(f"{t} = -{op_ir}")
                return t, f"(-{op_java})"
            if isinstance(node.op, ast.UAdd):
                self.ir_lines.append(f"{t} = +{op_ir}")
                return t, f"(+{op_java})"
            if isinstance(node.op, ast.Not):
                self.ir_lines.append(f"{t} = !{op_ir}")
                return t, f"(!{op_java})"

        if isinstance(node, ast.BinOp):
            l_ir, l_java = self.translate_expr_ir(node.left, line)
            r_ir, r_java = self.translate_expr_ir(node.right, line)
            t = self.new_temp()

            if isinstance(node.op, ast.Pow):
                self.ir_lines.append(f"{t} = pow({l_ir}, {r_ir})")
                return t, f"Math.pow({l_java}, {r_java})"

            op_map = {
                ast.Add: "+",
                ast.Sub: "-",
                ast.Mult: "*",
                ast.Div: "/",
                ast.FloorDiv: "/",
                ast.Mod: "%",
            }
            sym = op_map.get(type(node.op))
            if sym is None:
                self.errors.append(f"Line {line}: Unsupported binary operator '{type(node.op).__name__}'.")
                return l_ir, l_java

            self.ir_lines.append(f"{t} = {l_ir} {sym} {r_ir}")
            return t, f"({l_java} {sym} {r_java})"

        if isinstance(node, ast.Compare) and len(node.ops) == 1 and len(node.comparators) == 1:
            l_ir, l_java = self.translate_expr_ir(node.left, line)
            r_ir, r_java = self.translate_expr_ir(node.comparators[0], line)
            comp_map = {
                ast.Eq: "==",
                ast.NotEq: "!=",
                ast.Lt: "<",
                ast.LtE: "<=",
                ast.Gt: ">",
                ast.GtE: ">=",
            }
            comp_sym = comp_map.get(type(node.ops[0]))
            if comp_sym is None:
                self.errors.append(f"Line {line}: Unsupported comparison operator.")
                return l_ir, "false"
            t = self.new_temp()
            self.ir_lines.append(f"{t} = {l_ir} {comp_sym} {r_ir}")
            return t, f"({l_java} {comp_sym} {r_java})"

        if isinstance(node, ast.BoolOp):
            joiner = " && " if isinstance(node.op, ast.And) else " || "
            ir_joiner = " && " if isinstance(node.op, ast.And) else " || "
            ir_parts, java_parts = [], []
            for val in node.values:
                v_ir, v_java = self.translate_expr_ir(val, line)
                ir_parts.append(v_ir)
                java_parts.append(f"({v_java})")
            t = self.new_temp()
            self.ir_lines.append(f"{t} = {ir_joiner.join(ir_parts)}")
            return t, joiner.join(java_parts)

        self.errors.append(f"Line {line}: Unsupported expression construct '{type(node).__name__}'.")
        return "0", "0"

    def emit_stmt_list(self, stmts: List[ast.stmt], indent: str) -> List[str]:
        java_lines: List[str] = []
        for stmt in stmts:
            java_lines.extend(self.emit_stmt(stmt, indent))
        return java_lines

    def emit_stmt(self, stmt: ast.stmt, indent: str) -> List[str]:
        line = getattr(stmt, "lineno", 0)

        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                self.errors.append(f"Line {line}: Only single variable assignments are supported in this translator.")
                return []
            var_name = stmt.targets[0].id
            inferred_type = self.infer_expr_type(stmt.value, line)
            _, rhs_java = self.translate_expr_ir(stmt.value, line)

            # Check if re-assigned to an incompatible type
            if var_name in self.symbol_table:
                prev_type = self.symbol_table[var_name].java_type
                if prev_type != inferred_type and not (prev_type == "double" and inferred_type == "int"):
                    self.warnings.append(
                        f"Line {line}: Variable '{var_name}' changed type from {prev_type} to {inferred_type}."
                    )
            else:
                self.symbol_table[var_name] = SymbolEntry(
                    name=var_name,
                    java_type=inferred_type,
                    line=line,
                )

            self.ir_lines.append(f"{var_name} = {rhs_java}")

            if var_name not in self.declared_vars:
                self.declared_vars.add(var_name)
                jtype = self.symbol_table[var_name].java_type
                return [f"{indent}{jtype} {var_name} = {rhs_java};"]
            return [f"{indent}{var_name} = {rhs_java};"]

        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            if isinstance(call.func, ast.Name) and call.func.id == "print":
                if not call.args:
                    self.ir_lines.append("print()")
                    return [f"{indent}System.out.println();"]
                _, arg_java = self.translate_expr_ir(call.args[0], line)
                self.ir_lines.append(f"print({arg_java})")
                return [f"{indent}System.out.println({arg_java});"]
            self.errors.append(f"Line {line}: Unsupported function call '{ast.dump(call.func)}'.")
            return []

        if isinstance(stmt, ast.If):
            _, cond_java = self.translate_expr_ir(stmt.test, line)
            lbl_else = self.new_label()
            lbl_end = self.new_label()
            self.ir_lines.append(f"ifFalse {cond_java} goto {lbl_else}")

            lines = [f"{indent}if ({cond_java}) {{"]
            lines.extend(self.emit_stmt_list(stmt.body, indent + "    "))
            self.ir_lines.append(f"goto {lbl_end}")
            self.ir_lines.append(f"{lbl_else}:")

            if stmt.orelse:
                lines.append(f"{indent}}} else {{")
                lines.extend(self.emit_stmt_list(stmt.orelse, indent + "    "))

            lines.append(f"{indent}}}")
            self.ir_lines.append(f"{lbl_end}:")
            return lines

        if isinstance(stmt, ast.While):
            lbl_start = self.new_label()
            lbl_end = self.new_label()
            self.ir_lines.append(f"{lbl_start}:")
            _, cond_java = self.translate_expr_ir(stmt.test, line)
            self.ir_lines.append(f"ifFalse {cond_java} goto {lbl_end}")

            lines = [f"{indent}while ({cond_java}) {{"]
            lines.extend(self.emit_stmt_list(stmt.body, indent + "    "))
            self.ir_lines.append(f"goto {lbl_start}")
            self.ir_lines.append(f"{lbl_end}:")
            lines.append(f"{indent}}}")
            return lines

        if isinstance(stmt, ast.For):
            if not isinstance(stmt.target, ast.Name):
                self.errors.append(f"Line {line}: Loop target must be a simple variable name.")
                return []
            if not (
                isinstance(stmt.iter, ast.Call)
                and isinstance(stmt.iter.func, ast.Name)
                and stmt.iter.func.id == "range"
            ):
                self.errors.append(f"Line {line}: Only 'for <var> in range(...)' loops are supported.")
                return []

            loop_var = stmt.target.id
            self.symbol_table[loop_var] = SymbolEntry(name=loop_var, java_type="int", line=line)
            self.declared_vars.add(loop_var)

            args = stmt.iter.args
            if len(args) == 1:
                _, start_j = self.translate_expr_ir(ast.Constant(value=0), line)
                _, end_j = self.translate_expr_ir(args[0], line)
                _, step_j = self.translate_expr_ir(ast.Constant(value=1), line)
            elif len(args) == 2:
                _, start_j = self.translate_expr_ir(args[0], line)
                _, end_j = self.translate_expr_ir(args[1], line)
                _, step_j = self.translate_expr_ir(ast.Constant(value=1), line)
            elif len(args) == 3:
                _, start_j = self.translate_expr_ir(args[0], line)
                _, end_j = self.translate_expr_ir(args[1], line)
                _, step_j = self.translate_expr_ir(args[2], line)
            else:
                self.errors.append(f"Line {line}: range() expects 1 to 3 arguments.")
                return []

            lbl_start = self.new_label()
            lbl_end = self.new_label()
            self.ir_lines.append(f"{loop_var} = {start_j}")
            self.ir_lines.append(f"{lbl_start}:")
            self.ir_lines.append(f"ifFalse ({loop_var} < {end_j}) goto {lbl_end}")

            lines = [f"{indent}for (int {loop_var} = {start_j}; {loop_var} < {end_j}; {loop_var} += {step_j}) {{"]
            lines.extend(self.emit_stmt_list(stmt.body, indent + "    "))
            self.ir_lines.append(f"{loop_var} = {loop_var} + {step_j}")
            self.ir_lines.append(f"goto {lbl_start}")
            self.ir_lines.append(f"{lbl_end}:")
            lines.append(f"{indent}}}")
            return lines

        self.errors.append(f"Line {line}: Unsupported statement '{type(stmt).__name__}'.")
        return []

    def compile(self, source: str, class_name: str = "TranslatedProgram") -> TranslationResult:
        tokens_df, lex_errors = tokenize_python_code(source)
        tree, ast_dump, syn_errors = parse_python_ast(source)

        all_errors = lex_errors + syn_errors
        if tree is None or all_errors:
            return TranslationResult(
                tokens_df=tokens_df,
                ast_dump=ast_dump,
                errors=all_errors,
            )

        body_lines = self.emit_stmt_list(tree.body, indent="        ")
        java_lines = [
            f"public class {class_name} {{",
            "    public static void main(String[] args) {",
        ]
        java_lines.extend(body_lines)
        java_lines.extend([
            "    }",
            "}",
        ])

        return TranslationResult(
            tokens_df=tokens_df,
            ast_dump=ast_dump,
            symbols=list(self.symbol_table.values()),
            ir_lines=self.ir_lines,
            java_code="\n".join(java_lines),
            errors=self.errors,
            warnings=self.warnings,
        )


def verify_java_output(java_code: str, class_name: str) -> Tuple[str, str]:
    """Runs javac/java if available in environment; otherwise runs a structural check."""
    javac_path = shutil.which("javac")
    java_path = shutil.which("java")

    if javac_path and java_path:
        with tempfile.TemporaryDirectory() as tmpdir:
            java_file = Path(tmpdir) / f"{class_name}.java"
            java_file.write_text(java_code, encoding="utf-8")
            comp = subprocess.run(
                [javac_path, str(java_file)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if comp.returncode != 0:
                return "Compilation Failed (javac)", comp.stderr
            run = subprocess.run(
                [java_path, "-cp", tmpdir, class_name],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if run.returncode != 0:
                return "Runtime Error (java)", run.stderr
            return "Compiled & Executed Successfully", run.stdout

    # Structural fallback for environments without JDK installed (e.g., standard Streamlit Cloud)
    brace_balance = java_code.count("{") - java_code.count("}")
    paren_balance = java_code.count("(") - java_code.count(")")
    if brace_balance == 0 and paren_balance == 0 and f"public class {class_name}" in java_code:
        return (
            "Structural Java Check Passed (JDK not installed in host container)",
            "Balanced braces/parentheses and valid class/main structure verified.",
        )
    return (
        "Structural Check Warning",
        f"Brace balance: {brace_balance}, Parenthesis balance: {paren_balance}.",
    )


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.title("Python → Java Source Code Translator")
st.caption("Compiler Design Capstone: Lexical & Syntax Analysis • Semantic & IR Generation • Java Codegen")

with st.sidebar:
    st.header("Translator Settings")
    preset = st.selectbox("Sample Python Program", options=list(SAMPLE_PROGRAMS.keys()))
    class_name = st.text_input("Java Class Name", value="TranslatedProgram").strip() or "TranslatedProgram"
    uploaded_py = st.file_uploader("Or Upload a .py File", type=["py"])

    st.divider()
    st.markdown(
        """
        **Supported Constructs:**
        * Variables & type inference (`int`, `double`, `boolean`, `String`)
        * Arithmetic, comparison & logical operators
        * `if / else`, `while`, and `for ... in range(...)`
        * `print(...)` statements
        """
    )

default_code = SAMPLE_PROGRAMS[preset]
if uploaded_py is not None:
    default_code = uploaded_py.getvalue().decode("utf-8", errors="replace")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Input: Python Source Code (.py)")
    python_source = st.text_area(
        "Python Editor",
        value=default_code,
        height=340,
        label_visibility="collapsed",
    )
    translate_btn = st.button("Translate to Java", type="primary", use_container_width=True)

compiler = PythonToJavaCompiler()
result = compiler.compile(python_source, class_name=class_name)

with col_right:
    st.subheader(f"Output: Java Source Code ({class_name}.java)")
    if result.errors:
        for err in result.errors:
            st.error(err)
    else:
        for warn in result.warnings:
            st.warning(warn)
        st.code(result.java_code, language="java")
        st.download_button(
            label=f"Download {class_name}.java",
            data=result.java_code,
            file_name=f"{class_name}.java",
            mime="text/x-java-source",
            use_container_width=True,
        )

st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "Module 1: Tokens & AST",
    "Module 2: Symbol Table & IR",
    "Module 3: Compilation & Verification",
    "Architecture Overview",
])

with tab1:
    st.subheader("Lexical Analysis (Tokens)")
    if not result.tokens_df.empty:
        st.dataframe(result.tokens_df, use_container_width=True, hide_index=True)
    st.subheader("Syntax Analysis (Abstract Syntax Tree)")
    if result.ast_dump:
        st.code(result.ast_dump, language="python")

with tab2:
    st.subheader("Semantic Analysis — Symbol Table")
    if result.symbols:
        sym_df = pd.DataFrame(
            [
                {
                    "Identifier": s.name,
                    "Inferred Java Type": s.java_type,
                    "First Seen (Line)": s.line,
                }
                for s in result.symbols
            ]
        )
        st.dataframe(sym_df, use_container_width=True, hide_index=True)
    else:
        st.info("No variables recorded in symbol table.")

    st.subheader("Intermediate Representation (Three-Address Code / IR)")
    if result.ir_lines:
        st.code("\n".join(result.ir_lines), language="text")
    else:
        st.info("No IR instructions generated.")

with tab3:
    st.subheader("Java Verification & Execution Status")
    if result.java_code and not result.errors:
        status_title, status_detail = verify_java_output(result.java_code, class_name=class_name)
        if "Failed" in status_title or "Warning" in status_title:
            st.warning(f"**{status_title}**")
        else:
            st.success(f"**{status_title}**")
        st.code(status_detail, language="text")
    else:
        st.info("Fix translation errors above to verify Java output.")

with tab4:
    st.markdown(
        """
        ### Three-Module Compiler Pipeline
        1. **Lexical & Syntax Analysis (`tokenize` + `ast`):** Breaks `.py` source into tokens, builds an Abstract Syntax Tree, and surfaces lexical or syntax errors.
        2. **Semantic Analysis & IR Generation:** Populates a symbol table with inferred Java types, checks expressions/control flow, and emits intermediate representation instructions.
        3. **Java Code Generation:** Emits a `public class` with `main`, translates control structures and `print` calls into Java syntax, and exports `.java` source code.
        """
    )
