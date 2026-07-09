"""
Comment Stripper -- language-agnostic comment removal for line-level coverage.
================================================================================
Strips comments from source code so FS regex matching operates on actual
code, not comments. Uses tokenize for Python (handles edge cases like
# inside strings) and regex patterns for other languages.

Supports:
  - Python:   # comments, triple-quoted docstrings
  - C-like:   // line comments, /* block comments */
  - Generic:  configurable comment tokens
"""

import io
import re
import tokenize
from collections import defaultdict


# ============================================================
# Base class
# ============================================================

class CommentStripper:
    """Abstract base for language-specific comment strippers.

    Each stripper takes source code and returns a list of
    (line_number, text_after_stripping, is_significant: bool).

    is_significant=False means: blank lines, pure-comment lines,
    docstring-only lines, and import/boilerplate lines.
    These lines are excluded from line-level coverage analysis.
    """

    # Lines matching these patterns are auto-marked non-significant
    # even if they contain code. Subclasses can override.
    NON_SIGNIFICANT_PATTERNS: list[str] = [
        r'^\s*(import\s+|from\s+\S+\s+import\s+)',   # import statements
        r'^\s*$',                                       # blank lines
        r'^\s*[\)\]\}]+\s*$',                          # lone closing brackets
    ]

    def strip(self, source: str) -> list[tuple[int, str, bool]]:
        """Strip comments from source code.

        Args:
            source: Full source code string.

        Returns:
            List of (line_num: int, text: str, is_significant: bool).
            line_num is 1-based. text has comments removed.
        """
        raise NotImplementedError

    def _is_non_significant(self, line: str) -> bool:
        """Check if a line matches any non-significant pattern."""
        for pat in self.NON_SIGNIFICANT_PATTERNS:
            if re.match(pat, line):
                return True
        return False


# ============================================================
# Python comment stripper (tokenize-based, handles edge cases)
# ============================================================

class PythonCommentStripper(CommentStripper):
    """Strips Python comments using tokenize for precise handling.

    Correctly handles:
      - # line comments and trailing comments
      - ''' and \"\"\" multi-line strings (docstrings)
      - # inside string literals (NOT treated as comments)
      - Incomplete/broken code (tokenize errors are caught)
    """

    def strip(self, source: str) -> list[tuple[int, str, bool]]:
        """Strip Python comments from source."""
        lines = source.split('\n')
        n_lines = len(lines)

        # --- Pass 0: detect multi-line parenthesized imports ---
        # Lines like '    Flask,' inside 'from flask import (\n    Flask,\n...'
        # are pure boilerplate continuation lines -> non-significant.
        import_continuation_lines: set[int] = set()
        _detect_import_continuations(lines, import_continuation_lines)

        # --- Pass 1: use tokenize to find comment token spans ---
        # comment_spans[line_num] = [(col_start, col_end), ...]
        comment_spans: dict[int, list[tuple[int, int]]] = defaultdict(list)
        # docstring_lines: set of line numbers that are inside
        # a module/class/function docstring (excluded from coverage)
        docstring_lines: set[int] = set()

        try:
            readline = io.StringIO(source).readline
            tokens = tokenize.generate_tokens(readline)
            for tok in tokens:
                if tok.type == tokenize.COMMENT:
                    line = tok.start[0]
                    col = tok.start[1]
                    end_col = tok.end[1]
                    comment_spans[line].append((col, end_col))
                elif tok.type == tokenize.STRING:
                    t_start_line = tok.start[0]
                    t_end_line = tok.end[0]
                    t_start_col = tok.start[1]
                    text = tok.string
                    is_triple = (text.startswith('"""') or text.startswith("'''"))
                    is_multi = t_end_line > t_start_line
                    # Multi-line string at low indent -> likely docstring, exclude all lines
                    if is_triple and is_multi and t_start_col < 12:
                        for l in range(t_start_line, t_end_line + 1):
                            docstring_lines.add(l)
                    # Single-line docstring
                    elif is_triple and len(text) > 6 and t_start_col < 12:
                        docstring_lines.add(t_start_line)
                    # Multi-line string at higher indent (e.g. SQL query) ->
                    # only mark the delimiter-only lines as non-significant
                    elif is_triple and is_multi:
                        # Check if the start line is just the delimiter (plus optional closing paren)
                        start_line_text = lines[t_start_line - 1].strip() if t_start_line <= len(lines) else ''
                        if _is_delimiter_only(start_line_text):
                            docstring_lines.add(t_start_line)
                        end_line_text = lines[t_end_line - 1].strip() if t_end_line <= len(lines) else ''
                        if _is_delimiter_only(end_line_text):
                            docstring_lines.add(t_end_line)
        except (tokenize.TokenError, IndentationError):
            pass  # Gracefully handle broken/incomplete code

        # --- Pass 2: process each line ---
        # Pre-compute line start positions (character offsets)
        line_starts = _compute_line_starts(source)

        results: list[tuple[int, str, bool]] = []
        for i, original_line in enumerate(lines):
            line_num = i + 1
            stripped = original_line.strip()

            # Blank line
            if not stripped:
                results.append((line_num, '', False))
                continue

            # Inside a docstring
            if line_num in docstring_lines:
                results.append((line_num, '', False))
                continue

            # Inside a multi-line import continuation
            if line_num in import_continuation_lines:
                results.append((line_num, '', False))
                continue

            # Remove comment portions (replace with spaces to preserve column alignment)
            cleaned = original_line
            if line_num in comment_spans:
                chars = list(cleaned)
                for col_start, col_end in sorted(comment_spans[line_num], reverse=True):
                    for j in range(col_start, min(col_end, len(chars))):
                        chars[j] = ' '
                cleaned = ''.join(chars)

            # After stripping comments, check if anything remains
            cleaned_stripped = cleaned.strip()
            if not cleaned_stripped:
                results.append((line_num, '', False))
                continue

            # Check if the code content is non-significant (imports, etc.)
            if self._is_non_significant(cleaned_stripped):
                results.append((line_num, cleaned, False))
                continue

            results.append((line_num, cleaned, True))

        return results


def _is_delimiter_only(line: str) -> bool:
    """Check if a line is ONLY a triple-quote string delimiter."""
    stripped = line.strip()
    if not stripped:
        return False
    # """ or ''' alone (possibly with closing paren)
    if stripped in ('"""', "'''", ')"""', ")'''"):
        return True
    # """ with trailing content that is just delimiter chars
    if stripped.startswith('"""') and stripped.endswith('"""') and len(stripped) <= 6:
        return True
    if stripped.startswith("'''") and stripped.endswith("'''") and len(stripped) <= 6:
        return True
    # '"""' as complete delimiter on this line
    if stripped in ('co.execute(""")', 'cur.execute(""")', 'cursor.execute(""")',
                     "co.execute(''')", "cur.execute(''')", "cursor.execute(''')"):
        return False  # Has a function call -- not just a delimiter
    return False


def _detect_import_continuations(
    lines: list[str],
    out_set: set[int],
) -> None:
    """Detect lines that are continuations of parenthesized multi-line imports.

    Example:
        from flask import (      <- line L: opens paren
            Flask,               <- line L+1: continuation -> non-significant
            flash,               <- line L+2: continuation -> non-significant
            render_template,     <- line L+3: continuation -> non-significant
        )                        <- line L+4: closes paren -> non-significant

    Also handles:
        import (                 <- Python allows this too
            os,
            sys,
        )
    """
    import re as _re

    # Pattern: 'from X import (' or 'import (' -- opens a parenthesized block
    OPEN_PAT = _re.compile(r'^\s*(?:from\s+\S+\s+)?import\s*\(')
    # Track which lines are inside an open parenthesized import
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if OPEN_PAT.match(stripped):
            # Count parentheses from this line onward
            depth = stripped.count('(') - stripped.count(')')
            j = i + 1
            while j < len(lines) and depth > 0:
                out_set.add(j + 1)  # +1 for 1-based line numbers
                dline = lines[j]
                # Count parentheses (crude but effective for import blocks)
                depth += dline.count('(') - dline.count(')')
                j += 1
            i = j
        else:
            i += 1


# ============================================================
# Generic comment stripper (regex-based, for non-Python languages)
# ============================================================

class GenericCommentStripper(CommentStripper):
    """Regex-based comment stripper for C-like languages.

    Supports configurable comment tokens:
      - single_line: regex for single-line comments (e.g. '//' or '#')
      - block_open/block_close: delimiters for block comments (e.g. '/*' '*/')
    """

    def __init__(self,
                 single_line: str = '//',
                 block_open: str = r'/\*',
                 block_close: str = r'\*/',
                 string_delimiters: list[str] | None = None):
        """
        Args:
            single_line: Regex for single-line comment start.
            block_open: Regex for block comment start.
            block_close: Regex for block comment end.
            string_delimiters: List of string delimiter patterns to skip
                              (comments inside strings are NOT comments).
        """
        self.single_line = single_line
        self.block_open = block_open
        self.block_close = block_close
        self.string_delimiters = string_delimiters or ['"', "'"]

        # Build regex patterns
        sl = re.escape(single_line) if len(single_line) <= 2 else single_line
        bo = re.escape(block_open) if len(block_open) <= 2 else block_open
        bc = re.escape(block_close) if len(block_close) <= 2 else block_close

        self._single_pat = re.compile(f'({sl}.*)$')
        self._block_pat = re.compile(f'({bo}.*?{bc})', re.DOTALL)

    def strip(self, source: str) -> list[tuple[int, str, bool]]:
        """Strip comments from source using regex patterns."""
        lines = source.split('\n')

        # First, remove block comments from the entire source
        source_no_block = self._block_pat.sub(' ', source)
        lines_no_block = source_no_block.split('\n')

        # Track which lines were entirely inside a block comment
        # (replaced to empty/whitespace-only by the block comment removal)
        block_only_lines: set[int] = set()
        for i, (orig, stripped) in enumerate(zip(lines, lines_no_block)):
            if orig.strip() and not stripped.strip():
                block_only_lines.add(i + 1)

        results: list[tuple[int, str, bool]] = []
        for i, line in enumerate(lines_no_block):
            line_num = i + 1
            stripped = line.strip()

            if not stripped:
                results.append((line_num, '', False))
                continue

            if line_num in block_only_lines:
                results.append((line_num, '', False))
                continue

            # Remove single-line comments (preserving strings approximately)
            cleaned = self._strip_single_line_comment(line)

            cleaned_stripped = cleaned.strip()
            if not cleaned_stripped:
                results.append((line_num, '', False))
                continue

            if self._is_non_significant(cleaned_stripped):
                results.append((line_num, cleaned, False))
                continue

            results.append((line_num, cleaned, True))

        return results

    def _strip_single_line_comment(self, line: str) -> str:
        """Remove a single-line comment, with basic string awareness."""
        # Simple heuristic: find the comment marker NOT inside quotes
        # For robustness in the generic case, we do a simple state machine
        in_double = False
        in_single = False
        for i, ch in enumerate(line):
            if ch == '"' and not in_single:
                in_double = not in_double
            elif ch == "'" and not in_double:
                in_single = not in_single
            elif not in_double and not in_single:
                # Check for comment start
                if line[i:i+len(self.single_line)] == self.single_line:
                    return line[:i]
        return line


# ============================================================
# Factory & auto-detection
# ============================================================

def get_stripper(language: str = 'python') -> CommentStripper:
    """Factory: return the appropriate CommentStripper for a language.

    Args:
        language: 'python', 'java', 'cpp', 'javascript', 'generic',
                  or 'auto' (try auto-detection from source).

    Returns:
        CommentStripper instance.
    """
    if language == 'python':
        return PythonCommentStripper()
    elif language in ('java', 'cpp', 'c', 'javascript', 'js', 'typescript', 'ts',
                      'rust', 'swift', 'kotlin', 'scala', 'go', 'csharp', 'cs'):
        return GenericCommentStripper(single_line='//',
                                       block_open=r'/\*',
                                       block_close=r'\*/')
    elif language in ('ruby', 'bash', 'sh', 'perl', 'yaml'):
        return GenericCommentStripper(single_line='#',
                                       block_open=r'=begin',
                                       block_close=r'=end')
    else:
        # Generic fallback: try both // and # single-line, plus /* */
        return GenericCommentStripper()


def create_language_config_from_submissions(
    submissions: list[dict],
    sample_size: int = 5,
) -> dict:
    """Auto-detect language from student code using multi-signal scoring.

    Uses a weighted scoring system rather than simple comment counting.
    This avoids misclassification from edge cases like:
      - Python strings containing ``//`` (e.g., URLs)
      - ``#`` comments in non-Python languages (Ruby, YAML, shell)
      - Mixed comment styles in template/boilerplate code

    Detection signals (weighted by reliability):
      - Python: ``def``, ``class``, ``import``/``from X import``, ``:`` lines,
        ``self`` parameter, ``__name__``, lack of semicolons
      - Java/C/C++: ``//`` comments, ``/* */`` blocks, ``{``/``}`` braces,
        ``;`` terminators, ``public class``, ``System.out``
      - JavaScript/TS: ``//`` comments, ``const``/``let``/``var``, ``function``,
        ``=>`` arrows, ``console.log``

    Args:
        submissions: List of {student, code} dicts.
        sample_size: Number of submissions to sample.

    Returns:
        {'language': 'python' | 'java' | 'javascript' | 'generic',
         'single_line': detected single-line comment token,
         'has_block_comments': bool}
    """
    sample = submissions[:sample_size]
    if not sample:
        return {'language': 'python'}  # default

    py_score = 0
    c_score = 0
    js_score = 0

    for s in sample:
        code = s.get('code', '')
        lines = code.split('\n')
        full_text = code

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # --- Python signals (high-confidence) ---
            if stripped.startswith('def ') and '(' in stripped and ':' in stripped:
                py_score += 5  # Very strong Python indicator
            if stripped.startswith('class ') and ':' in stripped:
                py_score += 5
            if stripped.startswith('from ') and ' import ' in stripped:
                py_score += 4
            if stripped.startswith('import ') and not stripped.startswith('import ('):
                # Java also has 'import', but Python's is standalone
                py_score += 2
            if stripped.startswith('#'):
                py_score += 1  # Weak signal (many languages use #)
            if 'self' in stripped.split('(')[0] if '(' in stripped else False:
                pass  # Handled below
            if 'self' in stripped and ('def ' in stripped or 'self.' in stripped):
                py_score += 3
            if '__name__' in stripped:
                py_score += 3
            if stripped.startswith('@'):
                py_score += 3  # Decorators are very Python
            if stripped.startswith('elif '):
                py_score += 3
            if stripped == 'pass':
                py_score += 1
            # Python uses colons for blocks
            if stripped.rstrip().endswith(':') and not stripped.startswith('//'):
                if any(kw in stripped for kw in
                       ('if ', 'else', 'elif ', 'for ', 'while ', 'def ',
                        'class ', 'try', 'except', 'finally', 'with ')):
                    py_score += 1

            # --- C-like signals ---
            if stripped.startswith('//'):
                c_score += 3
                js_score += 3
            if stripped.startswith('/*') or stripped.endswith('*/'):
                c_score += 3
            if stripped.rstrip().endswith(';'):
                c_score += 2
                js_score += 1
            if stripped.startswith('{') or stripped == '}':
                c_score += 2
                js_score += 1
            if 'public class' in stripped or 'public static' in stripped:
                c_score += 5
            if 'System.out' in stripped:
                c_score += 4
            if '#include' in stripped:
                c_score += 5

            # --- JavaScript/TS signals ---
            if stripped.startswith('const ') or stripped.startswith('let ') or stripped.startswith('var '):
                js_score += 4
            if stripped.startswith('function '):
                js_score += 3
            if '=>' in stripped:
                js_score += 4
            if 'console.log' in stripped:
                js_score += 3
            if 'require(' in stripped:
                js_score += 2
            if 'export ' in stripped:
                js_score += 2

        # --- Whole-file signals ---
        if 'import java.' in full_text or 'package ' in full_text.split('\n')[0]:
            c_score += 8

    # Normalise by sample size
    py_score /= len(sample)
    c_score /= len(sample)
    js_score /= len(sample)

    # Decision: language with highest score wins, with minimum threshold
    if py_score >= max(c_score, js_score) and py_score >= 3:
        return {'language': 'python'}
    elif js_score >= max(py_score, c_score) and js_score >= 3:
        return {'language': 'javascript'}
    elif c_score >= max(py_score, js_score) and c_score >= 3:
        return {'language': 'java'}
    elif py_score > 0:
        return {'language': 'python'}  # Default to Python on any Python signal
    else:
        return {'language': 'generic'}


def _compute_line_starts(source: str) -> list[int]:
    """Compute character offsets where each physical line begins (0-indexed)."""
    starts = [0]
    for i, ch in enumerate(source):
        if ch == '\n':
            starts.append(i + 1)
    return starts
