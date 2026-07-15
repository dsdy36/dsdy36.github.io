"""
Pattern Matrix Builder — decompose rubric into good/bad pattern variants.
Uses hardcoded variants for known patterns, AI-driven generation as fallback.
"""
import hashlib
import json
import os
import re
import sys

# Try to load DeepSeek API for AI fallback
try:
    from dotenv import load_dotenv
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _candidates = [
        os.path.join(_script_dir, '.env'),
        os.path.join(_script_dir, '..', '.env'),
        os.path.join(_script_dir, '..', 'FS_generater-v1', '.env'),
    ]
    for _c in _candidates:
        _normalized = os.path.normpath(_c)
        if os.path.exists(_normalized):
            load_dotenv(_normalized, override=True)
            break
    DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
    DEEPSEEK_MODEL = 'deepseek-chat'
except Exception:
    DEEPSEEK_API_KEY = None

_VARIANT_CACHE: dict[str, list[dict]] = {}
_CACHE_PATH = ''


def _cache_key(description: str) -> str:
    """Stable cache key from pattern description."""
    return hashlib.md5(description.strip().lower().encode()).hexdigest()[:12]


def _load_cache(cache_path: str):
    global _VARIANT_CACHE, _CACHE_PATH
    _CACHE_PATH = cache_path
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                _VARIANT_CACHE = json.load(f)
        except Exception:
            _VARIANT_CACHE = {}


def _save_cache():
    if _CACHE_PATH:
        cache_file = os.path.abspath(_CACHE_PATH)
        cache_dir = os.path.dirname(cache_file)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(_VARIANT_CACHE, f, indent=2, ensure_ascii=False)


def _ai_generate_variants(criterion_id: str, ptype: str, index: int,
                           description: str) -> list[dict]:
    """AI-driven fallback: decompose a pattern description into concrete variants.

    Called when no hardcoded pattern in _define_variants matches.
    Results are cached per description hash.
    """
    # Check cache
    key = _cache_key(description)
    if key in _VARIANT_CACHE:
        return _VARIANT_CACHE[key]

    if not DEEPSEEK_API_KEY:
        # No API key available — return generic
        if ptype == 'good':
            return [{'id': 'A', 'instruction': f'Write code that does: {description}',
                     'check_regex': r'.*', 'difficulty': 'basic'}]
        else:
            return [{'id': 'A', 'instruction': f'Write code containing: {description}',
                     'check_regex': r'.*', 'difficulty': 'basic'}]

    # Call AI
    try:
        from openai import OpenAI
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url='https://api.deepseek.com/v1')

        ptype_label = 'CORRECT (good pattern)' if ptype == 'good' else 'INCORRECT (bad pattern)'
        system = """You decompose rubric patterns into concrete, DETECTABLE code-level variants.
Each variant must describe a SPECIFIC code behavior that can be verified by regex or code check.

CRITICAL for bad patterns: NEVER describe what's MISSING ("not using X", "didn't do Y").
ALWAYS describe what IS there — the specific wrong code the student wrote.
  WRONG: "Not using flash messages for errors"
  WRONG: "Deliberately implement incorrectly: not checking playlist existence"
  RIGHT: "Uses print() instead of flash() for error messages"
  RIGHT: "Calls cursor.execute(DELETE FROM Playlist) without SELECT COUNT check first"

CRITICAL for good patterns: Describe the SPECIFIC correct code, not a general category.
  WRONG: "Implements correctly"
  RIGHT: "Uses csv.DictReader(f, delimiter='\\t') and accesses row['PlaylistId']"

Output ONLY valid JSON. No markdown fences."""

        prompt = f"""Decompose this grading rubric pattern into 2-4 concrete, DETECTABLE code-level variants.

Criterion: {criterion_id}
Type: {ptype_label}
Pattern description: "{description}"

Each variant must have:
- id: "A", "B", "C", "D" (letter)
- instruction: A CONCRETE, DETECTABLE code instruction.
  MUST describe what code IS present (not what's missing).
  Use actual function names, table/column names, API calls from the assignment.
  For bad patterns: describe the EXACT wrong code (e.g., "uses f-string in cursor.execute()" not "writes insecure SQL").
- check_regex: A Python regex that can verify this variant was implemented.
  Use actual table/column/function names (not \\w+). Must NOT be ".*".
- difficulty: "basic" (common pattern), "common" (less common), or "edge" (rare/unusual)

Example for bad pattern "Uses string formatting for SQL queries":
[
  {{"id": "A", "instruction": "Use f-string: cursor.execute(f\\"INSERT INTO Playlist VALUES ({{name}})\\")",
    "check_regex": "f[\\"'].*\\\\bINSERT\\\\b", "difficulty": "basic"}},
  {{"id": "B", "instruction": "Use % formatting: cursor.execute(\\"INSERT INTO Playlist VALUES (%s)\\" % (name,))",
    "check_regex": "%.*\\\\(" , "difficulty": "basic"}},
  {{"id": "C", "instruction": "Use .format(): cursor.execute(\\"INSERT INTO Playlist VALUES ({{}})\\".format(name))",
    "check_regex": "\\.format\\s*\\(", "difficulty": "basic"}}
]

IMPORTANT:
- EVERY instruction MUST be a concrete, positive statement about what code EXISTS
- NEVER use "not", "without", "incorrectly", "deliberately", "missing" in the instruction
- check_regex must be specific (not ".*")
- Use ACTUAL function/table/column names from this assignment

Output: {{"variants": [...]}}"""

        resp = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{'role': 'system', 'content': system},
                      {'role': 'user', 'content': prompt}],
            max_tokens=2048, temperature=0.3,
        )
        text = resp.choices[0].message.content.strip()
        if text.startswith('```'):
            text = re.sub(r'^```(?:json)?\s*\n', '', text)
            text = re.sub(r'\n```\s*$', '', text)
        result = json.loads(text)
        variants = result.get('variants', [])

        # Ensure each variant has required fields
        for v in variants:
            v.setdefault('id', chr(65 + variants.index(v)))
            v.setdefault('difficulty', 'basic')

        _VARIANT_CACHE[key] = variants
        _save_cache()
        return variants

    except Exception as e:
        print(f'  [AI variant gen] Failed for "{description[:60]}": {e}')
        if ptype == 'good':
            return [{'id': 'A', 'instruction': f'Write code that does: {description}',
                     'check_regex': r'.*', 'difficulty': 'basic'}]
        else:
            return [{'id': 'A', 'instruction': f'Write code containing: {description}',
                     'check_regex': r'.*', 'difficulty': 'basic'}]


def build_pattern_matrix(rubric_path: str, variant_cache_path: str = '') -> dict:
    """Build pattern matrix from rubric cache.

    Args:
        rubric_path: Path to rubric_cache.json
        variant_cache_path: Path to AI variant cache JSON (enables AI fallback)
    """
    if variant_cache_path:
        _load_cache(variant_cache_path)

    with open(rubric_path) as f:
        rubric = json.load(f)

    matrix = {
        'criteria': {},
        'all_patterns': [],      # flat list of all pattern IDs
        'pattern_variants': {},  # pattern_id -> [variant_dicts]
    }

    for task in rubric['tasks']:
        for rc in task['rubric_criteria']:
            rid = rc['id']
            gps = rc.get('good_patterns', [])
            bps = rc.get('bad_patterns', [])

            patterns = []
            # Good patterns
            for i, gp in enumerate(gps):
                pid = f"{rid}_G{i+1}"
                variants = _define_variants(rid, 'good', i, gp)
                patterns.append({'id': pid, 'type': 'good', 'description': gp, 'variants': variants})
                matrix['pattern_variants'][pid] = variants
                matrix['all_patterns'].append(pid)

            # Bad patterns
            for i, bp in enumerate(bps):
                pid = f"{rid}_B{i+1}"
                variants = _define_variants(rid, 'bad', i, bp)
                patterns.append({'id': pid, 'type': 'bad', 'description': bp, 'variants': variants})
                matrix['pattern_variants'][pid] = variants
                matrix['all_patterns'].append(pid)

            matrix['criteria'][rid] = {
                'name': rc['name'],
                'marks': rc.get('marks', 1),
                'patterns': patterns,
            }

    return matrix


def _define_variants(criterion_id: str, ptype: str, index: int, description: str) -> list[dict]:
    """Define concrete implementation variants for a pattern.

    Each variant has:
      - id: unique variant identifier
      - instruction: concrete, specific instruction for the AI
      - check_regex: regex to verify this variant was generated
      - difficulty: 'basic' | 'common' | 'edge'
    """
    desc_lower = description.lower()

    # ── RQ1_1: File Reading ──
    if 'csv.reader' in desc_lower or 'dictreader' in desc_lower:
        return [
            {'id': 'A', 'instruction': 'Use csv.DictReader(f, delimiter="\\t") and access columns by name: row["PlaylistId"], row["TrackId"]',
             'check_regex': r'csv\.DictReader.*delimiter\s*=\s*["\']\\t["\']', 'difficulty': 'basic'},
            {'id': 'B', 'instruction': 'Use csv.reader(f, delimiter="\\t") with next(reader) to skip header, then access by index: row[0], row[1]',
             'check_regex': r'csv\.reader.*delimiter\s*=\s*["\']\\t["\']', 'difficulty': 'basic'},
        ]
    if 'delimiter' in desc_lower and 'tab' in desc_lower or '\\t' in description:
        return [
            {'id': 'A', 'instruction': 'Specify delimiter="\\t" (tab character) when creating the csv reader',
             'check_regex': r'delimiter\s*=\s*["\']\\t["\']', 'difficulty': 'basic'},
        ]
    if 'iterates' in desc_lower or 'extracting' in desc_lower:
        return [
            {'id': 'A', 'instruction': 'Use a for loop: for row in reader: playlist_id = row["PlaylistId"]; track_id = row["TrackId"]',
             'check_regex': r'for\s+row\s+in\s+reader', 'difficulty': 'basic'},
        ]

    # ── Bad: manual parsing without csv ──
    if 'manually parsing' in desc_lower or 'without using csv' in desc_lower:
        return [
            {'id': 'A', 'instruction': 'Read the file with f = open("PlaylistTracks.tsv", "r") and then lines = f.readlines(). For each line, call line.split("\\t") to get the fields.',
             'check_regex': r'\.split\s*\(\s*["\']\\t["\']\s*\)', 'difficulty': 'basic'},
            {'id': 'B', 'instruction': 'Use line.split("\\t") to parse each line of the TSV file. Do not include any import csv statement in the code.',
             'check_regex': r'\.split\s*\(.*\\t', 'difficulty': 'basic'},
        ]
    if 'pandas' in desc_lower or 'disallowed' in desc_lower:
        return [
            {'id': 'A', 'instruction': 'Import pandas as pd and use pd.read_csv() to read the TSV file.',
             'check_regex': r'import\s+pandas|from\s+pandas|pd\.read_csv', 'difficulty': 'edge'},
        ]
    if 'hardcoding' in desc_lower or 'assuming' in desc_lower:
        return [
            {'id': 'A', 'instruction': 'Write the file path as the string literal "PlaylistTracks.tsv" directly in the open() call. Do not reference the playlist_tracks_file parameter.',
             'check_regex': r'["\']PlaylistTracks\.tsv["\']', 'difficulty': 'basic'},
        ]

    # ── RQ1_2: Database Connection ──
    if 'sqlite3.connect' in desc_lower or 'connect()' in desc_lower:
        return [
            {'id': 'A', 'instruction': 'Use conn = sqlite3.connect(str(DB_FILE)) to connect',
             'check_regex': r'sqlite3\.connect\s*\(\s*str\s*\(\s*DB_FILE', 'difficulty': 'basic'},
            {'id': 'B', 'instruction': 'Write conn = sqlite3.connect(DB_FILE) where DB_FILE is a variable holding the database filename. Do not wrap DB_FILE with str().',
             'check_regex': r'sqlite3\.connect\s*\(\s*DB_FILE', 'difficulty': 'basic'},
        ]
    if 'cursor' in desc_lower or 'creates a cursor' in desc_lower:
        return [
            {'id': 'A', 'instruction': 'Create cursor: cursor = conn.cursor()',
             'check_regex': r'\.cursor\s*\(', 'difficulty': 'basic'},
        ]
    if 'closes connection' in desc_lower or 'context manager' in desc_lower:
        return [
            {'id': 'A', 'instruction': 'Call conn.close() after all database operations are complete',
             'check_regex': r'\.close\s*\(', 'difficulty': 'basic'},
            {'id': 'B', 'instruction': 'Use "with sqlite3.connect(...) as conn:" context manager',
             'check_regex': r'with\s+sqlite3\.connect', 'difficulty': 'basic'},
        ]

    # ── Bad: using other libraries ──
    if 'sqlalchemy' in desc_lower or 'other database' in desc_lower:
        return [
            {'id': 'A', 'instruction': 'Import sqlalchemy and use create_engine() to connect',
             'check_regex': r'from\s+sqlalchemy|import\s+sqlalchemy', 'difficulty': 'edge'},
        ]
    if 'hardcoding database' in desc_lower:
        return [
            {'id': 'A', 'instruction': 'Hardcode the database path as "data/iMusic.db" instead of using DB_FILE',
             'check_regex': r'["\']data/iMusic\.db["\']', 'difficulty': 'basic'},
        ]
    if 'not closing' in desc_lower or 'resource leak' in desc_lower:
        return [
            {'id': 'A', 'instruction': 'Open sqlite3 connection with conn = sqlite3.connect(...) and perform cursor operations. The function body ends after the INSERT loop with no conn.close() call present.',
             'check_regex': r'sqlite3\.connect', 'difficulty': 'basic'},
        ]

    # ── RQ1_3: Data Handling ──
    if 'select queries to check' in desc_lower and 'playlistid' in desc_lower:
        return [
            {'id': 'A', 'instruction': 'Use: cursor.execute("SELECT COUNT(*) FROM Playlist WHERE PlaylistId = ?", (pid,)); if cursor.fetchone()[0] == 0: continue',
             'check_regex': r'SELECT\s+COUNT\(\*\)\s+FROM\s+Playlist', 'difficulty': 'basic'},
            {'id': 'B', 'instruction': 'Execute cursor.execute("SELECT PlaylistId FROM Playlist WHERE PlaylistId = ?", (pid,)) and then check if cursor.fetchone() returns None. If it returns None, use continue to skip that row.',
             'check_regex': r'SELECT\s+PlaylistId\s+FROM\s+Playlist\s+WHERE\s+PlaylistId', 'difficulty': 'basic'},
        ]
    if 'select queries to check' in desc_lower and 'trackid' in desc_lower:
        return [
            {'id': 'A', 'instruction': 'Use: cursor.execute("SELECT COUNT(*) FROM Track WHERE TrackId = ?", (tid,)); if cursor.fetchone()[0] == 0: continue',
             'check_regex': r'SELECT\s+COUNT\(\*\)\s+FROM\s+Track', 'difficulty': 'basic'},
        ]
    if 'only inserts if both exist' in desc_lower or 'skips rows' in desc_lower:
        return [
            {'id': 'A', 'instruction': 'Check both IDs exist, use "continue" to skip invalid rows, only INSERT when both valid',
             'check_regex': r'continue', 'difficulty': 'basic'},
        ]
    if 'parameterized queries' in desc_lower and 'prevent sql injection' in desc_lower:
        return [
            {'id': 'A', 'instruction': 'Use ? placeholders: cursor.execute("SELECT ... WHERE id = ?", (val,))',
             'check_regex': r'\.execute\s*\(\s*["\'].*\?\s*["\']\s*,\s*\(', 'difficulty': 'basic'},
        ]

    # ── Bad: no validation ──
    if 'inserting without validation' in desc_lower:
        return [
            {'id': 'A', 'instruction': 'For each row parsed from the TSV, execute a bare INSERT INTO PlaylistTrack (PlaylistId, TrackId) VALUES (?, ?) statement directly. The function contains no SELECT queries on the Playlist or Track tables before the INSERT.',
             'check_regex': r'INSERT\s+INTO\s+PlaylistTrack', 'difficulty': 'basic'},
        ]
    if 'assuming existence without checking' in desc_lower:
        return [
            {'id': 'A', 'instruction': 'Convert each field from the TSV row to an integer using int(field) directly. Do not wrap the conversion in a try/except block.',
             'check_regex': r'int\s*\(\s*row\[', 'difficulty': 'basic'},
        ]
    if 'string formatting for sql' in desc_lower:
        return [
            {'id': 'A', 'instruction': 'Use f-string for one INSERT query: cursor.execute(f"INSERT INTO PlaylistTrack VALUES ({pid}, {tid})")',
             'check_regex': r'f["\'].*\bINSERT\b', 'difficulty': 'basic'},
            {'id': 'B', 'instruction': 'Use string concatenation: cursor.execute("INSERT INTO PlaylistTrack VALUES (" + str(pid) + ", " + str(tid) + ")")',
             'check_regex': r'["\'].*["\']\s*\+\s*str\(', 'difficulty': 'basic'},
            {'id': 'C', 'instruction': 'Use % formatting: cursor.execute("INSERT INTO PlaylistTrack VALUES (%s, %s)" % (pid, tid))',
             'check_regex': r'%\s*\(', 'difficulty': 'basic'},
        ]

    # ── RQ1_4: Accurate Update ──
    # NOTE: "checks for existing combination" and avoidance patterns (INSERT OR IGNORE,
    # SELECT COUNT before INSERT) are GOOD patterns — they're correct ways to avoid
    # duplicates. They should NOT be in the bad pattern section.
    # The actual BAD pattern is "inserting without any duplicate check at all."
    if 'inserting duplicate' in desc_lower:
        return [
            {'id': 'A', 'instruction': 'For each row from the TSV, execute a plain INSERT INTO PlaylistTrack (PlaylistId, TrackId) VALUES (?, ?). The function contains only bare INSERT statements with no SELECT COUNT, INSERT OR IGNORE, or NOT IN clauses referencing PlaylistTrack.',
             'check_regex': r'INSERT\s+INTO\s+PlaylistTrack', 'difficulty': 'basic'},
        ]
    # "checks for existing combination" is a good pattern description — skip it here
    if 'checks for existing combination' in desc_lower or 'duplicate' in desc_lower:
        return [
            {'id': 'A', 'instruction': 'Use INSERT OR IGNORE to avoid duplicates (alternative approach)',
             'check_regex': r'INSERT\s+OR\s+IGNORE', 'difficulty': 'basic'},
            {'id': 'B', 'instruction': 'SELECT COUNT(*) FROM PlaylistTrack WHERE PlaylistId=? AND TrackId=? first, only INSERT if count==0',
             'check_regex': r'SELECT.*FROM\s+PlaylistTrack.*WHERE.*PlaylistId.*AND.*TrackId', 'difficulty': 'basic'},
        ]
    if 'inserts only new' in desc_lower or 'valid combinations' in desc_lower:
        return [
            {'id': 'A', 'instruction': 'Before inserting a row into PlaylistTrack, first run a SELECT PlaylistId FROM Playlist WHERE PlaylistId = ? to verify the PlaylistId exists, and a SELECT TrackId FROM Track WHERE TrackId = ? to verify the TrackId exists. Then run a SELECT COUNT(*) FROM PlaylistTrack WHERE PlaylistId = ? AND TrackId = ? to ensure the combination is not already present. Only execute the INSERT if all three checks pass.',
             'check_regex': r'INSERT\s+INTO\s+PlaylistTrack', 'difficulty': 'basic'},
        ]
    if 'commits transaction' in desc_lower:
        return [
            {'id': 'A', 'instruction': 'Call conn.commit() after all inserts are done',
             'check_regex': r'\.commit\s*\(', 'difficulty': 'basic'},
        ]
    if 'not committing' in desc_lower:
        return [
            {'id': 'A', 'instruction': 'Execute multiple INSERT INTO PlaylistTrack statements successfully, but the conn.commit() line after the loop is deleted/commented out, so changes are lost',
             'check_regex': r'INSERT\s+INTO\s+PlaylistTrack', 'difficulty': 'basic'},
        ]

    # ── RQ2_1: Genre Retrieval ──
    if 'queries genre' in desc_lower and 'ascending' in desc_lower:
        return [
            {'id': 'A', 'instruction': 'cursor.execute("SELECT GenreId, Name FROM Genre ORDER BY Name ASC")',
             'check_regex': r'SELECT.*FROM\s+Genre.*ORDER\s+BY.*Name.*ASC', 'difficulty': 'basic'},
        ]
    if 'prepends' in desc_lower and 'all' in desc_lower and 'genreid=0' in desc_lower:
        return [
            {'id': 'A', 'instruction': 'genres.insert(0, {"GenreId": 0, "Name": "All"})',
             'check_regex': r'insert\s*\(\s*0\s*,\s*\{.*GenreId.*0', 'difficulty': 'basic'},
        ]
    if 'returns data' in desc_lower and 'format' in desc_lower:
        return [
            {'id': 'A', 'instruction': 'Return list of dicts: [{"GenreId": row[0], "Name": row[1]} for row in cursor.fetchall()]',
             'check_regex': r'GenreId.*Name|GenreId["\']\s*:\s*row', 'difficulty': 'basic'},
        ]
    if 'not including' in desc_lower and 'all' in desc_lower:
        return [
            {'id': 'A', 'instruction': 'Query the Genre table with SELECT GenreId, Name FROM Genre ORDER BY Name ASC. Store the results in a list of dictionaries. Return that list directly — the function contains no .insert(0, ...) call.',
             'check_regex': r'SELECT.*FROM\s+Genre', 'difficulty': 'basic'},
        ]
    if 'incorrect sort' in desc_lower:
        return [
            {'id': 'A', 'instruction': 'Use ORDER BY Name DESC instead of ASC.',
             'check_regex': r'ORDER\s+BY.*DESC', 'difficulty': 'basic'},
        ]

    # ── RQ2_2: Statistics ──
    if 'number of tracks' in desc_lower or 'duration' in desc_lower or 'total cost' in desc_lower:
        return [
            {'id': 'A', 'instruction': 'Calculate: COUNT(pt.TrackId) AS NumberOfTracks, SUM(t.Milliseconds)/60000.0 AS Duration, SUM(t.UnitPrice) AS TotalCost, AVG/CASE for AverageCost',
             'check_regex': r'COUNT\s*\(.*TrackId\).*SUM\s*\(.*Milliseconds\).*SUM\s*\(.*UnitPrice\)', 'difficulty': 'basic'},
        ]
    if 'coalesce' in desc_lower or 'ifnull' in desc_lower:
        return [
            {'id': 'A', 'instruction': 'Use COALESCE(SUM(t.Milliseconds), 0) to return 0 instead of NULL',
             'check_regex': r'COALESCE\s*\(', 'difficulty': 'basic'},
        ]
    if 'handles' in desc_lower and 'all' in desc_lower and 'genre' in desc_lower:
        return [
            {'id': 'A', 'instruction': 'Write code that checks if genre_id equals 0 or the string "All". If true, build a SQL query without a WHERE clause. Otherwise, append a WHERE clause that filters by genre_id.',
             'check_regex': r'if\s+.*genre.*==\s*0|if\s+.*genre.*==\s*["\']All["\']', 'difficulty': 'basic'},
        ]
    if 'integer division' in desc_lower:
        return [
            {'id': 'A', 'instruction': 'Calculate Duration as SUM(t.Milliseconds) / 60000 (integer division, no .0)',
             'check_regex': r'/\s*60000\b(?!\.)', 'difficulty': 'basic'},
        ]
    if 'not handling null' in desc_lower or 'returns none' in desc_lower:
        return [
            {'id': 'A', 'instruction': 'Use SUM(t.Milliseconds) and SUM(t.UnitPrice) directly in the SELECT clause. The function contains no COALESCE() or IFNULL() wrapper around the SUM calls, so the result shows NULL for playlists with no tracks.',
             'check_regex': r'SUM\s*\(', 'difficulty': 'basic'},
        ]

    # ── RQ2_3: Sort whitelist ──
    if 'whitelist validation' in desc_lower:
        return [
            {'id': 'A', 'instruction': 'Create ALLOWED_SORT = {"Name": "p.Name", ...} dict and use ALLOWED_SORT.get(sort_by, default)',
             'check_regex': r'(?:ALLOWED|SORT).*(?::|=)\s*\{', 'difficulty': 'basic'},
            {'id': 'B', 'instruction': 'Write code that checks if sort_by is not in the list ["Name", "NumberOfTracks"]. If so, set sort_by to "Name".',
             'check_regex': r'if\s+\w+\s+not\s+in\s+\(', 'difficulty': 'basic'},
        ]
    if 'directly interpolating' in desc_lower:
        return [
            {'id': 'A', 'instruction': 'Build the ORDER BY clause using an f-string: query += f" ORDER BY {sort_column} {sort_order}". The sort_column variable goes directly into the f-string with no preceding if-statement, dictionary lookup, or list membership check.',
             'check_regex': r'f["\'].*ORDER\s+BY\s*\{', 'difficulty': 'basic'},
        ]
    if 'not validating' in desc_lower:
        return [
            {'id': 'A', 'instruction': 'Pass sort_by and sort_order directly into the ORDER BY clause of the SQL query string as bare variables with no allowlist check or sanitization step.',
             'check_regex': r'ORDER\s+BY\s*\{', 'difficulty': 'basic'},
        ]

    # ── RQ2_4: Input Validation ──
    if 'validates genreid' in desc_lower or 'checks for 0' in desc_lower:
        return [
            {'id': 'A', 'instruction': 'Write code that checks if genre_id is not 0 and not the string "All". If true, query the Genre table to verify the genre exists. If the genre is not found, flash the message "Invalid genre ID" with category "danger".',
             'check_regex': r'Invalid\s+genre', 'difficulty': 'basic'},
        ]
    if 'flash message' in desc_lower and 'invalid genre' in desc_lower:
        return [
            {'id': 'A', 'instruction': 'Call flash("Invalid genre ID", "danger") when genre_id is not 0 or "All" and the genre is not found in the database.',
             'check_regex': r'flash\(.*Invalid\s+genre.*danger', 'difficulty': 'basic'},
        ]
    if 'validates sortby' in desc_lower or 'sortorder' in desc_lower:
        return [
            {'id': 'A', 'instruction': 'Check if SortBy in allowed list AND SortOrder in ("ASC", "DESC"). Flash "Invalid sorting parameters" danger if not.',
             'check_regex': r'Invalid\s+sorting', 'difficulty': 'basic'},
        ]
    if 'not validating genre' in desc_lower:
        return [
            {'id': 'A', 'instruction': 'Read GenreId from request.form and pass it directly to get_statistics(). The function contains no SELECT query on the Genre table before this call to verify the genre exists.',
             'check_regex': r'request\.form\.get\s*\(.*GenreId', 'difficulty': 'basic'},
        ]
    if 'using string formatting' in desc_lower:
        return [
            {'id': 'A', 'instruction': 'Use f-strings to build SQL queries in get_statistics()',
             'check_regex': r'f["\'].*\bSELECT\b', 'difficulty': 'basic'},
        ]

    # ── RQ3_1: Playlist/Genre Display ──
    if 'get_all_playlists queries' in desc_lower or 'playlist table' in desc_lower:
        return [
            {'id': 'A', 'instruction': 'cursor.execute("SELECT PlaylistId, Name FROM Playlist ORDER BY Name ASC")',
             'check_regex': r'SELECT.*FROM\s+Playlist.*ORDER\s+BY.*Name', 'difficulty': 'basic'},
        ]
    if 'passes both datasets' in desc_lower or 'render_template' in desc_lower:
        return [
            {'id': 'A', 'instruction': 'return render_template("playlists.html", playlists=playlists, genres=genres)',
             'check_regex': r'render_template.*playlists.*playlists\s*=\s*\w+.*genres\s*=\s*\w+', 'difficulty': 'basic'},
        ]
    if 'incorrect sort' in desc_lower:
        return [
            {'id': 'A', 'instruction': 'Execute the query SELECT PlaylistId, Name FROM Playlist (no ORDER BY clause). The playlists appear in database-default order rather than alphabetical.',
             'check_regex': r'SELECT.*FROM\s+Playlist', 'difficulty': 'basic'},
        ]

    # ── RQ3_2: Playlist Creation ──
    if 'validates that playlist name' in desc_lower:
        return [
            {'id': 'A', 'instruction': 'Write code that checks if name is falsy or if name.strip() is empty. If so, flash the message "Playlist name cannot be empty" with category "danger" and then return redirect(url_for("playlists")).',
             'check_regex': r'if\s+not\s+.*name.*strip|Playlist\s+name\s+cannot\s+be\s+empty', 'difficulty': 'basic'},
        ]
    if 'inserts new playlist' in desc_lower:
        return [
            {'id': 'A', 'instruction': 'cursor.execute("INSERT INTO Playlist (Name) VALUES (?)", (name,))',
             'check_regex': r'INSERT\s+INTO\s+Playlist.*VALUES\s*\(.*\?', 'difficulty': 'basic'},
        ]
    if 'flash message' in desc_lower and 'created successfully' in desc_lower:
        return [
            {'id': 'A', 'instruction': 'flash("Playlist created successfully", "success")',
             'check_regex': r'created\s+successfully.*success', 'difficulty': 'basic'},
        ]

    # ── RQ3_3: Rename/Delete ──
    if 'validates playlist exists' in desc_lower and 'rename' in desc_lower:
        return [
            {'id': 'A', 'instruction': 'Execute cursor.execute("SELECT COUNT(*) FROM Playlist WHERE PlaylistId = ?", (pid,)) and check if result[0] equals 0; if so, call flash("Playlist not found") and redirect to the playlists list page.',
             'check_regex': r'SELECT.*FROM\s+Playlist.*WHERE.*PlaylistId.*rename|Playlist\s+not\s+found', 'difficulty': 'basic'},
        ]
    if 'deletes associated records' in desc_lower or 'playlisttrack' in desc_lower:
        return [
            {'id': 'A', 'instruction': 'First: cursor.execute("DELETE FROM PlaylistTrack WHERE PlaylistId = ?", (pid,)). Then: cursor.execute("DELETE FROM Playlist WHERE PlaylistId = ?", (pid,))',
             'check_regex': r'DELETE\s+FROM\s+PlaylistTrack.*DELETE\s+FROM\s+Playlist', 'difficulty': 'basic'},
        ]

    # ── RQ3_4: Genre Track Management ──
    if 'validates playlist and genre exist' in desc_lower:
        return [
            {'id': 'A', 'instruction': 'Check SELECT COUNT(*) FROM Playlist WHERE... and SELECT COUNT(*) FROM Genre WHERE... before operating',
             'check_regex': r'SELECT.*FROM\s+Playlist.*SELECT.*FROM\s+Genre|SELECT.*FROM\s+Genre.*SELECT.*FROM\s+Playlist', 'difficulty': 'basic'},
        ]
    if 'inserts only tracks not already' in desc_lower or 'avoids duplicates' in desc_lower:
        return [
            {'id': 'A', 'instruction': 'Write the SQL query with a WHERE clause that includes TrackId NOT IN (SELECT TrackId FROM PlaylistTrack WHERE PlaylistId = ?) to filter out tracks already in the playlist.',
             'check_regex': r'NOT\s+IN\s*\(.*SELECT.*TrackId.*FROM\s+PlaylistTrack', 'difficulty': 'basic'},
        ]

    # ── RQ3_5: Error Handling ──
    if 'uses flash()' in desc_lower:
        return [
            {'id': 'A', 'instruction': 'Use flash("message", "success") or flash("message", "danger") for every operation',
             'check_regex': r'flash\s*\([^,]+,\s*["\']success|flash\s*\([^,]+,\s*["\']danger', 'difficulty': 'basic'},
        ]
    if 'redirects to' in desc_lower and 'playlists' in desc_lower:
        return [
            {'id': 'A', 'instruction': 'return redirect(url_for("playlists")) after every POST operation',
             'check_regex': r'redirect\(url_for\(["\']playlists["\']\)', 'difficulty': 'basic'},
        ]

    # ── RQ3_6: Security ──
    if 'parameterized queries' in desc_lower:
        return [
            {'id': 'A', 'instruction': 'Use ? placeholders for ALL SQL in Task 3 functions',
             'check_regex': r'\.execute\s*\(\s*["\'].*\?\s*["\']\s*,\s*\(', 'difficulty': 'basic'},
        ]
    if 'validates playlist id exists' in desc_lower:
        return [
            {'id': 'A', 'instruction': 'Check playlist existence with SELECT before UPDATE/DELETE',
             'check_regex': r'SELECT.*FROM\s+Playlist.*WHERE.*PlaylistId', 'difficulty': 'basic'},
        ]

    # ── Fallback: AI-driven variant generation ──
    # When no hardcoded pattern matches, use AI to decompose the description
    # into concrete code-level variants.
    return _ai_generate_variants(criterion_id, ptype, index, description)


# ── Coverage target: min number of students per variant ──
MIN_COVERAGE_PER_VARIANT = 2  # each variant should appear in at least 2 students
