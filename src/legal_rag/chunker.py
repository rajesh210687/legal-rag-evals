"""Clause-aware chunker for contract text.

Boundaries are found from numbering (1., 1.1, A.) validated by sequence, so
cross-references like "Section 3.5" are not mistaken for headings. Every chunk
carries a context header (title, parties, section) and stays under MAX_TOKENS
as counted by the bge tokenizer.
"""
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache

TOKENIZER_NAME = "BAAI/bge-small-en-v1.5"
TARGET_TOKENS = 400
MAX_TOKENS = 500  # bge truncates at 512 including [CLS]/[SEP]
MIN_PARA_TOKENS = 60  # shorter paragraphs are merged with their neighbour

# A number/letter preceded by one of these words is a cross-reference, not a heading.
XREF_WORDS = {
    "section", "sections", "article", "articles", "clause", "clauses", "point",
    "topic", "paragraph", "exhibit", "attachment", "schedule", "and", "or", "to",
    "of", "in", "under", "with", "by", "see", "per", "than", "through", "from",
    "at", "no", "pursuant", "ss",
}
SMALL_WORDS = r"and|of|the|for|to|or|in|on|with|by|a|an"
CAPS_WORD = r"[A-Z][A-Z0-9,'’&/\-()]*"
CAPS_RUN = re.compile(rf"{CAPS_WORD}(?: (?:{CAPS_WORD}|&))*")
TITLE_WORD = rf"(?:[A-Z][\w\-’'/&,]*|{SMALL_WORDS})"
TITLE_RUN = re.compile(rf"{TITLE_WORD}(?: {TITLE_WORD})*")
CAPTION = re.compile(rf"({TITLE_WORD}(?: {TITLE_WORD}){{0,7}})\.(?=\s|$)")
DEFINED_TERM = re.compile(r"[“\"]([^”\"]{1,40})[”\"]")
CAND = re.compile(
    r"(?<![\w.§(])(?:(?P<top>\d{1,2})\.(?=\s)"
    r"|(?P<n>\d{1,2})\.(?P<m>\d{1,3})\.?(?=\s)"
    r"|(?P<let>[A-H])\.(?=\s+[A-Z]))"
)
# A heading with no period after its number, alone on its own line: "7 Confidentiality".
TOP_NODOT = re.compile(r"(?m)^(?P<n>\d{1,2})[ \t]+(?P<h>[A-Z][^\n]{2,80})$")
BACK_MATTER = re.compile(
    r"IN WITNESS WHEREOF|(?:ATTACHMENT|EXHIBIT|SCHEDULE|ANNEX) [A-Z0-9]+\b"
    r"|(?m:^(?:Attachment|Exhibit|Schedule|Annex) [A-Z0-9]+\b)"
)
SENTENCE_END = re.compile(r"(?<=[.;:?!])\s+(?=[A-Z“\"(\[])")
LETTER_MARK = re.compile(r"(?<![\w(])\(([a-z])\)(?=\s)")


@lru_cache(maxsize=1)
def _tokenizer():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    tok.model_max_length = 10**9  # we enforce the real limit ourselves
    return tok


def n_tokens(text: str) -> int:
    return len(_tokenizer().encode(text, add_special_tokens=True))


@dataclass
class Chunk:
    chunk_id: str
    contract_id: str
    cuad_file: str
    clause_id: str
    clause_ids: list = field(default_factory=list)  # clause plus lettered sub-clauses inside
    section_heading: str = ""
    part: int = 1
    n_parts: int = 1
    token_count: int = 0
    char_start: int = 0
    char_end: int = 0
    body: str = ""
    text: str = ""  # header + body: this is what gets embedded

    def to_dict(self):
        return asdict(self)


def normalize(raw: str) -> str:
    """Drop page furniture, collapse spaces; keep line breaks as \\n and blank lines as \\n\\n."""
    t = raw.replace("\r", "")
    t = re.sub(r"(?<!\w)-\s?\d{1,3}\s?-(?!\w)", " ", t)  # page markers like -17-
    t = re.sub(r"INGRAM[ \t]+NETGEAR[ \t]+Distributor[ \t]+Agreement[ \t]+\d+(?:[ \t]+\d{4,6})?", " ", t, flags=re.I)
    t = re.sub(r"Source: GPAQ ACQUISITION HOLDINGS, INC\., S-4/A, 1/23/2020", " ", t)  # EDGAR footer
    t = re.sub(r"(?m)^[ \t]*\d{1,3}[ \t]*$", "", t)  # lines that are only a page number
    t = re.sub(r"[ \t\f\v]+", " ", t)
    t = re.sub(r"[ ]*\n[ ]*(?:\n[ ]*)+", "\n\n", t)  # blank lines -> one paragraph break
    t = re.sub(r"[ ]*\n[ ]*", "\n", t)
    return t.strip()


def _prev_ok(text: str, pos: int) -> bool:
    m = re.search(r"([A-Za-z]+)[\s]*$", text[max(0, pos - 24):pos])
    return not (m and m.group(1).lower() in XREF_WORDS)


def _top_heading(text: str, pos: int) -> str:
    rest = text[pos:].lstrip(" ")
    line = rest.split("\n", 1)[0]
    # heading on a line of its own (e.g. "14. Governing Law")
    if len(line) <= 90 and len(line.split()) <= 10 and not re.search(r"\.\s", line):
        return line.strip(" .:")
    m = CAPS_RUN.match(rest)
    if m and len(m.group(0)) >= 3:
        words = m.group(0).split(" ")
        if len(words) > 1 and words[-1] in {"A", "I"}:
            words = words[:-1]
        return " ".join(words)
    m = TITLE_RUN.match(rest)
    return m.group(0) if m else ""


def _caption(text: str, pos: int, parent: str) -> str:
    rest = text[pos:].lstrip(" ")
    m = DEFINED_TERM.match(rest)
    if m:
        return m.group(1)
    m = CAPTION.match(rest.split("\n", 1)[0])
    return m.group(1) if m else parent


def find_boundaries(text: str):
    """Return [(pos, clause_id, heading, level)] in document order."""
    out, top, sub, letter, heading_top = [], 0, 0, -1, ""
    found = sorted(list(CAND.finditer(text)) + list(TOP_NODOT.finditer(text)), key=lambda m: m.start())
    for m in found:
        if m.re is TOP_NODOT:
            head = m.group("h").strip()
            if int(m.group("n")) == top + 1 and len(head.split()) <= 8 and not re.search(r"[.;:,]$", head):
                top, sub, letter, heading_top = top + 1, 0, -1, head
                out.append((m.start(), str(top), head, 1))
            continue
        if not _prev_ok(text, m.start()):
            continue
        nxt = text[m.end():].lstrip()[:1]
        if m.group("top"):
            n = int(m.group("top"))
            if n != top + 1 or not (nxt.isupper() or nxt in "“\""):
                continue
            heading_top = _top_heading(text, m.end())
            top, sub, letter = n, 0, -1
            out.append((m.start(), str(n), heading_top, 1))
        elif m.group("n"):
            n, k = int(m.group("n")), int(m.group("m"))
            if not (nxt.isupper() or nxt in "“\"(*["):
                continue
            if n == top and sub < k <= sub + 3:
                sub = k
            elif n == top + 1 and k == 1:
                top, sub, letter, heading_top = n, 1, -1, ""
            else:
                continue
            out.append((m.start(), f"{n}.{k}", _caption(text, m.end(), heading_top), 2))
        else:
            idx = ord(m.group("let")) - 65
            at_line_start = m.start() == 0 or text[m.start() - 1] == "\n"
            if top and sub == 0 and (idx == letter + 1 or (at_line_start and letter < idx <= letter + 2)):
                letter = idx
                cap = _caption(text, m.end(), heading_top)
                out.append((m.start(), f"{top}.{m.group('let')}", cap, 2))
    return out


def _letter_marks(body: str):
    """Strict (a), (b), (c)... sequence inside a clause -> [(pos, letter)]."""
    marks, want = [], 0
    for m in LETTER_MARK.finditer(body):
        if ord(m.group(1)) - 97 == want:
            marks.append((m.start(), m.group(1)))
            want += 1
    return marks


def _segments(text: str):
    """Split text into (clause_id, heading, start, end) incl. preamble and back matter."""
    bounds = find_boundaries(text)
    if not bounds:
        return [("preamble", "Full text (no structure found)", 0, len(text))]
    segs = []
    if bounds[0][0] > 0:
        segs.append(("preamble", "Preamble and recitals", 0, bounds[0][0]))
    last_pos = bounds[-1][0]
    back = [m for m in BACK_MATTER.finditer(text, last_pos)]
    end_of_body = back[0].start() if back else len(text)
    for i, (pos, cid, head, _lvl) in enumerate(bounds):
        end = bounds[i + 1][0] if i + 1 < len(bounds) else end_of_body
        segs.append((cid, head, pos, end))
    seen = {}
    for i, m in enumerate(back):
        end = back[i + 1].start() if i + 1 < len(back) else len(text)
        label = m.group(0).title()
        seen[label] = seen.get(label, 0) + 1
        cid = f"back:{label}" + (f"-{seen[label]}" if seen[label] > 1 else "")
        segs.append((cid, label, m.start(), end))
    return segs


def _word_windows(text: str, budget: int):
    out, cur = [], ""
    for w in text.split(" "):
        cand = f"{cur} {w}" if cur else w
        if cur and n_tokens(cand) > budget:
            out.append(cur)
            cur = w
        else:
            cur = cand
    if cur:
        out.append(cur)
    return out


def _units(body: str, budget: int):
    """Smallest natural pieces: sub-clauses first, then sentences, then word windows."""
    cuts = [p for p, _ in _letter_marks(body)]
    pieces = [body[a:b] for a, b in zip([0] + cuts, cuts + [len(body)]) if body[a:b].strip()]
    units = []
    for p in pieces:
        if n_tokens(p) <= budget:
            units.append(p.strip())
            continue
        for s in SENTENCE_END.split(p):
            units.extend([s.strip()] if n_tokens(s) <= budget else _word_windows(s, budget))
    return units


def _pack(units, budget: int):
    out, cur = [], ""
    for u in units:
        cand = f"{cur} {u}" if cur else u
        if cur and n_tokens(cand) > budget:
            out.append(cur)
            cur = u
        else:
            cur = cand
    if cur:
        out.append(cur)
    return out


def _paragraphs(seg: str):
    """Blank-line paragraphs. A break that falls mid-sentence (a page break) is not a paragraph."""
    out = []
    for p in seg.split("\n\n"):
        p = p.replace("\n", " ").strip()
        if not p:
            continue
        if out and (not re.search(r"[.;:?!\"”)\]]$", out[-1]) or p[0].islower()):
            out[-1] += " " + p
        else:
            out.append(p)
    return out


def _merge_small(paras):
    """Join paragraphs until each group has MIN_PARA_TOKENS; a short tail joins the previous group."""
    groups, cur = [], ""
    for p in paras:
        cur = f"{cur} {p}" if cur else p
        if n_tokens(cur) >= MIN_PARA_TOKENS:
            groups.append(cur)
            cur = ""
    if cur:
        if groups:
            groups[-1] += " " + cur
        else:
            groups.append(cur)
    return groups


def chunk_contract(raw: str, contract_id: str, cuad_file: str, title: str, parties: str):
    """char_start/char_end are approximate offsets into the normalized text (a paragraph break counts as one space)."""
    text = normalize(raw)
    chunks = []
    for cid, heading, start, end in _segments(text):
        seg = text[start:end]
        body = " ".join(_paragraphs(seg))
        body_start = start + (len(seg) - len(seg.lstrip()))
        if cid != "preamble" and len(body) - len(cid) - len(heading) < 40:
            continue  # a heading with nothing under it; its sub-clauses carry the text
        label = f"{cid} {heading}".strip() if not cid.startswith("back:") else heading
        header = f"Contract: {title} | Parties: {parties} | Section: {label}"
        marks = _letter_marks(body)

        groups = _merge_small(_paragraphs(seg))
        tag = " (10/10)" if len(groups) > 1 else ""
        budget = TARGET_TOKENS - n_tokens(f"{header}{tag}\n")
        cursor, pieces = 0, []
        for grp in groups:
            off = body.find(grp[:40], cursor)
            off = cursor if off < 0 else off
            if n_tokens(f"{header}{tag}\n{grp}") <= MAX_TOKENS:
                pieces.append((off, grp))
            else:
                c2 = off
                for p in _pack(_units(grp, budget), budget):
                    o = body.find(p[:40], c2)
                    o = c2 if o < 0 else o
                    pieces.append((o, p))
                    c2 = o + len(p)
            cursor = off + len(grp)
        n = len(pieces)
        for k, (off, p) in enumerate(pieces, 1):
            part_tag = f" ({k}/{n})" if n > 1 else ""
            txt = f"{header}{part_tag}\n{p}"
            ids = [cid] + [f"{cid}({l})" for pos, l in marks if off <= pos < off + len(p)]
            chunks.append(Chunk(
                chunk_id=f"{contract_id}:{cid}" + (f"#{k}" if n > 1 else ""),
                contract_id=contract_id, cuad_file=cuad_file, clause_id=cid, clause_ids=ids,
                section_heading=heading, part=k, n_parts=n, token_count=n_tokens(txt),
                char_start=body_start + off, char_end=body_start + off + len(p),
                body=p, text=txt,
            ))
    return chunks


def section_matches(expected: str, clause_ids) -> bool:
    """Golden-set section vs chunk ids: equal, or one is an ancestor of the other."""
    for c in clause_ids:
        for a, b in ((expected, c), (c, expected)):
            if a == b or (b.startswith(a) and b[len(a)] in ".("):
                return True
    return False
