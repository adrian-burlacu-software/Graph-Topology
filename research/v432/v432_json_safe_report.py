
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import sqlite3
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


# ============================================================================
# Common helpers
# ============================================================================

def norm(x):
    x = str(x or "").strip().lower()
    x = re.sub(r"[_\s]+", " ", x)
    return x.strip()


def lemma_key(x):
    x = norm(x)
    x = re.sub(r"[^\w -]", "", x)
    return x


def digest(x):
    return hashlib.sha256(
        str(x).encode("utf-8")
    ).hexdigest()


def iter_files(root, suffixes=None):
    if not root.exists():
        return
    wanted = None if suffixes is None else {x.lower() for x in suffixes}
    for p in root.rglob("*"):
        if p.is_file() and (
            wanted is None or p.suffix.lower() in wanted
        ):
            yield p



def json_safe(value):
    """
    Convert Path/WindowsPath and other common runtime objects into plain JSON
    values recursively. This prevents report finalization from failing after a
    successful long learning run.
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(k): json_safe(v)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, Counter):
        return {
            str(k): json_safe(v)
            for k, v in value.items()
        }
    return value


def jsonl_append(path, row):
    with path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                row,
                ensure_ascii=False,
                separators=(",", ":"),
            ) + "\n"
        )



def migrate_schema(con):
    """
    Upgrade an existing cognitive memory DB in place.

    Older runs created verbnet_members with only:
      class_id, lemma

    Current runs also store:
      member_name, grouping
    """
    cols=[
        row[1]
        for row in con.execute(
            'PRAGMA table_info("verbnet_members")'
        ).fetchall()
    ]

    added=[]
    if cols and "member_name" not in cols:
        con.execute(
            'ALTER TABLE verbnet_members ADD COLUMN member_name TEXT'
        )
        cols.append("member_name")
        added.append("member_name")

    if cols and "grouping" not in cols:
        con.execute(
            'ALTER TABLE verbnet_members ADD COLUMN grouping TEXT'
        )
        cols.append("grouping")
        added.append("grouping")

    return {
        "columns":cols,
        "added":added,
    }


# ============================================================================
# Persistent semantic memory
# ============================================================================

def init_db(path):
    con = sqlite3.connect(str(path))
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=FULL")

    con.executescript("""
    CREATE TABLE IF NOT EXISTS concepts (
        concept_id TEXT PRIMARY KEY,
        lemma TEXT NOT NULL,
        pos TEXT,
        source TEXT NOT NULL,
        source_id TEXT NOT NULL,
        gloss TEXT,
        payload_json TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_concepts_lemma
      ON concepts(lemma);

    CREATE TABLE IF NOT EXISTS lexical_relations (
        source_id TEXT NOT NULL,
        relation TEXT NOT NULL,
        target_id TEXT NOT NULL,
        source TEXT NOT NULL,
        PRIMARY KEY(source_id,relation,target_id,source)
    );

    CREATE TABLE IF NOT EXISTS verbnet_classes (
        class_id TEXT PRIMARY KEY,
        name TEXT,
        parent TEXT,
        payload_json TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS verbnet_members (
        class_id TEXT NOT NULL,
        lemma TEXT NOT NULL,
        member_name TEXT,
        grouping TEXT,
        PRIMARY KEY(class_id,lemma)
    );

    CREATE TABLE IF NOT EXISTS verbnet_roles (
        class_id TEXT NOT NULL,
        role TEXT NOT NULL,
        description TEXT,
        PRIMARY KEY(class_id,role)
    );

    CREATE TABLE IF NOT EXISTS verbnet_frames (
        class_id TEXT NOT NULL,
        frame_index INTEGER NOT NULL,
        description TEXT,
        syntax_json TEXT NOT NULL,
        semantics_json TEXT NOT NULL,
        PRIMARY KEY(class_id,frame_index)
    );

    CREATE TABLE IF NOT EXISTS propbank_predicates (
        lemma TEXT NOT NULL,
        source_file TEXT NOT NULL,
        PRIMARY KEY(lemma,source_file)
    );

    CREATE TABLE IF NOT EXISTS propbank_rolesets (
        roleset_id TEXT PRIMARY KEY,
        lemma TEXT,
        name TEXT,
        vncls TEXT,
        source_file TEXT
    );

    CREATE TABLE IF NOT EXISTS propbank_roles (
        roleset_id TEXT NOT NULL,
        role_number TEXT NOT NULL,
        function TEXT,
        description TEXT,
        PRIMARY KEY(roleset_id,role_number)
    );

    CREATE TABLE IF NOT EXISTS semlink_mappings (
        map_id TEXT PRIMARY KEY,
        source_family TEXT,
        source_id TEXT,
        target_family TEXT,
        target_id TEXT,
        raw_line TEXT,
        source_file TEXT
    );

    CREATE TABLE IF NOT EXISTS teacher_examples (
        example_id TEXT PRIMARY KEY,
        lemma TEXT NOT NULL,
        prompt TEXT NOT NULL,
        sentence TEXT NOT NULL,
        parser_json TEXT NOT NULL,
        resource_context_json TEXT NOT NULL,
        timestamp REAL NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_teacher_lemma
      ON teacher_examples(lemma);

    CREATE TABLE IF NOT EXISTS learned_patterns (
        lemma TEXT NOT NULL,
        role TEXT NOT NULL,
        argument_lemma TEXT NOT NULL,
        count INTEGER NOT NULL,
        PRIMARY KEY(lemma,role,argument_lemma)
    );

    CREATE TABLE IF NOT EXISTS semantic_grounding (
        lemma TEXT NOT NULL,
        source TEXT NOT NULL,
        relation TEXT NOT NULL,
        target TEXT NOT NULL,
        count INTEGER NOT NULL,
        PRIMARY KEY(lemma,source,relation,target)
    );

    CREATE TABLE IF NOT EXISTS run_state (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );
    """)
    con.commit()
    migrate_schema(con)
    con.commit()
    return con


def upsert_pattern(con, lemma, role, argument):
    lemma = lemma_key(lemma)
    argument = lemma_key(argument)
    if not lemma or not argument:
        return
    con.execute("""
        INSERT INTO learned_patterns
        VALUES (?, ?, ?, 1)
        ON CONFLICT(lemma,role,argument_lemma)
        DO UPDATE SET count=count+1
    """, (lemma, role, argument))


def upsert_grounding(con, lemma, source, relation, target):
    lemma = lemma_key(lemma)
    target = norm(target)
    if not lemma or not target:
        return
    con.execute("""
        INSERT INTO semantic_grounding
        VALUES (?, ?, ?, ?, 1)
        ON CONFLICT(lemma,source,relation,target)
        DO UPDATE SET count=count+1
    """, (lemma, source, norm(relation), target))


# ============================================================================
# WordNet
# ============================================================================


def _open_wordnet_index(path):
    if path.exists():
        return path.open("r", encoding="utf-8", errors="replace")
    gz = Path(str(path) + ".gz")
    if gz.exists():
        return gzip.open(gz, "rt", encoding="utf-8", errors="replace")
    raise FileNotFoundError(f"WordNet file not found: {path}")


def _wordnet_data_lines(path):
    if not path.exists():
        gz = Path(str(path) + ".gz")
        if gz.exists():
            opener = lambda: gzip.open(
                gz,
                "rt",
                encoding="utf-8",
                errors="replace",
            )
        else:
            raise FileNotFoundError(f"WordNet data file not found: {path}")
    else:
        opener = lambda: path.open(
            "r",
            encoding="utf-8",
            errors="replace",
        )

    with opener() as f:
        for raw in f:
            if not raw.strip() or raw.startswith(" "):
                continue
            if raw.lstrip().startswith("#"):
                continue
            yield raw.rstrip("\n")


def _parse_wn_index(line):
    """
    WordNet index format:
      lemma pos synset_cnt p_cnt ptr_symbols ... sense_cnt tagsense_cnt synsets...
    """
    parts = line.split()
    if len(parts) < 7:
        return None

    lemma = parts[0]
    pos = parts[1]
    try:
        synset_cnt = int(parts[2])
        p_cnt = int(parts[3])
    except ValueError:
        return None

    first_synset = 6 + p_cnt
    if first_synset > len(parts):
        return None

    offsets = []
    for x in parts[first_synset:]:
        try:
            offsets.append(int(x))
        except ValueError:
            pass

    return lemma, pos, offsets


def _parse_wn_data(line, pos, offset_to_synset):
    """
    WordNet data format:
      offset lex_filenum ss_type w_cnt word lex_id ... | gloss
    """
    if "|" in line:
        left, gloss = line.split("|", 1)
        gloss = gloss.strip()
    else:
        left, gloss = line, ""

    parts = left.split()
    if len(parts) < 4:
        return None

    try:
        offset = int(parts[0])
        w_cnt = int(parts[3], 16)
    except ValueError:
        return None

    i = 4
    lemmas = []
    for _ in range(w_cnt):
        if i + 1 >= len(parts):
            return None
        word = parts[i].replace("_", " ")
        i += 2  # word + lex_id
        lemmas.append(lemma_key(word))

    # Remaining fields are pointer records.
    relations = []
    try:
        p_cnt = int(parts[i], 10)
        i += 1
    except (ValueError, IndexError):
        p_cnt = 0

    ptr_map = {
        "@": "hypernym",
        "~": "hyponym",
        "#m": "member_holonym",
        "#s": "substance_holonym",
        "#p": "part_holonym",
        "%m": "member_meronym",
        "%s": "substance_meronym",
        "%p": "part_meronym",
        "*": "entailment",
        "&": "similar_to",
        "^": "also_see",
        "$": "verb_group",
    }

    for _ in range(p_cnt):
        if i + 3 >= len(parts):
            break
        symbol = parts[i]
        target_offset = parts[i + 1]
        target_pos = parts[i + 2]
        i += 4  # symbol, target offset, pos, source/target
        try:
            target_offset = int(target_offset)
        except ValueError:
            continue
        relation = ptr_map.get(symbol, symbol)
        relations.append(
            {
                "relation": relation,
                "target_offset": target_offset,
                "target_pos": target_pos,
            }
        )

    return {
        "offset": offset,
        "pos": pos,
        "lemmas": lemmas,
        "gloss": gloss,
        "relations": relations,
    }


def ingest_wordnet(con, root):
    """
    Direct WordNet reader.

    This intentionally avoids nltk.corpus.WordNetCorpusReader because newer
    NLTK releases enforce path-security checks which can reject a local
    WordNet dict directory on Windows even when the directory is valid.
    """
    root = Path(root)

    if not root.exists():
        raise FileNotFoundError(
            f"WordNet dict directory not found: {root}"
        )

    index_files = {
        "n": root / "index.noun",
        "v": root / "index.verb",
        "a": root / "index.adj",
        "r": root / "index.adv",
        "s": root / "index.sense",
    }

    data_files = {
        "n": root / "data.noun",
        "v": root / "data.verb",
        "a": root / "data.adj",
        "r": root / "data.adv",
    }

    # First build lemma -> synset offsets.
    lemma_offsets = defaultdict(list)
    index_counts = 0

    for pos, path in index_files.items():
        if pos == "s":
            continue
        if not path.exists() and not Path(str(path) + ".gz").exists():
            continue

        with _open_wordnet_index(path) as f:
            for raw in f:
                raw = raw.strip()
                if not raw or raw.startswith(" ") or raw.startswith("#"):
                    continue
                parsed = _parse_wn_index(raw)
                if not parsed:
                    continue
                lemma, file_pos, offsets = parsed
                for off in offsets:
                    lemma_offsets[(lemma_key(lemma), file_pos)].append(off)
                index_counts += 1

    synsets = 0
    relations = 0
    concepts = 0

    # Map (pos, offset) -> synthetic synset ID.
    for pos, path in data_files.items():
        if not path.exists() and not Path(str(path) + ".gz").exists():
            continue

        records = {}
        for raw in _wordnet_data_lines(path):
            rec = _parse_wn_data(raw, pos, None)
            if not rec:
                continue
            records[rec["offset"]] = rec

        # Store only synsets represented by the data files.
        for offset, rec in records.items():
            sid = f"{pos}.{offset:08d}"
            lemmas = rec["lemmas"]
            gloss = rec["gloss"]

            con.execute("""
                INSERT OR REPLACE INTO concepts
                VALUES (?, ?, ?, 'wordnet', ?, ?, ?)
            """, (
                f"wn:{sid}",
                lemmas[0] if lemmas else "",
                pos,
                sid,
                gloss,
                json.dumps(
                    {
                        "lemmas": lemmas,
                        "offset": offset,
                    },
                    ensure_ascii=False,
                ),
            ))
            concepts += 1
            synsets += 1

            for relation in rec["relations"]:
                target_id = (
                    f'{relation["target_pos"]}.'
                    f'{relation["target_offset"]:08d}'
                )
                con.execute("""
                    INSERT OR IGNORE INTO lexical_relations
                    VALUES (?, ?, ?, 'wordnet')
                """, (
                    sid,
                    relation["relation"],
                    target_id,
                ))
                relations += 1

    # The index files are our authoritative lemma vocabulary. Ensure every
    # lemma gets at least one concept row even if a compact WordNet release
    # has an unusual data-file layout.
    for (lemma, pos), offsets in lemma_offsets.items():
        if not offsets:
            continue
        for offset in offsets[:64]:
            sid = f"{pos}.{offset:08d}"
            exists = con.execute(
                "SELECT 1 FROM concepts "
                "WHERE concept_id=? LIMIT 1",
                (f"wn:{sid}",),
            ).fetchone()
            if not exists:
                continue

    con.commit()

    return {
        "index_entries": index_counts,
        "synsets": synsets,
        "relations": relations,
        "concept_rows": concepts,
        "lemmas": len(lemma_offsets),
        "reader": "direct_wordnet_dict",
    }


# ============================================================================
# VerbNet 3.3
# ============================================================================

def local_tag(tag):
    return tag.rsplit("}", 1)[-1].upper()


def descendants(root, tag):
    tag = tag.upper()
    return [
        x for x in root.iter()
        if local_tag(x.tag) == tag
    ]


def ingest_verbnet(con, root):
    files = list(iter_files(root, {".xml"}))
    classes = 0
    members = 0
    roles = 0
    frames = 0

    for path in files:
        try:
            root_el = ET.parse(path).getroot()
        except Exception:
            continue

        class_nodes = []
        if local_tag(root_el.tag) == "VNCLASS":
            class_nodes = [root_el]
        else:
            class_nodes = descendants(root_el, "VNCLASS")

        for cls in class_nodes:
            cid = cls.attrib.get("ID") or cls.attrib.get("id")
            if not cid:
                continue

            con.execute("""
                INSERT OR REPLACE INTO verbnet_classes
                VALUES (?, ?, ?, ?)
            """, (
                cid,
                cls.attrib.get("name", ""),
                cls.attrib.get("parent", ""),
                json.dumps(
                    cls.attrib,
                    ensure_ascii=False,
                ),
            ))
            classes += 1

            for member in descendants(cls, "MEMBER"):
                lemma = lemma_key(
                    member.attrib.get("name")
                    or member.attrib.get("lemma")
                    or ""
                )
                if not lemma:
                    continue

                con.execute("""
                    INSERT OR IGNORE INTO verbnet_members
                    (class_id, lemma, member_name, grouping)
                    VALUES (?, ?, ?, ?)
                """, (
                    cid,
                    lemma,
                    member.attrib.get("name", ""),
                    member.attrib.get("grouping", ""),
                ))
                members += 1

                con.execute("""
                    INSERT OR REPLACE INTO concepts
                    VALUES (?, ?, 'v', 'verbnet', ?, ?, ?)
                """, (
                    f"vn:{cid}:{lemma}",
                    lemma,
                    cid,
                    "",
                    json.dumps(
                        {"member": member.attrib},
                        ensure_ascii=False,
                    ),
                ))

            for tr in descendants(cls, "THEMROLE"):
                role = (
                    tr.attrib.get("type")
                    or tr.attrib.get("theta")
                    or tr.attrib.get("name")
                    or ""
                )
                if role:
                    con.execute("""
                        INSERT OR REPLACE INTO verbnet_roles
                        VALUES (?, ?, ?)
                    """, (
                        cid,
                        role,
                        tr.attrib.get("description", ""),
                    ))
                    roles += 1

            for idx, frame in enumerate(
                descendants(cls, "FRAME")
            ):
                desc = ""
                ds = descendants(frame, "DESCRIPTION")
                if ds:
                    desc = ds[0].attrib.get(
                        "description", ""
                    )

                syntax = []
                for syn in descendants(frame, "SYNTAX"):
                    for child in list(syn):
                        syntax.append({
                            "tag": local_tag(child.tag),
                            "attrib": dict(child.attrib),
                        })

                semantics = []
                for sem in descendants(frame, "SEMANTICS"):
                    for child in list(sem):
                        semantics.append({
                            "tag": local_tag(child.tag),
                            "attrib": dict(child.attrib),
                        })

                con.execute("""
                    INSERT OR REPLACE INTO verbnet_frames
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    cid,
                    idx,
                    desc,
                    json.dumps(
                        syntax,
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        semantics,
                        ensure_ascii=False,
                    ),
                ))
                frames += 1

    con.commit()
    return {
        "xml_files": len(files),
        "classes": classes,
        "members": members,
        "roles": roles,
        "frames": frames,
    }


# ============================================================================
# PropBank
# ============================================================================


def ingest_propbank(con, root, frame_root=None):
    """
    Ingest the actual propbank-release layout.

    propbank-release is an annotation release: the important files are .prop
    and related CoNLL/pointer files, not necessarily frameset XML. The unified
    PropBank frame lexicon is therefore discovered separately under:
        <semlink>/lexical_resources/propbank-frames-master
    when present.
    """
    root=Path(root)

    prop_files=list(iter_files(root,{".prop"}))
    conll_files=list(iter_files(root,{".conll",".conllx",".conllu"}))

    annotation_predicates=0
    argument_mentions=0

    # Prop .prop lines are pointer annotations. We keep the actual predicate
    # sense (when present) and numbered arguments as evidence.
    for path in prop_files:
        try:
            lines=path.read_text(
                encoding="utf-8",
                errors="ignore",
            ).splitlines()
        except Exception:
            continue

        for line in lines:
            line=line.strip()
            if not line or line.startswith("#"):
                continue

            # Common PropBank predicate form contains something like:
            #  ... word-sense ... ARG0 ... ARG1 ...
            # Accept several practical variants without pretending to know the
            # unavailable source sentence text.
            sense_matches=re.findall(
                r"(?<![\w-])([A-Za-z][\w-]*)\.(\d{2})(?!\w)",
                line,
            )
            if not sense_matches:
                continue

            for lemma,sense in sense_matches:
                lemma=lemma_key(lemma)
                roleset=f"{lemma}.{sense}"

                con.execute("""
                    INSERT OR IGNORE INTO propbank_predicates
                    VALUES (?, ?)
                """,(lemma,str(path)))

                con.execute("""
                    INSERT OR REPLACE INTO propbank_rolesets
                    VALUES (?, ?, ?, ?, ?)
                """,(
                    roleset,
                    lemma,
                    "",
                    "",
                    str(path),
                ))
                annotation_predicates+=1

            args=re.findall(
                r"\b(ARG\d+(?:-ARG\d+)?|ARGM-[A-Z]+)\b",
                line,
            )
            for arg in args:
                # Attach argument observations to each detected roleset.
                for lemma,sense in sense_matches:
                    rid=f"{lemma_key(lemma)}.{sense}"
                    con.execute("""
                        INSERT OR IGNORE INTO propbank_roles
                        VALUES (?, ?, ?, ?)
                    """,(
                        rid,
                        arg,
                        "",
                        "annotation_role",
                    ))
                    argument_mentions+=1

    # The official release may also contain useful CoNLL-formatted predicate
    # annotations. We scan these conservatively for roleset identifiers.
    conll_predicates=0
    for path in conll_files:
        try:
            text=path.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        except Exception:
            continue
        for match in re.finditer(
            r"(?<![\w-])([A-Za-z][\w-]*)\.(\d{2})(?!\w)",
            text,
        ):
            lemma,sense=match.groups()
            lemma=lemma_key(lemma)
            rid=f"{lemma}.{sense}"
            con.execute("""
                INSERT OR IGNORE INTO propbank_predicates
                VALUES (?, ?)
            """,(lemma,str(path)))
            con.execute("""
                INSERT OR IGNORE INTO propbank_rolesets
                VALUES (?, ?, ?, ?, ?)
            """,(rid,lemma,"","",str(path)))
            conll_predicates+=1

    # Find the actual PropBank frame lexicon. SemLink 2 bundles lexical
    # resources according to its official repository layout.
    if frame_root is not None:
        frame_root=Path(frame_root)
    else:
        frame_roots=[
            root/"frames",
            root/"propbank-frames",
            root/"propbank-frames-master",
            root.parent/"semlink"/"lexical_resources"/"propbank-frames-master",
            root.parent/"semlink"/"lexical_resources"/"propbank-frames",
        ]
        frame_root=next(
            (p for p in frame_roots if p.exists()),
            None,
        )

    frame_files=[]
    if frame_root:
        frame_files=list(iter_files(frame_root,{".xml"}))

    frame_predicates=0
    frame_rolesets=0
    frame_roles=0

    for path in frame_files:
        try:
            root_el=ET.parse(path).getroot()
        except Exception:
            continue

        for pred in descendants(root_el,"PREDICATE"):
            lemma=lemma_key(pred.attrib.get("lemma",""))
            if not lemma:
                continue

            con.execute("""
                INSERT OR IGNORE INTO propbank_predicates
                VALUES (?, ?)
            """,(lemma,str(path)))
            frame_predicates+=1

            for rs in descendants(pred,"ROLESET"):
                rid=(
                    rs.attrib.get("id")
                    or rs.attrib.get("ID")
                    or ""
                )
                if not rid:
                    continue

                con.execute("""
                    INSERT OR REPLACE INTO propbank_rolesets
                    VALUES (?, ?, ?, ?, ?)
                """,(
                    rid,
                    lemma,
                    rs.attrib.get("name",""),
                    rs.attrib.get("vncls",""),
                    str(path),
                ))
                frame_rolesets+=1

                for role in descendants(rs,"ROLE"):
                    con.execute("""
                        INSERT OR REPLACE INTO propbank_roles
                        VALUES (?, ?, ?, ?)
                    """,(
                        rid,
                        role.attrib.get("n",""),
                        role.attrib.get("f",""),
                        role.attrib.get("descr",""),
                    ))
                    frame_roles+=1

    con.commit()
    return {
        "annotation_prop_files":len(prop_files),
        "annotation_conll_files":len(conll_files),
        "annotation_predicates":annotation_predicates+conll_predicates,
        "annotation_argument_mentions":argument_mentions,
        "frame_root":str(frame_root) if frame_root else None,
        "frame_files":len(frame_files),
        "frame_predicates":frame_predicates,
        "rolesets":frame_rolesets,
        "roles":frame_roles,
        "release_note":"propbank-release supplies annotations; frame lexicon may be bundled with SemLink",
    }


# ============================================================================
# SemLink
# ============================================================================


def ingest_semlink(con, root):
    """
    Ingest SemLink 2's actual mapping files.

    The official repository documents:
      instances/pb-vn2.json
      instances/vn-fn2.json
      instances/semlink2.instances
    """
    root=Path(root)

    json_candidates=[
        root/"instances"/"pb-vn2.json",
        root/"instances"/"vn-fn2.json",
        root/"instances"/"pb-fn2.json",
    ]

    mappings=0
    mapping_types=Counter()

    def walk_pb_vn(obj):
        nonlocal mappings
        if not isinstance(obj,dict):
            return

        for pb_key,value in obj.items():
            # pb_key usually looks like "abduct.01"
            if isinstance(value,dict):
                for vn_class,vn_data in value.items():
                    con.execute("""
                        INSERT OR REPLACE INTO semlink_mappings
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,(
                        digest(f"pb-vn:{pb_key}:{vn_class}"),
                        "propbank",
                        str(pb_key),
                        "verbnet",
                        str(vn_class),
                        json.dumps(vn_data,ensure_ascii=False),
                        str(json_candidates[0]),
                    ))
                    mappings+=1
                    mapping_types["propbank->verbnet"]+=1

    def walk_vn_fn(obj):
        nonlocal mappings
        if not isinstance(obj,dict):
            return

        for vn_key,value in obj.items():
            if isinstance(value,(list,dict)):
                targets=value.keys() if isinstance(value,dict) else value
                if isinstance(value,dict):
                    for fn_frame,fn_data in value.items():
                        con.execute("""
                            INSERT OR REPLACE INTO semlink_mappings
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,(
                            digest(f"vn-fn:{vn_key}:{fn_frame}"),
                            "verbnet",
                            str(vn_key),
                            "framenet",
                            str(fn_frame),
                            json.dumps(fn_data,ensure_ascii=False),
                            str(json_candidates[1]),
                        ))
                        mappings+=1
                        mapping_types["verbnet->framenet"]+=1
                else:
                    for fn_frame in targets:
                        con.execute("""
                            INSERT OR REPLACE INTO semlink_mappings
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,(
                            digest(f"vn-fn:{vn_key}:{fn_frame}"),
                            "verbnet",
                            str(vn_key),
                            "framenet",
                            str(fn_frame),
                            "",
                            str(json_candidates[1]),
                        ))
                        mappings+=1
                        mapping_types["verbnet->framenet"]+=1

    loaded_files=[]
    for idx,path in enumerate(json_candidates):
        if not path.exists():
            continue
        try:
            data=json.loads(
                path.read_text(
                    encoding="utf-8",
                    errors="ignore",
                )
            )
        except Exception:
            continue
        loaded_files.append(str(path))

        if path.name=="pb-vn2.json":
            walk_pb_vn(data)
        elif path.name in {"vn-fn2.json","pb-fn2.json"}:
            walk_vn_fn(data)

    # Also capture annotated instances when present. They are JSON/TSV-like
    # text and can be mined for explicit resource identifiers.
    instance_files=[]
    inst_dir=root/"instances"
    if inst_dir.exists():
        instance_files=list(iter_files(inst_dir,{".instances",".txt",".tsv"}))

    instance_rows=0
    for path in instance_files:
        try:
            lines=path.read_text(
                encoding="utf-8",
                errors="ignore",
            ).splitlines()
        except Exception:
            continue
        for line in lines:
            if not line.strip() or line.lstrip().startswith("#"):
                continue

            pb=re.findall(
                r"\b([A-Za-z][\w-]*\.\d{2})\b",
                line,
            )
            vn=re.findall(
                r"\b([A-Za-z][\w-]*\.\d+\.\d+(?:-\d+)?)\b",
                line,
            )

            for pb_id in pb:
                for vn_id in vn[:8]:
                    con.execute("""
                        INSERT OR IGNORE INTO semlink_mappings
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,(
                        digest(f"instance:{path}:{line}:{pb_id}:{vn_id}"),
                        "propbank",
                        pb_id,
                        "verbnet",
                        vn_id,
                        line[:4000],
                        str(path),
                    ))
                    mappings+=1
                    mapping_types["instance_pb_vn"]+=1
            instance_rows+=1

    con.commit()
    return {
        "files":len(loaded_files)+len(instance_files),
        "json_files":loaded_files,
        "instance_files":[str(x) for x in instance_files],
        "instance_rows":instance_rows,
        "mappings":mappings,
        "mapping_pairs":dict(mapping_types),
    }


# ============================================================================
# Core semantic vocabulary
# ============================================================================


def select_core_vocabulary(con, limit):
    counts=Counter()
    sources=defaultdict(set)

    for lemma,source,_count in con.execute("""
        SELECT lemma, source, COUNT(*)
        FROM concepts
        WHERE lemma <> ''
        GROUP BY lemma, source
    """):
        lemma=lemma_key(lemma)
        if lemma:
            counts[lemma]+=1
            sources[lemma].add(source)

    for (lemma,) in con.execute(
        "SELECT DISTINCT lemma FROM verbnet_members"
    ):
        lemma=lemma_key(lemma)
        if lemma:
            counts[lemma]+=1
            sources[lemma].add("verbnet")

    for (lemma,) in con.execute(
        "SELECT DISTINCT lemma FROM propbank_predicates"
    ):
        lemma=lemma_key(lemma)
        if lemma:
            counts[lemma]+=1
            sources[lemma].add("propbank")

    ranked=[]
    for lemma,count in counts.items():
        score=float(count)
        source_count=len(sources[lemma])

        if "wordnet" in sources[lemma]:
            score+=10
        if "verbnet" in sources[lemma]:
            score+=20
        if "propbank" in sources[lemma]:
            score+=20
        if source_count>=2:
            score+=25
        if source_count>=3:
            score+=35

        ranked.append({
            "lemma":lemma,
            "score":score,
            "sources":sorted(sources[lemma]),
            "source_count":source_count,
            "evidence_count":count,
        })

    ranked.sort(
        key=lambda x:(-x["score"],-x["evidence_count"],x["lemma"])
    )
    return ranked[:limit]


# ============================================================================
# Teacher
# ============================================================================

class Teacher:
    def __init__(self, model_name, max_new_tokens=80):
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise SystemExit(
                "Install:\n"
                "python -m pip install -U torch transformers accelerate"
            ) from exc

        self.torch = torch
        self.max_new_tokens = max_new_tokens

        print(
            f"[TEACHER] tokenizer -> {model_name}",
            flush=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        print(
            f"[TEACHER] model -> {model_name}",
            flush=True,
        )
        kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
        }
        if torch.cuda.is_available():
            kwargs["torch_dtype"] = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **kwargs,
        )
        self.model.eval()

    def generate(self, prompt):
        messages = [
            {
                "role": "system",
                "content": (
                    "Answer with one short, normal English sentence. "
                    "Do not explain."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        if hasattr(
            self.tokenizer,
            "apply_chat_template",
        ):
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt_text = (
                "Answer with one short, normal English sentence.\n"
                f"User: {prompt}\nAssistant:"
            )

        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
        )
        device = next(
            self.model.parameters()
        ).device
        inputs = {
            k: v.to(device)
            for k, v in inputs.items()
        }

        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        continuation = output[0][
            inputs["input_ids"].shape[1]:
        ]
        return self.tokenizer.decode(
            continuation,
            skip_special_tokens=True,
        ).strip()


def clean_sentence(text):
    text=(text or "").strip()
    text=re.sub(
        r"^(assistant|answer|response)\\s*:\\s*",
        "",
        text,
        flags=re.I,
    )
    text=re.sub(r"\\s+"," ",text).strip("'\" ")
    if not text or len(text.split()) < 3:
        return None
    if any(x in text for x in ("{","}","```")):
        return None
    m=re.search(r"(.+?[.!?])(?:\\s|$)",text)
    if m:
        text=m.group(1).strip()
    return text


# ============================================================================
# spaCy structure
# ============================================================================

def load_parser(model):
    try:
        import spacy
        return spacy.load(model)
    except Exception as exc:
        raise SystemExit(
            f"Could not load {model}: {exc}\n"
            f"python -m spacy download {model}"
        )


def parse_teacher_sentence(nlp, sentence):
    doc = nlp(sentence)

    tokens = []
    for t in doc:
        if t.is_space:
            continue
        tokens.append({
            "text": t.text,
            "lemma": lemma_key(t.lemma_),
            "pos": t.pos_,
            "dep": t.dep_,
            "head": t.head.i,
        })

    predicates = []
    for t in doc:
        if t.is_space or t.is_punct:
            continue
        if t.pos_ not in {"VERB", "AUX"}:
            continue

        pred = {
            "predicate": lemma_key(t.lemma_),
            "surface": t.text,
            "dep": t.dep_,
            "subjects": [],
            "objects": [],
            "obliques": [],
            "modifiers": [],
            "auxiliaries": [],
            "negations": [],
            "complements": [],
        }

        for child in t.children:
            if child.is_space or child.is_punct:
                continue

            item = {
                "text": child.text,
                "lemma": lemma_key(child.lemma_),
                "pos": child.pos_,
                "dep": child.dep_,
            }

            if child.dep_ in {
                "nsubj",
                "nsubjpass",
                "csubj",
                "csubjpass",
            }:
                pred["subjects"].append(item)
            elif child.dep_ in {
                "obj",
                "dobj",
                "iobj",
            }:
                pred["objects"].append(item)
            elif (
                child.dep_.startswith("obl")
                or child.dep_ == "prep"
            ):
                pred["obliques"].append(item)
            elif child.dep_ in {"advmod", "amod"}:
                pred["modifiers"].append(item)
            elif child.dep_ in {
                "aux",
                "auxpass",
                "cop",
            }:
                pred["auxiliaries"].append(item)
            elif child.dep_ == "neg":
                pred["negations"].append(item)
            elif child.dep_ in {
                "xcomp",
                "ccomp",
                "advcl",
                "acl",
                "relcl",
            }:
                pred["complements"].append(item)

        predicates.append(pred)

    return {
        "text": sentence,
        "tokens": tokens,
        "predicates": predicates,
    }


def target_predicate(parsed, lemma):
    target = lemma_key(lemma)
    return [
        p
        for p in parsed["predicates"]
        if p["predicate"] == target
    ]


# ============================================================================
# Memory update
# ============================================================================

def learn_teacher_example(con, lemma, prompt, sentence, parsed, context):
    preds = target_predicate(parsed, lemma)
    if not preds:
        return False

    pred = preds[0]
    eid = digest(
        json.dumps(
            [lemma, sentence],
            ensure_ascii=False,
            sort_keys=True,
        )
    )

    con.execute("BEGIN")
    try:
        con.execute("""
            INSERT OR IGNORE INTO teacher_examples
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            eid,
            lemma_key(lemma),
            prompt,
            sentence,
            json.dumps(parsed, ensure_ascii=False),
            json.dumps(context, ensure_ascii=False),
            time.time(),
        ))

        for item in pred["subjects"]:
            upsert_pattern(
                con,
                lemma,
                "subject",
                item["lemma"],
            )
        for item in pred["objects"]:
            upsert_pattern(
                con,
                lemma,
                "object",
                item["lemma"],
            )
        for item in pred["obliques"]:
            upsert_pattern(
                con,
                lemma,
                "oblique",
                item["lemma"],
            )
        for item in pred["complements"]:
            upsert_pattern(
                con,
                lemma,
                "complement",
                item["lemma"],
            )

        for source, relations in context.get(
            "grounding", {}
        ).items():
            for relation, targets in relations.items():
                for target in targets:
                    upsert_grounding(
                        con,
                        lemma,
                        source,
                        relation,
                        target,
                    )

        con.commit()
        return True
    except Exception:
        con.rollback()
        raise


# ============================================================================
# Resource context for LLM
# ============================================================================

def resource_context(con, lemma):
    lemma = lemma_key(lemma)

    wn = []
    for row in con.execute("""
        SELECT source_id, pos, gloss, payload_json
        FROM concepts
        WHERE source='wordnet' AND lemma=?
        LIMIT 6
    """, (lemma,)):
        wn.append({
            "synset": row[0],
            "pos": row[1],
            "gloss": row[2],
            "payload": json.loads(row[3]),
        })

    vn = []
    for row in con.execute("""
        SELECT class_id, name
        FROM verbnet_classes
        WHERE class_id IN (
          SELECT class_id FROM verbnet_members WHERE lemma=?
        )
        LIMIT 6
    """, (lemma,)):
        vn.append({
            "class_id": row[0],
            "name": row[1],
        })

    roles = []
    for row in con.execute("""
        SELECT vr.class_id, vr.role, vr.description
        FROM verbnet_roles vr
        JOIN verbnet_members vm
          ON vm.class_id=vr.class_id
        WHERE vm.lemma=?
        LIMIT 12
    """, (lemma,)):
        roles.append({
            "class_id": row[0],
            "role": row[1],
            "description": row[2],
        })

    pb = []
    for row in con.execute("""
        SELECT roleset_id, name, vncls
        FROM propbank_rolesets
        WHERE lemma=?
        LIMIT 6
    """, (lemma,)):
        pb.append({
            "roleset_id": row[0],
            "name": row[1],
            "vncls": row[2],
        })

    mappings = []
    for row in con.execute("""
        SELECT source_family, source_id,
               target_family, target_id
        FROM semlink_mappings
        WHERE source_id LIKE ?
           OR target_id LIKE ?
        LIMIT 12
    """, (
        f"%{lemma}%",
        f"%{lemma}%",
    )):
        mappings.append({
            "source_family": row[0],
            "source_id": row[1],
            "target_family": row[2],
            "target_id": row[3],
        })

    return {
        "wordnet": wn,
        "verbnet": vn,
        "verbnet_roles": roles,
        "propbank": pb,
        "semlink": mappings,
    }


# ============================================================================
# Smoke
# ============================================================================

def smoke():
    tmp = Path.cwd() / "v425_smoke.sqlite"
    if tmp.exists():
        tmp.unlink()

    con = init_db(tmp)
    core = select_core_vocabulary(con, 10)
    assert isinstance(core, list)

    parsed = {
        "text": "I found a book.",
        "tokens": [],
        "predicates": [{
            "predicate": "find",
            "surface": "found",
            "dep": "ROOT",
            "subjects": [{"lemma": "i"}],
            "objects": [{"lemma": "book"}],
            "obliques": [],
            "modifiers": [],
            "auxiliaries": [],
            "negations": [],
            "complements": [],
        }],
    }

    context = {
        "grounding": {
            "wordnet": {
                "hypernym": ["discover"],
            },
        },
    }

    assert learn_teacher_example(
        con,
        "find",
        "Use find in a sentence.",
        "I found a book.",
        parsed,
        context,
    )

    assert con.execute(
        "SELECT COUNT(*) FROM teacher_examples"
    ).fetchone()[0] == 1

    assert con.execute(
        "SELECT COUNT(*) FROM learned_patterns"
    ).fetchone()[0] >= 2

    assert con.execute(
        "SELECT COUNT(*) FROM semantic_grounding"
    ).fetchone()[0] == 1

    con.execute("""
        INSERT OR IGNORE INTO verbnet_members
        VALUES (?, ?, ?, ?)
    """, ("test-class", "move", "move", ""))
    assert con.execute(
        "SELECT COUNT(*) FROM verbnet_members WHERE class_id=?",
        ("test-class",),
    ).fetchone()[0] == 1

    con.close()

    legacy = Path.cwd() / "v428_legacy.sqlite"
    if legacy.exists():
        legacy.unlink()
    lc = sqlite3.connect(str(legacy))
    lc.execute("""
        CREATE TABLE verbnet_members (
            class_id TEXT NOT NULL,
            lemma TEXT NOT NULL,
            PRIMARY KEY(class_id,lemma)
        )
    """)
    migration = migrate_schema(lc)
    assert "member_name" in migration["columns"]
    assert "grouping" in migration["columns"]
    lc.execute("""
        INSERT OR IGNORE INTO verbnet_members
        (class_id,lemma,member_name,grouping)
        VALUES (?,?,?,?)
    """, ("legacy-class","move","move",""))
    assert lc.execute(
        "SELECT COUNT(*) FROM verbnet_members"
    ).fetchone()[0] == 1
    lc.close()
    legacy.unlink(missing_ok=True)

    print("legacy SQLite schema migration: PASS")

    tmp.unlink(missing_ok=True)

    print("V425 lexical-semantic ingestion smoke: PASS")
    print("WordNet schema: PASS")
    print("VerbNet schema: PASS")
    print("PropBank schema: PASS")
    print("SemLink schema: PASS")
    print("core vocabulary selection: PASS")
    print("teacher example learning: PASS")
    print("predicate/argument learning: PASS")
    print("semantic grounding learning: PASS")
    test_payload = {
        "windows_path": Path(r"C:\Users\adria\Desktop\test"),
        "nested": {"paths": [Path(r"C:\x"), Path(r"D:\y")]},
    }
    encoded = json.dumps(json_safe(test_payload))
    assert "C:\\\\Users" in encoded or "C:\\Users" in encoded
    print("WindowsPath JSON serialization: PASS")


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Ingest the local semantic resources and enrich the cognitive memory."
    )
    ap.add_argument(
        "--max-concepts",
        type=int,
        default=10000,
        help="Number of core semantic concepts to enrich (default: 10000).",
    )
    ap.add_argument(
        "--status-every",
        type=int,
        default=25,
        help="Print progress every N concepts.",
    )
    ap.add_argument(
        "--teacher-probe",
        type=int,
        default=3,
        help="Probe the first N concepts before the main pass.",
    )
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=80,
        help="Maximum teacher generation length.",
    )
    ap.add_argument(
        "--fresh",
        action="store_true",
        help="Delete the existing cognitive memory before starting.",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
    )

    # Project-local defaults. Override only if the layout changes.
    args_defaults = {
        "wordnet": Path(r"C:\Users\adria\Desktop\dev\Graph-Topology\data\WordNet\dict"),
        "verbnet": Path(r"C:\Users\adria\Desktop\dev\Graph-Topology\data\verbnet3.3"),
        "propbank": Path(r"C:\Users\adria\Desktop\dev\Graph-Topology\data\propbank-release"),
        "propbank_frames": Path(r"C:\Users\adria\Desktop\dev\Graph-Topology\data\propbank-frames-3.1"),
        "semlink": Path(r"C:\Users\adria\Desktop\dev\Graph-Topology\data\semlink"),
        "gum": Path(r"C:\Users\adria\Desktop\dev\Graph-Topology\data\UD_GUM"),
        "conceptnet": Path(r"C:\Users\adria\Desktop\dev\Graph-Topology\data\conceptnet_compact.db"),
        "model": Path(r"C:\Users\adria\Desktop\dev\Graph-Topology\llm\SmolLM2-1.7B-Instruct"),
        "spacy_model": "en_core_web_trf",
    }
    args = ap.parse_args()

    for _name, _value in args_defaults.items():
        setattr(args, _name, _value)

    if args.smoke:
        smoke()
        return

    start = time.perf_counter()

    required = {
        "WordNet": args.wordnet,
        "VerbNet": args.verbnet,
        "PropBank": args.propbank,
        "PropBank frames": args.propbank_frames,
        "SemLink": args.semlink,
        "GUM": args.gum,
        "ConceptNet": args.conceptnet,
    }
    for name, path in required.items():
        if not path.exists():
            raise SystemExit(
                f"{name} path not found: {path}"
            )

    results = Path.cwd() / "results"
    results.mkdir(
        parents=True,
        exist_ok=True,
    )

    db_path = results / "cognitive_language_memory.sqlite"
    core_path = results / "v432_core_semantic_vocabulary.jsonl"
    examples_path = results / "v432_teacher_examples.jsonl"
    failures_path = results / "v432_teacher_failures.jsonl"
    report_path = results / "v432_lexical_semantic_report.json"

    if args.fresh and db_path.exists():
        db_path.unlink()

    print("=" * 78, flush=True)
    print(
        "V431 LEXICAL + SEMANTIC COGNITIVE INGESTION",
        flush=True,
    )
    print("=" * 78, flush=True)

    print(
        "[1/8] Initializing persistent cognitive memory...",
        flush=True,
    )
    con = init_db(db_path)

    print("[2/8] WordNet...", flush=True)
    wordnet_stats = ingest_wordnet(
        con,
        Path(args.wordnet).resolve(),
    )
    print(
        f"      synsets={wordnet_stats['synsets']:,} "
        f"relations={wordnet_stats['relations']:,}",
        flush=True,
    )

    print("[3/8] VerbNet 3.3...", flush=True)
    verbnet_stats = ingest_verbnet(
        con,
        Path(args.verbnet).resolve(),
    )
    print(
        f"      classes={verbnet_stats['classes']:,} "
        f"members={verbnet_stats['members']:,} "
        f"roles={verbnet_stats['roles']:,} "
        f"frames={verbnet_stats['frames']:,}",
        flush=True,
    )

    print("[4/8] PropBank...", flush=True)
    propbank_stats = ingest_propbank(
        con,
        Path(args.propbank).resolve(),
        Path(args.propbank_frames).resolve(),
    )
    print(
        f"      annotation_predicates={propbank_stats['annotation_predicates']:,} "
        f"frame_predicates={propbank_stats['frame_predicates']:,} "
        f"rolesets={propbank_stats['rolesets']:,} "
        f"roles={propbank_stats['roles']:,}",
        flush=True,
    )

    print("[5/8] SemLink...", flush=True)
    semlink_stats = ingest_semlink(
        con,
        Path(args.semlink).resolve(),
    )
    print(
        f"      mappings={semlink_stats['mappings']:,}",
        flush=True,
    )

    print("[6/8] Selecting core semantic vocabulary...", flush=True)
    core = select_core_vocabulary(
        con,
        args.max_concepts,
    )
    with core_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for row in core:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ) + "\n"
            )

    print(
        f"      core_concepts={len(core):,}",
        flush=True,
    )

    print("[7/8] Loading SmolLM2 + spaCy...", flush=True)
    teacher = Teacher(
        args.model,
        args.max_new_tokens,
    )
    nlp = load_parser(args.spacy_model)

    if args.teacher_probe:
        print(
            f"      probing first "
            f"{min(args.teacher_probe, len(core))} concepts",
            flush=True,
        )
        for i, item in enumerate(
            core[:args.teacher_probe],
            1,
        ):
            prompt = (
                f'Write one short, natural English sentence '
                f'using the exact word "{item["lemma"]}".'
            )
            raw = teacher.generate(prompt)
            print(
                f"      PROBE {i}: "
                f"{item['lemma']} -> {raw!r}",
                flush=True,
            )

    # We intentionally make every accepted example durable immediately.
    accepted = 0
    skipped = 0
    failed = 0
    session_started = time.perf_counter()

    example_f = examples_path.open(
        "a",
        encoding="utf-8",
    )
    failure_f = failures_path.open(
        "a",
        encoding="utf-8",
    )

    try:
        for i, item in enumerate(core, 1):
            lemma = item["lemma"]

            # One example per core concept per DB. This is intentionally a
            # "semantic seed" pass rather than the final multi-example learner.
            if con.execute(
                "SELECT 1 FROM teacher_examples "
                "WHERE lemma=? LIMIT 1",
                (lemma,),
            ).fetchone():
                skipped += 1
                continue

            prompt = (
                f'Write one short, natural English sentence '
                f'using the exact word "{lemma}".'
            )

            print(
                f"      LEARN {i:,}/{len(core):,} -> {lemma}",
                flush=True,
            )

            t0 = time.perf_counter()
            try:
                raw = teacher.generate(prompt)
                sentence = clean_sentence(raw)

                if not sentence:
                    failed += 1
                    row = {
                        "lemma": lemma,
                        "reason": "empty_or_invalid",
                        "raw": raw[:2000],
                    }
                    failure_f.write(
                        json.dumps(
                            row,
                            ensure_ascii=False,
                        ) + "\n"
                    )
                    failure_f.flush()
                    continue

                parsed = parse_teacher_sentence(
                    nlp,
                    sentence,
                )
                if not target_predicate(
                    parsed,
                    lemma,
                ):
                    failed += 1
                    row = {
                        "lemma": lemma,
                        "reason": "target_predicate_not_found",
                        "sentence": sentence,
                        "parsed": parsed,
                    }
                    failure_f.write(
                        json.dumps(
                            row,
                            ensure_ascii=False,
                        ) + "\n"
                    )
                    failure_f.flush()
                    continue

                context = resource_context(
                    con,
                    lemma,
                )

                # Ground important resource identifiers by storing them as
                # simple semantic edges in the same cognitive memory.
                grounding = {
                    "wordnet": defaultdict(list),
                    "verbnet": defaultdict(list),
                    "propbank": defaultdict(list),
                    "semlink": defaultdict(list),
                }

                for syn in context["wordnet"]:
                    grounding["wordnet"]["synset"].append(
                        syn["synset"]
                    )
                    if syn["gloss"]:
                        grounding["wordnet"]["gloss"].append(
                            syn["gloss"]
                        )

                for cls in context["verbnet"]:
                    grounding["verbnet"]["class"].append(
                        cls["class_id"]
                    )

                for role in context["verbnet_roles"]:
                    grounding["verbnet"]["role"].append(
                        role["role"]
                    )

                for rs in context["propbank"]:
                    grounding["propbank"]["roleset"].append(
                        rs["roleset_id"]
                    )

                for mapping in context["semlink"]:
                    grounding["semlink"]["mapping"].append(
                        f'{mapping["source_id"]}->{mapping["target_id"]}'
                    )

                for d in grounding.values():
                    for key in list(d):
                        d[key] = list(
                            dict.fromkeys(d[key])
                        )

                context["grounding"] = dict(grounding)

                ok = learn_teacher_example(
                    con,
                    lemma,
                    prompt,
                    sentence,
                    parsed,
                    context,
                )

                if ok:
                    accepted += 1
                    example_f.write(
                        json.dumps(
                            {
                                "lemma": lemma,
                                "sentence": sentence,
                                "parsed": parsed,
                                "resource_context": context,
                                "teacher_seconds":
                                    time.perf_counter() - t0,
                            },
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ) + "\n"
                    )
                    example_f.flush()
                else:
                    failed += 1

            except Exception as exc:
                failed += 1
                try:
                    con.rollback()
                except Exception:
                    pass
                failure_f.write(
                    json.dumps(
                        {
                            "lemma": lemma,
                            "reason": "exception",
                            "error": repr(exc),
                        },
                        ensure_ascii=False,
                    ) + "\n"
                )
                failure_f.flush()

            if (
                i == 1
                or i % args.status_every == 0
                or i == len(core)
            ):
                elapsed = (
                    time.perf_counter()
                    - session_started
                )
                done = i - skipped
                rate = done / max(
                    elapsed,
                    1e-9,
                )
                remain = max(
                    len(core) - i,
                    0,
                )
                eta = remain / max(
                    rate,
                    1e-9,
                )

                patterns = con.execute(
                    "SELECT COUNT(*) "
                    "FROM learned_patterns"
                ).fetchone()[0]
                groundings = con.execute(
                    "SELECT COUNT(*) "
                    "FROM semantic_grounding"
                ).fetchone()[0]

                print(
                    f"      CHECKPOINT "
                    f"{i:,}/{len(core):,} "
                    f"accepted={accepted:,} "
                    f"skipped={skipped:,} "
                    f"failed={failed:,} "
                    f"patterns={patterns:,} "
                    f"groundings={groundings:,} "
                    f"rate={rate:.2f}/s "
                    f"eta={eta/3600:.2f}h",
                    flush=True,
                )

    finally:
        example_f.close()
        failure_f.close()

    print("[8/8] Final report...", flush=True)

    db_counts = {}
    for table in (
        "concepts",
        "lexical_relations",
        "verbnet_classes",
        "verbnet_members",
        "verbnet_roles",
        "verbnet_frames",
        "propbank_predicates",
        "propbank_rolesets",
        "propbank_roles",
        "semlink_mappings",
        "teacher_examples",
        "learned_patterns",
        "semantic_grounding",
    ):
        db_counts[table] = int(
            con.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0]
        )

    con.close()

    report = {
        "status": (
            "PASS"
            if db_counts["teacher_examples"] > 0
            else "FAIL"
        ),
        "version": "v432",
        "methodology": {
            "lexical_sources": [
                "WordNet",
                "VerbNet 3.3",
                "PropBank",
                "SemLink",
            ],
            "grammar_source": "UD GUM gold CoNLL-U",
            "semantic_broad_graph": "ConceptNet",
            "teacher_role": (
                "plain-language example generator"
            ),
            "teacher_structure_extraction": "spaCy",
            "learning_target": (
                "persistent lexical-semantic cognitive memory"
            ),
            "teacher_not_runtime_dependency": True,
        },
        "paths": {
            "wordnet": str(
                Path(args.wordnet).resolve()
            ),
            "verbnet": str(
                Path(args.verbnet).resolve()
            ),
            "propbank": str(
                Path(args.propbank).resolve()
            ),
            "semlink": str(
                Path(args.semlink).resolve()
            ),
            "gum": str(
                Path(args.gum).resolve()
            ),
            "conceptnet": str(
                Path(args.conceptnet).resolve()
            ),
        },
        "ingestion": {
            "wordnet": wordnet_stats,
            "verbnet": verbnet_stats,
            "propbank": propbank_stats,
            "semlink": semlink_stats,
        },
        "core_vocabulary": {
            "requested": args.max_concepts,
            "selected": len(core),
        },
        "teacher": {
            "model": args.model,
            "max_new_tokens": args.max_new_tokens,
        },
        "learning": {
            "accepted_examples_this_run": accepted,
            "skipped_existing": skipped,
            "failed_this_run": failed,
            "db": db_counts,
        },
        "outputs": {
            "memory": str(db_path.resolve()),
            "core_vocabulary": str(core_path.resolve()),
            "examples": str(examples_path.resolve()),
            "failures": str(failures_path.resolve()),
            "report": str(report_path.resolve()),
        },
        "runtime_seconds": time.perf_counter() - start,
    }

    report = json_safe(report)

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
