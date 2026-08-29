
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


def status(msg):
    print(msg,flush=True)


def make_smoke_files(root):
    corpus=root/"smoke_corpus.txt"
    corpus.write_text(
        "the dog chases the cat\n"
        "the cat sees the dog\n"
        "the dog eats the cat\n"
        "the cat chases the dog\n"
        "the dog sees the cat\n"
        "the cat eats the dog\n"
        "the dog likes the cat\n"
        "the cat likes the dog\n",
        encoding="utf-8",
    )

    db=root/"smoke_conceptnet.db"
    import sqlite3
    conn=sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE edges("
        "source TEXT, relation TEXT, target TEXT)"
    )
    rows=[
        ("dog","IsA","animal"),
        ("cat","IsA","animal"),
        ("dog","CapableOf","chase"),
        ("cat","CapableOf","chase"),
        ("chase","RelatedTo","pursuit"),
        ("eat","RelatedTo","food"),
    ]
    conn.executemany(
        "INSERT INTO edges VALUES(?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()
    return corpus,db


def run(corpus,conceptnet_db,train_limit=None,heldout=1000):
    from conceptnet import load_conceptnet
    from grammar_induction import (
        GrammarInducer,
        evaluate_grammar,
    )
    from semantic_grounding import SemanticGrounder

    status("[1/7] Loading ConceptNet semantic graph...")
    cn=load_conceptnet(conceptnet_db)
    cn_stats=cn.stats()
    status(
        f"      FOUND {cn_stats['database']}"
    )
    status(
        f"      edge table={cn_stats['edge_table']} "
        f"edges={cn_stats['edge_count']:,}"
    )

    status("[2/7] Loading BabyLM grammar corpus...")
    from grammar_induction import CorpusReader
    reader=CorpusReader()
    files=reader.files(corpus)
    status(f"      files discovered={len(files):,}")

    status("[3/7] Inducing grammar from corpus...")
    inducer=GrammarInducer(min_count=3)
    grammar,heldout_lines=inducer.induce(
        corpus,
        train_limit=train_limit,
        heldout_limit=heldout,
        semantic_graph=cn,
    )
    status(
        f"      training sentences={grammar.sentences:,}"
    )
    status(
        f"      training tokens={grammar.tokens:,}"
    )
    status(
        f"      vocabulary={grammar.vocabulary_size:,}"
    )
    status(
        f"      induced constructions={grammar.rule_count():,}"
    )

    status("[4/7] Grounding grammar into semantic graph...")
    grounder=SemanticGrounder(cn)
    grounding,ground_stats=grounder.ground(grammar)
    status(
        f"      grounded constructions="
        f"{ground_stats['grounded_constructions']:,}/"
        f"{ground_stats['total_constructions']:,}"
    )
    status(
        f"      semantic coverage="
        f"{ground_stats['semantic_coverage']:.3f}"
    )

    status("[5/7] Evaluating on held-out language...")
    heldout_metrics=evaluate_grammar(
        grammar,
        heldout_lines,
    )
    status(
        f"      construction recall="
        f"{heldout_metrics['construction_recall']:.3f}"
    )
    status(
        f"      DET+NOUN recall="
        f"{heldout_metrics['det_noun_recall']:.3f}"
    )
    status(
        f"      DET+NOUN+VERB rate="
        f"{heldout_metrics['det_noun_verb_rate']:.3f}"
    )

    status("[6/7] Checking grammar -> semantic integration...")
    integration_ok=(
        grammar.rule_count()>0
        and ground_stats["total_constructions"]>0
        and cn_stats["edge_count"]>0
    )
    status(
        f"      integration={'PASS' if integration_ok else 'FAIL'}"
    )

    # Regression check for the previous failure mode: no grounded sample may
    # come from punctuation-only anchors, and no match may be a substring-only
    # accident.
    bad_anchor_matches=[
        m for m in (
            grounding_match
            for values in grounding.values()
            for grounding_match in values
        )
        if not re.search(r"[A-Za-z]{2,}", m.anchor)
    ]
    integration_ok = integration_ok and not bad_anchor_matches

    status("[7/7] Grounding regression checks...")
    status(
        f"      invalid-anchor matches={len(bad_anchor_matches)}"
    )
    status(
        f"      grounding safety="
        f"{'PASS' if not bad_anchor_matches else 'FAIL'}"
    )

    status("[8/8] RESULT")
    result={
        "status":"PASS" if integration_ok else "FAIL",
        "conceptnet":cn_stats,
        "grammar":{
            "sentences":grammar.sentences,
            "tokens":grammar.tokens,
            "vocabulary":grammar.vocabulary_size,
            "constructions":grammar.rule_count(),
            "top_rules":[
                {
                    "lhs":c.lhs,
                    "rhs":c.rhs,
                    "count":c.count,
                    "probability":c.probability,
                    "anchors":c.semantic_anchors,
                }
                for c in grammar.top_rules(20)
            ],
        },
        "heldout":heldout_metrics,
        "grounding":ground_stats,
        "grounded_samples":[
            {
                "construction":k,
                "matches":[
                    {
                        "anchor":m.anchor,
                        "relation":m.relation,
                        "source":m.source,
                        "target":m.target,
                    }
                    for m in v[:3]
                ],
            }
            for k,v in list(grounding.items())[:12]
            if v
        ],
    }
    cn.close()
    return result


def smoke():
    root=Path(__file__).resolve().parent
    corpus,db=make_smoke_files(root)
    try:
        return run(
            corpus,
            db,
            train_limit=None,
            heldout=2,
        )
    finally:
        corpus.unlink(missing_ok=True)
        db.unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument(
        "corpus",
        nargs="?",
        type=Path,
    )
    parser.add_argument(
        "--conceptnet",
        type=Path,
        default=Path(".\\data\\conceptnet_compact.db"),
    )
    parser.add_argument(
        "--train-limit",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--heldout",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
    )
    args=parser.parse_args()

    if args.smoke or args.corpus is None:
        print(json.dumps(smoke(),indent=2,default=str))
        return

    corpus=args.corpus.expanduser().resolve()
    cn=args.conceptnet.expanduser().resolve()

    if not corpus.exists():
        raise SystemExit(
            f"BabyLM corpus not found: {corpus}"
        )
    if not cn.exists():
        raise SystemExit(
            f"ConceptNet database not found: {cn}"
        )

    result=run(
        corpus,
        cn,
        train_limit=args.train_limit,
        heldout=args.heldout,
    )
    print(json.dumps(result,indent=2,default=str))


if __name__=="__main__":
    main()
