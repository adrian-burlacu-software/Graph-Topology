import sqlite3
import tempfile
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
import graph_knowledge_discovery as g


def build_db(path):
    con = sqlite3.connect(path)
    con.executescript('''
    CREATE TABLE concepts(concept_id INTEGER PRIMARY KEY, canonical TEXT, display TEXT, concept_type TEXT);
    CREATE TABLE facts(
        fact_id INTEGER PRIMARY KEY, subject_id INTEGER, predicate TEXT, object_id INTEGER,
        object_text TEXT, fact_type TEXT, domain TEXT, source_id INTEGER,
        confidence REAL DEFAULT 1, frequency REAL DEFAULT 1, answerable INTEGER DEFAULT 1
    );
    CREATE TABLE live_facts(session_id TEXT, subject TEXT, predicate TEXT, object_text TEXT);
    ''')
    con.executemany('INSERT INTO concepts VALUES(?,?,?,?)', [
        (1,'person','person','entity'),(2,'human','human','entity'),(3,'hand','hand','entity'),
        (4,'body','body','entity'),(5,'tail','tail','entity'),(6,'wing','wing','entity')
    ])
    con.executemany('INSERT INTO facts VALUES(?,?,?,?,?,?,?,?,?,?,?)', [
        (1,1,'is_a',2,None,'semantic',None,1,1,1,1),
        (2,1,'has_part',3,None,'semantic',None,1,1,1,1),
        (3,2,'has_not',5,None,'semantic',None,1,1,1,1),
        (4,2,'capable_of',None,'shake hands','semantic',None,1,1,1,1),
    ])
    con.commit(); con.close()


def main():
    with tempfile.TemporaryDirectory() as td:
        db = str(Path(td)/'test.sqlite')
        build_db(db)
        con = g.connect(db)
        info = g.inspect_schema(con)
        adapter, _ = g.pick_adapter(con, info, None)
        assert adapter.table == 'facts'
        assert adapter.mode == 'canonical'
        rows = g.sample_direct(con, adapter, 4, 541)
        assert rows
        neg = g.discover_negative(con, adapter, 10)
        assert neg and neg[0]['status'] == 'REFUTED'
        print('V541 schema/discovery smoke test: PASS')


if __name__ == '__main__':
    main()
