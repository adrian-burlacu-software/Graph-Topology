import sqlite3
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from topic_catalog import inspect_topic, format_topic_inspection
from schema import SCHEMA

def main():
    con=sqlite3.connect(":memory:")
    con.row_factory=sqlite3.Row
    con.executescript(SCHEMA)
    con.executemany("INSERT INTO concepts(concept_id,canonical,display,concept_type) VALUES(?,?,?,?)", [(1,'person','person','noun'),(2,'hand','hand','noun'),(3,'finger','finger','noun')])
    con.execute("INSERT INTO sources(source_id,dataset,source_path,content_type) VALUES(1,'test','memory','semantic')")
    rows=[
        (1,1,'hasA',2,'hand','semantic',1,1.0,1.0,1),
        (2,2,'PartOf',1,'person','semantic',1,1.0,1.0,1),
        (3,2,'hasA',3,'finger','semantic',1,1.0,1.0,1),
    ]
    con.executemany("INSERT INTO facts(fact_id,subject_id,predicate,object_id,object_text,fact_type,source_id,confidence,frequency,answerable) VALUES(?,?,?,?,?,?,?,?,?,?)", rows)
    r=inspect_topic(con,'person',2)
    assert r['levels']
    out=format_topic_inspection(r)
    assert 'person' in out and 'hand' in out
    print('V530 knowledge inspection test: PASS')

if __name__=='__main__': main()
