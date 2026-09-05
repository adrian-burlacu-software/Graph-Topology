import argparse
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from v679_memory import RamSemanticMemory
from v679_offline_learning import run_lane, worker_main


def source_graph(path):
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE nodes(node TEXT, node_type TEXT);
        CREATE TABLE edges(subject TEXT, relation TEXT, object TEXT);
        """
    )
    return connection


class OfflineLearningTests(unittest.TestCase):
    def test_composition_reuses_prior_derived_path_at_depth_three(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "graph.sqlite"
            connection = source_graph(database)
            connection.executemany(
                "INSERT INTO edges VALUES(?,?,?)",
                [
                    ("en:a", "is_a", "en:b"),
                    ("en:b", "has_part", "en:c"),
                    ("en:c", "capable_of", "en:d"),
                ],
            )
            connection.commit()
            ram = RamSemanticMemory(0)
            ram.composition_fanout = 4
            ram.composition_max = 100
            ram.composition_max_depth = 3

            run_lane(connection, ram, "relation_composition", 1, 0, 0, ["en:b"])
            run_lane(connection, ram, "relation_composition", 1, 0, 1, ["en:a"])

            row = ram.conn.execute(
                """SELECT derivation_depth,feature_json FROM semantic_knowledge
                   WHERE kind='relation_composition' AND subject='en:a'
                     AND relation='is_a->has_part->capable_of' AND object='en:d'"""
            ).fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row[0], 3)
            self.assertEqual(
                json.loads(row[1])["nodes"], ["en:a", "en:b", "en:c", "en:d"]
            )
            ram.conn.close()
            connection.close()

    def test_composition_reuses_imported_shared_path(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "graph.sqlite"
            connection = source_graph(database)
            connection.execute(
                "INSERT INTO edges VALUES(?,?,?)", ("en:a", "is_a", "en:b")
            )
            connection.commit()
            producer = RamSemanticMemory(0)
            producer.upsert_knowledge(
                "relation_composition",
                "en:b",
                "has_part->capable_of",
                "en:d",
                {
                    "middle": "en:c",
                    "nodes": ["en:b", "en:c", "en:d"],
                    "depth": 2,
                    "relations": ["has_part", "capable_of"],
                },
                positive=1,
                confidence=0.4,
                provenance="derived",
                derivation_depth=2,
            )
            ram = RamSemanticMemory(1)
            ram.import_records(producer.export_records())
            ram.composition_fanout = 4
            ram.composition_max = 100
            ram.composition_max_depth = 3

            run_lane(connection, ram, "relation_composition", 1, 1, 0, ["en:a"])

            self.assertIsNotNone(
                ram.conn.execute(
                    """SELECT 1 FROM semantic_knowledge
                       WHERE kind='relation_composition' AND subject='en:a'
                         AND relation='is_a->has_part->capable_of' AND object='en:d'"""
                ).fetchone()
            )
            producer.conn.close()
            ram.conn.close()
            connection.close()

    def test_worker_stops_after_configured_no_new_streak(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "graph.sqlite"
            source_graph(database).close()
            args = argparse.Namespace(
                database=str(database),
                shared_memory=str(root / "shared.sqlite"),
                worker_log_dir=str(root / "workers"),
                total_workers=1,
                checkpoint_seconds=60,
                seed=1,
                batch_sleep=0,
                duration_seconds=0,
                composition_fanout=4,
                composition_max=100,
                composition_max_depth=3,
                max_no_new_batches=1,
                worker_query_batch_subjects=1,
                task_poll_seconds=0.01,
            )

            worker_main(args, 0)

            rows = [
                json.loads(line)
                for line in (root / "workers" / "worker_00.jsonl").read_text().splitlines()
            ]
            stopped = next(row for row in rows if row["event"] == "worker_stop")
            self.assertEqual(stopped["termination_reason"], "max_no_new_batches")
            self.assertEqual(stopped["no_new_streak"], 1)


if __name__ == "__main__":
    unittest.main()
