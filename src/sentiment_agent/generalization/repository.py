from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from sentiment_agent.attribution.models import Attribution
from sentiment_agent.evidence.models import CaseEvidence
from sentiment_agent.generalization.models import GeneralizedExperience


def _rule_json(rule: GeneralizedExperience) -> str:
    return rule.model_dump_json(exclude_computed_fields=True)


class EvolutionRepository:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS case_evidence(
                id TEXT PRIMARY KEY,
                sample_id TEXT NOT NULL,
                batch_id INTEGER NOT NULL,
                payload_json TEXT NOT NULL,
                UNIQUE(sample_id, batch_id)
            );
            CREATE TABLE IF NOT EXISTS attributions(
                id TEXT PRIMARY KEY,
                case_id TEXT NOT NULL UNIQUE REFERENCES case_evidence(id),
                payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS generalized_experiences(
                id TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS generalized_experience_evidence(
                experience_id TEXT NOT NULL REFERENCES generalized_experiences(id),
                case_id TEXT NOT NULL REFERENCES case_evidence(id),
                attribution_id TEXT REFERENCES attributions(id),
                relation TEXT NOT NULL CHECK(relation IN ('support','contradiction')),
                batch_id INTEGER NOT NULL,
                PRIMARY KEY(experience_id, case_id)
            );
            CREATE TABLE IF NOT EXISTS generalized_experience_events(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experience_id TEXT NOT NULL REFERENCES generalized_experiences(id),
                event_type TEXT NOT NULL,
                batch_id INTEGER NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS generalized_experience_outcomes(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experience_id TEXT NOT NULL REFERENCES generalized_experiences(id),
                sample_id TEXT NOT NULL,
                batch_id INTEGER NOT NULL,
                correct INTEGER NOT NULL
            );
            """
        )
        self.connection.commit()

    def create_case(self, case: CaseEvidence) -> None:
        try:
            with self.connection:
                self.connection.execute(
                    "INSERT INTO case_evidence(id,sample_id,batch_id,payload_json) VALUES(?,?,?,?)",
                    (case.id, case.sample_id, case.batch_id, case.model_dump_json()),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError("case evidence already exists for sample and batch") from exc

    def get_case(self, case_id: str) -> CaseEvidence:
        row = self.connection.execute(
            "SELECT payload_json FROM case_evidence WHERE id=?", (case_id,)
        ).fetchone()
        if row is None:
            raise KeyError(case_id)
        return CaseEvidence.model_validate_json(row[0])

    def create_attribution(self, attribution: Attribution) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT INTO attributions(id,case_id,payload_json) VALUES(?,?,?)",
                (attribution.id, attribution.case_id, attribution.model_dump_json()),
            )

    def get_attribution(self, attribution_id: str) -> Attribution:
        row = self.connection.execute(
            "SELECT payload_json FROM attributions WHERE id=?", (attribution_id,)
        ).fetchone()
        if row is None:
            raise KeyError(attribution_id)
        return Attribution.model_validate_json(row[0])

    def list_attributions(self) -> list[Attribution]:
        rows = self.connection.execute("SELECT payload_json FROM attributions ORDER BY id").fetchall()
        return [Attribution.model_validate_json(row[0]) for row in rows]

    def create_rule(self, rule: GeneralizedExperience) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT INTO generalized_experiences(id,payload_json) VALUES(?,?)",
                (rule.id, _rule_json(rule)),
            )
            self._event(rule.id, "created", rule.created_batch, rule)

    def get_rule(self, rule_id: str) -> GeneralizedExperience:
        row = self.connection.execute(
            "SELECT payload_json FROM generalized_experiences WHERE id=?", (rule_id,)
        ).fetchone()
        if row is None:
            raise KeyError(rule_id)
        return GeneralizedExperience.model_validate_json(row[0])

    def list_rules(self, *, status: str | None = None) -> list[GeneralizedExperience]:
        rules = [GeneralizedExperience.model_validate_json(row[0]) for row in self.connection.execute(
            "SELECT payload_json FROM generalized_experiences ORDER BY id").fetchall()]
        return rules if status is None else [rule for rule in rules if rule.status == status]

    def update_rule(self, rule: GeneralizedExperience, *, event_type: str, batch_id: int) -> None:
        with self.connection:
            changed = self.connection.execute(
                "UPDATE generalized_experiences SET payload_json=? WHERE id=?",
                (_rule_json(rule), rule.id),
            ).rowcount
            if not changed:
                raise KeyError(rule.id)
            self._event(rule.id, event_type, batch_id, rule)

    def add_evidence(self, experience_id: str, case_id: str, *, relation: str,
                     batch_id: int, attribution_id: str | None = None) -> bool:
        if relation not in {"support", "contradiction"}:
            raise ValueError("relation must be support or contradiction")
        try:
            with self.connection:
                self.connection.execute(
                    "INSERT INTO generalized_experience_evidence(experience_id,case_id,attribution_id,relation,batch_id) VALUES(?,?,?,?,?)",
                    (experience_id, case_id, attribution_id, relation, batch_id),
                )
                rule = self.get_rule(experience_id)
                batches = tuple(sorted(set(rule.supporting_batches) | ({batch_id} if relation == "support" else set())))
                updated = rule.model_copy(update={
                    "support_count": rule.support_count + int(relation == "support"),
                    "contradiction_count": rule.contradiction_count + int(relation == "contradiction"),
                    "supporting_batches": batches,
                    "last_updated_batch": batch_id,
                    "version": rule.version + 1,
                })
                self.connection.execute(
                    "UPDATE generalized_experiences SET payload_json=? WHERE id=?",
                    (_rule_json(updated), experience_id),
                )
                self._event(experience_id, "reinforced" if relation == "support" else "contradicted",
                            batch_id, updated)
            return True
        except sqlite3.IntegrityError:
            return False

    def record_outcome(self, experience_id: str, *, sample_id: str,
                       batch_id: int, correct: bool) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT INTO generalized_experience_outcomes(experience_id,sample_id,batch_id,correct) VALUES(?,?,?,?)",
                (experience_id, sample_id, batch_id, int(correct)),
            )

    def history(self, experience_id: str) -> list[dict]:
        rows = self.connection.execute(
            "SELECT id,event_type,batch_id,payload_json,created_at FROM generalized_experience_events WHERE experience_id=? ORDER BY id",
            (experience_id,),
        ).fetchall()
        return [{"id": row[0], "event_type": row[1], "batch_id": row[2],
                 "payload": json.loads(row[3]), "created_at": row[4]} for row in rows]

    def stats(self) -> dict[str, int]:
        def count(table: str) -> int:
            return int(self.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        relations = dict(self.connection.execute(
            "SELECT relation,COUNT(*) FROM generalized_experience_evidence GROUP BY relation").fetchall())
        rules = self.list_rules()
        return {
            "case_count": count("case_evidence"),
            "attribution_count": count("attributions"),
            "candidate_count": sum(rule.status == "candidate" for rule in rules),
            "active_count": sum(rule.status == "active" for rule in rules),
            "conflicted_count": sum(rule.status == "conflicted" for rule in rules),
            "suppressed_count": sum(rule.status == "suppressed" for rule in rules),
            "support_count": int(relations.get("support", 0)),
            "contradiction_count": int(relations.get("contradiction", 0)),
        }

    def _event(self, experience_id: str, event_type: str, batch_id: int,
               rule: GeneralizedExperience) -> None:
        self.connection.execute(
            "INSERT INTO generalized_experience_events(experience_id,event_type,batch_id,payload_json) VALUES(?,?,?,?)",
            (experience_id, event_type, batch_id, _rule_json(rule)),
        )

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> EvolutionRepository:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
