"""
Rhythm Oscillation: explicit control between theta (PFC) and gamma (HPC/ACC).

States:
- continue: ACC checks all pass; keep running gamma on next subquestion.
- retrieval: stay in gamma; run HPC path replay to improve retrieval.
- repair: switch to theta; repair only the current subquestion.
- replan: switch to theta; replan the full framework.
"""

from dataclasses import dataclass
from typing import List, Dict


STATE_CONTINUE = "continue"
STATE_RETRIEVAL = "retrieval"
STATE_REPAIR = "repair"
STATE_REPLAN = "replan"


ACTION_MAP = {
    STATE_CONTINUE: "gamma_continue",
    STATE_RETRIEVAL: "gamma_retrieval",
    STATE_REPAIR: "theta_repair",
    STATE_REPLAN: "theta_replan",
}


@dataclass
class RhythmOscillation:
    """
    System state controller based on ACC check results.
    """

    imax: int = 3
    retrieval_count: int = 0
    state: str = STATE_CONTINUE

    def reset(self) -> None:
        self.retrieval_count = 0
        self.state = STATE_CONTINUE

    def _normalize_checks(self, checks: List[int]) -> List[int]:
        if not isinstance(checks, list) or len(checks) != 3:
            raise ValueError("checks must be a list of three integers (0/1).")
        normed: List[int] = []
        for v in checks:
            if v not in (0, 1):
                raise ValueError("checks must contain only 0 or 1.")
            normed.append(int(v))
        return normed

    def update(self, checks: List[int]) -> Dict[str, int]:
        """
        Update state based on ACC checks and return the decision snapshot.
        """
        c1, c2, c3 = self._normalize_checks(checks)

        # All pass -> continue and reset retrieval counter
        if c1 == 1 and c2 == 1 and c3 == 1:
            self.state = STATE_CONTINUE
            self.retrieval_count = 0
            return self._snapshot("all_checks_pass")

        # Prioritize retrieval for missing core entity evidence
        if c1 == 0:
            self.retrieval_count += 1
            if self.retrieval_count > self.imax:
                self.state = STATE_REPLAN
                self.retrieval_count = 0
                return self._snapshot("retrieval_exceeded_imax")
            self.state = STATE_RETRIEVAL
            return self._snapshot("core_entity_missing_in_evidence")

        # Type mismatch -> repair the current subquestion
        if c2 == 0:
            self.state = STATE_REPAIR
            self.retrieval_count = 0
            return self._snapshot("expected_type_mismatch")

        # Reason mismatch -> treat as retrieval issue
        self.retrieval_count += 1
        if self.retrieval_count > self.imax:
            self.state = STATE_REPLAN
            self.retrieval_count = 0
            return self._snapshot("retrieval_exceeded_imax")
        self.state = STATE_RETRIEVAL
        return self._snapshot("reason_not_supported")

    def _snapshot(self, reason: str) -> Dict[str, int]:
        return {
            "state": self.state,
            "action": ACTION_MAP[self.state],
            "retrieval_count": self.retrieval_count,
            "imax": self.imax,
            "reason": reason,
        }
