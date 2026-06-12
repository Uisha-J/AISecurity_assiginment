"""Trial protocol construction for the ASV-bypass evaluation.

Single source of truth for *who is enrolled* and *who is tested*. Turns a pool
of speaker utterances (+ optional cloned audio) into the explicit 4-population
protocol the proposal requires:

    enrollment      victim's registration utterance
    trial_genuine   victim's *other* utterance      label "target"     expect ACCEPT
    trial_impostor  a *different* speaker's utt.     label "nontarget"  expect REJECT
    trial_spoof     cloned audio of the victim       label "spoof"      expect REJECT

Design principle — speaker disjointness: the impostor test speakers never
overlap with the victim being enrolled, so impostor trials are genuine
zero-effort impostors. Genuine/impostor trials calibrate the ASV operating
point (see ``attack.verify.calibrate``); spoof trials drive the tandem attack
evaluation (see ``pipeline.tandem``).

The module is pure-Python (no torch / no models) so the protocol can be built
and unit-tested with synthetic data.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Mapping, Optional, Sequence

# label/kind vocabulary -------------------------------------------------------
KIND_GENUINE = "genuine"
KIND_IMPOSTOR = "impostor"
KIND_SPOOF = "spoof"

LABEL_FOR_KIND = {
    KIND_GENUINE: "target",
    KIND_IMPOSTOR: "nontarget",
    KIND_SPOOF: "spoof",
}
# ASV ground-truth: 1 == should be accepted (target), 0 == should be rejected.
ASV_TARGET_FOR_KIND = {KIND_GENUINE: 1, KIND_IMPOSTOR: 0, KIND_SPOOF: 0}


@dataclass
class Trial:
    """One enroll/test pair."""
    enroll_path: str
    test_path: str
    target_speaker: str          # the enrolled (victim) speaker
    test_speaker: str            # speaker the test utterance actually belongs to
    kind: str                    # genuine | impostor | spoof
    label: str = ""              # target | nontarget | spoof  (derived)

    def __post_init__(self) -> None:
        if not self.label:
            self.label = LABEL_FOR_KIND[self.kind]

    @property
    def asv_target(self) -> int:
        """1 if ASV *should* accept this trial, else 0."""
        return ASV_TARGET_FOR_KIND[self.kind]


@dataclass
class TrialSet:
    trials: list[Trial] = field(default_factory=list)

    # -- filtered views -------------------------------------------------------
    def of_kind(self, kind: str) -> list[Trial]:
        return [t for t in self.trials if t.kind == kind]

    def genuine(self) -> list[Trial]:
        return self.of_kind(KIND_GENUINE)

    def impostor(self) -> list[Trial]:
        return self.of_kind(KIND_IMPOSTOR)

    def spoof(self) -> list[Trial]:
        return self.of_kind(KIND_SPOOF)

    def enrolled_speakers(self) -> list[str]:
        return sorted({t.target_speaker for t in self.trials})

    def summary(self) -> dict:
        return {
            "n_trials": len(self.trials),
            "n_genuine": len(self.genuine()),
            "n_impostor": len(self.impostor()),
            "n_spoof": len(self.spoof()),
            "n_enrolled_speakers": len(self.enrolled_speakers()),
        }

    # -- serialization --------------------------------------------------------
    _FIELDS = ["enroll_path", "test_path", "target_speaker",
               "test_speaker", "kind", "label"]

    def to_csv(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self._FIELDS)
            w.writeheader()
            for t in self.trials:
                w.writerow({k: getattr(t, k) for k in self._FIELDS})
        return path

    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump([asdict(t) for t in self.trials], f, indent=2)
        return path

    @classmethod
    def from_csv(cls, path: str | Path) -> "TrialSet":
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        trials = [
            Trial(
                enroll_path=r["enroll_path"], test_path=r["test_path"],
                target_speaker=r["target_speaker"], test_speaker=r["test_speaker"],
                kind=r["kind"], label=r.get("label", ""),
            )
            for r in rows
        ]
        return cls(trials=trials)


def build_trial_protocol(
    speaker_utterances: Mapping[str, Sequence[str]],
    cloned_by_speaker: Optional[Mapping[str, Sequence[str]]] = None,
    *,
    seed: int = 42,
    n_enroll_speakers: Optional[int] = None,
    n_genuine_per_speaker: int = 3,
    n_impostor_per_speaker: int = 5,
) -> TrialSet:
    """Build a speaker-disjoint 4-population trial protocol.

    Args:
        speaker_utterances: ``{speaker_id: [utt_path, ...]}``. Each enrolled
            speaker needs >= 2 utterances (one to enroll, one+ for genuine).
        cloned_by_speaker: ``{speaker_id: [cloned_wav, ...]}`` produced by the
            attack. Speakers present here get spoof trials.
        seed: RNG seed (reproducible sampling).
        n_enroll_speakers: cap on how many speakers to enroll (None = all eligible).
        n_genuine_per_speaker: genuine trials per victim.
        n_impostor_per_speaker: zero-effort impostor trials per victim.

    Returns:
        TrialSet with genuine/impostor/spoof trials.
    """
    rng = random.Random(seed)
    cloned_by_speaker = cloned_by_speaker or {}

    eligible = sorted(s for s, u in speaker_utterances.items() if len(u) >= 2)
    if not eligible:
        raise ValueError("Need at least one speaker with >= 2 utterances.")

    victims = eligible if n_enroll_speakers is None else eligible[:n_enroll_speakers]
    trials: list[Trial] = []

    for victim in victims:
        utts = list(speaker_utterances[victim])
        rng.shuffle(utts)
        enroll = utts[0]
        rest = utts[1:]

        # genuine: victim's own remaining utterances
        for test in rest[:n_genuine_per_speaker]:
            trials.append(Trial(enroll, test, victim, victim, KIND_GENUINE))

        # impostor: utterances from *other* speakers (speaker-disjoint)
        impostor_pool = [
            (spk, p)
            for spk in speaker_utterances
            if spk != victim
            for p in speaker_utterances[spk]
        ]
        rng.shuffle(impostor_pool)
        for spk, test in impostor_pool[:n_impostor_per_speaker]:
            trials.append(Trial(enroll, test, victim, spk, KIND_IMPOSTOR))

        # spoof: cloned audio of the victim
        for test in cloned_by_speaker.get(victim, []):
            trials.append(Trial(enroll, test, victim, victim, KIND_SPOOF))

    return TrialSet(trials=trials)
