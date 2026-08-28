from __future__ import annotations

import json
import re
from typing import Any

from haystack import component

from .common import Context, Stage, evidence


CLAUSE_LABELS = [
    "term",
    "termination",
    "liability",
    "indemnity",
    "governing_law",
    "assignment",
    "security",
    "dpa",
    "sla",
    "payment",
]


def _section_map(text: str) -> dict[str, str]:
    matches = list(re.finditer(r"^\[([A-Z_]+)\]\s*$", text, flags=re.MULTILINE))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[match.group(1).lower()] = text[start:end].strip()
    return sections


@component
class ClauseExtractor(Stage):
    model_role = "text"

    @component.output_types(context=Context)
    def run(self, context: Context) -> dict[str, Context]:
        def extract(ctx: Context) -> None:
            text = ctx.get("normalized_text", "")
            if ctx.get("extraction_attempt", 1) > 1:
                text = ctx.get("raw_text", text)
                ctx["normalized_text"] = text
            response = self.models.call_json("text",
                system=(
                    "Extract the requested contract clauses. Return a JSON object keyed by clause label; "
                    "each value must contain source text or null."
                ),
                prompt=json.dumps({"labels": CLAUSE_LABELS, "text": text[:80_000]}),
            )
            raw_clauses: dict[str, Any] | None = None
            if response and isinstance(response.get("clauses"), dict):
                raw_clauses = response["clauses"]
            elif response and any(label in response for label in CLAUSE_LABELS):
                raw_clauses = {label: response.get(label) for label in CLAUSE_LABELS}
            if raw_clauses is not None:
                sections = {}
                for key, value in raw_clauses.items():
                    if isinstance(value, dict):
                        value = value.get("source_text") or value.get("text") or value.get("evidence")
                    if value:
                        sections[key] = str(value)
            else:
                sections = _section_map(text)
            clauses: list[dict[str, Any]] = []
            for label in CLAUSE_LABELS:
                snippet = sections.get(label)
                if snippet:
                    clauses.append(
                        {
                            "label": label,
                            "summary": snippet[:500],
                            "evidence": evidence(ctx, label, snippet, confidence=0.94),
                        }
                    )
            ctx["clauses"] = clauses
            ctx["clause_map"] = {clause["label"]: clause for clause in clauses}
            ctx["missing_clauses"] = [label for label in CLAUSE_LABELS if label not in ctx["clause_map"]]

        return {"context": self.execute(context, "clause_extractor", extract)}


def _integer(pattern: str, text: str) -> int | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    value = next((group for group in match.groups() if group is not None), None)
    return int(value) if value is not None else None


def _as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    match = re.search(r"\d+", str(value))
    return int(match.group()) if match else None


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool) or value is None:
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "yes", "present", "required"}:
        return True
    if normalized in {"false", "no", "absent", "not present"}:
        return False
    return None


def _as_risk(value: Any) -> str | None:
    normalized = str(value).lower() if value is not None else None
    return normalized if normalized in {"low", "medium", "high"} else None


def _canonical_live_terms(raw: dict[str, Any]) -> dict[str, Any]:
    term = raw.get("term") if isinstance(raw.get("term"), dict) else {}
    termination = raw.get("termination") if isinstance(raw.get("termination"), dict) else {}
    liability = raw.get("liability") if isinstance(raw.get("liability"), dict) else {}
    indemnity = raw.get("indemnity") if isinstance(raw.get("indemnity"), dict) else {}
    security = raw.get("security") if isinstance(raw.get("security"), dict) else {}
    sla = raw.get("sla") if isinstance(raw.get("sla"), dict) else {}
    availability = sla.get("availability")
    if isinstance(availability, (int, float)):
        availability = f"{availability}%"
    elif availability is not None:
        availability = str(availability)
    payment_terms = raw.get("payment_terms")
    if payment_terms is not None and not isinstance(payment_terms, str):
        payment_terms = str(payment_terms)
    payment_net_days = _as_int(raw.get("payment_net_days"))
    if payment_net_days is None and payment_terms:
        payment_net_days = _integer(r"net\s*(\d+)", payment_terms)
    return {
        "term": {
            "initial_months": _as_int(term.get("initial_months")),
            "auto_renewal": _as_bool(term.get("auto_renewal")),
            "renewal_months": _as_int(term.get("renewal_months")),
            "notice_days": _as_int(term.get("notice_days")),
        },
        "termination": {
            "for_cause": _as_bool(termination.get("for_cause")),
            "for_convenience": _as_bool(termination.get("for_convenience")),
            "notice_days": _as_int(termination.get("notice_days")),
        },
        "governing_law": str(raw["governing_law"]) if raw.get("governing_law") is not None else None,
        "assignment_restricted": _as_bool(raw.get("assignment_restricted")),
        "liability": {
            "cap_present": _as_bool(liability.get("cap_present")),
            "cap_description": str(liability["cap_description"]) if liability.get("cap_description") else None,
            "risk": _as_risk(liability.get("risk")),
        },
        "indemnity": {
            "summary": str(indemnity["summary"]) if indemnity.get("summary") else None,
            "risk": _as_risk(indemnity.get("risk")),
        },
        "security": {
            "security_clause_present": _as_bool(security.get("security_clause_present")),
            "dpa_language_present": _as_bool(security.get("dpa_language_present")),
        },
        "sla": {
            "availability": availability,
            "remedy_present": _as_bool(sla.get("remedy_present")),
        },
        "payment_terms": payment_terms,
        "payment_net_days": payment_net_days,
        "unusual_prepayment": bool(_as_bool(raw.get("unusual_prepayment"))),
        "unlimited_unilateral_indemnity": bool(
            _as_bool(raw.get("unlimited_unilateral_indemnity"))
        ),
    }


@component
class StructuredTermNormalizer(Stage):
    model_role = "text"

    @component.output_types(context=Context)
    def run(self, context: Context) -> dict[str, Context]:
        def normalize(ctx: Context) -> None:
            clauses = {key: value["summary"] for key, value in ctx.get("clause_map", {}).items()}
            response = self.models.call_json("text",
                system=(
                    "Normalize contract clauses into JSON with exactly one top-level key named terms. "
                    "terms must contain: term {initial_months, auto_renewal, renewal_months, notice_days}; "
                    "termination {for_cause, for_convenience, notice_days}; governing_law; "
                    "assignment_restricted; liability {cap_present, cap_description, risk}; "
                    "indemnity {summary, risk}; security {security_clause_present, dpa_language_present}; "
                    "sla {availability, remedy_present}; payment_terms; payment_net_days; "
                    "unusual_prepayment; unlimited_unilateral_indemnity. Convert years to months. "
                    "Use null when unresolved and return JSON only."
                ),
                prompt=json.dumps(clauses),
            )
            if response and isinstance(response.get("terms"), dict):
                ctx["terms"] = _canonical_live_terms(response["terms"])
                return
            term = clauses.get("term", "")
            termination = clauses.get("termination", "")
            liability = clauses.get("liability", "")
            indemnity = clauses.get("indemnity", "")
            security = clauses.get("security", "")
            dpa = clauses.get("dpa", "")
            sla = clauses.get("sla", "")
            payment = clauses.get("payment", "")
            governing = clauses.get("governing_law", "")
            assignment = clauses.get("assignment", "")
            availability = re.search(r"(\d{2,3}(?:\.\d+)?)%", sla)
            net_days = _integer(r"net\s*(\d+)", payment)
            ctx["terms"] = {
                "term": {
                    "initial_months": _integer(r"initial term(?: is|:)?\s*(\d+)\s*months", term),
                    "auto_renewal": bool(re.search(r"auto(?:matic(?:ally)?)?[- ]?renew", term, re.I)),
                    "renewal_months": _integer(r"renew(?:s|al)?(?: for|:)?\s*(\d+)\s*months", term),
                    "notice_days": _integer(r"notice(?: of)?\s*(\d+)\s*days|(?:at least\s*)?(\d+)\s*days[^.]*notice", term),
                },
                "termination": {
                    "for_cause": "for cause" in termination.lower(),
                    "for_convenience": "for convenience" in termination.lower(),
                    "notice_days": _integer(r"(\d+)\s*days(?:'| of)?\s*notice", termination),
                },
                "governing_law": (
                    "Delaware" if "delaware" in governing.lower() else
                    "New York" if "new york" in governing.lower() else
                    governing.strip() or None
                ),
                "assignment_restricted": (
                    None if not assignment else not bool(re.search(r"freely|without (?:consent|restriction)", assignment, re.I))
                ),
                "liability": {
                    "cap_present": bool(liability) and "unlimited" not in liability.lower(),
                    "cap_description": liability or None,
                    "risk": None,
                },
                "indemnity": {"summary": indemnity or None, "risk": None},
                "security": {
                    "security_clause_present": bool(security),
                    "dpa_language_present": bool(dpa),
                },
                "sla": {
                    "availability": f"{availability.group(1)}%" if availability else None,
                    "remedy_present": bool(re.search(r"service credit|remed", sla, re.I)),
                },
                "payment_terms": f"Net {net_days}" if net_days is not None else None,
                "payment_net_days": net_days,
                "unusual_prepayment": bool(
                    re.search(r"prepay|prepayment|annual advance", payment, re.I)
                    and not re.search(r"no\s+(?:advance\s+)?prepayment|prepayment\s+is\s+not\s+required", payment, re.I)
                ),
                "unlimited_unilateral_indemnity": bool(
                    re.search(r"customer.*unlimited|unlimited.*customer", indemnity, re.I | re.S)
                ),
            }
            # Regex with two alternative capture groups needs a small correction.
            notice_match = re.search(
                r"notice(?: of)?\s*(\d+)\s*days|(?:at least\s*)?(\d+)\s*days[^.]*notice",
                term,
                re.I,
            )
            if notice_match:
                ctx["terms"]["term"]["notice_days"] = int(notice_match.group(1) or notice_match.group(2))

        return {"context": self.execute(context, "structured_term_normalizer", normalize)}
