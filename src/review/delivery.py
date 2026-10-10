"""Aggregate explicit operation incompleteness without interpreting scientific statuses."""

from schemas.review import DeliveryCheck, FinalReview

STAGES = ("materials", "screening", "verification", "execution", "assessment", "report", "teaser")


def checked_delivery(review: FinalReview, *, stages=None, extraction_status=None, additional_checks=(), **coverage):
    result = review.model_copy(deep=True)
    records = [*result.delivery_checks, *additional_checks]

    def count(item, key):
        value = item.get(key)
        return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0

    def add(stage, component, state, reason, claim_id=None):
        records.append(DeliveryCheck(stage=stage, component=component, state=state,
                                     reason=reason, claim_id=claim_id))

    for stage, state in (stages or {}).items():
        if stage in STAGES and state == "failed":
            add(stage, stage, "failed", "The requested stage failed.")
    if extraction_status == "failed":
        add("screening", "claim_extraction", "failed", "Claim extraction failed.")
    for name in ("claim_coverage", "writing_coverage", "figure_coverage", "table_coverage",
                 "figure_context_coverage", "table_context_coverage"):
        item = coverage.get(name)
        if not isinstance(item, dict):
            continue
        if name == "claim_coverage":
            incomplete = item.get("status") in {"failed", "partial", "incomplete"}
            incomplete |= item.get("status") == "not_run" and item.get("requested") is True
            for required, done in (("windows_total", "windows_reviewed"),
                                   ("claim_checks_required", "claim_checks_completed"),
                                   ("original_claim_reviews_required", "original_claim_reviews_completed")):
                incomplete |= count(item, required) > count(item, done)
        else:
            incomplete = any(count(item, key) > 0 for key in ("failed", "unavailable", "unrecorded"))
            if "context" not in name:
                incomplete |= count(item, "total") > count(item, "checked") + count(item, "not_applicable")
        if incomplete:
            add("screening", name, "incomplete", f"Requested {name} checks are incomplete.")
    for claim in result.claims:
        for limitation in claim.verification_limitations:
            # Rejected plans can describe absent author materials or policy choices.
            # Only execution stage failure is unambiguously operational here.
            if limitation.kind == "plan_rejected":
                continue
            if limitation.stage == "execution" and limitation.kind != "stage_failed":
                continue
            stage = ("screening" if limitation.kind == "claim_extraction_incomplete" else
                     "execution" if limitation.stage == "execution" else "verification")
            add(stage, limitation.stage + "." + limitation.kind, "unavailable",
                limitation.reason, claim.id)
        if result.advice_requested is True and (claim.advice is None or claim.advice.state == "unavailable"):
            add("report", "advice", "unavailable", "Requested claim advice is unavailable.", claim.id)
    unique = {}
    for record in records:
        unique.setdefault((record.stage, record.component, record.state, record.claim_id), record)
    result.delivery_checks = list(unique.values())
    incomplete = set(result.incomplete_stages) | {record.stage for record in result.delivery_checks}
    result.incomplete_stages = [stage for stage in STAGES if stage in incomplete]
    result.incomplete_stages.extend(sorted(incomplete - set(STAGES)))
    if result.incomplete_stages:
        result.run_status = "partial"
    return result


def delivery_lines(review):
    if not review.delivery_checks:
        return []
    return ["", "Requested operations not completed:", "", *[
        f"- {check.stage}/{check.component}"
        + (f" ({check.claim_id})" if check.claim_id else "")
        + f": {check.state}. {check.reason}"
        for check in review.delivery_checks
    ], ""]
