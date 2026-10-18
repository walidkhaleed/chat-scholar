
import re


CITATION_PATTERN = re.compile(
    r"\[S(\d+)\]"
)

NUMBER_PATTERN = re.compile(
    r"(?<![\w.])-?\d+(?:[.,]\d+)?%?(?!\w)"
)


def extract_citations(answer: str) -> list[str]:
    """
    Extract unique source IDs in order of appearance.
    """
    return list(dict.fromkeys(
        f"S{number}"
        for number in CITATION_PATTERN.findall(answer)
    ))



def split_answer_sentences(answer: str) -> list[str]:
    """
    Split sentences while retaining trailing citations.
    """

    # Convert "Sentence. [S1]" into "Sentence [S1]."
    normalized = re.sub(
        r"([.!?])\s+(\[S\d+\])",
        r" \2\1",
        answer
    )

    return [
        sentence.strip()
        for sentence in re.split(
            r"(?<=[.!?])\s+|\n+",
            normalized
        )
        if sentence.strip()
    ]



def validate_citations(
    answer: str,
    sources: list
) -> dict:
    """
    Validate source references and perform basic
    evidence checks.

    This checks citation structure and numerical
    presence, not full semantic entailment.
    """

    source_map = {
        source["source_id"]: source
        for source in sources
    }

    cited_ids = extract_citations(answer)

    valid_ids = [
        source_id
        for source_id in cited_ids
        if source_id in source_map
    ]

    invalid_ids = [
        source_id
        for source_id in cited_ids
        if source_id not in source_map
    ]

    sentences = split_answer_sentences(answer)

    uncited_sentences = []

    for sentence in sentences:
        # Ignore short headings.
        if len(sentence.split()) < 5:
            continue

        if not CITATION_PATTERN.search(sentence):
            uncited_sentences.append(sentence)

    # Remove source markers before checking numbers.
    clean_answer = CITATION_PATTERN.sub("", answer)

    answer_numbers = set(
        NUMBER_PATTERN.findall(clean_answer)
    )

    evidence_text = " ".join(
        source["text"]
        for source in sources
    )

    evidence_numbers = set(
        NUMBER_PATTERN.findall(evidence_text)
    )

    unsupported_numbers = sorted(
        answer_numbers - evidence_numbers
    )

    warnings = []

    if invalid_ids:
        warnings.append(
            "The answer cites source IDs that were not retrieved."
        )

    if uncited_sentences:
        warnings.append(
            "Some answer sentences have no source citation."
        )

    if unsupported_numbers:
        warnings.append(
            "Some numerical values are absent from the "
            "retrieved excerpts."
        )

    if not cited_ids:
        warnings.append(
            "The answer contains no source citations."
        )

    citation_valid = (
        len(invalid_ids) == 0
        and len(cited_ids) > 0
    )

    evidence_checks_passed = (
        citation_valid
        and not uncited_sentences
        and not unsupported_numbers
    )

    return {
        "citation_valid": citation_valid,
        "evidence_checks_passed": evidence_checks_passed,
        "cited_source_ids": cited_ids,
        "valid_source_ids": valid_ids,
        "invalid_source_ids": invalid_ids,
        "uncited_sentences": uncited_sentences,
        "unsupported_numbers": unsupported_numbers,
        "warnings": warnings,
        "verification_level": (
            "structural_and_heuristic_only"
        )
    }
