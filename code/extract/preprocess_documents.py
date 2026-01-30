"""
Document Preprocessing Pipeline for NEPA Timeline Extraction

This module provides preprocessing functions to reduce document text before
sending to an LLM for timeline extraction. The goal is to reduce token count
while preserving timeline-relevant information (decision dates, application dates).

Usage:
    from preprocess_documents import preprocess_for_timeline

    result = preprocess_for_timeline(document_text)
    # result['original_text'] - full original text for validation
    # result['processed_text'] - extracted snippets for LLM
    # result['snippets'] - list of extracted snippet details
    # result['reduction_pct'] - percentage reduction achieved

Author: Created for NEPA timeline extraction project
Date: 2026-01-28
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


# =============================================================================
# EXTRACTION KEYWORDS BY CATEGORY
# =============================================================================

EXTRACTION_KEYWORDS = {
    # -------------------------------------------------------------------------
    # Decision/Approval signatures (HIGHEST PRIORITY - usually has decision date)
    # -------------------------------------------------------------------------
    'decision_signatures': [
        r'Authorizing\s+Official',
        r'Field\s+(?:Office\s+)?Manager',
        r'NEPA\s+Compliance\s+Officer',
        r'Environmental\s+Coordinator',
        r'Digitally\s+signed',
        r'D\.\s*Signature',
        r'FIELD\s+OFFICE\s+MANAGER\s+DETERMINATION',
        r'Program\s+Lead\s*\[?Signature',
        r'Reviewing\s+Official',
        r'Approving\s+Official',
        r'District\s+Manager',
        r'Forest\s+Supervisor',
        r'Regional\s+Forester',
    ],

    # -------------------------------------------------------------------------
    # Date labels (often precede dates directly)
    # -------------------------------------------------------------------------
    'date_labels': [
        r'Date\s+Determined',
        r'Date\s+Approved',
        r'Approval\s+Date',
        r'Effective\s+Date',
        r'Expiration\s+Date',
        r'Decision\s+Date',
        r'Date\s+of\s+Decision',
        r'Signed\s+Date',
        r'Issue\s+Date',
        r'Date\s+Issued',
        r'Date\s+Signed',
        r'Revised\s*:',
        r'Amended\s*:',
    ],

    # -------------------------------------------------------------------------
    # Section headers (indicate decision-relevant sections)
    # -------------------------------------------------------------------------
    'section_headers': [
        r'DECISION\s+RECORD',
        r'NEPA\s+DETERMINATION',
        r'NEPA\s+PROVISION',
        r'CATEGORICAL\s+EXCLUSION\s+DOCUMENTATION',
        r'CATEGORICAL\s+EXCLUSION\s+WORKSHEET',
        r'CATEGORICAL\s+EXCLUSION\s+REVIEW',
        r'C\.\s+Compliance\s+with\s+NEPA',
        r'COMPLIANCE\s+WITH\s+NEPA',
        r'A\.\s+Background',
        r'FINDING\s+OF\s+NO\s+SIGNIFICANT\s+IMPACT',
        r'RECORD\s+OF\s+DECISION',
    ],

    # -------------------------------------------------------------------------
    # Application/start date indicators
    # -------------------------------------------------------------------------
    'application_indicators': [
        r'application\s+(?:received|submitted|date)',
        r'received\s+on',
        r'submitted\s+on',
        r'Proposal\s+Date',
        r'Project\s+(?:Start|Initiation|Began)',
        r'Date\s+(?:Received|Submitted)',
        r'Application\s+Date',
        r'Request\s+(?:Date|Received)',
        r'(?:APD|Application)\s+received',
    ],

    # -------------------------------------------------------------------------
    # Plan conformance (RMP dates - provides context)
    # -------------------------------------------------------------------------
    'plan_dates': [
        r'RMP.*?Date\s+Approved',
        r'Resource\s+Management\s+Plan.*?(?:approved|Approved)',
        r'Land\s+Use\s+Plan.*?(?:approved|Approved)',
        r'Record\s+of\s+Decision.*?\d{4}',
    ],

    # -------------------------------------------------------------------------
    # Agency-specific terms
    # -------------------------------------------------------------------------
    'agency_specific': [
        # DOE (Department of Energy)
        r'DOE\s+NEPA',
        r'DOE\s+Order\s+451',
        r'EERE\s+PROJECT',
        r'NETL',
        r'Savannah\s+River',

        # BLM (Bureau of Land Management)
        r'BLM\s+Office\s*:',
        r'BUREAU\s+OF\s+LAND\s+MANAGEMENT',
        r'Case\s+File\s+No',
        r'Serial\s+(?:No|Number)',
        r'516\s+DM',

        # USFS (US Forest Service)
        r'USDA\s+Forest\s+Service',
        r'National\s+Forest',
        r'Ranger\s+District',

        # USACE (Army Corps of Engineers)
        r'Army\s+Corps',
        r'USACE',

        # USFWS (Fish and Wildlife Service)
        r'Fish\s+and\s+Wildlife\s+Service',
        r'USFWS',

        # General federal
        r'DEPARTMENT\s+OF\s+THE\s+INTERIOR',
        r'DEPARTMENT\s+OF\s+ENERGY',
        r'DEPARTMENT\s+OF\s+AGRICULTURE',
    ],

    # -------------------------------------------------------------------------
    # Project type indicators
    # -------------------------------------------------------------------------
    'project_types': [
        r'Right-of-Way',
        r'ROW\s+(?:Grant|Application|Amendment)',
        r'(?:Special\s+)?(?:Use\s+)?Permit',
        r'Lease\s+(?:Application|Amendment)',
        r'Land\s+Use\s+(?:Permit|Authorization)',
        r'Easement',
        r'License\s+(?:Application|Amendment)',
        r'(?:Oil|Gas|Mining)\s+(?:Lease|Permit|Application)',
        r'APD',  # Application for Permit to Drill
        r'Sundry\s+Notice',
        r'(?:Renewable|Solar|Wind)\s+Energy',
        r'Transmission\s+(?:Line|Project)',
        r'Pipeline',
    ],

    # -------------------------------------------------------------------------
    # Fiscal year and alternative date formats
    # -------------------------------------------------------------------------
    'date_formats': [
        r'FY\s*\d{2,4}',
        r'Fiscal\s+Year\s+\d{4}',
        r'Calendar\s+Year\s+\d{4}',
        r'(?:Winter|Spring|Summer|Fall)\s+\d{4}',
        r'Q[1-4]\s+\d{4}',
    ],
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ExtractedSnippet:
    """A single extracted text snippet with metadata."""
    text: str
    start_pos: int
    end_pos: int
    categories: List[str]
    keywords: List[str]

    @property
    def position_pct(self) -> float:
        """Position as percentage through document (requires doc_length)."""
        return 0.0  # Set externally


@dataclass
class PreprocessResult:
    """Result of preprocessing a document."""
    original_text: str
    processed_text: str
    snippets: List[ExtractedSnippet]
    reduction_pct: float
    keywords_found: Dict[str, int]

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'original_text': self.original_text,
            'processed_text': self.processed_text,
            'snippets': [
                {
                    'text': s.text,
                    'start_pos': s.start_pos,
                    'end_pos': s.end_pos,
                    'categories': s.categories,
                    'keywords': s.keywords,
                }
                for s in self.snippets
            ],
            'reduction_pct': self.reduction_pct,
            'keywords_found': self.keywords_found,
        }


# =============================================================================
# CORE PREPROCESSING FUNCTIONS
# =============================================================================

def extract_snippets(
    text: str,
    window_size: int = 250,
    keywords: Optional[Dict[str, List[str]]] = None,
) -> List[ExtractedSnippet]:
    """
    Extract text snippets around keyword matches.

    Args:
        text: The full document text
        window_size: Characters to include before and after each keyword match
        keywords: Dictionary of keyword categories and patterns.
                  Defaults to EXTRACTION_KEYWORDS if not provided.

    Returns:
        List of ExtractedSnippet objects, merged to avoid overlaps
    """
    if keywords is None:
        keywords = EXTRACTION_KEYWORDS

    # Collect all matches
    matches = []
    for category, patterns in keywords.items():
        for pattern in patterns:
            try:
                for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
                    matches.append({
                        'category': category,
                        'keyword': match.group(),
                        'position': match.start(),
                        'start': max(0, match.start() - window_size),
                        'end': min(len(text), match.end() + window_size),
                    })
            except re.error:
                # Skip invalid regex patterns
                continue

    if not matches:
        return []

    # Sort by position
    matches = sorted(matches, key=lambda x: x['position'])

    # Merge overlapping windows
    merged = []
    for m in matches:
        if merged and m['start'] <= merged[-1]['end']:
            # Merge with previous
            merged[-1]['end'] = max(merged[-1]['end'], m['end'])
            merged[-1]['categories'].add(m['category'])
            merged[-1]['keywords'].append(m['keyword'])
        else:
            merged.append({
                'start': m['start'],
                'end': m['end'],
                'categories': {m['category']},
                'keywords': [m['keyword']],
            })

    # Convert to ExtractedSnippet objects
    snippets = []
    for m in merged:
        snippet_text = text[m['start']:m['end']].strip()
        # Clean up excessive whitespace
        snippet_text = re.sub(r'\n{3,}', '\n\n', snippet_text)
        snippet_text = re.sub(r' {3,}', '  ', snippet_text)

        snippets.append(ExtractedSnippet(
            text=snippet_text,
            start_pos=m['start'],
            end_pos=m['end'],
            categories=list(m['categories']),
            keywords=m['keywords'],
        ))

    return snippets


def combine_snippets(snippets: List[ExtractedSnippet], separator: str = "\n\n[...]\n\n") -> str:
    """
    Combine extracted snippets into a single text string for LLM input.

    Args:
        snippets: List of ExtractedSnippet objects
        separator: Text to insert between snippets

    Returns:
        Combined text string
    """
    if not snippets:
        return ""
    return separator.join(s.text for s in snippets)


def preprocess_for_timeline(
    text: str,
    window_size: int = 250,
    min_output_chars: int = 200,
    max_output_chars: int = 8000,
    fallback_to_truncation: bool = True,
) -> PreprocessResult:
    """
    Main preprocessing function for timeline extraction.

    This function extracts relevant snippets from a document while preserving
    the original text for validation.

    Args:
        text: The full document text
        window_size: Characters to include around each keyword match
        min_output_chars: If extracted text is below this, use fallback
        max_output_chars: Maximum characters to return (truncate if exceeded)
        fallback_to_truncation: If True, fall back to truncation when extraction fails

    Returns:
        PreprocessResult with original and processed text
    """
    original_text = text
    doc_length = len(text)

    # Extract snippets
    snippets = extract_snippets(text, window_size=window_size)

    # Count keywords found by category
    keywords_found = {}
    for snippet in snippets:
        for cat in snippet.categories:
            keywords_found[cat] = keywords_found.get(cat, 0) + 1

    # Combine snippets
    processed_text = combine_snippets(snippets)

    # Fallback: if extraction didn't find enough, use truncation
    if len(processed_text) < min_output_chars and fallback_to_truncation:
        # Take first N characters (usually contains cover page with key info)
        fallback_chars = min(max_output_chars, doc_length)
        processed_text = text[:fallback_chars]
        if fallback_chars < doc_length:
            processed_text += "\n\n[... document truncated ...]"

    # Truncate if too long
    if len(processed_text) > max_output_chars:
        processed_text = processed_text[:max_output_chars]
        processed_text += "\n\n[... output truncated ...]"

    # Calculate reduction
    reduction_pct = 100 * (1 - len(processed_text) / doc_length) if doc_length > 0 else 0

    return PreprocessResult(
        original_text=original_text,
        processed_text=processed_text,
        snippets=snippets,
        reduction_pct=reduction_pct,
        keywords_found=keywords_found,
    )


def preprocess_batch(
    documents: List[Dict],
    text_column: str = 'page_text',
    id_column: str = 'document_id',
    **kwargs,
) -> List[Dict]:
    """
    Preprocess a batch of documents.

    Args:
        documents: List of document dictionaries
        text_column: Name of the column containing document text
        id_column: Name of the column containing document ID
        **kwargs: Additional arguments passed to preprocess_for_timeline

    Returns:
        List of dictionaries with original and processed text
    """
    results = []
    for doc in documents:
        doc_id = doc.get(id_column, 'unknown')
        text = doc.get(text_column, '')

        result = preprocess_for_timeline(text, **kwargs)

        results.append({
            id_column: doc_id,
            'original_text': result.original_text,
            'processed_text': result.processed_text,
            'reduction_pct': result.reduction_pct,
            'n_snippets': len(result.snippets),
            'keywords_found': result.keywords_found,
        })

    return results


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_keyword_summary() -> Dict[str, int]:
    """Return a summary of all keyword patterns by category."""
    return {cat: len(patterns) for cat, patterns in EXTRACTION_KEYWORDS.items()}


def test_keywords_on_text(text: str) -> Dict[str, List[str]]:
    """
    Test which keywords match in a given text.
    Useful for debugging and validation.
    """
    matches = {}
    for category, patterns in EXTRACTION_KEYWORDS.items():
        category_matches = []
        for pattern in patterns:
            try:
                found = re.findall(pattern, text, re.IGNORECASE)
                if found:
                    category_matches.extend(found)
            except re.error:
                continue
        if category_matches:
            matches[category] = category_matches
    return matches


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == '__main__':
    import sys

    print("Document Preprocessing Pipeline")
    print("=" * 60)
    print("\nKeyword categories and pattern counts:")
    for cat, count in get_keyword_summary().items():
        print(f"  {cat}: {count} patterns")

    # Test with sample text if provided
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        print(f"\nTesting on: {filepath}")

        with open(filepath, 'r') as f:
            text = f.read()

        result = preprocess_for_timeline(text)

        print(f"\nOriginal length: {len(result.original_text):,} chars")
        print(f"Processed length: {len(result.processed_text):,} chars")
        print(f"Reduction: {result.reduction_pct:.1f}%")
        print(f"Snippets extracted: {len(result.snippets)}")
        print(f"\nKeywords found by category:")
        for cat, count in result.keywords_found.items():
            print(f"  {cat}: {count}")

        print(f"\n{'='*60}")
        print("PROCESSED TEXT:")
        print("=" * 60)
        print(result.processed_text[:2000])
        if len(result.processed_text) > 2000:
            print(f"\n... [{len(result.processed_text) - 2000} more chars]")
