#!/usr/bin/env python3
"""
Compare LegalBERT vs E5 embedding models for document tagging.

This script runs the same documents through both models and compares:
- Semantic similarity scores
- Hybrid confidence scores
- Alignment with human feedback (confirmed/rejected tags)

Usage:
    cd /Users/jaredkaplan/Projects/back_tag_web/backend
    PYTHONPATH=. python -m app.scripts.compare_models [--limit N] [--verbose]

Example:
    PYTHONPATH=. python -m app.scripts.compare_models --limit 5
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

# Models to compare
MODELS = {
    "legalbert": "pile-of-law/legalbert-large-1.7M-2",
    "e5": "intfloat/e5-large-v2",
    "bge": "BAAI/bge-large-en-v1.5"
}


def load_models() -> Dict:
    """Load all models into memory."""
    from app.services.fast_tagger import load_model_fresh

    print("Loading models...")
    models = {}
    for name, model_id in MODELS.items():
        print(f"  Loading {name}: {model_id}")
        models[name] = load_model_fresh(model_id)
        print(f"    Embedding dimension: {models[name].get_sentence_embedding_dimension()}")

    return models


def load_taxonomy(db_session) -> Dict:
    """Load tag taxonomy from database."""
    from app.services.fast_tagger import load_taxonomy_tags

    print("\nLoading taxonomy...")
    tags = load_taxonomy_tags(db_session)
    print(f"  Loaded {len(tags)} tags")
    return tags


def compute_all_embeddings(models: Dict, tags: Dict) -> Dict[str, Dict]:
    """Compute tag embeddings for each model."""
    from app.services.fast_tagger import compute_embeddings_for_model

    print("\nComputing tag embeddings for each model...")
    embeddings = {}
    for name, model in models.items():
        print(f"  Computing embeddings for {name}...")
        embeddings[name] = compute_embeddings_for_model(model, tags)

    return embeddings


def get_documents_with_feedback(db_session, limit: int = None) -> List[Tuple]:
    """Get documents that have human feedback."""
    from app.database.db import Document, TagFeedback
    from sqlalchemy import func

    # Find documents with feedback (excluding special markers)
    query = db_session.query(
        Document,
        func.count(TagFeedback.id).label('feedback_count')
    ).join(
        TagFeedback, Document.id == TagFeedback.document_id
    ).filter(
        Document.status == 'completed',
        ~TagFeedback.tag_name.like('__%')  # Exclude special markers like __failed_reviewed__
    ).group_by(Document.id)

    if limit:
        query = query.limit(limit)

    results = query.all()
    print(f"\nFound {len(results)} documents with human feedback")
    return results


def get_completed_documents(db_session, limit: int = None) -> List:
    """Get completed documents (even without feedback) for raw comparison."""
    from app.database.db import Document

    query = db_session.query(Document).filter(
        Document.status == 'completed'
    ).order_by(Document.uploaded_at.desc())

    if limit:
        query = query.limit(limit)

    return query.all()


def get_feedback_for_document(db_session, doc_id: str) -> Dict[str, str]:
    """Get human feedback for a document: tag_name -> action (confirmed/rejected)."""
    from app.database.db import TagFeedback

    feedback = db_session.query(TagFeedback).filter(
        TagFeedback.document_id == doc_id,
        ~TagFeedback.tag_name.like('__%')
    ).all()

    return {fb.tag_name: fb.action for fb in feedback}


def compare_document(
    doc,
    models: Dict,
    embeddings: Dict[str, Dict],
    tags: Dict,
    db_session,
    verbose: bool = False
) -> Dict:
    """Compare model scores for a single document."""
    from app.services.fast_tagger import extract_text, score_text_with_model

    # Extract text
    try:
        text = extract_text(doc.filepath)
    except Exception as e:
        return {"error": str(e)}

    # Get human feedback
    feedback = get_feedback_for_document(db_session, doc.id)

    # Score with each model
    model_scores = {}
    for model_name, model in models.items():
        model_scores[model_name] = score_text_with_model(
            text, model, embeddings[model_name], tags
        )

    # Compare results
    comparison = {
        "document": doc.filename,
        "feedback": feedback,
        "tag_comparisons": [],
        "summary": {
            "e5_wins_confirmed": 0,
            "legalbert_wins_confirmed": 0,
            "ties_confirmed": 0,
            "e5_wins_rejected": 0,  # Lower score on rejected = win
            "legalbert_wins_rejected": 0,
        }
    }

    # Compare each tag with feedback
    for tag_name, action in feedback.items():
        lb_scores = model_scores.get("legalbert", {}).get(tag_name, {})
        e5_scores = model_scores.get("e5", {}).get(tag_name, {})

        if not lb_scores or not e5_scores:
            continue

        lb_sem = lb_scores.get("semantic_similarity", 0)
        e5_sem = e5_scores.get("semantic_similarity", 0)
        lb_hyb = lb_scores.get("hybrid_confidence", 0)
        e5_hyb = e5_scores.get("hybrid_confidence", 0)

        # Determine winner (for confirmed: higher is better; for rejected: lower is better)
        delta = e5_sem - lb_sem
        if action == "confirmed":
            if delta > 0.02:
                winner = "E5"
                comparison["summary"]["e5_wins_confirmed"] += 1
            elif delta < -0.02:
                winner = "LegalBERT"
                comparison["summary"]["legalbert_wins_confirmed"] += 1
            else:
                winner = "Tie"
                comparison["summary"]["ties_confirmed"] += 1
        else:  # rejected - lower is better
            if delta < -0.02:
                winner = "E5"  # E5 correctly scored lower
                comparison["summary"]["e5_wins_rejected"] += 1
            elif delta > 0.02:
                winner = "LegalBERT"  # LegalBERT correctly scored lower
                comparison["summary"]["legalbert_wins_rejected"] += 1
            else:
                winner = "Tie"

        comparison["tag_comparisons"].append({
            "tag": tag_name,
            "action": action,
            "legalbert_semantic": lb_sem,
            "e5_semantic": e5_sem,
            "delta": round(delta, 4),
            "winner": winner
        })

    return comparison


def print_document_comparison(comp: Dict, verbose: bool = False):
    """Print comparison results for a single document."""
    print(f"\n{'='*70}")
    print(f"Document: {comp['document']}")
    print(f"{'='*70}")

    if "error" in comp:
        print(f"  ERROR: {comp['error']}")
        return

    if not comp["tag_comparisons"]:
        print("  No tags with feedback to compare")
        return

    # Header
    print(f"{'Tag':<30} | {'LegalBERT':>9} | {'E5':>9} | {'Human':>8} | {'Winner':>10}")
    print("-" * 70)

    for tc in comp["tag_comparisons"]:
        human = "✓" if tc["action"] == "confirmed" else "✗"
        print(f"{tc['tag']:<30} | {tc['legalbert_semantic']*100:>8.1f}% | {tc['e5_semantic']*100:>8.1f}% | {human:>8} | {tc['winner']:>10}")

    # Summary
    s = comp["summary"]
    print("-" * 70)
    print(f"Confirmed tags: E5 wins {s['e5_wins_confirmed']}, LegalBERT wins {s['legalbert_wins_confirmed']}, Ties {s['ties_confirmed']}")


def print_overall_summary(all_comparisons: List[Dict]):
    """Print overall summary across all documents."""
    print(f"\n{'='*70}")
    print("OVERALL SUMMARY")
    print(f"{'='*70}")

    total_e5_wins_confirmed = 0
    total_lb_wins_confirmed = 0
    total_ties_confirmed = 0
    total_e5_wins_rejected = 0
    total_lb_wins_rejected = 0

    all_deltas = []
    confirmed_deltas = []
    rejected_deltas = []

    for comp in all_comparisons:
        if "error" in comp:
            continue

        s = comp.get("summary", {})
        total_e5_wins_confirmed += s.get("e5_wins_confirmed", 0)
        total_lb_wins_confirmed += s.get("legalbert_wins_confirmed", 0)
        total_ties_confirmed += s.get("ties_confirmed", 0)
        total_e5_wins_rejected += s.get("e5_wins_rejected", 0)
        total_lb_wins_rejected += s.get("legalbert_wins_rejected", 0)

        for tc in comp.get("tag_comparisons", []):
            all_deltas.append(tc["delta"])
            if tc["action"] == "confirmed":
                confirmed_deltas.append(tc["delta"])
            else:
                rejected_deltas.append(tc["delta"])

    total_confirmed = total_e5_wins_confirmed + total_lb_wins_confirmed + total_ties_confirmed
    total_rejected = total_e5_wins_rejected + total_lb_wins_rejected

    print(f"\nDocuments analyzed: {len(all_comparisons)}")
    print(f"Total tag comparisons: {len(all_deltas)}")

    if total_confirmed > 0:
        print(f"\n--- CONFIRMED TAGS (higher score = better) ---")
        print(f"  E5 wins:       {total_e5_wins_confirmed:>4} ({total_e5_wins_confirmed/total_confirmed*100:.1f}%)")
        print(f"  LegalBERT wins:{total_lb_wins_confirmed:>4} ({total_lb_wins_confirmed/total_confirmed*100:.1f}%)")
        print(f"  Ties:          {total_ties_confirmed:>4} ({total_ties_confirmed/total_confirmed*100:.1f}%)")

    if total_rejected > 0:
        print(f"\n--- REJECTED TAGS (lower score = better) ---")
        print(f"  E5 wins:       {total_e5_wins_rejected:>4}")
        print(f"  LegalBERT wins:{total_lb_wins_rejected:>4}")

    if all_deltas:
        avg_delta = sum(all_deltas) / len(all_deltas)
        print(f"\n--- SCORE DELTAS (E5 - LegalBERT) ---")
        print(f"  Average delta (all):       {avg_delta*100:+.2f}%")
        if confirmed_deltas:
            avg_conf = sum(confirmed_deltas) / len(confirmed_deltas)
            print(f"  Average delta (confirmed): {avg_conf*100:+.2f}%")
        if rejected_deltas:
            avg_rej = sum(rejected_deltas) / len(rejected_deltas)
            print(f"  Average delta (rejected):  {avg_rej*100:+.2f}%")

    # Final verdict
    print(f"\n{'='*70}")
    if total_confirmed > 0:
        e5_rate = total_e5_wins_confirmed / total_confirmed
        lb_rate = total_lb_wins_confirmed / total_confirmed

        if e5_rate > lb_rate + 0.1:
            print("VERDICT: E5 performs significantly better on confirmed tags")
        elif lb_rate > e5_rate + 0.1:
            print("VERDICT: LegalBERT performs significantly better on confirmed tags")
        else:
            print("VERDICT: Models perform similarly - consider other factors")
    print(f"{'='*70}")


def compare_document_raw(
    doc,
    models: Dict,
    embeddings: Dict[str, Dict],
    tags: Dict,
    threshold: float = 0.45
) -> Dict:
    """Compare model scores for a document without feedback - raw score comparison."""
    from app.services.fast_tagger import extract_text, score_text_with_model

    # Extract text
    try:
        text = extract_text(doc.filepath)
    except Exception as e:
        return {"error": str(e), "document": doc.filename}

    # Score with each model
    model_scores = {}
    for model_name, model in models.items():
        model_scores[model_name] = score_text_with_model(
            text, model, embeddings[model_name], tags
        )

    # Find tags where models disagree significantly
    comparison = {
        "document": doc.filename,
        "tag_scores": [],
        "above_threshold": {name: 0 for name in models.keys()},
        "model_names": list(models.keys())
    }

    # Get all tag names
    all_tags = set()
    for scores in model_scores.values():
        all_tags.update(scores.keys())

    for tag_name in sorted(all_tags):
        tag_data = {"tag": tag_name, "scores": {}}

        for model_name in models.keys():
            scores = model_scores.get(model_name, {}).get(tag_name, {})
            sem = scores.get("semantic_similarity", 0)
            tag_data["scores"][model_name] = sem

            if sem >= threshold:
                comparison["above_threshold"][model_name] += 1

        # Calculate deltas vs legalbert
        lb_sem = tag_data["scores"].get("legalbert", 0)
        tag_data["e5_delta"] = round(tag_data["scores"].get("e5", 0) - lb_sem, 4)
        tag_data["bge_delta"] = round(tag_data["scores"].get("bge", 0) - lb_sem, 4)

        # Include if any model shows significant difference or is above threshold
        max_delta = max(abs(tag_data["e5_delta"]), abs(tag_data["bge_delta"]))
        any_above = any(s >= threshold for s in tag_data["scores"].values())

        if max_delta > 0.03 or any_above:
            comparison["tag_scores"].append(tag_data)

    return comparison


def print_raw_comparison(comp: Dict):
    """Print raw score comparison for a document."""
    print(f"\n{'='*100}")
    print(f"Document: {comp['document']}")
    print(f"{'='*100}")

    if "error" in comp:
        print(f"  ERROR: {comp['error']}")
        return

    # Summary
    above = comp["above_threshold"]
    print(f"Tags above threshold (0.45): LegalBERT={above.get('legalbert', 0)}, E5={above.get('e5', 0)}, BGE={above.get('bge', 0)}")

    # Detailed scores - 3 models
    print(f"\n{'Tag':<28} | {'LegalBERT':>9} | {'E5':>9} | {'BGE':>9} | {'E5 Δ':>7} | {'BGE Δ':>7}")
    print("-" * 100)

    # Sort by max delta
    for ts in sorted(comp["tag_scores"], key=lambda x: -max(abs(x.get("e5_delta", 0)), abs(x.get("bge_delta", 0)))):
        scores = ts["scores"]
        lb = scores.get("legalbert", 0) * 100
        e5 = scores.get("e5", 0) * 100
        bge = scores.get("bge", 0) * 100
        e5_d = ts.get("e5_delta", 0) * 100
        bge_d = ts.get("bge_delta", 0) * 100

        print(f"{ts['tag']:<28} | {lb:>8.1f}% | {e5:>8.1f}% | {bge:>8.1f}% | {e5_d:>+6.1f}% | {bge_d:>+6.1f}%")


def print_raw_summary(all_comparisons: List[Dict]):
    """Print summary of raw comparisons."""
    print(f"\n{'='*100}")
    print("RAW COMPARISON SUMMARY - LegalBERT vs E5 vs BGE")
    print(f"{'='*100}")

    # Aggregate stats
    total_above = {"legalbert": 0, "e5": 0, "bge": 0}
    e5_deltas = []
    bge_deltas = []
    valid_docs = 0

    for comp in all_comparisons:
        if "error" in comp:
            continue
        valid_docs += 1

        above = comp.get("above_threshold", {})
        for model in total_above:
            total_above[model] += above.get(model, 0)

        for ts in comp.get("tag_scores", []):
            e5_deltas.append(ts.get("e5_delta", 0))
            bge_deltas.append(ts.get("bge_delta", 0))

    print(f"\nDocuments analyzed: {valid_docs}")
    print(f"\nTags above threshold (0.45):")
    print(f"  LegalBERT: {total_above['legalbert']}")
    print(f"  E5:        {total_above['e5']}")
    print(f"  BGE:       {total_above['bge']}")

    if e5_deltas:
        avg_e5 = sum(e5_deltas) / len(e5_deltas)
        avg_bge = sum(bge_deltas) / len(bge_deltas)

        e5_wins = sum(1 for d in e5_deltas if d > 0.02)
        bge_wins = sum(1 for d in bge_deltas if d > 0.02)
        lb_wins_vs_e5 = sum(1 for d in e5_deltas if d < -0.02)
        lb_wins_vs_bge = sum(1 for d in bge_deltas if d < -0.02)

        print(f"\n{'='*50}")
        print("SCORE DELTAS vs LegalBERT (baseline)")
        print(f"{'='*50}")
        print(f"\n{'Model':<12} | {'Avg Delta':>12} | {'Wins (>2%)':>12} | {'Loses':>10}")
        print("-" * 50)
        print(f"{'E5':<12} | {avg_e5*100:>+11.2f}% | {e5_wins:>12} | {lb_wins_vs_e5:>10}")
        print(f"{'BGE':<12} | {avg_bge*100:>+11.2f}% | {bge_wins:>12} | {lb_wins_vs_bge:>10}")

        # E5 vs BGE comparison
        e5_vs_bge_wins = sum(1 for e, b in zip(e5_deltas, bge_deltas) if e > b + 0.02)
        bge_vs_e5_wins = sum(1 for e, b in zip(e5_deltas, bge_deltas) if b > e + 0.02)

        print(f"\n{'='*50}")
        print("E5 vs BGE HEAD-TO-HEAD")
        print(f"{'='*50}")
        print(f"  E5 higher than BGE:  {e5_vs_bge_wins} tags")
        print(f"  BGE higher than E5:  {bge_vs_e5_wins} tags")
        print(f"  Similar (within 2%): {len(e5_deltas) - e5_vs_bge_wins - bge_vs_e5_wins} tags")

    print(f"\n{'='*100}")
    # Determine winner
    if e5_deltas and bge_deltas:
        avg_e5 = sum(e5_deltas) / len(e5_deltas)
        avg_bge = sum(bge_deltas) / len(bge_deltas)

        if avg_e5 > avg_bge + 0.02:
            print("OBSERVATION: E5 produces highest semantic scores overall")
        elif avg_bge > avg_e5 + 0.02:
            print("OBSERVATION: BGE produces highest semantic scores overall")
        else:
            print("OBSERVATION: E5 and BGE perform similarly, both higher than LegalBERT")

    print("Note: Higher scores don't guarantee accuracy - need human feedback to validate.")
    print(f"{'='*100}")


def main():
    parser = argparse.ArgumentParser(description="Compare embedding models for document tagging")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of documents to analyze")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output per document")
    parser.add_argument("--raw", action="store_true", help="Compare raw scores (don't require feedback)")
    args = parser.parse_args()

    # Setup database
    from app.database.db import get_db
    db = next(get_db())

    try:
        # Load models
        models = load_models()

        # Load taxonomy
        tags = load_taxonomy(db)

        # Compute embeddings for each model
        embeddings = compute_all_embeddings(models, tags)

        # Get documents with feedback first
        docs_with_feedback = get_documents_with_feedback(db, args.limit)

        if docs_with_feedback and not args.raw:
            # Compare with human feedback
            all_comparisons = []
            for doc, feedback_count in docs_with_feedback:
                print(f"\nProcessing: {doc.filename} ({feedback_count} feedback items)")
                comp = compare_document(doc, models, embeddings, tags, db, args.verbose)
                all_comparisons.append(comp)

                if args.verbose:
                    print_document_comparison(comp, args.verbose)

            print_overall_summary(all_comparisons)

        else:
            # No feedback or --raw flag: do raw comparison
            if not args.raw:
                print("\nNo documents with human feedback found.")
                print("Running raw score comparison instead (use --raw to skip this message).\n")

            docs = get_completed_documents(db, args.limit)
            if not docs:
                print("No completed documents found.")
                return

            print(f"\nComparing raw scores for {len(docs)} documents...")

            all_comparisons = []
            for doc in docs:
                print(f"  Processing: {doc.filename}")
                comp = compare_document_raw(doc, models, embeddings, tags)
                all_comparisons.append(comp)

                if args.verbose:
                    print_raw_comparison(comp)

            print_raw_summary(all_comparisons)

    finally:
        db.close()


if __name__ == "__main__":
    main()
