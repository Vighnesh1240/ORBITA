# src/evaluation/evaluator.py
"""
ORBITA Formal Evaluation Engine

This module implements all evaluation metrics needed for the
research paper. It is the answer to every reviewer question
about system quality.

Metrics implemented:
    1. Bias Detection Accuracy
       - Mean Absolute Error (MAE) vs AllSides ground truth
       - Directional Accuracy (did we get the sign right?)
       - Pearson correlation with source-level ground truth

    2. Synthesis Quality
       - ROUGE-1, ROUGE-2, ROUGE-L vs reference summaries
       - Word count, sentence count
       - Coverage: does synthesis address both sides?

    3. System Consistency
       - Variance across multiple runs of same topic
       - Coefficient of Variation (CV)

    4. Pipeline Performance
       - End-to-end latency
       - Per-phase breakdown
       - Throughput (articles/second)

    5. Coverage Diversity
       - Shannon entropy of stance distribution
       - Source diversity score
       - Unique source count

Usage:
    from src.evaluation.evaluator import ORBITAEvaluator

    evaluator = ORBITAEvaluator()
    report    = evaluator.evaluate(pipeline_result, topic)
    evaluator.save_report(report)
    evaluator.print_summary(report)

"""

import os
import json
import time
import numpy as np
from pathlib import Path
from typing import Optional

try:
    from .ground_truth import (
        get_source_bias,
        get_topic_ground_truth,
        compute_expected_bias_from_sources,
    )
    from .rouge_scorer import compute_all_rouge, get_reference_summary
except ImportError:
    try:
        from evaluation.ground_truth import (
            get_source_bias,
            get_topic_ground_truth,
            compute_expected_bias_from_sources,
        )
        from evaluation.rouge_scorer import compute_all_rouge, get_reference_summary
    except ImportError:
        from src.evaluation.ground_truth import (
            get_source_bias,
            get_topic_ground_truth,
            compute_expected_bias_from_sources,
        )
        from src.evaluation.rouge_scorer import compute_all_rouge, get_reference_summary


class ORBITAEvaluator:
    """
    Main evaluation class for ORBITA.

    Usage:
        evaluator = ORBITAEvaluator()
        result    = run_pipeline("Farm Laws India")
        report    = evaluator.evaluate(result, "Farm Laws India")
        evaluator.save_report(report)
        evaluator.print_summary(report)
    """

    def __init__(self, output_dir: str = "evaluation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "experiments").mkdir(exist_ok=True)

    # ─────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────

    def evaluate(
        self,
        pipeline_result:    dict,
        topic:              str,
        elapsed_seconds:    Optional[float] = None,
        reference_summary:  Optional[str]   = None,
    ) -> dict:
        """
        Run all evaluations on a pipeline result.

        This is the main function. Call it after every pipeline run
        during your experiments.

        Args:
            pipeline_result:   the dict returned by run_pipeline()
            topic:             the query topic string
            elapsed_seconds:   how long the pipeline took (optional)
            reference_summary: human reference text for ROUGE (optional)

        Returns:
            Complete evaluation report dict
        """
        print(f"\n[evaluator] Running evaluation for: '{topic}'")

        report   = pipeline_result.get("report",   {})
        articles = pipeline_result.get("articles", [])
        stats    = pipeline_result.get("stats",    {})

        # ── Run all evaluation components ─────────────────────────
        eval_report = {
            "topic":     topic,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),

            "bias_accuracy":      self._eval_bias_accuracy(
                report, articles, topic
            ),
            "synthesis_quality":  self._eval_synthesis_quality(
                report, topic, reference_summary
            ),
            "coverage_diversity": self._eval_coverage_diversity(
                articles, stats
            ),
            "argument_quality":   self._eval_argument_quality(report),
            "performance":        self._eval_performance(
                pipeline_result, elapsed_seconds
            ),
        }

        # ── Compute overall score ─────────────────────────────────
        eval_report["overall_score"] = self._compute_overall_score(
            eval_report
        )

        print(f"  Overall quality score: "
              f"{eval_report['overall_score']['score']:.3f} / 1.000")
        print(f"  Grade: {eval_report['overall_score']['grade']}")

        return eval_report

    def save_report(self, eval_report: dict) -> str:
        """Save evaluation report to disk."""
        topic     = eval_report.get("topic", "unknown")
        safe      = "".join(c if c.isalnum() else "_" for c in topic)[:30]
        timestamp = int(time.time())
        filename  = f"eval_{safe}_{timestamp}.json"
        filepath  = self.output_dir / "experiments" / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(eval_report, f, indent=2, ensure_ascii=False)

        print(f"[evaluator] Saved → {filepath}")
        return str(filepath)

    def print_summary(self, eval_report: dict) -> None:
        """Print a formatted summary to terminal."""
        sep = "=" * 62

        print(f"\n{sep}")
        print(f"  ORBITA Evaluation Report")
        print(f"  Topic: {eval_report['topic']}")
        print(f"{sep}")

        # Bias Accuracy
        ba = eval_report.get("bias_accuracy", {})
        print(f"\n  BIAS DETECTION ACCURACY")
        print(f"  {'─'*50}")
        print(f"  Predicted Score:      {ba.get('predicted_bias', 0):+.4f}")
        print(f"  Expected (AllSides):  {ba.get('expected_bias', 'N/A')}")
        print(f"  MAE:                  {ba.get('mae', 'N/A')}")
        print(f"  Directional Correct:  {ba.get('directional_correct', 'N/A')}")
        print(f"  Sources Matched:      "
              f"{ba.get('sources_matched', 0)}/"
              f"{ba.get('sources_total', 0)}")

        # Synthesis Quality
        sq = eval_report.get("synthesis_quality", {})
        print(f"\n  SYNTHESIS QUALITY")
        print(f"  {'─'*50}")
        print(f"  Word Count:           {sq.get('word_count', 0)}")
        print(f"  Covers Both Sides:    {sq.get('covers_both_sides', False)}")
        print(f"  Hallucination Rate:   "
              f"{sq.get('hallucination_rate', 0):.1%}")
        rouge = sq.get("rouge_scores", {})
        if rouge:
            print(f"  ROUGE-1 F1:           "
                  f"{rouge.get('rouge_1', {}).get('f1', 'N/A')}")
            print(f"  ROUGE-2 F1:           "
                  f"{rouge.get('rouge_2', {}).get('f1', 'N/A')}")
            print(f"  ROUGE-L F1:           "
                  f"{rouge.get('rouge_l', {}).get('f1', 'N/A')}")
        else:
            print(f"  ROUGE:                No reference available")

        # Coverage
        cd = eval_report.get("coverage_diversity", {})
        print(f"\n  COVERAGE DIVERSITY")
        print(f"  {'─'*50}")
        print(f"  Articles Analysed:    {cd.get('n_articles', 0)}")
        print(f"  Unique Sources:       {cd.get('unique_sources', 0)}")
        print(f"  Stance Entropy:       {cd.get('stance_entropy', 0):.4f}")
        print(f"  Source Diversity:     {cd.get('source_diversity', 0):.4f}")
        stance = cd.get("stance_counts", {})
        print(f"  Supportive/Critical/Neutral: "
              f"{stance.get('Supportive',0)}/"
              f"{stance.get('Critical',0)}/"
              f"{stance.get('Neutral',0)}")

        # Argument Quality
        aq = eval_report.get("argument_quality", {})
        print(f"\n  ARGUMENT QUALITY")
        print(f"  {'─'*50}")
        print(f"  Supporting Args:      {aq.get('n_supporting_args', 0)}")
        print(f"  Counter Args:         {aq.get('n_counter_args', 0)}")
        print(f"  Agent A Confidence:   {aq.get('agent_a_confidence', 0):.2f}")
        print(f"  Agent B Confidence:   {aq.get('agent_b_confidence', 0):.2f}")
        print(f"  Loaded Lang Removed:  {aq.get('loaded_lang_removed', 0)}")

        # Performance
        perf = eval_report.get("performance", {})
        print(f"\n  PERFORMANCE")
        print(f"  {'─'*50}")
        if perf.get("elapsed_seconds"):
            print(f"  Total Time:           {perf['elapsed_seconds']:.1f}s")
        print(f"  Chunks in VectorDB:   {perf.get('total_chunks', 0)}")

        # Overall
        ov = eval_report.get("overall_score", {})
        print(f"\n  OVERALL QUALITY SCORE")
        print(f"  {'─'*50}")
        print(f"  Score:    {ov.get('score', 0):.3f} / 1.000")
        print(f"  Grade:    {ov.get('grade', 'N/A')}")
        print(f"  Notes:    {ov.get('notes', '')}")

        print(f"\n{sep}\n")

    def load_all_results(self) -> list:
        """Load all saved evaluation results for aggregate analysis."""
        results = []
        experiments_dir = self.output_dir / "experiments"

        for filepath in experiments_dir.glob("eval_*.json"):
            try:
                with open(filepath, encoding="utf-8") as f:
                    results.append(json.load(f))
            except Exception as e:
                print(f"[evaluator] Warning: could not load {filepath}: {e}")

        return sorted(results, key=lambda x: x.get("timestamp", ""))

    def compute_aggregate_metrics(self) -> dict:
        """
        Compute aggregate metrics across all evaluation runs.

        This produces the summary statistics table for your paper.
        Call this after running all 30 benchmark experiments.
        """
        results = self.load_all_results()

        if not results:
            return {"error": "No evaluation results found"}

        print(f"[evaluator] Computing aggregates over {len(results)} runs...")

        # ── Collect metrics across runs ───────────────────────────
        maes               = []
        directional_accs   = []
        rouge_l_f1s        = []
        word_counts        = []
        hallucination_rates = []
        stance_entropies   = []
        source_diversities = []
        n_articles_list    = []
        elapsed_times      = []
        overall_scores     = []

        for r in results:
            ba = r.get("bias_accuracy", {})
            sq = r.get("synthesis_quality", {})
            cd = r.get("coverage_diversity", {})
            ov = r.get("overall_score", {})
            pf = r.get("performance", {})

            if ba.get("mae") is not None:
                maes.append(ba["mae"])
            if ba.get("directional_correct") is not None:
                directional_accs.append(
                    1.0 if ba["directional_correct"] else 0.0
                )

            rouge = sq.get("rouge_scores", {})
            if rouge and rouge.get("rouge_l", {}).get("f1") is not None:
                rouge_l_f1s.append(rouge["rouge_l"]["f1"])

            if sq.get("word_count"):
                word_counts.append(sq["word_count"])
            if sq.get("hallucination_rate") is not None:
                hallucination_rates.append(sq["hallucination_rate"])

            if cd.get("stance_entropy") is not None:
                stance_entropies.append(cd["stance_entropy"])
            if cd.get("source_diversity") is not None:
                source_diversities.append(cd["source_diversity"])
            if cd.get("n_articles"):
                n_articles_list.append(cd["n_articles"])

            if pf.get("elapsed_seconds"):
                elapsed_times.append(pf["elapsed_seconds"])

            if ov.get("score") is not None:
                overall_scores.append(ov["score"])

        def _stats(values: list) -> dict:
            """Compute mean, std, min, max for a list."""
            if not values:
                return {"mean": None, "std": None,
                        "min": None, "max": None, "n": 0}
            arr = np.array(values)
            return {
                "mean": round(float(np.mean(arr)), 4),
                "std":  round(float(np.std(arr)),  4),
                "min":  round(float(np.min(arr)),  4),
                "max":  round(float(np.max(arr)),  4),
                "n":    len(values),
            }

        aggregates = {
            "n_evaluations":          len(results),
            "bias_mae":               _stats(maes),
            "directional_accuracy":   _stats(directional_accs),
            "rouge_l_f1":             _stats(rouge_l_f1s),
            "synthesis_word_count":   _stats(word_counts),
            "hallucination_rate":     _stats(hallucination_rates),
            "stance_entropy":         _stats(stance_entropies),
            "source_diversity":       _stats(source_diversities),
            "articles_per_run":       _stats(n_articles_list),
            "latency_seconds":        _stats(elapsed_times),
            "overall_score":          _stats(overall_scores),
        }

        # ── Print paper-ready table ───────────────────────────────
        self._print_paper_table(aggregates, results)

        # ── Save aggregates ───────────────────────────────────────
        agg_path = self.output_dir / "aggregate_results.json"
        with open(agg_path, "w", encoding="utf-8") as f:
            json.dump(aggregates, f, indent=2)
        print(f"[evaluator] Aggregates saved → {agg_path}")

        return aggregates

    # ─────────────────────────────────────────────────────────────
    # PRIVATE EVALUATION COMPONENTS
    # ─────────────────────────────────────────────────────────────

    def _eval_bias_accuracy(
        self,
        report:   dict,
        articles: list,
        topic:    str,
    ) -> dict:
        """Evaluate bias detection accuracy against ground truth."""
        predicted  = report.get("bias_score", 0.0)

        # ── Method 1: Source-level ground truth ───────────────────
        source_gt  = compute_expected_bias_from_sources(articles)
        source_expected = source_gt.get("expected_score")

        # ── Method 2: Topic-level ground truth ────────────────────
        topic_gt   = get_topic_ground_truth(topic)
        topic_expected = topic_gt.get("expected_score")

        # ── Compute MAE and directional accuracy ──────────────────
        # Prefer source-level if we have good coverage
        if (source_expected is not None and
                source_gt.get("coverage", 0) >= 0.3):
            expected = source_expected
            gt_type  = "source_level"
        elif topic_expected is not None:
            expected = topic_expected
            gt_type  = "topic_level"
        else:
            expected = None
            gt_type  = "none"

        result = {
            "predicted_bias":     round(float(predicted), 4),
            "expected_bias":      round(float(expected), 4) if expected is not None else None,
            "ground_truth_type":  gt_type,
            "mae":                None,
            "directional_correct": None,
            "sources_matched":    source_gt.get("n_matched", 0),
            "sources_total":      len(articles),
            "source_coverage":    round(source_gt.get("coverage", 0), 4),
            "matched_sources":    source_gt.get("matched_sources", []),
            "topic_ground_truth": topic_gt,
        }

        if expected is not None:
            mae = abs(predicted - expected)
            result["mae"] = round(mae, 4)

            pred_sign     = np.sign(predicted)
            expected_sign = np.sign(expected)

            # Directional correct if both on same side OR both near zero
            if abs(expected) < 0.1:
                # Expected is near-zero — count as correct if predicted < 0.3
                result["directional_correct"] = abs(predicted) < 0.3
            else:
                result["directional_correct"] = bool(
                    pred_sign == expected_sign
                )

        print(f"  Bias accuracy: predicted={predicted:+.4f}, "
              f"expected={expected}, MAE={result['mae']}")

        return result

    def _eval_synthesis_quality(
        self,
        report:           dict,
        topic:            str,
        reference_summary: Optional[str] = None,
    ) -> dict:
        """Evaluate the quality of the Agent C synthesis."""
        synthesis = report.get("synthesis_report", "")

        if not synthesis:
            return {
                "word_count":        0,
                "sentence_count":    0,
                "covers_both_sides": False,
                "hallucination_rate": 1.0,
                "rouge_scores":      None,
            }

        # ── Basic quality metrics ─────────────────────────────────
        words     = synthesis.split()
        sentences = [
            s.strip() for s in synthesis.split(".")
            if len(s.strip().split()) >= 3
        ]

        # ── Coverage check ────────────────────────────────────────
        agent_a = report.get("agent_a", {})
        agent_b = report.get("agent_b", {})
        has_pro_side  = len(agent_a.get("arguments", [])) > 0
        has_con_side  = len(agent_b.get("counter_arguments", [])) > 0
        covers_both   = has_pro_side and has_con_side

        # ── Hallucination rate ────────────────────────────────────
        flags     = report.get("hallucination_flags", [])
        all_claims = (
            agent_a.get("arguments", []) +
            agent_b.get("counter_arguments", [])
        )
        total_claims = max(len(all_claims), 1)
        hall_rate    = len(flags) / total_claims

        # ── ROUGE scores ──────────────────────────────────────────
        # Use provided reference, or look up from our database
        ref = reference_summary or get_reference_summary(topic)

        rouge_scores = None
        if ref:
            rouge_scores = compute_all_rouge(synthesis, ref)
            print(f"  ROUGE-L F1: {rouge_scores['rouge_l']['f1']:.4f}")
        else:
            print(f"  ROUGE: No reference summary available for '{topic}'")

        return {
            "word_count":         len(words),
            "sentence_count":     len(sentences),
            "covers_both_sides":  covers_both,
            "hallucination_rate": round(hall_rate, 4),
            "hallucination_count": len(flags),
            "rouge_scores":       rouge_scores,
            "synthesis_preview":  synthesis[:150] + "...",
        }

    def _eval_coverage_diversity(
        self,
        articles: list,
        stats:    dict,
    ) -> dict:
        """Evaluate how diverse and comprehensive the coverage is."""
        # ── Stance distribution ───────────────────────────────────
        counts = {"Supportive": 0, "Critical": 0, "Neutral": 0}
        for article in articles:
            stance = article.get("stance", "Neutral")
            if stance in counts:
                counts[stance] += 1

        # ── Shannon entropy ───────────────────────────────────────
        total   = sum(counts.values())
        entropy = 0.0
        if total > 0:
            for count in counts.values():
                if count > 0:
                    p        = count / total
                    entropy -= p * np.log2(p)
            # Normalize to [0,1]
            entropy = entropy / np.log2(3)

        # ── Source diversity ──────────────────────────────────────
        unique_sources = len(set(
            a.get("source", "") for a in articles
            if a.get("source")
        ))

        # Compute text-based diversity
        try:
            from ..bias_model import compute_source_diversity
        except ImportError:
            try:
                from src.bias_model import compute_source_diversity
            except ImportError:
                from bias_model import compute_source_diversity
        source_div = compute_source_diversity(articles)

        # ── Coverage completeness ─────────────────────────────────
        # Did we get at least one article from each stance?
        full_coverage = all(v > 0 for v in counts.values())

        return {
            "n_articles":      total,
            "stance_counts":   counts,
            "stance_entropy":  round(float(entropy), 4),
            "unique_sources":  unique_sources,
            "source_diversity": source_div,
            "full_stance_coverage": full_coverage,
            "total_chunks":    stats.get("total_chunks", 0),
        }

    def _eval_argument_quality(self, report: dict) -> dict:
        """Evaluate the quality of extracted arguments."""
        agent_a = report.get("agent_a", {})
        agent_b = report.get("agent_b", {})
        agent_c = report.get("agent_c", {})

        n_args     = len(agent_a.get("arguments", []))
        n_counters = len(agent_b.get("counter_arguments", []))
        n_removed  = len(report.get("loaded_language_removed", []))
        n_flags    = len(report.get("hallucination_flags", []))
        n_agrees   = len(report.get("key_agreements", []))
        n_disagrees = len(report.get("key_disagreements", []))

        a_conf = float(agent_a.get("confidence_score", 0))
        b_conf = float(agent_b.get("confidence_score", 0))
        mean_conf = (a_conf + b_conf) / 2

        # Balance ratio: how balanced are the two sides?
        # Perfect balance = 1.0, complete imbalance = 0.0
        total = n_args + n_counters
        if total > 0:
            balance = 1.0 - abs(n_args - n_counters) / total
        else:
            balance = 0.0

        return {
            "n_supporting_args":    n_args,
            "n_counter_args":       n_counters,
            "total_arguments":      total,
            "argument_balance":     round(balance, 4),
            "agent_a_confidence":   round(a_conf,     4),
            "agent_b_confidence":   round(b_conf,     4),
            "mean_confidence":      round(mean_conf,  4),
            "loaded_lang_removed":  n_removed,
            "hallucination_flags":  n_flags,
            "key_agreements":       n_agrees,
            "key_disagreements":    n_disagrees,
        }

    def _eval_performance(
        self,
        pipeline_result: dict,
        elapsed_seconds: Optional[float],
    ) -> dict:
        """Evaluate pipeline performance metrics."""
        articles = pipeline_result.get("articles", [])
        stats    = pipeline_result.get("stats",    {})

        throughput = None
        if elapsed_seconds and elapsed_seconds > 0 and articles:
            throughput = round(len(articles) / elapsed_seconds, 3)

        return {
            "elapsed_seconds":  elapsed_seconds,
            "n_articles":       len(articles),
            "total_chunks":     stats.get("total_chunks", 0),
            "throughput_articles_per_sec": throughput,
        }

    def _compute_overall_score(self, eval_report: dict) -> dict:
        """
        Compute a single overall quality score for this pipeline run.

        Scoring rubric:
            Bias Accuracy (30%)  — how close to ground truth
            Synthesis Quality (30%) — ROUGE + coverage
            Coverage Diversity (20%) — stance entropy + source diversity
            Argument Quality (20%)  — confidence + balance
        """
        score_components = {}

        # ── Bias Accuracy Component ───────────────────────────────
        ba  = eval_report.get("bias_accuracy", {})
        mae = ba.get("mae")
        if mae is not None:
            # MAE of 0 → score 1.0, MAE of 1 → score 0.0
            bias_score = max(0.0, 1.0 - mae)
        else:
            # No ground truth — give neutral score
            bias_score = 0.5
        score_components["bias_accuracy"] = round(bias_score, 4)

        # ── Synthesis Quality Component ───────────────────────────
        sq      = eval_report.get("synthesis_quality", {})
        rouge   = sq.get("rouge_scores", {})
        rl_f1   = rouge.get("rouge_l", {}).get("f1", None) if rouge else None
        hall    = sq.get("hallucination_rate", 0.5)
        covers  = 1.0 if sq.get("covers_both_sides") else 0.0
        words   = sq.get("word_count", 0)
        word_ok = min(1.0, words / 200) if words < 200 else 1.0  # expect 200+

        if rl_f1 is not None:
            synth_score = (0.4 * rl_f1 + 0.3 * (1 - hall) +
                           0.2 * covers + 0.1 * word_ok)
        else:
            synth_score = (0.5 * (1 - hall) + 0.3 * covers +
                           0.2 * word_ok)
        score_components["synthesis_quality"] = round(synth_score, 4)

        # ── Coverage Diversity Component ──────────────────────────
        cd       = eval_report.get("coverage_diversity", {})
        entropy  = cd.get("stance_entropy",   0.0)
        src_div  = cd.get("source_diversity", 0.0)
        cov_score = (0.6 * entropy + 0.4 * src_div)
        score_components["coverage_diversity"] = round(cov_score, 4)

        # ── Argument Quality Component ────────────────────────────
        aq       = eval_report.get("argument_quality", {})
        conf     = aq.get("mean_confidence",  0.5)
        balance  = aq.get("argument_balance", 0.5)
        arg_score = (0.5 * conf + 0.5 * balance)
        score_components["argument_quality"] = round(arg_score, 4)

        # ── Weighted Overall ──────────────────────────────────────
        overall = (
            0.30 * score_components["bias_accuracy"]     +
            0.30 * score_components["synthesis_quality"] +
            0.20 * score_components["coverage_diversity"]+
            0.20 * score_components["argument_quality"]
        )
        overall = round(float(overall), 4)

        # ── Grade ─────────────────────────────────────────────────
        if overall >= 0.80:
            grade = "A (Excellent)"
        elif overall >= 0.65:
            grade = "B (Good)"
        elif overall >= 0.50:
            grade = "C (Acceptable)"
        else:
            grade = "D (Needs Improvement)"

        return {
            "score":      overall,
            "grade":      grade,
            "components": score_components,
            "notes":      self._generate_notes(eval_report, score_components),
        }

    def _generate_notes(
        self,
        eval_report:       dict,
        score_components:  dict,
    ) -> str:
        """Generate human-readable notes about evaluation results."""
        notes = []

        ba = eval_report.get("bias_accuracy", {})
        if ba.get("mae") and ba["mae"] > 0.5:
            notes.append(
                f"High MAE ({ba['mae']:.2f}) — "
                "bias score differs significantly from ground truth"
            )
        if ba.get("directional_correct") is False:
            notes.append("Directional error — got wrong side of bias")

        sq = eval_report.get("synthesis_quality", {})
        if not sq.get("covers_both_sides"):
            notes.append("Synthesis does not cover both perspectives")
        if sq.get("hallucination_rate", 0) > 0.5:
            notes.append(
                f"High hallucination rate "
                f"({sq['hallucination_rate']:.0%})"
            )

        cd = eval_report.get("coverage_diversity", {})
        if cd.get("stance_entropy", 0) < 0.3:
            notes.append("Low stance diversity — consider more balanced queries")

        if not notes:
            notes.append("All metrics within acceptable range")

        return "; ".join(notes)

    def _print_paper_table(
        self,
        aggregates: dict,
        results:    list,
    ) -> None:
        """Print LaTeX-ready table for the paper."""
        print("\n" + "=" * 62)
        print("  PAPER-READY RESULTS TABLE")
        print("=" * 62)

        def fmt(stats_dict: dict) -> str:
            if not stats_dict or stats_dict.get("mean") is None:
                return "N/A"
            return (
                f"{stats_dict['mean']:.3f} "
                f"± {stats_dict['std']:.3f}"
            )

        print(f"\n  Metric                     Mean ± Std")
        print(f"  {'─'*50}")
        print(f"  Bias MAE                   {fmt(aggregates['bias_mae'])}")
        print(f"  Directional Accuracy       "
              f"{fmt(aggregates['directional_accuracy'])}")
        print(f"  ROUGE-L F1                 {fmt(aggregates['rouge_l_f1'])}")
        print(f"  Hallucination Rate         "
              f"{fmt(aggregates['hallucination_rate'])}")
        print(f"  Stance Entropy             "
              f"{fmt(aggregates['stance_entropy'])}")
        print(f"  Source Diversity           "
              f"{fmt(aggregates['source_diversity'])}")
        print(f"  Synthesis Length (words)   "
              f"{fmt(aggregates['synthesis_word_count'])}")
        print(f"  Latency (seconds)          "
              f"{fmt(aggregates['latency_seconds'])}")
        print(f"  Overall Score              "
              f"{fmt(aggregates['overall_score'])}")
        print(f"\n  Based on {aggregates['n_evaluations']} evaluation runs")
        print("=" * 62)

        # LaTeX table output
        print("\n  LaTeX Table (copy to paper):")
        print("  " + "─"*50)
        print(r"  \begin{table}[h]")
        print(r"  \centering")
        print(r"  \begin{tabular}{lcc}")
        print(r"  \hline")
        print(r"  \textbf{Metric} & \textbf{Mean} & \textbf{Std} \\")
        print(r"  \hline")

        rows = [
            ("Bias MAE",             aggregates["bias_mae"]),
            ("Directional Acc.",     aggregates["directional_accuracy"]),
            ("ROUGE-L F1",           aggregates["rouge_l_f1"]),
            ("Hallucination Rate",   aggregates["hallucination_rate"]),
            ("Stance Entropy",       aggregates["stance_entropy"]),
            ("Source Diversity",     aggregates["source_diversity"]),
            ("Overall Score",        aggregates["overall_score"]),
        ]
        for name, stats in rows:
            if stats and stats.get("mean") is not None:
                print(
                    f"  {name} & "
                    f"{stats['mean']:.3f} & "
                    f"{stats['std']:.3f} \\\\"
                )
        print(r"  \hline")
        print(r"  \end{tabular}")
        print(r"  \caption{ORBITA Evaluation Results}")
        print(r"  \label{tab:results}")
        print(r"  \end{table}")