"""Trainer package for RAG chatbot evaluation."""

from .eval_epoch import (
    evaluate_retrieval_performance,
    plot_retrieval_metrics,
    create_evaluation_report,
    save_results_to_csv
)

__all__ = [
    'evaluate_retrieval_performance',
    'plot_retrieval_metrics',
    'create_evaluation_report',
    'save_results_to_csv'
]

