"""Compatibility import for the BookNLP isolated adapter."""

from .booknlp import BOOKNLP_VERSION, BookNLPAnalyzer, BookNLPAnalyzerError, parse_booknlp_output

__all__ = ["BOOKNLP_VERSION", "BookNLPAnalyzer", "BookNLPAnalyzerError", "parse_booknlp_output"]
