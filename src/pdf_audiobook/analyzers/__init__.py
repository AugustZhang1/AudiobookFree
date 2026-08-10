"""Replaceable, isolated whole-book analyzer adapters."""

from .booknlp import BookNLPAnalyzer, BookNLPAnalyzerError, parse_booknlp_output

__all__ = ["BookNLPAnalyzer", "BookNLPAnalyzerError", "parse_booknlp_output"]
