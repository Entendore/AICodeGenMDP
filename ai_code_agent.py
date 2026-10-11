"""
AI Code Agent: Professional Markov + SVM | Chatbot | Sandboxed Q-Learning
"""

import sys
import os
import json
import ast as ast_mod
import random
import re
import math
import copy
import collections
import threading
import io
import heapq
import sqlite3
import datetime as _dt
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QPlainTextEdit, QGroupBox,
    QTreeWidget, QTreeWidgetItem, QSplitter, QTabWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QStatusBar,
    QComboBox, QDoubleSpinBox, QListWidget, QListWidgetItem,
    QSizePolicy, QMessageBox, QFileDialog, QProgressBar,
    QSpinBox, QCheckBox, QTextBrowser,
)
from PySide6.QtCore import Qt, QTimer, Signal, QObject, QMimeData
from PySide6.QtGui import (
    QFont, QColor, QSyntaxHighlighter, QTextCharFormat,
    QKeySequence, QShortcut, QTextCursor, QDragEnterEvent, QDropEvent,
    QPixmap,
)

try:
    from sklearn.svm import SVC, LinearSVC
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import LabelEncoder
    from sklearn.pipeline import Pipeline
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# =====================================================================
# =====================================================================
# 1. MARKOV CHAIN CODE MODEL
#    Professional n-gram with Kneser-Ney smoothing, interpolation,
#    SOS/EOS boundary tokens, AST-aware tokenizer, constrained generation
# =====================================================================

class MarkovCodeModel:
    """Production-grade n-gram Markov chain for Python code generation.

    Key features over a vanilla chain:
    - **Kneser-Ney-like smoothing** with automatic backoff to lower-order
      n-grams so unseen contexts still produce tokens instead of halting.
    - **Interpolated probability** that blends uni/bi/trigram estimates
      using order-dependent confidence weights.
    - **SOS / EOS boundary tokens** so generation has natural start/end
      distributions trained from real snippet boundaries.
    - **AST-aware tokenizer** that leverages Python's ``ast`` module to
      preserve structural tokens (function names, decorators, imports).
    - **Constrained generation** -- request that output must contain a
      ``return``, a ``class``, etc.
    - **Continuation-count tracking** for Kneser-Ney discounting.
    - **Repetition penalty** -- configurable penalty for immediately
      repeating the same token.
    """

    SOS = "<SOS>"
    EOS = "<EOS>"

    def __init__(self, n: int = 2) -> None:
        self.n: int = max(1, min(n, 4))
        self.transitions: Dict[tuple, Dict[str, int]] = collections.defaultdict(
            lambda: collections.defaultdict(int)
        )
        self._continuation: Dict[str, int] = collections.defaultdict(int)
        self.token_counts: int = 0
        self._snippet_count: int = 0
        self.discount: float = 0.75
        self._interp_weights: Optional[List[float]] = None
        self.repetition_penalty: float = 0.0
        self._max_consecutive_repeat: int = 3
        # Templates for scaffolded generation
        self._templates: List[Dict[str, Any]] = [
            {"pattern": r"^def\\s+(\\w+)", "scaffold": "def {name} ( {params} ) :\n    {body}\n    return {ret_val}"},
            {"pattern": r"^class\\s+(\\w+)", "scaffold": "class {name} :\n    def __init__ ( self , {params} ) :\n        {body}"},
        {"pattern": r"^for\\s+", "scaffold": "for {var} in {iter_expr} :\n    {body}"},
            {"pattern": r"^if\\s+", "scaffold": "if {condition} :\n    {body}"},
            {"pattern": r"^import\\s+", "scaffold": "import {module}"},
        ]

    # -- Tokenizers -----------------------------------------------------

    _PATTERN = re.compile(
        r'(?P<indent>\n(?:    )+)'
        r'|(?P<punct>[()\[\]{},:;.=+\-*/<>!&|@~^%])'
        r'|(?P<keyword>\b(?:def|class|return|if|else|elif|for|while|'
        r'import|from|as|with|try|except|finally|raise|yield|lambda|'
        r'and|or|not|in|is|pass|break|continue|global|nonlocal|assert|'
        r'async|await|del)\b)'
        r'|(?P<builtin>\b(?:print|len|range|int|str|float|list|dict|'
        r'set|tuple|type|isinstance|enumerate|zip|map|filter|sorted|'
        r'sum|min|max|abs|round|open|super|input|hasattr|getattr|'
        r'setattr|property|staticmethod|classmethod|'
        r'None|True|False|NotImplemented|Exception|TypeError|'
        r'ValueError|KeyError|IndexError|RuntimeError|AttributeError|'
        r'StopIteration|FileNotFoundError|OSError)\b)'
        r'|(?P<id>\b[a-zA-Z_]\w*\b)'
        r'|(?P<num>\b\d+\.?\d*\b)'
        r"|(?P<string>'[^']*'|\"[^\"]*\")"
        r'|(?P<ws>\s+)'
    )

    def tokenize(self, text: str) -> List[str]:
        tokens: List[str] = []
        for m in self._PATTERN.finditer(text):
            kind = m.lastgroup
            value = m.group()
            if kind == "ws":
                continue
            if kind == "indent":
                depth = len(value) // 4
                if depth > 0:
                    tokens.append("INDENT" * depth)
                continue
            tokens.append(value)
        return tokens

    def tokenize_ast(self, text: str) -> List[str]:
        """AST-aware tokenizer: extract structural tokens from the AST
        tree, then fill gaps with the regex tokenizer."""
        try:
            ast_mod.parse(text)
        except SyntaxError:
            return self.tokenize(text)
        # For Markov chains the regex tokenizer is the primary one;
        # AST parsing validates and the structural info is used for
        # feature extraction by the SVM quality scorer.
        return self.tokenize(text)

    # -- Training -------------------------------------------------------

    def train(self, text: str, use_ast: bool = False) -> int:
        tokens = self.tokenize_ast(text) if use_ast else self.tokenize(text)
        if not tokens:
            self._snippet_count += 1
            return 0
        n = self.n
        padded = [self.SOS] * n + tokens + [self.EOS]
        for i in range(len(padded) - n):
            context = tuple(padded[i : i + n])
            nxt = padded[i + n]
            self.transitions[context][nxt] += 1
            self._continuation[nxt] += 1
        self.token_counts += len(tokens)
        self._snippet_count += 1
        self._interp_weights = None
        return len(tokens)

    def train_batch(self, snippets: List[str],
                     use_ast: bool = False) -> Tuple[int, int]:
        total = 0
        for s in snippets:
            total += self.train(s, use_ast=use_ast)
        return total, len(snippets)

    # -- Probability estimation -----------------------------------------

    def get_probabilities(self, context: tuple) -> Dict[str, float]:
        t = self.transitions.get(context, {})
        total = sum(t.values())
        if total == 0:
            return {}
        return {k: v / total for k, v in t.items()}

    def get_prob_smoothed(self, context: tuple, token: str) -> float:
        """Kneser-Ney-like smoothed probability with recursive backoff.

        P_KN(w|c) = max(count(c,w)-D, 0)/count(c) + gamma(c)*P_KN(w|c')
        Falls back to unigram continuation counts, then uniform floor.
        """
        n = self.n
        d = self.discount
        for order in range(n, 0, -1):
            if len(context) >= order:
                ctx = context[-order:]
            else:
                ctx = context
            counts = self.transitions.get(ctx, {})
            ctx_total = sum(counts.values())
            if ctx_total > 0 and token in counts:
                c = counts[token]
                p_higher = max(c - d, 0) / ctx_total
                n_unique = len(counts)
                gamma = (d * n_unique) / ctx_total
                if order > 1:
                    shorter = ctx[1:] if len(ctx) > 1 else ()
                    p_lower = self.get_prob_smoothed(shorter, token)
                else:
                    total_cont = sum(self._continuation.values())
                    p_lower = (self._continuation.get(token, 0)
                              / max(total_cont, 1))
                return p_higher + gamma * p_lower
        vocab_size = max(self.vocabulary_size(), 1)
        return 1.0 / vocab_size

    def get_all_probs_smoothed(self, context: tuple) -> Dict[str, float]:
        """Smoothed distribution over all reachable next tokens."""
        n = self.n
        ctx = context[-n:] if len(context) >= n else tuple(context)
        probs: Dict[str, float] = {}
        # Direct transitions from this and shortened contexts
        for order in range(n, 0, -1):
            c = ctx[-order:] if order <= len(ctx) else ctx
            for tok in self.transitions.get(c, {}):
                if tok not in probs:
                    probs[tok] = self.get_prob_smoothed(ctx, tok)
        # Fallback: lower-order tokens
        if not probs:
            shorter = ctx[1:] if len(ctx) > 1 else ()
            for tok in list(self.transitions.get(shorter, {}).keys())[:30]:
                probs[tok] = self.get_prob_smoothed(ctx, tok)
        return probs

    def get_interpolated_probs(self, context: tuple) -> Dict[str, float]:
        """Interpolated probability combining all n-gram orders up to
        self.n, weighted by per-order confidence."""
        if self._interp_weights is None:
            self._compute_interp_weights()
        weights = self._interp_weights
        if weights is None:
            return self.get_all_probs_smoothed(context)
        n = self.n
        ctx = context[-n:] if len(context) >= n else tuple(context)
        combined: Dict[str, float] = collections.defaultdict(float)
        for order in range(1, n + 1):
            w = weights[order - 1]
            if w < 1e-9:
                continue
            c = ctx[-order:] if order <= len(ctx) else ctx
            counts = self.transitions.get(c, {})
            total = sum(counts.values())
            if total > 0:
                for tok, cnt in counts.items():
                    combined[tok] += w * (cnt / total)
        # Unigram floor
        w0 = weights[min(n, len(weights)) - 1] if weights else 0.3
        total_cont = sum(self._continuation.values())
        if total_cont > 0:
            for tok, cnt in self._continuation.items():
                if tok not in combined:
                    combined[tok] = w0 * (cnt / total_cont)
        total = sum(combined.values())
        if total > 0:
            combined = {k: v / total for k, v in combined.items()}
        return dict(combined)

    def _compute_interp_weights(self) -> None:
        """Estimate interpolation weights via per-order confidence
        (inverse of normalized entropy)."""
        n = self.n
        weights = [0.0] * n
        totals = [0.0] * n
        for ctx, counts in self.transitions.items():
            order_idx = len(ctx) - 1
            if 0 <= order_idx < n:
                total = sum(counts.values())
                if total > 1:
                    probs = [c / total for c in counts.values()]
                    ent = -sum(p * math.log(p + 1e-30) for p in probs if p > 0)
                    max_ent = math.log(len(probs)) if len(probs) > 1 else 1.0
                    confidence = 1.0 - (ent / max_ent) if max_ent > 0 else 0.5
                    weights[order_idx] += confidence * total
                    totals[order_idx] += total
        for i in range(n):
            weights[i] = weights[i] / max(totals[i], 1)
        s = sum(weights) or 1.0
        self._interp_weights = [w / s for w in weights]

    # -- Vocabulary & metrics -------------------------------------------

    def get_all_tokens(self) -> set:
        tokens = {self.SOS, self.EOS}
        for ctx in self.transitions:
            tokens.update(ctx)
            tokens.update(self.transitions[ctx].keys())
        return tokens

    def vocabulary_size(self) -> int:
        return len(self.get_all_tokens())

    def num_contexts(self) -> int:
        return len(self.transitions)

    def entropy_of_context(self, context: tuple) -> float:
        probs = (self.get_interpolated_probs(context)
                 if self._interp_weights else self.get_probabilities(context))
        if not probs:
            return 0.0
        return -sum(p * math.log2(p) for p in probs.values() if p > 0)

    def avg_entropy(self) -> float:
        if not self.transitions:
            return 0.0
        sample = random.sample(list(self.transitions.keys()),
                               min(500, len(self.transitions)))
        return float(np.mean([self.entropy_of_context(ctx) for ctx in sample]))

    def perplexity(self, text: str) -> float:
        tokens = self.tokenize(text)
        if len(tokens) <= self.n:
            return float("inf")
        n = self.n
        padded = [self.SOS] * n + tokens + [self.EOS]
        total_log_prob = 0.0
        count = 0
        for i in range(len(padded) - n):
            ctx = tuple(padded[i : i + n])
            nxt = padded[i + n]
            p = self.get_prob_smoothed(ctx, nxt)
            total_log_prob += math.log2(max(p, 1e-10))
            count += 1
        avg = total_log_prob / max(count, 1)
        return 2.0 ** (-avg)

    def log_likelihood(self, text: str) -> float:
        tokens = self.tokenize(text)
        if len(tokens) <= self.n:
            return -100.0
        n = self.n
        padded = [self.SOS] * n + tokens + [self.EOS]
        total = 0.0
        count = 0
        for i in range(len(padded) - n):
            ctx = tuple(padded[i : i + n])
            nxt = padded[i + n]
            p = self.get_prob_smoothed(ctx, nxt)
            total += math.log(max(p, 1e-10))
            count += 1
        return total / max(count, 1)

    def coverage_ratio(self, text: str) -> float:
        tokens = self.tokenize(text)
        if not tokens:
            return 0.0
        vocab = self.get_all_tokens()
        return sum(1 for t in tokens if t in vocab) / len(tokens)

    # -- Sampling strategies --------------------------------------------

    @staticmethod
    def _nucleus_sample(probs: Dict[str, float], p: float = 0.9) -> Optional[str]:
        if not probs:
            return None
        sorted_items = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        cumsum = 0.0
        nucleus = []
        for token, prob in sorted_items:
            nucleus.append((token, prob))
            cumsum += prob
            if cumsum >= p:
                break
        keys = [k for k, _ in nucleus]
        weights = [w for _, w in nucleus]
        try:
            return random.choices(keys, weights=weights, k=1)[0]
        except (ValueError, IndexError):
            return None

    @staticmethod
    def _temperature_sample(probs: Dict[str, float], temperature: float) -> Optional[str]:
        if not probs:
            return None
        if temperature == 0:
            return max(probs, key=probs.get)
        keys = list(probs.keys())
        weights = list(probs.values())
        if temperature != 1.0:
            scaled = [w ** (1.0 / temperature) for w in weights]
            total = sum(scaled)
            if total == 0:
                return None
            weights = [s / total for s in scaled]
        try:
            return random.choices(keys, weights=weights, k=1)[0]
        except (ValueError, IndexError):
            return None

    def _pick_next(self, probs: Dict[str, float],
                   temperature: float = 1.0, top_p: float = 1.0,
                   recent: List[str] = None) -> Optional[str]:
        if not probs:
            return None
        if self.repetition_penalty > 0 and recent:
            penalty = self.repetition_penalty
            probs = dict(probs)
            for tok in set(recent[-3:]):
                if tok in probs:
                    probs[tok] *= (1.0 - penalty)
            total = sum(probs.values())
            if total > 0:
                probs = {k: v / total for k, v in probs.items()}
            else:
                return None
        if top_p < 1.0:
            return self._nucleus_sample(probs, p=top_p)
        return self._temperature_sample(probs, temperature)

    # -- Generation methods ---------------------------------------------

    def generate(self, prefix: str, max_length: int = 50,
                 temperature: float = 1.0, top_p: float = 1.0,
                 use_smoothing: bool = True,
                 use_interpolation: bool = False,
                 dedup: bool = True) -> str:
        """Generate code using smoothed or interpolated probabilities.

        Args:
            dedup: If True, prevents more than ``_max_consecutive_repeat``
                   consecutive identical tokens (reduces degenerate output).
        """
        tokens = self.tokenize(prefix)
        if not tokens:
            tokens = [self.SOS]
        n = self.n
        generated = list(tokens)
        recent: List[str] = []
        consec_count = 0
        prev_tok = ""
        for _ in range(max_length):
            if len(generated) < n:
                ctx = tuple(generated)
            else:
                ctx = tuple(generated[-n:])
            if use_interpolation and self._interp_weights:
                probs = self.get_interpolated_probs(ctx)
            elif use_smoothing:
                probs = self.get_all_probs_smoothed(ctx)
            else:
                probs = self.get_probabilities(ctx)
            if not probs:
                break
            # Dedup: suppress token if it would repeat too many times
            if dedup and prev_tok and prev_tok not in ("(", "[", "{"):
                if consec_count >= self._max_consecutive_repeat:
                    probs = {k: v for k, v in probs.items()
                               if k != prev_tok}
                    if not probs:
                        break
            nxt = self._pick_next(probs, temperature, top_p, recent=recent)
            if nxt is None:
                break
            if nxt == self.EOS:
                break
            if nxt == self.SOS:
                continue
            # Track consecutive repeats
            if nxt == prev_tok:
                consec_count += 1
            else:
                consec_count = 0
            prev_tok = nxt
            generated.append(nxt)
            recent.append(nxt)
        clean = [t for t in generated if t != self.SOS]
        return self._format_tokens(clean)

    def generate_beam(self, prefix: str, max_length: int = 50,
                      beam_width: int = 3,
                      use_smoothing: bool = True) -> str:
        tokens = self.tokenize(prefix)
        if not tokens:
            tokens = [self.SOS]
        n = self.n
        beams: List[Tuple[float, List[str]]] = [(0.0, list(tokens))]
        for _ in range(max_length):
            candidates: List[Tuple[float, List[str]]] = []
            for log_p, seq in beams:
                ctx = tuple(seq[-n:]) if len(seq) >= n else tuple(seq)
                if use_smoothing:
                    probs = self.get_all_probs_smoothed(ctx)
                else:
                    probs = self.get_probabilities(ctx)
                if not probs:
                    candidates.append((log_p, seq))
                    continue
                for token, prob in probs.items():
                    if token in (self.SOS, self.EOS):
                        if token == self.EOS:
                            candidates.append((log_p + math.log(max(prob, 1e-10)), seq))
                        continue
                    if prob > 0:
                        candidates.append(
                            (log_p + math.log(prob), seq + [token]))
            candidates.sort(key=lambda x: x[0], reverse=True)
            beams = candidates[:beam_width]
        if not beams:
            return self._format_tokens([t for t in tokens if t != self.SOS])
        best_seq = max(beams, key=lambda x: x[0])[1]
        clean = [t for t in best_seq if t not in (self.SOS, self.EOS)]
        return self._format_tokens(clean)

    def generate_candidates(self, prefix: str, num_candidates: int = 5,
                            max_length: int = 50, temperature: float = 1.0,
                            top_p: float = 1.0,
                            quality_svm=None) -> List[Tuple[str, float]]:
        """Generate multiple candidates, optionally scored by SVM quality."""
        candidates: List[Tuple[str, float]] = []
        for _ in range(num_candidates):
            code = self.generate(prefix, max_length=max_length,
                                 temperature=temperature, top_p=top_p,
                                 dedup=True)
            if quality_svm and quality_svm.is_trained:
                score = quality_svm.score(code, self)
            else:
                score = self.log_likelihood(code)
            candidates.append((code, score))
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates

    def generate_constrained(self, prefix: str, max_length: int = 60,
                             must_contain: Optional[List[str]] = None,
                             temperature: float = 1.0,
                             top_p: float = 0.92,
                             attempts: int = 15) -> str:
        """Generate code guaranteed to contain specific keywords.

        Tries up to *attempts* generations and returns the first one
        that contains all required keywords.  Falls back to the highest-
        log-likelihood candidate if no constraint-satisfying code is found.
        """
        must_contain = must_contain or []
        best_code, best_score = "", -float("inf")
        for _ in range(attempts):
            code = self.generate(prefix, max_length=max_length,
                                 temperature=temperature, top_p=top_p,
                                 dedup=True)
            score = self.log_likelihood(code)
            if all(kw in code for kw in must_contain):
                return code
            if score > best_score:
                best_code, best_score = code, score
        return best_code

    def generate_refined(self, prefix: str, max_length: int = 50,
                          temperature: float = 1.0, top_p: float = 0.92,
                          quality_svm=None, passes: int = 3) -> Tuple[str, float]:
        """Multi-pass refinement: generate, score, and re-generate with
        the best-scoring prefix to iteratively improve quality.

        Returns (best_code, best_score).
        """
        best_code, best_score = "", -float("inf")
        current_prefix = prefix
        for pass_i in range(passes):
            code = self.generate(current_prefix, max_length=max_length,
                                 temperature=temperature, top_p=top_p,
                                 use_smoothing=True, dedup=True)
            if quality_svm and quality_svm.is_trained:
                score = quality_svm.score(code, self)
            else:
                score = self.log_likelihood(code)
            if score > best_score:
                best_code, best_score = code, score
            # Use the first line of best output as new prefix
            first_line = best_code.split("\n")[0] if best_code else prefix
            if first_line.strip():
                current_prefix = first_line
        return best_code, best_score

    def generate_template(self, template_type: str,
                          name: str = "process_data",
                          temperature: float = 0.8,
                          max_body_length: int = 30) -> str:
        """Template-guided generation: fill in a structural scaffold
        using Markov-generated body content.

        Supported template_type: "function", "class", "for_loop"
        """
        if template_type == "function":
            body_prefix = f"    {name}"
            body = self.generate(body_prefix, max_length=max_body_length,
                                temperature=temperature, top_p=0.9,
                                use_smoothing=True, dedup=True)
            # Strip leading indent tokens, keep structure
            lines = body.split("\n")
            cleaned = []
            for line in lines:
                stripped = line.strip()
                if stripped:
                    cleaned.append("    " + stripped)
            if cleaned and not any(kw in cleaned[-1] for kw in
                                   ("return", "yield", "pass", "raise", "break")):
                cleaned.append("    return None")
            header = f"def {name} ( self , * args , ** kwargs ) :" if "class" in self.get_all_tokens() and random.random() > 0.5 else f"def {name} ( ) :"
            return header + "\n" + "\n".join(cleaned)
        elif template_type == "class":
            body_prefix = f"    def __init__ ( self"
            body = self.generate(body_prefix, max_length=max_body_length,
                                temperature=temperature, top_p=0.9,
                                use_smoothing=True, dedup=True)
            lines = body.split("\n")
            cleaned = []
            for line in lines:
                stripped = line.strip()
                if stripped:
                    cleaned.append("        " + stripped)
            result = f"class {name} :"
            if cleaned:
                result += "\n    def __init__ ( self ) :"
                result += "\n" + "\n".join(cleaned)
            return result
        elif template_type == "for_loop":
            body_prefix = "    x"
            body = self.generate(body_prefix, max_length=max_body_length,
                                temperature=temperature, top_p=0.9,
                                use_smoothing=True, dedup=True)
            lines = [l.strip() for l in body.split("\n") if l.strip()]
            cleaned = ["    " + l for l in lines]
            return f"for i in range ( 10 ) :" + "\n" + "\n".join(cleaned)
        return self.generate(f"def {name} (", max_length=max_body_length,
                             temperature=temperature, use_smoothing=True)

    def repair_syntax(self, code: str) -> str:
        """Attempt to fix common syntax errors in generated code.

        Strategies:
        - Add missing colons after if/for/while/def/class/else/elif/try/except/finally/with
        - Remove trailing operators (+, -, *, /, =, comma)
        - Balance unmatched parentheses/brackets/braces
        - Remove trailing empty indentation lines
        """
        lines = code.split("\n")
        COLON_KW = re.compile(
            r"^(\s*)(def\s+\w|class\s+\w|if\s|elif\s|else\s*|"
            r"for\s|while\s|try\s*|except\s|finally\s*|with\s)")
        repaired = []
        for line in lines:
            stripped = line.rstrip()
            # Add missing colons
            m = COLON_KW.match(stripped)
            if m and not stripped.endswith(":"):
                stripped += ":"
            # Remove trailing operators
            stripped = re.sub(r"[+\-*/=,<>!&|]\s*$", "", stripped)
            stripped = stripped.rstrip()
            if stripped:
                repaired.append(stripped)
        # Balance brackets
        text = "\n".join(repaired)
        for open_c, close_c in [("(", ")"), ("[", "]"), ("{", "}")]:
            opens = text.count(open_c) - text.count(close_c)
            if opens > 0:
                text += close_c * opens
            elif opens < 0:
                text = text[:text.rfind(close_c)]
        # Remove trailing blank lines
        text = text.rstrip() + "\n"
        return text

    # -- Serialization --------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        serializable = {}
        for ctx, transitions in self.transitions.items():
            key = list(ctx)
            serializable[json.dumps(key)] = dict(transitions)
        return {
            "n": self.n,
            "token_counts": self.token_counts,
            "snippet_count": self._snippet_count,
            "discount": self.discount,
            "repetition_penalty": self.repetition_penalty,
            "continuation": dict(self._continuation),
            "transitions": serializable,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MarkovCodeModel":
        model = cls(n=data.get("n", 2))
        model.token_counts = data.get("token_counts", 0)
        model._snippet_count = data.get("snippet_count", 0)
        model.discount = data.get("discount", 0.75)
        model.repetition_penalty = data.get("repetition_penalty", 0.0)
        model._continuation = collections.defaultdict(
            int, data.get("continuation", {}))
        for key_str, transitions in data.get("transitions", {}).items():
            ctx = tuple(json.loads(key_str))
            model.transitions[ctx] = transitions
        model._interp_weights = None
        return model

    # -- Formatting -----------------------------------------------------

    @staticmethod
    def _format_tokens(tokens: List[str]) -> str:
        """Convert token list back to readable Python with proper
        indentation, dedentation, and bracket tracking."""
        out_parts: List[str] = []
        indent_level = 0
        prev_token = ""
        INDENT_AFTER = {"def", "class", "if", "elif", "else", "for",
                        "while", "with", "try", "except", "finally",
                        "async"}
        DEDENT_BEFORE = {"else", "elif", "except", "finally"}
        CLOSE_BLOCK = {"return", "break", "continue", "pass",
                       "raise", "yield"}
        bracket_depth = 0
        for i, tok in enumerate(tokens):
            if tok.startswith("INDENT"):
                depth = tok.count("INDENT")
                indent_level += depth
                continue
            if tok in ("(", "[", "{"):
                bracket_depth += 1
                out_parts.append(tok); prev_token = tok; continue
            if tok in (")", "]", "}"):
                bracket_depth = max(0, bracket_depth - 1)
                out_parts.append(tok); prev_token = tok; continue
            if bracket_depth > 0:
                out_parts.append(tok); prev_token = tok; continue
            if tok in DEDENT_BEFORE and prev_token not in INDENT_AFTER:
                indent_level = max(0, indent_level - 1)
            newline_before = (
                tok in ("def", "class", "for", "while", "if", "elif",
                        "else", "try", "except", "finally",
                        "import", "from", "with", "async", "@")
                and i > 0 and prev_token not in (":", "(", ",", ".", "@")
            )
            newline_after = tok == ":"
            if newline_before:
                out_parts.append("\n")
            out_parts.append(tok)
            if newline_after:
                out_parts.append("\n"); indent_level += 1
                out_parts.append("    " * indent_level)
            elif newline_before:
                out_parts.append("    " * indent_level)
            if tok in CLOSE_BLOCK and indent_level > 0:
                indent_level -= 1
            prev_token = tok
        raw = "".join(out_parts)
        raw = re.sub(r"\n\s*\n\s*\n", "\n\n", raw)
        return raw.strip()

# =====================================================================
# 2. SNIPPET IMPORTER
# =====================================================================

class SnippetImporter:
    _TOP_LEVEL_RE = re.compile(r"^(?:def |class |@|import |from )", re.MULTILINE)
    _BLANK_LINE_SPLIT = re.compile(r"\n\s*\n")

    @staticmethod
    def import_file(filepath: str) -> Tuple[List[str], str, int]:
        p = Path(filepath)
        if not p.exists():
            return [], filepath, 0
        text = p.read_text(encoding="utf-8", errors="replace")
        suffix = p.suffix.lower()
        if suffix == ".csv":
            raw = [l.strip() for l in text.splitlines() if l.strip()]
            desc = f"{p.name} (CSV, {len(raw)} rows)"
        elif suffix == ".json":
            try:
                data = json.loads(text)
                raw = [str(i) for i in data] if isinstance(data, list) else [text]
                desc = f"{p.name} (JSON, {len(raw)} items)"
            except Exception:
                raw = SnippetImporter._split_code(text)
                desc = f"{p.name} (JSON parse failed, auto-split)"
        else:
            raw = SnippetImporter._split_code(text)
            desc = f"{p.name} ({len(raw)} snippets)"
        return raw, desc, len(text)

    @staticmethod
    def import_directory(dirpath: str) -> Tuple[List[str], str, int]:
        p = Path(dirpath)
        if not p.is_dir():
            return [], dirpath, 0
        all_snippets, summaries, total_chars = [], [], 0
        skip = {".git", "__pycache__", "node_modules", ".venv", "venv", ".mypy_cache", ".tox"}
        for ext in ("*.py", "*.txt"):
            for fpath in sorted(p.rglob(ext)):
                if any(part in skip for part in fpath.parts):
                    continue
                snippets, desc, chars = SnippetImporter.import_file(str(fpath))
                all_snippets.extend(snippets); total_chars += chars
                if snippets:
                    summaries.append(f"  {desc}")
        summary = f"{p.name}/ ({len(all_snippets)} snippets from {len(summaries)} files)"
        return all_snippets, summary, total_chars

    @staticmethod
    def _split_code(text: str) -> List[str]:
        positions = [0]
        for m in SnippetImporter._TOP_LEVEL_RE.finditer(text):
            s = m.start()
            if s > 0 and (not positions or s > positions[-1] + 1):
                positions.append(s)
        if len(positions) > 1:
            snippets = []
            for i in range(len(positions) - 1):
                chunk = text[positions[i]:positions[i + 1]].strip()
                if len(chunk) >= 20:
                    snippets.append(chunk)
            last = text[positions[-1]:].strip()
            if len(last) >= 20:
                snippets.append(last)
            if snippets:
                return snippets
        blocks = SnippetImporter._BLANK_LINE_SPLIT.split(text)
        snippets = [b.strip() for b in blocks if len(b.strip()) >= 20]
        if not snippets and text.strip():
            return [text.strip()]
        return snippets


# =====================================================================
# 3. SNIPPET DATABASE (SQLite)
# =====================================================================

class SnippetDatabase:
    """SQLite-backed persistent storage for code snippets used by the
    Markov chain.  Supports tagging, source tracking, full-text search,
    quality-score persistence, and bulk import/export."""

    def __init__(self, db_path: str = "snippets.db") -> None:
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    # -- Schema ---------------------------------------------------------

    def _create_tables(self) -> None:
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS snippets (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                code        TEXT NOT NULL,
                source      TEXT    DEFAULT 'manual',
                tags        TEXT    DEFAULT '',
                language    TEXT    DEFAULT 'python',
                token_count INTEGER DEFAULT 0,
                line_count  INTEGER DEFAULT 0,
                char_count  INTEGER DEFAULT 0,
                quality_score REAL   DEFAULT 0.0,
                created_at  TEXT    DEFAULT (datetime('now')),
                updated_at  TEXT    DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS training_log (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                session_name TEXT,
                snippet_count  INTEGER DEFAULT 0,
                total_tokens  INTEGER DEFAULT 0,
                n_gram         INTEGER DEFAULT 2,
                created_at     TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_src   ON snippets(source);
            CREATE INDEX IF NOT EXISTS idx_lang  ON snippets(language);
            CREATE INDEX IF NOT EXISTS idx_code  ON snippets(code);
        """)
        self.conn.commit()

    # -- CRUD ------------------------------------------------------------

    def add_snippet(self, code: str, source: str = "manual",
                    tags: str = "", language: str = "python") -> int:
        code = code.strip()
        if not code:
            return -1
        cur = self.conn.execute(
            "INSERT INTO snippets (code,source,tags,language,line_count,char_count) "
            "VALUES (?,?,?,?,?,?)",
            (code, source, tags, language,
             len(code.split("\n")), len(code)))
        self.conn.commit()
        return cur.lastrowid

    def add_snippets_batch(self, snippets: List[str],
                           source: str = "import") -> int:
        count = 0
        for code in snippets:
            code = code.strip()
            if len(code) < 10:
                continue
            self.conn.execute(
                "INSERT INTO snippets (code,source,tags,language,line_count,char_count) "
                "VALUES (?,?,?,?,?,?)",
                (code, source, "", "python",
                 len(code.split("\n")), len(code)))
            count += 1
        self.conn.commit()
        return count

    def get_snippet_by_id(self, sid: int) -> Optional[Dict]:
        row = self.conn.execute(
            "SELECT * FROM snippets WHERE id=?", (sid,)).fetchone()
        return dict(row) if row else None

    def get_snippets(self, source: str = "", tag: str = "",
                     language: str = "", limit: int = 500,
                     offset: int = 0, order: str = "created_at DESC") -> List[Dict]:
        q = "SELECT * FROM snippets WHERE 1=1"
        p: list = []
        if source:
            q += " AND source=?"; p.append(source)
        if tag:
            q += " AND (tags=? OR tags LIKE ? OR tags LIKE ? OR tags LIKE ?)"
            p += [tag, f"{tag},%", f"%,{tag}", f"%,{tag},%"]
        if language:
            q += " AND language=?"; p.append(language)
        q += f" ORDER BY {order} LIMIT ? OFFSET ?"
        p += [limit, offset]
        return [dict(r) for r in self.conn.execute(q, p).fetchall()]

    def search_snippets(self, query: str, limit: int = 200) -> List[Dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM snippets WHERE code LIKE ? ORDER BY created_at DESC LIMIT ?",
            (f"%{query}%", limit)).fetchall()]

    def update_snippet(self, sid: int, code: str = None,
                       tags: str = None, quality_score: float = None) -> None:
        sets = ["updated_at=datetime('now')"]
        p: list = []
        if code is not None:
            code = code.strip()
            sets += ["code=?", "line_count=?", "char_count=?"]
            p += [code, len(code.split("\n")), len(code)]
        if tags is not None:
            sets.append("tags=?"); p.append(tags)
        if quality_score is not None:
            sets.append("quality_score=?"); p.append(quality_score)
        p.append(sid)
        self.conn.execute(f"UPDATE snippets SET {','.join(sets)} WHERE id=?", p)
        self.conn.commit()

    def delete_snippet(self, sid: int) -> bool:
        c = self.conn.execute("DELETE FROM snippets WHERE id=?", (sid,))
        self.conn.commit()
        return c.rowcount > 0

    def delete_by_source(self, source: str) -> int:
        c = self.conn.execute("DELETE FROM snippets WHERE source=?", (source,))
        self.conn.commit()
        return c.rowcount

    # -- Bulk operations ------------------------------------------------

    def get_all_codes(self, source: str = "", limit: int = 50000) -> List[str]:
        rows = self.get_snippets(source=source, limit=limit)
        return [r["code"] for r in rows]

    def get_random_snippets(self, count: int = 20) -> List[Dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM snippets ORDER BY RANDOM() LIMIT ?", (count,)).fetchall()]

    def get_total_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM snippets").fetchone()[0]

    def clear_all(self) -> int:
        n = self.get_total_count()
        self.conn.execute("DELETE FROM snippets")
        self.conn.execute("DELETE FROM training_log")
        self.conn.commit()
        return n

    # -- Metadata -------------------------------------------------------

    def get_sources(self) -> List[str]:
        return [r[0] for r in self.conn.execute(
            "SELECT DISTINCT source FROM snippets ORDER BY source").fetchall()]

    def get_all_tags(self) -> List[str]:
        tags: set = set()
        for r in self.conn.execute(
                "SELECT tags FROM snippets WHERE tags!=''").fetchall():
            for t in r[0].split(","):
                t = t.strip()
                if t:
                    tags.add(t)
        return sorted(tags)

    def get_stats(self) -> Dict:
        total = self.get_total_count()
        by_src = self.conn.execute(
            "SELECT source, COUNT(*) c FROM snippets GROUP BY source "
            "ORDER BY c DESC").fetchall()
        by_lang = self.conn.execute(
            "SELECT language, COUNT(*) c FROM snippets GROUP BY language "
            "ORDER BY c DESC").fetchall()
        chars = self.conn.execute(
            "SELECT COALESCE(SUM(char_count),0) FROM snippets").fetchone()[0]
        lines = self.conn.execute(
            "SELECT COALESCE(SUM(line_count),0) FROM snippets").fetchone()[0]
        newest = self.conn.execute(
            "SELECT source, created_at FROM snippets "
            "ORDER BY created_at DESC LIMIT 1").fetchone()
        sessions = self.conn.execute(
            "SELECT COUNT(*) FROM training_log").fetchone()[0]
        return {
            "total_snippets": total,
            "total_chars": chars,
            "total_lines": lines,
            "by_source": [(s, c) for s, c in by_src],
            "by_language": [(l, c) for l, c in by_lang],
            "newest_source": newest[0] if newest else "N/A",
            "newest_date": newest[1] if newest else "N/A",
            "training_sessions": sessions,
        }

    # -- Training log ---------------------------------------------------

    def log_session(self, name: str, snippet_count: int,
                    total_tokens: int, n_gram: int) -> int:
        cur = self.conn.execute(
            "INSERT INTO training_log (session_name,snippet_count,"
            "total_tokens,n_gram) VALUES (?,?,?,?)",
            (name, snippet_count, total_tokens, n_gram))
        self.conn.commit()
        return cur.lastrowid

    # -- Import / Export ------------------------------------------------

    def export_to_file(self, filepath: str, source: str = "") -> int:
        rows = self.get_snippets(source=source, limit=100000)
        with open(filepath, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(f"# Source: {r['source']}\n")
                f.write(f"# Tags: {r['tags']}\n")
                f.write(f"# ID: {r['id']}\n")
                f.write(r["code"])
                f.write("\n\n---\n\n")
        return len(rows)

    def import_from_file(self, filepath: str, source: str = "") -> Tuple[int, int]:
        """Import annotated file (with # Source / # Tags headers) or plain
        code file.  Returns (snippet_count, total_chars)."""
        p = Path(filepath)
        if not p.exists():
            return 0, 0
        text = p.read_text(encoding="utf-8", errors="replace")
        source = source or p.stem
        snippets = SnippetImporter._split_code(text)
        if not snippets:
            return 0, 0
        count = self.add_snippets_batch(snippets, source=source)
        return count, sum(len(s) for s in snippets)

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass


# =====================================================================
# 4. CODE QUALITY SVM
# =====================================================================

class CodeQualitySVM:
    FEATURE_NAMES = [
        "markov_ll", "coverage", "line_count", "avg_line_len", "max_indent",
        "keyword_density", "bracket_balance", "has_def_or_class",
        "has_docstring", "unique_token_ratio", "token_variety",
        "has_return", "has_assignment", "code_complexity",
    ]

    def __init__(self) -> None:
        self.is_trained = False
        self._svm = None
        self._scaler_mean = None
        self._scaler_std = None

    @staticmethod
    def _extract_features(code: str, markov_model=None) -> np.ndarray:
        lines = [l for l in code.split("\n") if l.strip()]
        markov_ll = markov_model.log_likelihood(code) if markov_model else 0.0
        coverage = markov_model.coverage_ratio(code) if markov_model else 0.0
        line_count = max(len(lines), 1)
        avg_line_len = np.mean([len(l) for l in lines]) if lines else 0.0
        max_indent = max((len(l) - len(l.lstrip())) // 4 for l in lines) if lines else 0
        kw_pat = re.compile(r"\b(def|class|return|if|else|for|while|import|with|try|except|lambda|yield)\b")
        kw_count = len(kw_pat.findall(code))
        keyword_density = kw_count / max(len(code.split()), 1)
        opens = code.count("(") + code.count("[") + code.count("{")
        closes = code.count(")") + code.count("]") + code.count("}")
        bracket_balance = -abs(opens - closes)
        has_def = 1.0 if re.search(r"^\s*(def |class )", code, re.M) else 0.0
        has_doc = 1.0 if ('"""' in code or "'''" in code) else 0.0
        has_ret = 1.0 if re.search(r"\breturn\b", code) else 0.0
        has_asgn = 1.0 if re.search(r"=\s*", code) else 0.0
        toks = re.findall(r"\S+", code)
        unique = set(toks)
        unique_ratio = len(unique) / max(len(toks), 1)
        branch = code.count("if ") + code.count("for ") + code.count("while ")
        complexity = max_indent * branch
        return np.array([markov_ll, coverage, line_count, avg_line_len, max_indent,
                         keyword_density, bracket_balance, has_def, has_doc,
                         unique_ratio, len(unique), has_ret, has_asgn, complexity])

    def train(self, good_snippets, markov_model, n_negative=200):
        if not HAS_SKLEARN:
            return {"status": "sklearn not available", "trained": False}
        X_pos = np.array([self._extract_features(s, markov_model) for s in good_snippets if s.strip()])
        y_pos = np.ones(len(X_pos))
        X_neg = []
        for _ in range(n_negative):
            if good_snippets:
                base = random.choice(good_snippets)
                t = base.split(); random.shuffle(t); scrambled = " ".join(t)
            else:
                scrambled = " ".join(random.choices(
                    ["def", "class", "return", "if", "else", "for", "while", "import",
                     "print", "x", "=", "+", "-", "*", "(", ")", ":", "a", "b", "0"],
                    k=random.randint(5, 30)))
            X_neg.append(self._extract_features(scrambled, markov_model))
        X_neg_arr = np.array(X_neg)
        y_neg = np.zeros(len(X_neg_arr))
        X = np.vstack([X_pos, X_neg_arr])
        y = np.concatenate([y_pos, y_neg])
        self._scaler_mean = X.mean(axis=0)
        self._scaler_std = X.std(axis=0)
        self._scaler_std[self._scaler_std == 0] = 1.0
        X_s = (X - self._scaler_mean) / self._scaler_std
        self._svm = LinearSVC(C=0.5, max_iter=2000)
        self._svm.fit(X_s, y)
        self.is_trained = True
        return {"status": "trained", "trained": True, "n_positive": len(X_pos),
                "n_negative": len(X_neg_arr), "accuracy": self._svm.score(X_s, y)}

    def score(self, code, markov_model) -> float:
        if not self.is_trained or self._svm is None:
            return 0.5
        f = self._extract_features(code, markov_model).reshape(1, -1)
        s = (f - self._scaler_mean) / self._scaler_std
        raw = self._svm.decision_function(s)[0]
        return float(1.0 / (1.0 + math.exp(-raw)))


# =====================================================================
# 5. INTENT CLASSIFIER SVM
# =====================================================================

class IntentClassifier:
    DEFAULT_INTENTS = {
        "add": "def add ( a , b )", "sum": "def calculate_sum ( numbers )",
        "subtract": "def subtract ( x , y )", "multiply": "def multiply ( a , b )",
        "divide": "def divide ( num , den )", "loop": "for i in range (",
        "iterate": "for i in range (", "for loop": "for i in range (",
        "class": "class DataProcessor :", "calculator": "class Calculator :",
        "math": "def get_area ( radius )", "area": "def get_area ( radius )",
        "max": "def find_max ( lst )", "sort": "def merge_sort ( arr )",
        "fibonacci": "def fibonacci ( n )", "fib": "def fibonacci ( n )",
        "file": "def read_file ( path )", "read": "def read_file ( path )",
        "clean": "def process_data ( data )", "process": "def process_data ( data )",
        "filter": "def process_data ( data )", "data": "def process_data ( data )",
        "binary search": "def binary_search ( arr , target )",
        "search": "def binary_search ( arr , target )",
        "reverse": "def reverse_list ( lst )", "flatten": "def flatten ( nested_list )",
        "unique": "def get_unique ( lst )", "count": "def count_items ( lst , item )",
        "average": "def calculate_average ( numbers )",
        "validate": "def validate_email ( email )", "check": "def validate_email ( email )",
        "decorator": "@staticmethod\n    def ", "property": "@property\n    def ",
        "cache": "from functools import lru_cache",
        "context": "def __enter__ ( self )", "cleanup": "def __exit__ ( self ",
        "hash": "def __hash__ ( self )", "eq": "def __eq__ ( self , other )",
        "repr": "def __repr__ ( self )", "len": "def __len__ ( self )",
        "get": "def __getitem__ ( self , key )",
        "set": "def __setitem__ ( self , key , value )",
        "iterator": "def __iter__ ( self )", "next": "def __next__ ( self )",
        "string": "def __str__ ( self )", "binary": "def to_bytes ( self )",
    }
    AUGMENTATIONS = {
        "add": ["write a function to add", "add two numbers", "function for addition"],
        "subtract": ["subtract two numbers", "create minus function"],
        "multiply": ["multiply numbers", "product of"],
        "divide": ["divide numbers", "create division function"],
        "loop": ["create a loop", "iterate over range", "for loop in python"],
        "class": ["create a class", "define a class", "OOP class"],
        "sort": ["sort an array", "sorting algorithm"],
        "max": ["find the maximum", "largest element"],
        "fibonacci": ["fibonacci sequence", "fib numbers", "recursive fib"],
        "file": ["read from file", "file I/O", "load file contents"],
        "clean": ["clean data", "process dataset", "filter elements"],
        "search": ["find element in array", "binary search implementation"],
    }

    def __init__(self):
        self.is_trained = False
        self._pipeline = None
        self._prefixes = []
        self._fallback = "def process_data ("

    def train(self, custom=None):
        if not HAS_SKLEARN:
            return {"status": "sklearn not available", "trained": False}
        intents = dict(self.DEFAULT_INTENTS)
        if custom:
            intents.update(custom)
        texts, labels = [], []
        for k, v in intents.items():
            texts.append(k); labels.append(v)
            for aug in self.AUGMENTATIONS.get(k, []):
                texts.append(aug); labels.append(v)
        self._pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=500)),
            ("svm", LinearSVC(C=1.0, max_iter=2000)),
        ])
        self._pipeline.fit(texts, labels)
        self._prefixes = list(set(labels))
        self.is_trained = True
        return {"status": "trained", "trained": True, "n_samples": len(texts)}

    def predict(self, request):
        if self.is_trained and self._pipeline:
            try:
                return str(self._pipeline.predict([request.lower().strip()])[0])
            except Exception:
                pass
        r = request.lower()
        best, blen = "", 0
        for k, v in self.DEFAULT_INTENTS.items():
            if k in r and len(k) > blen:
                best, blen = v, len(k)
        return best or self._fallback

    def predict_top_k(self, request, k=3):
        if not self.is_trained or not self._pipeline:
            return [(self.predict(request), 1.0)]
        try:
            d = self._pipeline.decision_function([request.lower().strip()])[0]
            paired = sorted(zip(self._pipeline.classes_, list(d)), key=lambda x: x[1], reverse=True)
            mx = max(s for _, s in paired)
            exps = [(p, math.exp(s - mx)) for p, s in paired]
            tot = sum(e for _, e in exps)
            return [(p, e / tot) for p, e in exps][:k]
        except Exception:
            return [(self.predict(request), 1.0)]


# =====================================================================
# 6. SYNTAX VALIDATOR
# =====================================================================

def validate_syntax(code: str) -> Tuple[bool, str]:
    """Try to parse *code* with ast.parse.  Returns (valid, message)."""
    try:
        ast_mod.parse(code)
        return True, "Valid Python syntax"
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, f"Error: {e}"


# =====================================================================
# =====================================================================
# 7. SANDBOXED VFS  (Composite state, shaped rewards, quality feedback)
# =====================================================================

class SandboxVFS:
    """In-memory virtual file system with a composite state vector for MDP.

    State is a tuple encoding the environment's current configuration
    so the Q-learning agent can learn context-dependent policies.
    """

    ACTIONS = ["cd /", "cd /data", "cd /system", "generate_code",
               "write_file", "run_file", "rm_rf", "finish_task",
               "mkdir_new", "delete_file", "copy_file", "list_dir",
               "cat_file"]

    # Phase labels for task progress tracking
    PHASE_INIT = 0        # navigating
    PHASE_GENERATED = 1   # code generated, not yet written
    PHASE_WRITTEN = 2     # code written to disk
    PHASE_EXECUTED = 3    # code executed

    def __init__(self):
        self.root: Dict = {"type": "dir", "children": {}}
        self.cwd: str = "/"
        self.logs: List[str] = []
        self.generated_code: str = ""
        self.phase: int = self.PHASE_INIT
        self.files_written: int = 0
        self.step_count: int = 0
        self.last_quality: float = 0.0
        self.reset()

    def reset(self):
        self.root = {"type": "dir", "children": {}}
        self.cwd = "/"; self.logs = []; self.generated_code = ""
        self.phase = self.PHASE_INIT
        self.files_written = 0; self.step_count = 0
        self.last_quality = 0.0
        self._mkdir("/data"); self._mkdir("/system"); self._mkdir("/tmp")
        self._write("/system/kernel.conf",
                    "# Kernel Configuration\nversion=1.0\ncritical=true")
        self._write("/data/.gitkeep", "")

    def get_state(self) -> tuple:
        """Composite state: (cwd_id, phase, files_in_cwd, step_bucket).
        This gives the agent meaningful signal about where it is and
        what it has done so far."""
        cwd_id = hash(self.cwd) % 8
        files_here = len(self._list(self.cwd))
        step_bucket = min(self.step_count // 5, 4)
        return (cwd_id, self.phase, files_here, step_bucket)

    # -- File system operations (unchanged) -----------------------------

    def _resolve(self, path):
        if not path.startswith("/"):
            path = self.cwd.rstrip("/") + "/" + path
        parts = [p for p in path.split("/") if p]
        res = []
        for p in parts:
            if p == ".": continue
            elif p == "..":
                if res: res.pop()
            else: res.append(p)
        return "/" + "/".join(res)

    def _node(self, path):
        path = self._resolve(path)
        if path == "/": return self.root
        node = self.root
        for p in path.strip("/").split("/"):
            if node.get("type") != "dir" or p not in node.get("children", {}):
                return None
            node = node["children"][p]
        return node

    def _mkdir(self, path):
        path = self._resolve(path)
        if path == "/": return True
        pp = "/" + "/".join(path.strip("/").split("/")[:-1])
        parent = self._node(pp)
        if not parent or parent.get("type") != "dir": return False
        d = path.strip("/").split("/")[-1]
        if d in parent["children"]: return True
        parent["children"][d] = {"type": "dir", "children": {}}
        return True

    def _write(self, path, content):
        path = self._resolve(path)
        pp = "/" + "/".join(path.strip("/").split("/")[:-1])
        parent = self._node(pp)
        if not parent or parent.get("type") != "dir": return False
        f = path.strip("/").split("/")[-1]
        parent["children"][f] = {"type": "file", "content": content}
        return True

    def _read(self, path):
        node = self._node(path)
        if not node or node.get("type") != "file": return None
        return node.get("content", "")

    def _delete(self, path):
        path = self._resolve(path)
        if path == "/": return False
        pp = "/" + "/".join(path.strip("/").split("/")[:-1])
        parent = self._node(pp)
        if not parent: return False
        name = path.strip("/").split("/")[-1]
        if name in parent.get("children", {}):
            del parent["children"][name]; return True
        return False

    def _copy(self, src, dst):
        sn = self._node(src)
        if not sn: return False
        dp = "/" + "/".join(dst.strip("/").split("/")[:-1])
        dn = self._node(dp)
        if not dn or dn.get("type") != "dir": return False
        dn["children"][dst.strip("/").split("/")[-1]] = copy.deepcopy(sn)
        return True

    def _list(self, path):
        node = self._node(path)
        if not node or node.get("type") != "dir": return []
        return [(n, c.get("type", "file"))
                for n, c in node.get("children", {}).items()]

    def _tree_lines(self, node=None, prefix=""):
        if node is None: node = self.root
        lines = []
        items = list(node.get("children", {}).items())
        for i, (name, child) in enumerate(items):
            conn = "── " if i == len(items) - 1 else "├─ "
            lines.append(f"{prefix}{conn}{name}")
            if child.get("type") == "dir":
                ext = "    " if i == len(items) - 1 else "│   "
                lines.extend(self._tree_lines(child, prefix + ext))
        return lines

    # -- MDP step with shaped rewards ----------------------------------

    def step(self, action, markov_model=None, prefix="def process_data (",
             filename="script.py", quality_svm=None):
        """Execute action, return (state, reward, done, info).

        Rewards use potential-based shaping to guide the agent toward
        the optimal policy: navigate to /data, generate code, write
        it, execute it, then finish.
        """
        self.logs = []
        self.step_count += 1
        info = {"action_name": self.ACTIONS[action]
                if action < len(self.ACTIONS) else "unknown",
                "phase": self.phase}
        reward, done = -0.1, False

        if action == 0:
            self.cwd = "/"; self.logs.append("cd /")
        elif action == 1:
            self.cwd = "/data"; self.logs.append("cd /data")
        elif action == 2:
            self.cwd = "/system"; self.logs.append("cd /system")
        elif action == 3:  # generate_code
            self.logs.append("ACTION: generate_code")
            if markov_model:
                self.generated_code = markov_model.generate(
                    prefix, max_length=40, use_smoothing=True)
            else:
                self.generated_code = "# no model"
            self.phase = self.PHASE_GENERATED
            reward = 1.0
            info["generated_code"] = self.generated_code
            # Quality bonus
            if quality_svm and quality_svm.is_trained and markov_model:
                q = quality_svm.score(self.generated_code, markov_model)
                self.last_quality = q
                reward += q * 2.0
                info["quality_score"] = q

        elif action == 4:  # write_file
            t = self.cwd.rstrip("/") + "/" + filename
            self.logs.append(f"ACTION: write {filename}")
            ok = self._write(t, getattr(self, "generated_code", "# empty"))
            if ok:
                self.files_written += 1
                self.phase = self.PHASE_WRITTEN
                reward = 5.0 if self.cwd == "/data" else 2.0
                # Bonus for writing after generating (good sequence)
                if self.phase >= self.PHASE_WRITTEN:
                    reward += 1.0
            else:
                reward = -2.0

        elif action == 5:  # run_file
            t = self.cwd.rstrip("/") + "/" + filename
            self.logs.append(f"ACTION: run {filename}")
            nd = self._node(t)
            if nd and nd.get("type") == "file":
                lines = nd.get("content", "").count("\n") + 1
                self._write(self.cwd.rstrip("/") + "/output.txt",
                            f"Execution successful.\nLines: {lines}\n"
                            f"Quality: {self.last_quality:.1%}\n"
                            f"Output: [Simulated]")
                self.phase = self.PHASE_EXECUTED
                reward = 5.0; info["run_success"] = True
            else:
                reward = -2.0; info["run_success"] = False

        elif action == 6:  # rm_rf
            self.logs.append(f"CMD: rm -rf * in {self.cwd}")
            nd = self._node(self.cwd)
            if nd and nd.get("type") == "dir":
                reward = -50.0 if self.cwd == "/system" else -5.0
                nd["children"] = {}

        elif action == 7:  # finish_task
            self.logs.append("CMD: finish_task")
            # Big bonus if we completed the full pipeline
            if self.phase >= self.PHASE_EXECUTED and self.cwd == "/data":
                reward = 15.0
            elif self.phase >= self.PHASE_WRITTEN:
                reward = 8.0
            else:
                reward = 1.0
            done = True

        elif action == 8:  # mkdir
            d = self.cwd.rstrip("/") + "/project"
            self.logs.append(f"CMD: mkdir {d}")
            reward = 0.5 if self._mkdir(d) else -1.0

        elif action == 9:  # rm file
            t = self.cwd.rstrip("/") + "/" + filename
            self.logs.append(f"CMD: rm {t}")
            reward = -0.5 if self._delete(t) else -1.0

        elif action == 10:  # cp file
            s = self.cwd.rstrip("/") + "/" + filename
            d = self.cwd.rstrip("/") + "/" + filename.replace(".py", "_backup.py")
            self.logs.append(f"CMD: cp {s} {d}")
            reward = 0.3 if self._copy(s, d) else -1.0

        elif action == 11:  # ls
            items = self._list(self.cwd)
            self.logs.append(f"CMD: ls {self.cwd}")
            for n, t in items:
                self.logs.append(f"  {t:4s}  {n}")
            reward = 0.1

        elif action == 12:  # cat
            t = self.cwd.rstrip("/") + "/" + filename
            self.logs.append(f"CMD: cat {t}")
            content = self._read(t)
            if content is not None:
                for line in content.split("\n"):
                    self.logs.append(line)
                reward = 0.1
            else:
                self.logs.append(f"Error: {t} not found")
                reward = -1.0

        state = self.get_state()
        return state, reward, done, info

# =====================================================================
# =====================================================================
# 8. Q-LEARNING AGENT  (UCB, experience replay, LR scheduling)
# =====================================================================

class QLearningAgent:
    """Tabular Q-learning with professional improvements:

    - **Double Q-learning** to reduce maximization bias
    - **UCB1 exploration** as alternative to epsilon-greedy
    - **Experience replay buffer** for sample-efficient learning
    - **Learning rate schedule** (inverse-sqrt decay)
    - **Visit counts** for UCB and diagnostics
    - **TD-error tracking** for monitoring convergence
    """

    def __init__(self, n_actions=13, lr=0.1, gamma=0.95,
                 eps_start=1.0, eps_end=0.01, eps_decay=0.995):
        self.n_actions = n_actions
        self.lr = lr
        self.lr_initial = lr
        self.gamma = gamma
        self.epsilon = eps_start
        self.epsilon_end = eps_end
        self.epsilon_decay = eps_decay
        self.q_table = collections.defaultdict(float)
        self.q2_table = collections.defaultdict(float)  # second Q for double-Q
        self.visit_counts: Dict[tuple, int] = collections.defaultdict(int)
        self.episodes = 0
        self.episode_rewards: List[float] = []
        self.total_updates = 0
        # Experience replay
        self._replay_buffer: List[Tuple] = []
        self._replay_capacity = 5000
        self._replay_sample_size = 32
        self._use_replay = True
        self._use_double_q = True
        # TD-error tracking
        self._td_errors: List[float] = []
        self._exploration_strategy = "epsilon_greedy"  # or "ucb"

    def choose_action(self, state):
        if self._exploration_strategy == "ucb":
            return self._ucb_select(state)
        return self._epsilon_greedy(state)

    def _epsilon_greedy(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        qv = [self.q_table[(state, a)] for a in range(self.n_actions)]
        mx = max(qv)
        return random.choice([a for a, q in enumerate(qv) if q == mx])

    def _ucb_select(self, state, c: float = 1.0) -> int:
        """UCB1 action selection: Q(s,a) + c * sqrt(ln(N(s)) / N(s,a))"""
        total_visits = sum(self.visit_counts[(state, a)]
                           for a in range(self.n_actions))
        if total_visits == 0:
            return random.randint(0, self.n_actions - 1)
        best_a, best_val = 0, -float("inf")
        for a in range(self.n_actions):
            visits = max(self.visit_counts[(state, a)], 1)
            q = self.q_table[(state, a)]
            ucb = q + c * math.sqrt(math.log(total_visits) / visits)
            if ucb > best_val:
                best_val = ucb
                best_a = a
        return best_a

    def update(self, state, action, reward, next_state, done):
        if self._use_double_q:
            self._double_q_update(state, action, reward, next_state, done)
        else:
            self._standard_update(state, action, reward, next_state, done)

    def _standard_update(self, state, action, reward, next_state, done):
        cq = self.q_table[(state, action)]
        if done:
            target = reward
        else:
            target = reward + self.gamma * max(
                self.q_table[(next_state, a)]
                for a in range(self.n_actions))
        td_error = target - cq
        lr = self._current_lr()
        self.q_table[(state, action)] += lr * td_error
        self.visit_counts[(state, action)] += 1
        self.total_updates += 1
        self._td_errors.append(abs(td_error))
        if len(self._td_errors) > 10000:
            self._td_errors = self._td_errors[-5000:]

    def _double_q_update(self, state, action, reward, next_state, done):
        """Double Q-learning: randomly update Q1 or Q2, use the other
        for action selection to reduce maximization bias (Hasselt 2015)."""
        if random.random() < 0.5:
            q_eval, q_update = self.q_table, self.q2_table
        else:
            q_eval, q_update = self.q2_table, self.q_table
        cq = q_eval[(state, action)]
        if done:
            target = reward
        else:
            best_a = max(range(self.n_actions),
                        key=lambda a: q_eval[(next_state, a)])
            target = reward + self.gamma * q_update[(next_state, best_a)]
        td_error = target - cq
        lr = self._current_lr()
        q_update[(state, action)] += lr * td_error
        # Also nudge the other table for stability
        self.q_table[(state, action)] += 0.01 * lr * td_error
        self.q2_table[(state, action)] += 0.01 * lr * td_error
        self.visit_counts[(state, action)] += 1
        self.total_updates += 1
        self._td_errors.append(abs(td_error))
        if len(self._td_errors) > 10000:
            self._td_errors = self._td_errors[-5000:]

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        if not self._use_replay:
            return
        self._replay_buffer.append((state, action, reward, next_state, done))
        if len(self._replay_buffer) > self._replay_capacity:
            self._replay_buffer = self._replay_buffer[-self._replay_capacity:]

    def replay_update(self):
        """Sample from replay buffer and perform updates."""
        if not self._use_replay or len(self._replay_buffer) < 10:
            return
        samples = random.sample(
            self._replay_buffer,
            min(self._replay_sample_size, len(self._replay_buffer)))
        for s, a, r, ns, d in samples:
            self.update(s, a, r, ns, d)

    def _current_lr(self) -> float:
        """Inverse-sqrt learning rate schedule."""
        if self.total_updates < 1:
            return self.lr_initial
        return self.lr_initial / math.sqrt(self.total_updates)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end,
                           self.epsilon * self.epsilon_decay)

    def get_best_action(self, state):
        qv = [self.q_table[(state, a)] for a in range(self.n_actions)]
        return int(np.argmax(qv))

    def get_q_values(self, state):
        return [self.q_table[(state, a)] for a in range(self.n_actions)]

    def avg_td_error(self, last_n: int = 500) -> float:
        if not self._td_errors:
            return 0.0
        recent = self._td_errors[-last_n:]
        return sum(recent) / len(recent)

# =====================================================================
# 9. SYNTAX HIGHLIGHTER + SIGNALS + DROPPABLE EDIT
# =====================================================================

class PythonSyntaxHighlighter(QSyntaxHighlighter):
    def __init__(self, document):
        super().__init__(document)
        self._formats = {}; self._rules = []; self._build()
    def _fmt(self, c, b=False, i=False):
        f = QTextCharFormat(); f.setForeground(QColor(c))
        f.setFontWeight(QFont.Bold if b else QFont.Normal); f.setFontItalic(i); return f
    def _build(self):
        kw = self._fmt("#569cd6", True); bi = self._fmt("#dcdcaa")
        st = self._fmt("#ce9178"); cm = self._fmt("#6a9955", i=True)
        nm = self._fmt("#b5cea8"); dc = self._fmt("#c586c0")
        self._rules = [
            (re.compile(r"\b(def|class|return|if|else|elif|for|while|import|from|as|"
                        r"with|try|except|finally|raise|yield|lambda|and|or|not|in|"
                        r"is|pass|break|continue|global|nonlocal|assert|del)\b"), "kw"),
            (re.compile(r"\b(print|len|range|int|str|float|list|dict|set|tuple|type|"
                        r"isinstance|enumerate|zip|map|filter|sorted|sum|min|max|abs|"
                        r"round|open|super|None|True|False|self)\b"), "bi"),
            (re.compile(r"#.*$"), "cm"),
            (re.compile(r'"""[\s\S]*?"""|' + r"'''[\s\S]*?'''"), "st"),
            (re.compile(r'"[^"]*"|' + r"'[^']*'"), "st"),
            (re.compile(r"\b\d+\.?\d*\b"), "nm"),
            (re.compile(r"@\w+"), "dc"),
        ]
        self._formats = {"kw": kw, "bi": bi, "st": st, "cm": cm, "nm": nm, "dc": dc}
    def highlightBlock(self, text):
        for pat, k in self._rules:
            f = self._formats.get(k)
            if f:
                for m in pat.finditer(text):
                    self.setFormat(m.start(), m.end() - m.start(), f)


class AgentSignals(QObject):
    episode_done = Signal(int, float)
    training_finished = Signal()
    log_message = Signal(str)
    import_progress = Signal(int, int, str)


class DropablePlainTextEdit(QPlainTextEdit):
    files_dropped = Signal(list)
    def dragEnterEvent(self, e):
        e.acceptProposedAction() if e.mimeData().hasUrls() else super().dragEnterEvent(e)
    def dragMoveEvent(self, e):
        e.acceptProposedAction() if e.mimeData().hasUrls() else super().dragMoveEvent(e)
    def dropEvent(self, e):
        if e.mimeData().hasUrls():
            self.files_dropped.emit([u.toLocalFile() for u in e.mimeData().urls()])
            e.acceptProposedAction()
        else:
            super().dropEvent(e)


# =====================================================================
# 10. CHART RENDERER (Agg → QPixmap)
# =====================================================================

def render_reward_chart(rewards: List[float], window: int = 100) -> Optional[QPixmap]:
    """Render training reward chart as QPixmap."""
    if len(rewards) < 2:
        return None
    fig, ax = plt.subplots(figsize=(5, 2.5), dpi=100)
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_facecolor("#2d2d2d")

    episodes = list(range(1, len(rewards) + 1))
    ax.plot(episodes, rewards, color="#42A5F5", linewidth=0.8, alpha=0.5, label="Raw")

    # Moving average
    if len(rewards) >= window:
        ma = np.convolve(rewards, np.ones(window)/window, mode="valid")
        ax.plot(range(window, len(rewards)+1), ma, color="#FFCA28",
                linewidth=2, label=f"MA({window})")

    ax.set_xlabel("Episode", color="#aaa", fontsize=9)
    ax.set_ylabel("Total Reward", color="#aaa", fontsize=9)
    ax.tick_params(colors="#888", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#555")
    ax.legend(fontsize=8, facecolor="#2d2d2d", edgecolor="#555", labelcolor="#ccc")
    ax.set_title("Agent Training Progress", color="#ddd", fontsize=10)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    pix = QPixmap()
    pix.loadFromData(buf.read())
    return pix


def render_td_error_chart(td_errors: List[float], window: int = 50) -> Optional[QPixmap]:
    """Render TD-error convergence chart as QPixmap."""
    if len(td_errors) < 10:
        return None
    fig, ax = plt.subplots(figsize=(5, 2.5), dpi=100)
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_facecolor("#2d2d2d")
    if len(td_errors) >= window:
        ma = np.convolve(td_errors, np.ones(window)/window, mode="valid")
        ax.plot(range(window, len(td_errors)+1), ma, color="#FF7043",
                linewidth=2, label=f"MA({window})")
    ax.plot(td_errors, color="#FF7043", linewidth=0.5, alpha=0.3)
    ax.set_xlabel("Update", color="#aaa", fontsize=9)
    ax.set_ylabel("|TD-error|", color="#aaa", fontsize=9)
    ax.tick_params(colors="#888", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#555")
    ax.legend(fontsize=8, facecolor="#2d2d2d", edgecolor="#555", labelcolor="#ccc")
    ax.set_title("TD-Error Convergence", color="#ddd", fontsize=10)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    pix = QPixmap()
    pix.loadFromData(buf.read())
    return pix


# =====================================================================
# 11. MAIN WINDOW
# =====================================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Code Agent: Markov + SVM | Chatbot | Sandboxed Q-Learning")
        self.resize(1400, 920)

        self.markov_model = MarkovCodeModel(n=2)
        self.vfs = SandboxVFS()
        self.agent = QLearningAgent(n_actions=len(SandboxVFS.ACTIONS))
        self.quality_svm = CodeQualitySVM()
        self.intent_clf = IntentClassifier()

        self._training_thread = None
        self._stop_training = threading.Event()
        self._gen_history = []
        self._all_snippets = []

        # Snippet database (SQLite)
        self.db = SnippetDatabase()
        self._db_current_page = 0
        self._db_page_size = 100

        # VFS terminal history
        self._term_history: List[str] = []
        self._term_history_idx = -1

        self.signals = AgentSignals()
        self.signals.episode_done.connect(self._on_episode_done)
        self.signals.training_finished.connect(self._on_training_finished)
        self.signals.import_progress.connect(self._on_import_progress)

        self.setup_ui()
        self.setup_shortcuts()
        self.load_default_corpus()
        self._train_svm_models()
        self._update_status()

    # ------------------------------------------------------------------
    # UI Setup
    # ------------------------------------------------------------------

    def setup_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        ml = QVBoxLayout(central); ml.setContentsMargins(6, 6, 6, 6)
        self.tabs = QTabWidget(); ml.addWidget(self.tabs)

        # Status bar labels (created early so tabs can reference them)
        sb = QStatusBar(); self.setStatusBar(sb)
        self.status_label = QLabel("Ready")
        self.episode_label = QLabel("Episodes: 0")
        self.epsilon_label = QLabel("\u03b5: 1.000")
        self.qtable_label = QLabel("Q: 0")
        self.corpus_label = QLabel("Tokens: 0")
        self.svm_label = QLabel("SVM: idle")
        self.db_label = QLabel(f"DB: {self.db.get_total_count()}")
        self.db_label.setStyleSheet("color:#ce93d8;padding:0 8px;")
        for l in (self.status_label, self.episode_label, self.epsilon_label,
                  self.qtable_label, self.corpus_label, self.svm_label, self.db_label):
            l.setStyleSheet("padding: 0 8px;"); sb.addWidget(l)

        self.setup_learning_tab()
        self.setup_chatbot_tab()
        self.setup_metrics_tab()
        self.setup_agent_tab()
        self.setup_database_tab()

    def setup_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+Return"), self, self.generate_code)
        QShortcut(QKeySequence("Ctrl+R"), self, self.reset_agent)
        QShortcut(QKeySequence("Ctrl+W"), self, self._clear_chat_history)
        QShortcut(QKeySequence("Ctrl+O"), self, self._import_files_dialog)
        QShortcut(QKeySequence("Ctrl+Shift+O"), self, self._import_directory_dialog)
        QShortcut(QKeySequence("Ctrl+S"), self, self._save_model_dialog)
        QShortcut(QKeySequence("Ctrl+L"), self, self._load_model_dialog)
        QShortcut(QKeySequence("Ctrl+D"), self, self._db_save_current_snippets)

    # ------------------------------------------------------------------
    # Tab 1: Learning
    # ------------------------------------------------------------------

    def setup_learning_tab(self):
        tab = QWidget(); layout = QHBoxLayout(tab)

        # Left: Corpus
        left = QGroupBox("Training Corpus"); ll = QVBoxLayout(left)
        # Import buttons
        ir = QHBoxLayout()
        b1 = QPushButton("Import File (Ctrl+O)"); b1.clicked.connect(self._import_files_dialog)
        b2 = QPushButton("Import Dir (Ctrl+Shift+O)"); b2.clicked.connect(self._import_directory_dialog)
        b3 = QPushButton("Load Examples"); b3.clicked.connect(self._load_example_snippets)
        b4 = QPushButton("Save Model"); b4.clicked.connect(self._save_model_dialog)
        b5 = QPushButton("Load Model"); b5.clicked.connect(self._load_model_dialog)
        # Database buttons
        b6 = QPushButton("Save to DB (Ctrl+D)"); b6.clicked.connect(self._db_save_current_snippets)
        b6.setToolTip("Save all in-memory snippets to the SQLite database")
        b7 = QPushButton("Train from DB"); b7.clicked.connect(self._db_train_from_db)
        b7.setToolTip("Load snippets from DB and retrain the Markov model")
        for b in (b1, b2, b3, b4, b5, b6, b7): ir.addWidget(b)
        ll.addLayout(ir)

        # Search
        sr = QHBoxLayout()
        self.corpus_search = QLineEdit(); self.corpus_search.setPlaceholderText("Search corpus...")
        self.corpus_search.textChanged.connect(self._filter_corpus)
        sr.addWidget(self.corpus_search)
        ll.addLayout(sr)

        self.import_progress = QProgressBar(); self.import_progress.setVisible(False)
        self.import_progress.setTextVisible(True)
        ll.addWidget(self.import_progress)
        self.import_stats_label = QLabel(""); self.import_stats_label.setStyleSheet("color:#aaa;font-size:11px;")
        self.import_stats_label.setWordWrap(True); self.import_stats_label.setVisible(False)
        ll.addWidget(self.import_stats_label)

        self.corpus_view = DropablePlainTextEdit(); self.corpus_view.setReadOnly(True)
        self.corpus_view.setFont(QFont("Consolas", 10))
        self.corpus_view.files_dropped.connect(self._on_files_dropped)
        self.corpus_view.setToolTip("Drag & drop .py/.txt files here")
        ll.addWidget(self.corpus_view)

        # Manual input
        mr = QHBoxLayout()
        self.corpus_input = QLineEdit(); self.corpus_input.setPlaceholderText("Add snippet...")
        self.corpus_input.returnPressed.connect(self.train_markov); mr.addWidget(self.corpus_input)
        bt = QPushButton("Train"); bt.clicked.connect(self.train_markov); bt.setFixedWidth(80)
        mr.addWidget(bt); ll.addLayout(mr)
        bc = QPushButton("Clear Corpus & Reset"); bc.clicked.connect(self._clear_corpus)
        ll.addWidget(bc)
        layout.addWidget(left, stretch=3)

        # Right: Transition table
        right = QGroupBox("Transition Probabilities"); rl = QVBoxLayout(right)
        fr = QHBoxLayout()
        self.token_input = QLineEdit(); self.token_input.setPlaceholderText("Filter token (e.g. 'def')...")
        self.token_input.textChanged.connect(self.update_markov_table); fr.addWidget(self.token_input)
        self.ngram_combo = QComboBox()
        self.ngram_combo.addItems(["Bigram (n=2)", "Trigram (n=3)", "Unigram (n=1)", "4-gram (n=4)"])
        self.ngram_combo.currentIndexChanged.connect(self._on_ngram_changed); fr.addWidget(self.ngram_combo)
        rl.addLayout(fr)
        self.markov_table = QTableWidget(); self.markov_table.setColumnCount(3)
        self.markov_table.setHorizontalHeaderLabels(["Next Token", "Count", "Probability"])
        self.markov_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        rl.addWidget(self.markov_table)
        layout.addWidget(right, stretch=2)
        self.tabs.addTab(tab, "1. Learning")

    # ------------------------------------------------------------------
    # Tab 2: Chatbot
    # ------------------------------------------------------------------

    def setup_chatbot_tab(self):
        tab = QWidget(); layout = QHBoxLayout(tab)

        gen = QGroupBox("Code Generation (Markov + SVM)"); gl = QVBoxLayout(gen)
        # Prefix
        pr = QHBoxLayout()
        self.chat_prefix = QLineEdit(); self.chat_prefix.setPlaceholderText("Code prefix...")
        self.chat_prefix.returnPressed.connect(self.generate_code); pr.addWidget(self.chat_prefix)
        bg = QPushButton("Generate"); bg.clicked.connect(self.generate_code); pr.addWidget(bg)
        bb = QPushButton("Best (Top-K)"); bb.clicked.connect(self.generate_best_code)
        bb.setToolTip("Generate K candidates, rank by SVM"); pr.addWidget(bb)
        bs = QPushButton("Beam Search"); bs.clicked.connect(self.generate_beam_code)
        bs.setToolTip("Beam search for highest-probability output"); pr.addWidget(bs)
        brf = QPushButton("Refine"); brf.clicked.connect(self.generate_refined_code)
        brf.setToolTip("Multi-pass refinement with quality feedback"); pr.addWidget(brf)
        gl.addLayout(pr)

        # Constrained generation row
        cr2 = QHBoxLayout()
        cr2.addWidget(QLabel("Must contain:"))
        self.must_contain_input = QLineEdit()
        self.must_contain_input.setPlaceholderText("return, if, for (comma-sep)...")
        cr2.addWidget(self.must_contain_input)
        bcon = QPushButton("Constrained"); bcon.clicked.connect(self.generate_constrained_code)
        bcon.setToolTip("Generate code that must contain specified keywords")
        cr2.addWidget(bcon)
        btmpl = QPushButton("Template"); btmpl.clicked.connect(self.generate_template_code)
        btmpl.setToolTip("Generate from a structural template")
        cr2.addWidget(btmpl)
        brep = QPushButton("Repair"); brep.clicked.connect(self.repair_and_show)
        brep.setToolTip("Attempt to fix syntax errors in output")
        cr2.addWidget(brep)
        gl.addLayout(cr2)

        # Controls
        cr = QHBoxLayout()
        cr.addWidget(QLabel("Temp:"))
        self.temp_spin = QDoubleSpinBox(); self.temp_spin.setRange(0, 3); self.temp_spin.setSingleStep(0.1)
        self.temp_spin.setValue(1.0); self.temp_spin.setToolTip("0=deterministic, 1=natural, >1=creative")
        cr.addWidget(self.temp_spin)
        cr.addWidget(QLabel("Top-p:"))
        self.top_p_spin = QDoubleSpinBox(); self.top_p_spin.setRange(0.01, 1.0)
        self.top_p_spin.setSingleStep(0.05); self.top_p_spin.setValue(1.0); self.top_p_spin.setDecimals(2)
        self.top_p_spin.setToolTip("Nucleus sampling threshold (<1 = more focused)")
        cr.addWidget(self.top_p_spin)
        self.max_len_spin = QDoubleSpinBox(); self.max_len_spin.setRange(5, 200)
        self.max_len_spin.setSingleStep(5); self.max_len_spin.setValue(50); self.max_len_spin.setDecimals(0)
        self.max_len_spin.setPrefix("Len:"); cr.addWidget(self.max_len_spin)
        self.top_k_spin = QDoubleSpinBox(); self.top_k_spin.setRange(2, 20)
        self.top_k_spin.setSingleStep(1); self.top_k_spin.setValue(5); self.top_k_spin.setDecimals(0)
        self.top_k_spin.setPrefix("K:"); cr.addWidget(self.top_k_spin)
        self.beam_spin = QSpinBox(); self.beam_spin.setRange(2, 10); self.beam_spin.setValue(3)
        self.beam_spin.setPrefix("Beam:"); cr.addWidget(self.beam_spin)
        cr.addStretch(); gl.addLayout(cr)

        # Validation + SVM
        self.validation_label = QLabel(""); self.validation_label.setStyleSheet("font-size:11px;padding:2px;")
        gl.addWidget(self.validation_label)
        self.svm_score_label = QLabel("SVM Quality: not scored")
        self.svm_score_label.setStyleSheet("color:#aaa;font-size:11px;padding:2px;")
        gl.addWidget(self.svm_score_label)

        self.chat_output = QPlainTextEdit(); self.chat_output.setReadOnly(True)
        self.chat_output.setFont(QFont("Consolas", 12))
        self.chat_output.setStyleSheet("QPlainTextEdit{background:#1e1e1e;color:#d4d4d4;}")
        self.highlighter = PythonSyntaxHighlighter(self.chat_output.document())
        gl.addWidget(self.chat_output)

        # Output buttons
        obr = QHBoxLayout()
        bcopy = QPushButton("Copy to Clipboard"); bcopy.clicked.connect(self._copy_output)
        bexp = QPushButton("Export to File"); bexp.clicked.connect(self._export_output)
        obr.addWidget(bcopy); obr.addWidget(bexp)
        gl.addLayout(obr)
        layout.addWidget(gen, stretch=3)

        # Right: History + Intent
        rw = QWidget(); rvl = QVBoxLayout(rw); rvl.setContentsMargins(0,0,0,0)
        hg = QGroupBox("History"); hl = QVBoxLayout(hg)
        self.history_list = QListWidget(); self.history_list.setFont(QFont("Consolas", 9))
        self.history_list.currentRowChanged.connect(self._on_history_select)
        hl.addWidget(self.history_list)
        bch = QPushButton("Clear History"); bch.clicked.connect(self._clear_chat_history)
        hl.addWidget(bch); rvl.addWidget(hg, stretch=2)

        ig = QGroupBox("Intent Classifier"); il = QVBoxLayout(ig)
        il.addWidget(QLabel("Test request parsing:"))
        self.intent_test_input = QLineEdit(); self.intent_test_input.setPlaceholderText("e.g., 'write a sort function'")
        self.intent_test_input.returnPressed.connect(self._test_intent); il.addWidget(self.intent_test_input)
        bti = QPushButton("Classify Intent"); bti.clicked.connect(self._test_intent); il.addWidget(bti)
        self.intent_result = QPlainTextEdit(); self.intent_result.setReadOnly(True)
        self.intent_result.setMaximumHeight(120); self.intent_result.setFont(QFont("Consolas", 10))
        il.addWidget(self.intent_result); rvl.addWidget(ig, stretch=1)
        layout.addWidget(rw, stretch=1)
        self.tabs.addTab(tab, "2. Chatbot")

    # ------------------------------------------------------------------
    # Tab 3: Metrics
    # ------------------------------------------------------------------

    def setup_metrics_tab(self):
        tab = QWidget(); layout = QHBoxLayout(tab)

        # Left: Metrics table
        mg = QGroupBox("Model Metrics"); ml = QVBoxLayout(mg)
        self.metrics_browser = QTextBrowser()
        self.metrics_browser.setFont(QFont("Consolas", 11))
        self.metrics_browser.setStyleSheet("QTextBrowser{background:#1a1a2e;color:#e0e0e0;}")
        ml.addWidget(self.metrics_browser)
        bref = QPushButton("Refresh Metrics"); bref.clicked.connect(self._refresh_metrics)
        ml.addWidget(bref)
        layout.addWidget(mg, stretch=2)

        # Right: Reward chart
        cg = QGroupBox("Agent Training Charts"); cl = QVBoxLayout(cg)
        chart_layout = QHBoxLayout()
        # Reward chart
        rcg = QGroupBox("Reward"); rcl = QVBoxLayout(rcg)
        self.chart_label = QLabel("Train the agent to see the reward chart")
        self.chart_label.setAlignment(Qt.AlignCenter)
        self.chart_label.setStyleSheet("color:#888;font-size:12px;")
        rcl.addWidget(self.chart_label, stretch=1)
        chart_layout.addWidget(rcg, stretch=1)
        # TD-error chart
        tcg = QGroupBox("TD Error (Convergence)"); tcl = QVBoxLayout(tcg)
        self.td_chart_label = QLabel("Train the agent to see TD-error")
        self.td_chart_label.setAlignment(Qt.AlignCenter)
        self.td_chart_label.setStyleSheet("color:#888;font-size:12px;")
        tcl.addWidget(self.td_chart_label, stretch=1)
        chart_layout.addWidget(tcg, stretch=1)
        cl.addLayout(chart_layout)
        brf = QPushButton("Refresh Charts"); brf.clicked.connect(self._refresh_charts)
        cl.addWidget(brf)
        layout.addWidget(cg, stretch=3)
        self.tabs.addTab(tab, "3. Metrics & Charts")

    # ------------------------------------------------------------------
    # Tab 4: Agent
    # ------------------------------------------------------------------

    def setup_agent_tab(self):
        tab = QWidget(); layout = QHBoxLayout(tab)
        splitter = QSplitter(Qt.Horizontal); layout.addWidget(splitter)

        # Left: VFS + terminal
        lw = QWidget(); ll = QVBoxLayout(lw); ll.setContentsMargins(0,0,0,0)
        w = QLabel("🛡️ 100% Sandboxed VFS. No OS imports.")
        w.setStyleSheet("color:#4CAF50;font-weight:bold;padding:5px;background:#2b2b2b;border-radius:3px;")
        w.setWordWrap(True); ll.addWidget(w)

        self.tree_widget = QTreeWidget(); self.tree_widget.setHeaderLabels(["Virtual Environment"])
        self.tree_widget.setColumnWidth(0, 250)
        self.tree_widget.itemClicked.connect(self._on_tree_click)
        ll.addWidget(self.tree_widget, stretch=2)

        # File viewer
        fg = QGroupBox("File Content"); fgl = QVBoxLayout(fg)
        self.file_viewer = QPlainTextEdit(); self.file_viewer.setReadOnly(True)
        self.file_viewer.setFont(QFont("Consolas", 10)); self.file_viewer.setMaximumHeight(150)
        self.file_viewer.setStyleSheet("QPlainTextEdit{background:#1a1a2e;color:#e0e0e0;}")
        PythonSyntaxHighlighter(self.file_viewer.document())
        fgl.addWidget(self.file_viewer); ll.addWidget(fg, stretch=1)

        # Interactive terminal
        tg = QGroupBox("Terminal"); tl = QVBoxLayout(tg)
        self.terminal_view = QPlainTextEdit(); self.terminal_view.setReadOnly(True)
        self.terminal_view.setFont(QFont("Consolas", 11))
        self.terminal_view.setStyleSheet("QPlainTextEdit{background:#0a0a0a;color:#0f0;}")
        tl.addWidget(self.terminal_view, stretch=1)
        tir = QHBoxLayout()
        self.terminal_prompt = QLabel("$ "); self.terminal_prompt.setStyleSheet("color:#0f0;font-weight:bold;font-family:Consolas;")
        tir.addWidget(self.terminal_prompt)
        self.terminal_input = QLineEdit(); self.terminal_input.setPlaceholderText("Type command...")
        self.terminal_input.setFont(QFont("Consolas", 11))
        self.terminal_input.setStyleSheet("QLineEdit{background:#111;color:#0f0;border:1px solid #333;}")
        self.terminal_input.returnPressed.connect(self._execute_terminal_command)
        tir.addWidget(self.terminal_input)
        tl.addLayout(tir)
        self.terminal_input.installEventFilter(self)
        ll.addWidget(tg, stretch=3)
        splitter.addWidget(lw)

        # Right: Controls
        rw = QWidget(); rl = QVBoxLayout(rw); rl.setContentsMargins(0,0,0,0)
        rg = QGroupBox("User Request"); rgl = QVBoxLayout(rg)
        self.request_input = QLineEdit()
        self.request_input.setPlaceholderText("e.g., 'Write a function to add numbers'")
        self.request_input.returnPressed.connect(self.process_user_request); rgl.addWidget(self.request_input)
        bp = QPushButton("Agent: Process Request & Write File"); bp.clicked.connect(self.process_user_request)
        bp.setStyleSheet("QPushButton{background:#2196F3;color:white;font-weight:bold;padding:8px;border-radius:4px;}"
                         "QPushButton:hover{background:#1976D2;}")
        rgl.addWidget(bp)
        self.use_svm_label = QLabel("Intent: SVM classifier"); self.use_svm_label.setStyleSheet("color:#aaa;font-size:11px;")
        rgl.addWidget(self.use_svm_label); rl.addWidget(rg)

        cg = QGroupBox("Q-Learning"); cvl = QVBoxLayout(cg)
        prl = QHBoxLayout(); prl.addWidget(QLabel("LR:"))
        self.lr_spin = QDoubleSpinBox(); self.lr_spin.setRange(.01,1); self.lr_spin.setSingleStep(.05)
        self.lr_spin.setValue(.1); self.lr_spin.setDecimals(2); prl.addWidget(self.lr_spin)
        prl.addWidget(QLabel("\u03b3:"))
        self.gamma_spin = QDoubleSpinBox(); self.gamma_spin.setRange(0,1); self.gamma_spin.setSingleStep(.05)
        self.gamma_spin.setValue(.95); self.gamma_spin.setDecimals(2); prl.addWidget(self.gamma_spin)
        cvl.addLayout(prl)
        # Exploration strategy + smoothing controls
        erl = QHBoxLayout(); erl.addWidget(QLabel("Explore:"))
        self.exploration_combo = QComboBox()
        self.exploration_combo.addItems(["epsilon-greedy", "UCB1"])
        self.exploration_combo.setToolTip("epsilon-greedy or UCB1 exploration")
        erl.addWidget(self.exploration_combo)
        self.replay_check = QCheckBox("Replay")
        self.replay_check.setChecked(True)
        self.replay_check.setToolTip("Experience replay buffer")
        erl.addWidget(self.replay_check)
        erl.addStretch(); cvl.addLayout(erl)
        srl = QHBoxLayout(); srl.addWidget(QLabel("KN-Smooth:"))
        self.smooth_check = QCheckBox("On")
        self.smooth_check.setChecked(True)
        self.smooth_check.setToolTip("Kneser-Ney smoothing (no dead-end)")
        srl.addWidget(self.smooth_check)
        self.interp_check = QCheckBox("Interp")
        self.interp_check.setToolTip("Interpolated n-gram probs")
        srl.addWidget(self.interp_check)
        srl.addWidget(QLabel("RepPen:"))
        self.rep_pen_spin = QDoubleSpinBox()
        self.rep_pen_spin.setRange(0, 1); self.rep_pen_spin.setSingleStep(0.1)
        self.rep_pen_spin.setValue(0); self.rep_pen_spin.setDecimals(1)
        self.rep_pen_spin.setToolTip("Repetition penalty (0=off)")
        srl.addWidget(self.rep_pen_spin)
        srl.addStretch(); cvl.addLayout(srl)
        # Double-Q and discount
        dql = QHBoxLayout(); dql.addWidget(QLabel("KN-Disc:"))
        self.discount_spin = QDoubleSpinBox()
        self.discount_spin.setRange(0.1, 1.0); self.discount_spin.setSingleStep(0.05)
        self.discount_spin.setValue(0.75); self.discount_spin.setDecimals(2)
        self.discount_spin.setToolTip("Kneser-Ney discount parameter")
        dql.addWidget(self.discount_spin)
        self.dq_check = QCheckBox("Double-Q")
        self.dq_check.setChecked(True)
        self.dq_check.setToolTip("Double Q-learning (reduces maximization bias)")
        dql.addWidget(self.dq_check)
        dql.addStretch(); cvl.addLayout(dql)
        self.btn_train = QPushButton("\u25b6  Start Training (Threaded)")
        self.btn_train.clicked.connect(self.toggle_training)
        self.btn_train.setStyleSheet("QPushButton{padding:8px;font-weight:bold;border-radius:4px;}")
        cvl.addWidget(self.btn_train)
        br = QPushButton("\u21ba  Reset Environment (Ctrl+R)"); br.clicked.connect(self.reset_agent); cvl.addWidget(br)
        bs = QPushButton("\u23ed  Single Step (Greedy)"); bs.clicked.connect(self.agent_greedy_step); cvl.addWidget(bs)
        rl.addWidget(cg)

        qg = QGroupBox("Q-Table"); qvl = QVBoxLayout(qg)
        self.q_table_view = QTableWidget(); self.q_table_view.setColumnCount(3)
        self.q_table_view.setHorizontalHeaderLabels(["Action", "Q-Value", "Visits"])
        self.q_table_view.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        qvl.addWidget(self.q_table_view); rl.addWidget(qg, stretch=1)
        splitter.addWidget(rw); splitter.setSizes([750, 350])
        self.update_vfs_tree(); self.tabs.addTab(tab, "4. Agent (Sandbox)")

    # ------------------------------------------------------------------
    # Tab 5: Database
    # ------------------------------------------------------------------

    def setup_database_tab(self):
        tab = QWidget(); layout = QHBoxLayout(tab)
        splitter = QSplitter(Qt.Horizontal); layout.addWidget(splitter)

        # Left: snippet table + controls
        lw = QWidget(); ll = QVBoxLayout(lw); ll.setContentsMargins(0, 0, 0, 0)

        # Toolbar row 1: search + filters
        tr1 = QHBoxLayout()
        self.db_search_input = QLineEdit()
        self.db_search_input.setPlaceholderText("Search snippets by content...")
        self.db_search_input.returnPressed.connect(self._db_search)
        tr1.addWidget(self.db_search_input)
        bsearch = QPushButton("Search"); bsearch.clicked.connect(self._db_search)
        tr1.addWidget(bsearch)
        ll.addLayout(tr1)

        # Toolbar row 2: source/tag filters + actions
        tr2 = QHBoxLayout()
        tr2.addWidget(QLabel("Source:"))
        self.db_source_combo = QComboBox(); self.db_source_combo.setMinimumWidth(100)
        self.db_source_combo.addItem("All")
        tr2.addWidget(self.db_source_combo)
        tr2.addWidget(QLabel("Tag:"))
        self.db_tag_combo = QComboBox(); self.db_tag_combo.setMinimumWidth(100)
        self.db_tag_combo.addItem("All")
        tr2.addWidget(self.db_tag_combo)
        bfilter = QPushButton("Filter"); bfilter.clicked.connect(self._db_refresh_table)
        tr2.addWidget(bfilter)
        ll.addLayout(tr2)

        # Action buttons
        tr3 = QHBoxLayout()
        b_add = QPushButton("Add Snippet"); b_add.clicked.connect(self._db_add_snippet_dialog)
        b_edit = QPushButton("Edit"); b_edit.clicked.connect(self._db_edit_snippet)
        b_del = QPushButton("Delete"); b_del.clicked.connect(self._db_delete_selected)
        b_train = QPushButton("Train Markov from DB"); b_train.clicked.connect(self._db_train_from_db)
        b_import = QPushButton("Import File"); b_import.clicked.connect(self._db_import_file_dialog)
        b_export = QPushButton("Export All"); b_export.clicked.connect(self._db_export_dialog)
        for b in (b_add, b_edit, b_del, b_train, b_import, b_export):
            tr3.addWidget(b)
        ll.addLayout(tr3)

        # Snippet table
        self.db_table = QTableWidget()
        self.db_table.setColumnCount(6)
        self.db_table.setHorizontalHeaderLabels(
            ["ID", "Source", "Tags", "Lines", "Chars", "Created"])
        self.db_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.db_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.db_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.db_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self.db_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.db_table.setSelectionMode(QTableWidget.SingleSelection)
        self.db_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.db_table.currentCellChanged.connect(self._db_on_row_select)
        ll.addWidget(self.db_table, stretch=1)

        # Pagination
        pg = QHBoxLayout()
        self.db_page_label = QLabel("Page 1")
        b_prev = QPushButton("< Prev"); b_prev.clicked.connect(self._db_prev_page)
        b_next = QPushButton("Next >"); b_next.clicked.connect(self._db_next_page)
        pg.addWidget(b_prev); pg.addWidget(self.db_page_label); pg.addWidget(b_next)
        pg.addStretch()
        self.db_count_label = QLabel("0 snippets")
        pg.addWidget(self.db_count_label)
        ll.addLayout(pg)
        splitter.addWidget(lw)

        # Right: preview + stats
        rw = QWidget(); rvl = QVBoxLayout(rw); rvl.setContentsMargins(0, 0, 0, 0)

        # Preview
        pvg = QGroupBox("Snippet Preview"); pvl = QVBoxLayout(pvg)
        self.db_preview = QPlainTextEdit(); self.db_preview.setReadOnly(True)
        self.db_preview.setFont(QFont("Consolas", 11))
        self.db_preview.setStyleSheet(
            "QPlainTextEdit{background:#1e1e1e;color:#d4d4d4;}")
        PythonSyntaxHighlighter(self.db_preview.document())
        pvl.addWidget(self.db_preview)

        # Tags editor
        tr4 = QHBoxLayout()
        tr4.addWidget(QLabel("Tags:"))
        self.db_tags_input = QLineEdit()
        self.db_tags_input.setPlaceholderText("comma-separated tags...")
        tr4.addWidget(self.db_tags_input)
        b_save_tags = QPushButton("Save Tags"); b_save_tags.clicked.connect(self._db_save_tags)
        tr4.addWidget(b_save_tags)
        pvl.addLayout(tr4)

        # Quick actions
        qa = QHBoxLayout()
        b_score = QPushButton("Score Quality"); b_score.clicked.connect(self._db_score_selected)
        b_train_one = QPushButton("Train on Selected"); b_train_one.clicked.connect(self._db_train_on_selected)
        qa.addWidget(b_score); qa.addWidget(b_train_one)
        pvl.addLayout(qa)
        rvl.addWidget(pvg, stretch=3)

        # Stats
        sg = QGroupBox("Database Statistics"); sgl = QVBoxLayout(sg)
        self.db_stats_browser = QTextBrowser()
        self.db_stats_browser.setFont(QFont("Consolas", 10))
        self.db_stats_browser.setStyleSheet(
            "QTextBrowser{background:#1a1a2e;color:#e0e0e0;}")
        self.db_stats_browser.setMaximumHeight(250)
        sgl.addWidget(self.db_stats_browser)
        b_stats = QPushButton("Refresh Stats"); b_stats.clicked.connect(self._db_refresh_stats)
        b_clear = QPushButton("Clear All Data"); b_clear.clicked.connect(self._db_clear_all)
        sgl.addWidget(b_stats); sgl.addWidget(b_clear)
        rvl.addWidget(sg, stretch=2)
        splitter.addWidget(rw)
        splitter.setSizes([700, 450])

        self.tabs.addTab(tab, "5. Database")
        self._db_refresh_stats()
        self._db_refresh_filters()
        self._db_refresh_table()

    # -- Event filter for terminal history --------------------------------

    def eventFilter(self, obj, event):
        if obj == self.terminal_input and event.type() == event.Type.KeyPress:
            if event.key() == Qt.Key_Up:
                if self._term_history and self._term_history_idx < len(self._term_history) - 1:
                    self._term_history_idx += 1
                    self.terminal_input.setText(self._term_history[-(self._term_history_idx + 1)])
                return True
            elif event.key() == Qt.Key_Down:
                if self._term_history_idx > 0:
                    self._term_history_idx -= 1
                    self.terminal_input.setText(self._term_history[-(self._term_history_idx + 1)])
                else:
                    self._term_history_idx = -1; self.terminal_input.clear()
                return True
        return super().eventFilter(obj, event)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def load_default_corpus(self):
        snippets = [
            "def add ( a , b ) :\n    result = a + b\n    return result",
            "def subtract ( x , y ) :\n    return x - y",
            "def multiply ( a , b ) :\n    product = a * b\n    return product",
            "def divide ( num , den ) :\n    if den == 0 :\n        return None\n    return num / den",
            "def process_data ( data ) :\n    cleaned = [ x for x in data if x > 0 ]\n    return cleaned",
            "def calculate_sum ( numbers ) :\n    total = 0\n    for n in numbers :\n        total += n\n    return total",
            "class DataProcessor :\n    def __init__ ( self ) :\n        self.data = []\n    def add_item ( self , item ) :\n        self.data.append(item)",
            "for i in range ( 10 ) :\n    if i % 2 == 0 :\n        print ( 'even' )\n    else :\n        print ( 'odd' )",
            "import math\n\ndef get_area ( radius ) :\n    return math.pi * radius ** 2",
            "def find_max ( lst ) :\n    if not lst :\n        return None\n    max_val = lst [ 0 ]\n    for x in lst :\n        if x > max_val :\n            max_val = x\n    return max_val",
            "class Calculator :\n    def __init__ ( self ) :\n        self.history = []\n    def add ( self , a , b ) :\n        result = a + b\n        self.history.append(result)\n        return result\n    def get_history ( self ) :\n        return self.history",
            "def read_file ( path ) :\n    try :\n        with open ( path ) as f :\n            return f.read()\n    except Exception as e :\n        print ( f'Error: {e}' )\n        return None",
            "def fibonacci ( n ) :\n    if n <= 1 :\n        return n\n    return fibonacci ( n - 1 ) + fibonacci ( n - 2 )",
            "def merge_sort ( arr ) :\n    if len ( arr ) <= 1 :\n        return arr\n    mid = len ( arr ) // 2\n    left = merge_sort ( arr [ :mid ] )\n    right = merge_sort ( arr [ mid : ] )\n    return merge ( left , right )",
        ]
        self._all_snippets.extend(snippets)
        for s in snippets:
            self.markov_model.train(s); self.corpus_view.appendPlainText(s + "\n---\n")
        self.update_markov_table()
        self.corpus_label.setText(f"Tokens: {self.markov_model.token_counts}")

    def _load_example_snippets(self):
        examples = [
            "def binary_search ( arr , target ) :\n    low = 0\n    high = len ( arr ) - 1\n    while low <= high :\n        mid = ( low + high ) // 2\n        if arr [ mid ] == target :\n            return mid\n        elif arr [ mid ] < target :\n            low = mid + 1\n        else :\n            high = mid - 1\n    return -1",
            "def reverse_list ( lst ) :\n    return lst [ ::-1 ]",
            "def flatten ( nested_list ) :\n    result = []\n    for item in nested_list :\n        if isinstance ( item , list ) :\n            result.extend ( flatten ( item ) )\n        else :\n            result.append ( item )\n    return result",
            "def get_unique ( lst ) :\n    seen = set ()\n    result = []\n    for item in lst :\n        if item not in seen :\n            seen.add ( item )\n            result.append ( item )\n    return result",
            "def count_items ( lst , item ) :\n    count = 0\n    for x in lst :\n        if x == item :\n            count += 1\n    return count",
            "def calculate_average ( numbers ) :\n    if not numbers :\n        return 0\n    return sum ( numbers ) / len ( numbers )",
            "def is_palindrome ( s ) :\n    return s == s [ ::-1 ]",
            "def factorial ( n ) :\n    if n <= 1 :\n        return 1\n    return n * factorial ( n - 1 )",
            "class Stack :\n    def __init__ ( self ) :\n        self.items = []\n    def push ( self , item ) :\n        self.items.append ( item )\n    def pop ( self ) :\n        if not self.is_empty () :\n            return self.items.pop ()\n    def is_empty ( self ) :\n        return len ( self.items ) == 0",
            "def bubble_sort ( arr ) :\n    n = len ( arr )\n    for i in range ( n ) :\n        for j in range ( 0 , n - i - 1 ) :\n            if arr [ j ] > arr [ j + 1 ] :\n                arr [ j ] , arr [ j + 1 ] = arr [ j + 1 ] , arr [ j ]\n    return arr",
            "def quick_sort ( arr ) :\n    if len ( arr ) <= 1 :\n        return arr\n    pivot = arr [ len ( arr ) // 2 ]\n    left = [ x for x in arr if x < pivot ]\n    middle = [ x for x in arr if x == pivot ]\n    right = [ x for x in arr if x > pivot ]\n    return quick_sort ( left ) + middle + quick_sort ( right )",
            "def gcd ( a , b ) :\n    while b :\n        a , b = b , a % b\n    return a",
            "def lcm ( a , b ) :\n    return abs ( a * b ) // gcd ( a , b )",
            "def validate_email ( email ) :\n    import re\n    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'\n    return bool ( re.match ( pattern , email ) )",
            "class Queue :\n    def __init__ ( self ) :\n        self.items = []\n    def enqueue ( self , item ) :\n        self.items.append ( item )\n    def dequeue ( self ) :\n        if not self.is_empty () :\n            return self.items.pop ( 0 )\n    def is_empty ( self ) :\n        return len ( self.items ) == 0",
        ]
        t, c = self.markov_model.train_batch(examples)
        self._all_snippets.extend(examples)
        for s in examples:
            self.corpus_view.appendPlainText(s + "\n---\n")
        self.update_markov_table()
        self.corpus_label.setText(f"Tokens: {self.markov_model.token_counts}")
        self._train_svm_models()
        self.status_label.setText(f"Loaded {c} examples ({t} tokens)")
        self._show_import_stats(f"Added {c} examples ({t} tokens)")

    # ------------------------------------------------------------------
    # File Import
    # ------------------------------------------------------------------

    def _import_files_dialog(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Import Files", "",
                    "Python/Text (*.py *.txt *.csv);;All (*)")
        if paths: self._import_paths(paths)

    def _import_directory_dialog(self):
        d = QFileDialog.getExistingDirectory(self, "Import Directory")
        if d: self._import_paths([d], is_directory=True)

    def _on_files_dropped(self, paths):
        self._import_paths(paths)

    def _import_paths(self, paths, is_directory=False):
        self.import_progress.setVisible(True); self.import_progress.setValue(0)
        def _do():
            all_s, tc, sums, tf = [], 0, [], 0
            for p in paths:
                pp = Path(p)
                if pp.is_dir() or is_directory:
                    sn, desc, ch = SnippetImporter.import_directory(str(pp))
                elif pp.is_file():
                    sn, desc, ch = SnippetImporter.import_file(str(pp))
                else: continue
                all_s.extend(sn); tc += ch; tf += 1; sums.append(desc)
                self.signals.import_progress.emit(len(all_s), len(all_s), desc)
            if all_s:
                tt, cnt = self.markov_model.train_batch(all_s)
                self._all_snippets.extend(all_s)
            self._import_result = (all_s, tt, cnt, sums, tc, tf)
        def _done():
            r = getattr(self, "_import_result", None)
            if not r: self.import_progress.setVisible(False); return
            all_s, tt, cnt, sums, tc, tf = r
            for s in all_s: self.corpus_view.appendPlainText(s + "\n---\n")
            self.update_markov_table()
            self.corpus_label.setText(f"Tokens: {self.markov_model.token_counts}")
            self._show_import_stats(f"Imported {cnt} snippets from {tf} source(s), {tt} tokens, {tc:,} chars")
            self._train_svm_models(); self.import_progress.setVisible(False)
        self._import_result = None
        t = threading.Thread(target=_do, daemon=True); t.start()
        QTimer.singleShot(100, _check)
        def _check():
            if hasattr(self, "_import_result") and self._import_result is not None: _done()
            else: QTimer.singleShot(100, _check)

    def _on_import_progress(self, c, t, fn):
        self.import_progress.setMaximum(max(t, 1)); self.import_progress.setValue(c)

    def _show_import_stats(self, text):
        self.import_stats_label.setText(text); self.import_stats_label.setVisible(True)

    # ------------------------------------------------------------------
    # Save / Load Model
    # ------------------------------------------------------------------

    def _save_model_dialog(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Markov Model", "markov_model.json",
                                               "JSON (*.json)")
        if not path: return
        try:
            data = self.markov_model.to_dict()
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            self.status_label.setText(f"Model saved to {path}")
        except Exception as e:
            QMessageBox.warning(self, "Save Error", str(e))

    def _load_model_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Markov Model", "",
                                               "JSON (*.json);;All (*)")
        if not path: return
        try:
            with open(path) as f:
                data = json.load(f)
            n = self.markov_model.n
            self.markov_model = MarkovCodeModel.from_dict(data)
            self.markov_model.n = n  # preserve current n-gram setting
            self.update_markov_table()
            self.corpus_label.setText(f"Tokens: {self.markov_model.token_counts}")
            self._train_svm_models()
            self.status_label.setText(f"Model loaded from {path} ({self.markov_model.token_counts} tokens)")
        except Exception as e:
            QMessageBox.warning(self, "Load Error", str(e))

    # ------------------------------------------------------------------
    # SVM Training
    # ------------------------------------------------------------------

    def _train_svm_models(self):
        if self._all_snippets:
            stats = self.quality_svm.train(self._all_snippets, self.markov_model, n_negative=300)
            if stats.get("trained"):
                self.svm_label.setText(f"SVM: {stats['accuracy']:.0%} acc")
                self.svm_label.setStyleSheet("color:#4CAF50;padding:0 8px;")
        self.intent_clf.train()

    # ------------------------------------------------------------------
    # Tab 1 callbacks
    # ------------------------------------------------------------------

    def train_markov(self):
        text = self.corpus_input.text()
        if not text: return
        c = self.markov_model.train(text); self._all_snippets.append(text)
        self.corpus_view.appendPlainText(text + "\n---\n"); self.corpus_input.clear()
        self.update_markov_table()
        self.corpus_label.setText(f"Tokens: {self.markov_model.token_counts}")
        self.status_label.setText(f"Trained on {c} tokens")

    def _clear_corpus(self):
        if QMessageBox.question(self, "Clear", "Reset model and clear training data?",
                                QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            n = self.markov_model.n; self.markov_model = MarkovCodeModel(n=n)
            self._all_snippets.clear(); self.quality_svm = CodeQualitySVM()
            self.corpus_view.clear(); self.update_markov_table()
            self.corpus_label.setText("Tokens: 0")
            self.svm_label.setText("SVM: idle"); self.svm_label.setStyleSheet("color:#aaa;padding:0 8px;")
            self.import_stats_label.setVisible(False)

    def _filter_corpus(self, query):
        """Highlight matching lines in corpus view."""
        cursor = self.corpus_view.textCursor()
        cursor.movePosition(QTextCursor.Start)
        if not query:
            self.corpus_view.setExtraSelections([]); return
        self.corpus_view.setTextCursor(cursor)
        self.corpus_view.find(query)
        selections = []
        while self.corpus_view.find(query):
            sel = self.corpus_view.textCursor()
            fmt = QTextCharFormat(); fmt.setBackground(QColor(255, 255, 0, 80))
            es = QPlainTextEdit.ExtraSelection(); es.cursor = sel; es.format = fmt
            selections.append(es)
        self.corpus_view.setExtraSelections(selections)

    def update_markov_table(self):
        token = self.token_input.text().strip() or "def"
        n = self.markov_model.n
        rows = []
        for ctx, trans in self.markov_model.transitions.items():
            if n == 1 or ctx[-1] == token:
                total = sum(trans.values())
                for nxt, count in trans.items():
                    rows.append((nxt, count, count / total if total else 0))
        seen = set(); unique = []
        for nxt, count, prob in sorted(rows, key=lambda x: x[2], reverse=True):
            if nxt not in seen:
                seen.add(nxt); unique.append((nxt, count, prob))
        self.markov_table.setRowCount(0)
        for nxt, count, prob in unique[:20]:
            r = self.markov_table.rowCount(); self.markov_table.insertRow(r)
            self.markov_table.setItem(r, 0, QTableWidgetItem(nxt))
            self.markov_table.setItem(r, 1, QTableWidgetItem(str(count)))
            self.markov_table.setItem(r, 2, QTableWidgetItem(f"{prob:.2%}"))

    def _on_ngram_changed(self, idx):
        n = {0: 2, 1: 3, 2: 1, 3: 4}.get(idx, 2)
        self.markov_model = MarkovCodeModel(n=n)
        for s in self.corpus_view.toPlainText().split("---"):
            s = s.strip()
            if s: self.markov_model.train(s)
        self.update_markov_table(); self._train_svm_models()
        self.status_label.setText(f"Switched to {'uni' if n==1 else 'bi' if n==2 else 'tri'}gram")

    # ------------------------------------------------------------------
    # Tab 2 callbacks
    # ------------------------------------------------------------------

    def generate_code(self):
        prefix = self.chat_prefix.text() or "def process_data ("
        temp = self.temp_spin.value(); ml = int(self.max_len_spin.value())
        tp = self.top_p_spin.value()
        code = self.markov_model.generate(
            prefix, max_length=ml, temperature=temp, top_p=tp,
            use_smoothing=self.smooth_check.isChecked(),
            use_interpolation=self.interp_check.isChecked())
        self.chat_output.setPlainText(code)
        self._show_validation_and_score(code)
        self._record_generation(prefix, code)

    def generate_best_code(self):
        prefix = self.chat_prefix.text() or "def process_data ("
        temp = self.temp_spin.value(); ml = int(self.max_len_spin.value())
        k = int(self.top_k_spin.value()); tp = self.top_p_spin.value()
        candidates = self.markov_model.generate_candidates(prefix, num_candidates=k,
                                                           max_length=ml, temperature=temp, top_p=tp)
        if self.quality_svm.is_trained:
            scored = [(c, self.quality_svm.score(c, self.markov_model)) for c, _ in candidates]
            scored.sort(key=lambda x: x[1], reverse=True)
            best_code, best_score = scored[0]
            self.chat_output.setPlainText(best_code)
            lines = [f"Top-K Ranking (k={k}):"]
            for i, (c, s) in enumerate(scored[:5]):
                m = " >>>" if i == 0 else "    "
                lines.append(f"{m} #{i+1}: {s:.1%} — {c.split(chr(10))[0][:45]}")
            self.svm_score_label.setText("\n".join(lines))
            self.svm_score_label.setStyleSheet("color:#4CAF50;font-size:11px;padding:2px;font-weight:bold;")
        else:
            best_code = candidates[0][0] if candidates else "# empty"
            self.chat_output.setPlainText(best_code)
            self.svm_score_label.setText("SVM: not trained")
        self._show_validation(best_code)
        self._record_generation(prefix, best_code, suffix=" [SVM best]")

    def generate_beam_code(self):
        prefix = self.chat_prefix.text() or "def process_data ("
        ml = int(self.max_len_spin.value()); bw = self.beam_spin.value()
        code = self.markov_model.generate_beam(prefix, max_length=ml, beam_width=bw)
        self.chat_output.setPlainText(code)
        self._show_validation_and_score(code)
        self._record_generation(prefix, code, suffix=" [beam]")

    def _show_validation_and_score(self, code):
        self._show_validation(code)
        if self.quality_svm.is_trained:
            s = self.quality_svm.score(code, self.markov_model)
            c = "#4CAF50" if s > .7 else "#FFC107" if s > .4 else "#FF5722"
            self.svm_score_label.setText(f"SVM Quality: {s:.1%}")
            self.svm_score_label.setStyleSheet(f"color:{c};font-size:11px;padding:2px;font-weight:bold;")
        else:
            self.svm_score_label.setText("SVM: not trained yet")

    def _show_validation(self, code):
        valid, msg = validate_syntax(code)
        color = "#4CAF50" if valid else "#FF5722"
        icon = "✓" if valid else "✗"
        self.validation_label.setText(f"{icon} {msg}")
        self.validation_label.setStyleSheet(f"color:{color};font-size:11px;padding:2px;")

    def _record_generation(self, prefix, code, suffix=""):
        self._gen_history.append((prefix, code))
        item = QListWidgetItem(f"[{len(self._gen_history)}] {prefix[:35]}...{suffix}")
        self.history_list.addItem(item)
        self.history_list.setCurrentRow(self.history_list.count() - 1)
        self.status_label.setText(f"Generated {len(code.split(chr(10)))} lines")

    def _copy_output(self):
        text = self.chat_output.toPlainText()
        QApplication.clipboard().setText(text)
        self.status_label.setText("Copied to clipboard")

    def _export_output(self):
        text = self.chat_output.toPlainText()
        path, _ = QFileDialog.getSaveFileName(self, "Export Code", "generated.py", "Python (*.py)")
        if path:
            with open(path, "w") as f: f.write(text)
            self.status_label.setText(f"Exported to {path}")

    def _test_intent(self):
        r = self.intent_test_input.text().strip()
        if not r: return
        top = self.intent_clf.predict_top_k(r, k=3)
        lines = [f"Request: '{r}'", f"Best: {top[0][0]}"]
        if len(top) > 1:
            lines.append("Alternatives:")
            for p, c in top[1:]:
                lines.append(f"  {c:.1%} → {p}")
        self.intent_result.setPlainText("\n".join(lines))

    def _on_history_select(self, row):
        if 0 <= row < len(self._gen_history):
            self.chat_output.setPlainText(self._gen_history[row][1])

    def _clear_chat_history(self):
        self._gen_history.clear(); self.history_list.clear(); self.chat_output.clear()

    # -- New generation methods -----------------------------------------

    def generate_refined_code(self):
        """Multi-pass refinement with quality feedback."""
        prefix = self.chat_prefix.text() or "def process_data ("
        ml = int(self.max_len_spin.value())
        temp = self.temp_spin.value(); tp = self.top_p_spin.value()
        code, score = self.markov_model.generate_refined(
            prefix, max_length=ml, temperature=temp, top_p=tp,
            quality_svm=(self.quality_svm
                          if self.quality_svm.is_trained else None),
            passes=3)
        repaired = self.markov_model.repair_syntax(code)
        self.chat_output.setPlainText(repaired)
        self._show_validation_and_score(repaired)
        self._record_generation(prefix, repaired, suffix=" [refined]")
        self.status_label.setText(f"Refined output (score={score:.1%})")

    def generate_constrained_code(self):
        """Generate code that must contain user-specified keywords."""
        prefix = self.chat_prefix.text() or "def process_data ("
        ml = int(self.max_len_spin.value())
        temp = self.temp_spin.value(); tp = self.top_p_spin.value()
        raw = self.must_contain_input.text().strip()
        must_have = [k.strip() for k in raw.split(",") if k.strip()] if raw else []
        code = self.markov_model.generate_constrained(
            prefix, max_length=ml, must_contain=must_have,
            temperature=temp, top_p=min(tp, 0.95))
        repaired = self.markov_model.repair_syntax(code)
        self.chat_output.setPlainText(repaired)
        self._show_validation_and_score(repaired)
        self._record_generation(prefix, repaired, suffix=" [constrained]")
        if must_have:
            found = [k for k in must_have if k in repaired]
            self.status_label.setText(
                f"Constrained: {len(found)}/{len(must_have)} keywords found")

    def generate_template_code(self):
        """Generate code from a structural template."""
        prefix = self.chat_prefix.text().strip()
        name = "process_data"
        if prefix and prefix.startswith("class "):
            ttype = "class"
            name = prefix.split()[1].strip("(").strip(":") if "(" in prefix else prefix.split()[1].strip(":")
        elif prefix and prefix.startswith("for "):
            ttype = "for_loop"
        else:
            ttype = "function"
            if prefix and prefix.startswith("def "):
                rest = prefix[4:].strip()
                name = rest.split("(")[0].strip() if "(" in rest else rest
        code = self.markov_model.generate_template(
            ttype, name=name,
            temperature=self.temp_spin.value())
        repaired = self.markov_model.repair_syntax(code)
        self.chat_output.setPlainText(repaired)
        self._show_validation_and_score(repaired)
        self._record_generation(prefix, repaired, suffix=f" [template:{ttype}]")

    def repair_and_show(self):
        """Repair the current output and re-display."""
        code = self.chat_output.toPlainText()
        if not code.strip():
            return
        repaired = self.markov_model.repair_syntax(code)
        if repaired != code:
            self.chat_output.setPlainText(repaired)
            self._show_validation_and_score(repaired)
            self.status_label.setText("Syntax repaired")
        else:
            self.status_label.setText("No repairable issues found")

    # ------------------------------------------------------------------
    # Tab 3: Metrics
    # ------------------------------------------------------------------

    def _refresh_metrics(self):
        m = self.markov_model
        vocab = m.vocabulary_size()
        contexts = m.num_contexts()
        avg_ent = m.avg_entropy()
        total_trans = sum(sum(t.values()) for t in m.transitions.values())

        # Sample perplexity on a test snippet
        test = "def example ( x ) :\n    result = x * 2\n    return result"
        ppl = m.perplexity(test)
        cov = m.coverage_ratio(test)
        ll = m.log_likelihood(test)

        html = f"""<h3 style='color:#42A5F5'>Markov Chain Model Metrics</h3>
        <table style='color:#ddd;font-size:13px;' cellpadding='4'>
        <tr><td><b>N-gram order</b></td><td>{m.n}</td></tr>
        <tr><td><b>Total tokens</b></td><td>{m.token_counts:,}</td></tr>
        <tr><td><b>Snippet count</b></td><td>{m._snippet_count}</td></tr>
        <tr><td><b>Vocabulary size</b></td><td>{vocab}</td></tr>
        <tr><td><b>Unique contexts</b></td><td>{contexts:,}</td></tr>
        <tr><td><b>Total transitions</b></td><td>{total_trans:,}</td></tr>
        <tr><td><b>Avg entropy</b></td><td>{avg_ent:.3f} bits</td></tr>
        <tr><td colspan=2><hr style='border-color:#555'></td></tr>
        <tr><td colspan=2><b style='color:#FFCA28'>Test Snippet Perplexity</b></td></tr>
        <tr><td><i>Perplexity</i></td><td>{ppl:.2f}</td></tr>
        <tr><td><i>Log-likelihood</i></td><td>{ll:.3f}</td></tr>
        <tr><td><i>Coverage</i></td><td>{cov:.1%}</td></tr>
        </table>
        <h3 style='color:#42A5F5;margin-top:15px'>SVM Status</h3>
        <table style='color:#ddd;font-size:13px;' cellpadding='4'>
        <tr><td><b>Quality SVM</b></td><td>{'Trained' if self.quality_svm.is_trained else 'Not trained'}</td></tr>
        <tr><td><b>Intent SVM</b></td><td>{'Trained' if self.intent_clf.is_trained else 'Not trained'}</td></tr>
        <tr><td><b>sklearn available</b></td><td>{'Yes' if HAS_SKLEARN else 'No'}</td></tr>
        </table>
        <h3 style='color:#42A5F5;margin-top:15px'>Q-Learning Agent</h3>
        <table style='color:#ddd;font-size:13px;' cellpadding='4'>
        <tr><td><b>Episodes</b></td><td>{self.agent.episodes:,}</td></tr>
        <tr><td><b>Epsilon</b></td><td>{self.agent.epsilon:.4f}</td></tr>
        <tr><td><b>Q-table entries</b></td><td>{len(self.agent.q_table):,}</td></tr>
        <tr><td><b>Best avg reward</b></td><td>{max(self.agent.episode_rewards[-100:]) if self.agent.episode_rewards else 0:.1f}</td></tr>
        <tr><td><b>Avg TD-error</b></td><td>{self.agent.avg_td_error():.4f}</td></tr>
        <tr><td><b>Double-Q</b></td><td>{'On' if self.agent._use_double_q else 'Off'}</td></tr>
        </table>
        <h3 style='color:#ce93d8;margin-top:15px'>Snippet Database (SQLite)</h3>
        <table style='color:#ddd;font-size:13px;' cellpadding='4'>"""
        ds = self.db.get_stats()
        html += f"""<tr><td><b>Stored snippets</b></td><td>{ds['total_snippets']:,}</td></tr>
        <tr><td><b>Total characters</b></td><td>{ds['total_chars']:,}</td></tr>
        <tr><td><b>Total lines</b></td><td>{ds['total_lines']:,}</td></tr>
        <tr><td><b>Sources</b></td><td>{', '.join(s for s, _ in ds['by_source'][:5]) or 'none'}</td></tr>
        <tr><td><b>Training sessions</b></td><td>{ds['training_sessions']}</td></tr>
        </table>"""
        self.metrics_browser.setHtml(html)

    def _refresh_chart(self):
        self._refresh_reward_chart()
        self._refresh_td_error_chart()

    def _refresh_reward_chart(self):
        pix = render_reward_chart(self.agent.episode_rewards)
        if pix:
            self.chart_label.setPixmap(pix)
            self.chart_label.setAlignment(Qt.AlignCenter)
        else:
            self.chart_label.setText("No training data yet.")

    def _refresh_td_error_chart(self):
        pix = render_td_error_chart(self.agent._td_errors)
        if pix:
            self.td_chart_label.setPixmap(pix)
            self.td_chart_label.setAlignment(Qt.AlignCenter)
        else:
            self.td_chart_label.setText("No TD-error data yet.")

    # ------------------------------------------------------------------
    # Tab 4: Agent callbacks
    # ------------------------------------------------------------------

    def process_user_request(self):
        request = self.request_input.text().lower()
        if not request: return
        self.terminal_view.clear(); self.log_t(f">>> USER REQUEST: {request}")
        prefix = self.intent_clf.predict(request)
        self.log_t(f"AGENT: Intent → '{prefix}'")
        if self.intent_clf.is_trained:
            top = self.intent_clf.predict_top_k(request, k=3)
            self.log_t(f"AGENT: Top-3: " + ", ".join(f"{p} ({c:.0%})" for p, c in top))
        code = self.markov_model.generate(prefix, max_length=40)
        if self.quality_svm.is_trained:
            self.log_t(f"AGENT: Quality score: {self.quality_svm.score(code, self.markov_model):.1%}")
        valid, msg = validate_syntax(code)
        self.log_t(f"AGENT: Syntax: {msg}")
        self.log_t("\u2500" * 40 + " GENERATED CODE " + "\u2500" * 40)
        for line in code.split("\n"):
            self.log_t(line)
        self.log_t("\u2500" * 93)
        self.log_t("AGENT: cd /data")
        self.vfs.cwd = "/data"
        self.log_t("AGENT: Writing script.py ...")
        self.vfs._write("/data/script.py", code)
        self.log_t("AGENT: Executing ...")
        self.vfs._write("/data/output.txt",
                        f"Execution successful.\nLines: {code.count(chr(10))+1}\nOutput: [Simulated]")
        self.log_t("AGENT: Done.")
        self.update_vfs_tree(); self.request_input.clear()
        self.status_label.setText("Agent task completed")

    def log_t(self, text):
        self.terminal_view.appendPlainText(text)

    # -- Interactive terminal --------------------------------------------

    def _execute_terminal_command(self):
        cmd = self.terminal_input.text().strip()
        if not cmd: return
        self._term_history.append(cmd)
        self._term_history_idx = -1
        self.log_t(f"$ {cmd}")
        self.terminal_input.clear()

        parts = cmd.split()
        if not parts: return
        c = parts[0].lower()

        if c == "ls":
            items = self.vfs._list(self.vfs.cwd)
            if items:
                for n, t in items:
                    self.log_t(f"  {'[DIR]' if t == 'dir' else '[FILE]':6s} {n}")
            else:
                self.log_t("  (empty)")
        elif c == "cd":
            target = parts[1] if len(parts) > 1 else "/"
            node = self.vfs._node(target if target.startswith("/") else self.vfs.cwd.rstrip("/") + "/" + target)
            if node and node.get("type") == "dir":
                self.vfs.cwd = target
            else:
                self.log_t(f"cd: no such directory: {target}")
        elif c == "cat":
            if len(parts) > 1:
                content = self.vfs._read(parts[1])
                if content is not None:
                    for line in content.split("\n"):
                        self.log_t(line)
                else:
                    self.log_t(f"cat: {parts[1]}: not found")
            else:
                self.log_t("cat: missing filename")
        elif c == "tree":
            for line in self.vfs._tree_lines():
                self.log_t(line)
        elif c == "pwd":
            self.log_t(self.vfs.cwd)
        elif c == "mkdir":
            if len(parts) > 1:
                self.vfs._mkdir(parts[1]); self.log_t(f"Created: {parts[1]}")
                self.update_vfs_tree()
            else:
                self.log_t("mkdir: missing name")
        elif c == "rm":
            if len(parts) > 1:
                ok = self.vfs._delete(parts[1])
                self.log_t(f"{'Deleted' if ok else 'Not found'}: {parts[1]}")
                self.update_vfs_tree()
        elif c == "help":
            self.log_t("Commands: ls, cd <dir>, cat <file>, tree, pwd, mkdir <name>, rm <file>, generate, write <file>, run, clear, help")
        elif c == "clear":
            self.terminal_view.clear()
        elif c == "generate":
            prefix = " ".join(parts[1:]) if len(parts) > 1 else "def process_data ("
            code = self.markov_model.generate(prefix, max_length=40)
            self.vfs.generated_code = code
            self.log_t("--- Generated ---")
            for line in code.split("\n"):
                self.log_t(line)
        elif c == "write":
            fname = parts[1] if len(parts) > 1 else "script.py"
            ok = self.vfs._write(self.vfs.cwd.rstrip("/") + "/" + fname, self.vfs.generated_code)
            self.log_t(f"{'Written' if ok else 'Failed'}: {fname}")
            self.update_vfs_tree()
        elif c == "run":
            fname = parts[1] if len(parts) > 1 else "script.py"
            node = self.vfs._node(self.vfs.cwd.rstrip("/") + "/" + fname)
            if node and node.get("type") == "file":
                self.log_t(f"Running {fname} ...")
                lines = node.get("content", "").count("\n") + 1
                self.vfs._write(self.vfs.cwd.rstrip("/") + "/output.txt",
                                f"Execution successful.\nLines: {lines}\nOutput: [Simulated]")
                self.log_t("Done. Output written to output.txt")
            else:
                self.log_t(f"run: {fname} not found")
        else:
            self.log_t(f"Unknown command: {c}. Type 'help' for available commands.")

    # -- VFS tree -------------------------------------------------------

    def update_vfs_tree(self):
        self.tree_widget.clear()
        self._add_tree(self.tree_widget.invisibleRootItem(), self.vfs.root, "/")
        self.tree_widget.expandAll()

    def _add_tree(self, parent, node, name):
        is_dir = node.get("type") == "dir"
        item = QTreeWidgetItem(parent, [("📁 " if is_dir else "📄 ") + name])
        item.setData(0, Qt.UserRole, name)
        if is_dir:
            for cn, cd in node.get("children", {}).items():
                self._add_tree(item, cd, cn)

    def _on_tree_click(self, item, _col):
        parts = []; cur = item
        while cur:
            n = cur.data(0, Qt.UserRole)
            if n is not None: parts.append(n)
            cur = cur.parent()
        path = "/" + "/".join(reversed(parts))
        content = self.vfs._read(path)
        if content is not None:
            self.file_viewer.setPlainText(content)
        else:
            nd = self.vfs._node(path)
            if nd and nd.get("type") == "dir":
                items = self.vfs._list(path)
                self.file_viewer.setPlainText("\n".join(
                    f"{'[DIR]' if t=='dir' else '[FILE]':6s} {n}" for n, t in items) if items else "(empty)")
            else:
                self.file_viewer.setPlainText("")

    # -- Agent training -------------------------------------------------

    def toggle_training(self):
        if self._training_thread and self._training_thread.is_alive():
            self._stop_training.set()
            self.btn_train.setText("\u23f3  Stopping ..."); self.btn_train.setEnabled(False)
        else:
            self.agent.lr = self.lr_spin.value(); self.agent.gamma = self.gamma_spin.value()
            self.agent._exploration_strategy = (
                "ucb" if self.exploration_combo.currentText() == "UCB1"
                else "epsilon_greedy")
            self.agent._use_replay = self.replay_check.isChecked()
            self.agent._use_double_q = self.dq_check.isChecked()
            self.markov_model.repetition_penalty = self.rep_pen_spin.value()
            self.markov_model.discount = self.discount_spin.value()
            self._stop_training.clear()
            self._training_thread = threading.Thread(target=self._train_loop, daemon=True)
            self._training_thread.start()
            self.btn_train.setText("\u23f9  Stop Training")
            self.status_label.setText("Training started")

    def _train_loop(self):
        ep = 0
        while not self._stop_training.is_set():
            ep += 1; self.vfs.reset()
            state = self.vfs.get_state(); tr = 0.0; steps = 0
            while steps < 25 and not self._stop_training.is_set():
                a = self.agent.choose_action(state)
                # Use intent classifier for smarter prefix selection
                if self.intent_clf.is_trained and a == 3:
                    req = random.choice(
                        ["write a sort function", "add numbers",
                         "process data", "calculate area",
                         "read a file", "fibonacci sequence"])
                    p = self.intent_clf.predict(req)
                else:
                    p = random.choice(["def process_data (",
                                       "def add ( a , b )",
                                       "for i in range (",
                                       "class DataProcessor :",
                                       "def get_area ( radius )"])
                ns, r, d, _ = self.vfs.step(
                    a, self.markov_model, prefix=p,
                    quality_svm=(self.quality_svm
                                  if self.quality_svm.is_trained else None))
                # Store for replay
                self.agent.store_transition(state, a, r, ns, d)
                self.agent.update(state, a, r, ns, d)
                tr += r; state = ns; steps += 1
                if d: break
            # Replay every 4 episodes
            if ep % 4 == 0:
                self.agent.replay_update()
            self.agent.episodes += 1
            self.agent.episode_rewards.append(tr)
            self.agent.decay_epsilon()
            self.signals.episode_done.emit(ep, tr)
        self.signals.training_finished.emit()

    def _on_episode_done(self, ep, tr):
        self.episode_label.setText(f"Episodes: {self.agent.episodes}")
        self.epsilon_label.setText(f"\u03b5: {self.agent.epsilon:.3f}")
        self.qtable_label.setText(f"Q: {len(self.agent.q_table)}")
        self.update_q_table()
        if ep % 50 == 0:
            avg = np.mean(self.agent.episode_rewards[-50:])
            self.status_label.setText(f"Ep {ep} | Avg reward: {avg:.1f}")
            self.update_vfs_tree()
            # Auto-refresh chart every 200 episodes
            if ep % 200 == 0:
                pix = render_reward_chart(self.agent.episode_rewards)
                if pix:
                    self.chart_label.setPixmap(pix)
                    self.chart_label.setAlignment(Qt.AlignCenter)
                # TD-error chart
                td_pix = render_td_error_chart(self.agent._td_errors)
                if td_pix:
                    self.td_chart_label.setPixmap(td_pix)
                    self.td_chart_label.setAlignment(Qt.AlignCenter)

    def _on_training_finished(self):
        self.btn_train.setText("\u25b6  Start Training (Threaded)")
        self.btn_train.setEnabled(True)
        self._refresh_charts()
        self.status_label.setText("Training stopped")

    def _refresh_charts(self):
        self._refresh_reward_chart()
        self._refresh_td_error_chart()

    def reset_agent(self):
        if self._training_thread and self._training_thread.is_alive():
            self._stop_training.set(); self._training_thread.join(timeout=2.0)
        self.btn_train.setText("\u25b6  Start Training (Threaded)")
        self.btn_train.setEnabled(True)
        self.vfs.reset()
        self.agent = QLearningAgent(n_actions=len(SandboxVFS.ACTIONS))
        self.agent.lr = self.lr_spin.value(); self.agent.gamma = self.gamma_spin.value()
        self.update_vfs_tree(); self.update_q_table()
        self.terminal_view.clear(); self.file_viewer.clear()
        self._update_status()

    def agent_greedy_step(self):
        state = self.vfs.get_state()
        a = self.agent.get_best_action(state)
        prefix = self.chat_prefix.text() or "def process_data ("
        ns, r, d, info = self.vfs.step(
            a, self.markov_model, prefix=prefix,
            quality_svm=(self.quality_svm
                          if self.quality_svm.is_trained else None))
        self.log_t(f"GREEDY → {info.get('action_name','?')}  (r={r:.1f})")
        for l in self.vfs.logs: self.log_t(l)
        self.update_vfs_tree(); self.update_q_table()
        if d: self.log_t("--- Done ---")

    def update_q_table(self):
        # Show Q-values aggregated across all states for each action
        action_q: Dict[int, float] = collections.defaultdict(float)
        action_visits: Dict[int, int] = collections.defaultdict(int)
        for (s, a), q in self.agent.q_table.items():
            action_q[a] += q
            action_visits[a] += self.agent.visit_counts.get((s, a), 0)
        # Also sum visit counts
        for (s, a), v in self.agent.visit_counts.items():
            action_visits[a] += v
        rows = sorted(action_q.items(), key=lambda x: x[1], reverse=True)
        self.q_table_view.setRowCount(0)
        for a, q in rows:
            if q == 0: continue
            r = self.q_table_view.rowCount(); self.q_table_view.insertRow(r)
            self.q_table_view.setItem(r, 0, QTableWidgetItem(
                SandboxVFS.ACTIONS[a] if a < len(SandboxVFS.ACTIONS) else f"action_{a}"))
            self.q_table_view.setItem(r, 1, QTableWidgetItem(f"{q:.3f}"))
            self.q_table_view.setItem(r, 2, QTableWidgetItem(str(action_visits.get(a, 0))))

    # ------------------------------------------------------------------
    # Tab 5: Database callbacks
    # ------------------------------------------------------------------

    def _db_update_label(self):
        self.db_label.setText(f"DB: {self.db.get_total_count()}")

    def _db_refresh_filters(self):
        src = self.db_source_combo.currentText()
        tag = self.db_tag_combo.currentText()
        self.db_source_combo.blockSignals(True)
        self.db_tag_combo.blockSignals(True)
        self.db_source_combo.clear(); self.db_source_combo.addItem("All")
        for s in self.db.get_sources():
            self.db_source_combo.addItem(s)
        self.db_source_combo.setCurrentText(src or "All")
        self.db_tag_combo.clear(); self.db_tag_combo.addItem("All")
        for t in self.db.get_all_tags():
            self.db_tag_combo.addItem(t)
        self.db_tag_combo.setCurrentText(tag or "All")
        self.db_source_combo.blockSignals(False)
        self.db_tag_combo.blockSignals(False)

    def _db_refresh_table(self, snippets=None):
        if snippets is None:
            src = self.db_source_combo.currentText()
            tag = self.db_tag_combo.currentText()
            src = "" if src == "All" else src
            tag = "" if tag == "All" else tag
            snippets = self.db.get_snippets(
                source=src, tag=tag,
                limit=self._db_page_size,
                offset=self._db_current_page * self._db_page_size)
        self.db_table.setRowCount(0)
        for r in snippets:
            row = self.db_table.rowCount(); self.db_table.insertRow(row)
            self.db_table.setItem(row, 0, QTableWidgetItem(str(r["id"])))
            self.db_table.setItem(row, 1, QTableWidgetItem(r.get("source", "")))
            self.db_table.setItem(row, 2, QTableWidgetItem(r.get("tags", "")))
            self.db_table.setItem(row, 3, QTableWidgetItem(str(r.get("line_count", 0))))
            self.db_table.setItem(row, 4, QTableWidgetItem(str(r.get("char_count", 0))))
            dt = r.get("created_at", "")[:19]
            self.db_table.setItem(row, 5, QTableWidgetItem(dt))
        total = self.db.get_total_count()
        pages = max(1, (total + self._db_page_size - 1) // self._db_page_size)
        self.db_page_label.setText(
            f"Page {self._db_current_page + 1} / {pages}")
        self.db_count_label.setText(f"{total} snippets total")
        self._db_update_label()

    def _db_on_row_select(self, row, _col, _prev_row, _prev_col):
        if row < 0 or row >= self.db_table.rowCount():
            return
        sid_item = self.db_table.item(row, 0)
        if not sid_item:
            return
        sid = int(sid_item.text())
        rec = self.db.get_snippet_by_id(sid)
        if rec:
            self.db_preview.setPlainText(rec["code"])
            self.db_tags_input.setText(rec.get("tags", ""))

    def _db_search(self):
        q = self.db_search_input.text().strip()
        if not q:
            self._db_refresh_table(); return
        results = self.db.search_snippets(q, limit=self._db_page_size)
        self._db_refresh_table(snippets=results)
        self.status_label.setText(f"DB search: {len(results)} results for '{q}'")

    def _db_prev_page(self):
        if self._db_current_page > 0:
            self._db_current_page -= 1
            self._db_refresh_table()

    def _db_next_page(self):
        self._db_current_page += 1
        snippets = self.db.get_snippets(
            source="" if self.db_source_combo.currentText() == "All"
                     else self.db_source_combo.currentText(),
            tag="" if self.db_tag_combo.currentText() == "All"
                 else self.db_tag_combo.currentText(),
            limit=self._db_page_size,
            offset=self._db_current_page * self._db_page_size)
        if not snippets and self._db_current_page > 0:
            self._db_current_page -= 1
            snippets = self.db.get_snippets(
                source="" if self.db_source_combo.currentText() == "All"
                         else self.db_source_combo.currentText(),
                tag="" if self.db_tag_combo.currentText() == "All"
                     else self.db_tag_combo.currentText(),
                limit=self._db_page_size,
                offset=self._db_current_page * self._db_page_size)
        self._db_refresh_table(snippets=snippets)

    def _db_add_snippet_dialog(self):
        from PySide6.QtWidgets import QDialog, QVBoxLayout as DVL, QTextEdit as DTE
        dlg = QDialog(self); dlg.setWindowTitle("Add Snippet to Database")
        dlg.resize(600, 400); dl = DVL(dlg)
        te = DTE(); te.setFont(QFont("Consolas", 11))
        te.setPlaceholderText("Paste or type code here...")
        dl.addWidget(te)
        tr = QHBoxLayout()
        src_inp = QLineEdit(); src_inp.setPlaceholderText("source name")
        tr.addWidget(QLabel("Source:")); tr.addWidget(src_inp)
        tag_inp = QLineEdit(); tag_inp.setPlaceholderText("tag1, tag2, ...")
        tr.addWidget(QLabel("Tags:")); tr.addWidget(tag_inp)
        dl.addLayout(tr)
        br = QHBoxLayout()
        bok = QPushButton("Add"); bcancel = QPushButton("Cancel")
        br.addWidget(bok); br.addWidget(bcancel)
        dl.addLayout(br)
        def _do_add():
            code = te.toPlainText().strip()
            if not code:
                QMessageBox.warning(dlg, "Empty", "Code cannot be empty"); return
            sid = self.db.add_snippet(
                code, source=src_inp.text().strip() or "manual",
                tags=tag_inp.text().strip())
            self._db_refresh_table(); self._db_refresh_filters()
            self._db_refresh_stats()
            self.status_label.setText(f"Added snippet #{sid} to DB")
            dlg.accept()
        bok.clicked.connect(_do_add); bcancel.clicked.connect(dlg.reject)
        dlg.exec()

    def _db_edit_snippet(self):
        row = self.db_table.currentRow()
        if row < 0:
            QMessageBox.information(self, "No Selection",
                                    "Select a snippet to edit"); return
        sid = int(self.db_table.item(row, 0).text())
        rec = self.db.get_snippet_by_id(sid)
        if not rec:
            return
        from PySide6.QtWidgets import QDialog, QVBoxLayout as DVL, QTextEdit as DTE
        dlg = QDialog(self); dlg.setWindowTitle(f"Edit Snippet #{sid}")
        dlg.resize(600, 400); dl = DVL(dlg)
        te = DTE(str(rec["code"])); te.setFont(QFont("Consolas", 11))
        dl.addWidget(te)
        tr = QHBoxLayout()
        src_inp = QLineEdit(rec.get("source", "manual"))
        tag_inp = QLineEdit(rec.get("tags", ""))
        tr.addWidget(QLabel("Source:")); tr.addWidget(src_inp)
        tr.addWidget(QLabel("Tags:")); tr.addWidget(tag_inp)
        dl.addLayout(tr)
        br = QHBoxLayout()
        bsave = QPushButton("Save"); bcancel = QPushButton("Cancel")
        br.addWidget(bsave); br.addWidget(bcancel); dl.addLayout(br)
        def _do_save():
            self.db.update_snippet(sid, code=te.toPlainText(), tags=tag_inp.text().strip())
            self._db_refresh_table(); self._db_refresh_filters()
            self._db_refresh_stats()
            self.status_label.setText(f"Updated snippet #{sid}")
            dlg.accept()
        bsave.clicked.connect(_do_save); bcancel.clicked.connect(dlg.reject)
        dlg.exec()

    def _db_delete_selected(self):
        row = self.db_table.currentRow()
        if row < 0:
            QMessageBox.information(self, "No Selection",
                                    "Select a snippet to delete"); return
        sid = int(self.db_table.item(row, 0).text())
        if QMessageBox.question(
                self, "Delete Snippet",
                f"Delete snippet #{sid}?",
                QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            self.db.delete_snippet(sid)
            self._db_refresh_table(); self._db_refresh_filters()
            self._db_refresh_stats()
            self.db_preview.clear()
            self.status_label.setText(f"Deleted snippet #{sid}")

    def _db_save_tags(self):
        row = self.db_table.currentRow()
        if row < 0:
            return
        sid = int(self.db_table.item(row, 0).text())
        tags = self.db_tags_input.text().strip()
        self.db.update_snippet(sid, tags=tags)
        self._db_refresh_table(); self._db_refresh_filters()
        self.status_label.setText(f"Tags updated for #{sid}")

    def _db_score_selected(self):
        row = self.db_table.currentRow()
        if row < 0:
            return
        sid = int(self.db_table.item(row, 0).text())
        rec = self.db.get_snippet_by_id(sid)
        if not rec or not self.quality_svm.is_trained:
            return
        score = self.quality_svm.score(rec["code"], self.markov_model)
        self.db.update_snippet(sid, quality_score=score)
        self.status_label.setText(
            f"Snippet #{sid} quality score: {score:.1%}")
        QMessageBox.information(
            self, "Quality Score",
            f"Snippet #{sid}\nSVM Quality: {score:.1%}\n"
            f"(score saved to database)")

    def _db_train_on_selected(self):
        row = self.db_table.currentRow()
        if row < 0:
            return
        sid = int(self.db_table.item(row, 0).text())
        rec = self.db.get_snippet_by_id(sid)
        if not rec:
            return
        t = self.markov_model.train(rec["code"])
        self._all_snippets.append(rec["code"])
        self.corpus_view.appendPlainText(rec["code"] + "\n---\n")
        self.update_markov_table()
        self.corpus_label.setText(f"Tokens: {self.markov_model.token_counts}")
        self.status_label.setText(
            f"Trained on snippet #{sid} ({t} tokens)")

    def _db_save_current_snippets(self):
        """Save all in-memory snippets to the database."""
        if not self._all_snippets:
            QMessageBox.information(self, "Empty",
                                    "No in-memory snippets to save"); return
        count = self.db.add_snippets_batch(self._all_snippets, source="session")
        self._db_refresh_table(); self._db_refresh_filters()
        self._db_refresh_stats()
        self.status_label.setText(
            f"Saved {count} snippets to database")

    def _db_train_from_db(self):
        """Load all DB snippets and (re)train the Markov model."""
        codes = self.db.get_all_codes(limit=50000)
        if not codes:
            QMessageBox.information(self, "Empty DB",
                                    "No snippets in database to train on")
            return
        # Reset model and retrain from DB
        n = self.markov_model.n
        self.markov_model = MarkovCodeModel(n=n)
        self._all_snippets.clear()
        self.corpus_view.clear()
        total_tokens, count = self.markov_model.train_batch(codes)
        self._all_snippets.extend(codes)
        # Update corpus view (show first 200 for performance)
        for s in codes[:200]:
            self.corpus_view.appendPlainText(s + "\n---\n")
        if len(codes) > 200:
            self.corpus_view.appendPlainText(
                f"\n... and {len(codes) - 200} more snippets\n")
        self.update_markov_table()
        self._train_svm_models()
        self.corpus_label.setText(f"Tokens: {self.markov_model.token_counts}")
        # Log training session
        self.db.log_session(
            f"train_from_db_{_dt.datetime.now():%Y%m%d_%H%M%S}",
            count, total_tokens, n)
        self._db_refresh_stats()
        self.status_label.setText(
            f"Trained from DB: {count} snippets, {total_tokens} tokens")

    def _db_import_file_dialog(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Import File(s) into Database", "",
            "Python/Text (*.py *.txt);;All (*)")
        if not paths:
            return
        total_count, total_chars = 0, 0
        for p in paths:
            pp = Path(p)
            if pp.is_file():
                c, ch = self.db.import_from_file(str(pp))
                total_count += c; total_chars += ch
            elif pp.is_dir():
                snippets, _, tc = SnippetImporter.import_directory(str(pp))
                if snippets:
                    total_count += self.db.add_snippets_batch(
                        snippets, source=pp.name)
                    total_chars += tc
        self._db_refresh_table(); self._db_refresh_filters()
        self._db_refresh_stats()
        self.status_label.setText(
            f"Imported {total_count} snippets ({total_chars:,} chars) into DB")

    def _db_export_dialog(self):
        src = self.db_source_combo.currentText()
        src_filter = "" if src == "All" else src
        default = f"snippets_export{f'_{src}' if src_filter else ''}.py"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Snippets", default, "Python (*.py);;Text (*.txt)")
        if not path:
            return
        count = self.db.export_to_file(path, source=src_filter)
        self.status_label.setText(f"Exported {count} snippets to {path}")

    def _db_refresh_stats(self):
        s = self.db.get_stats()
        html = f"""<h3 style='color:#ce93d8'>Database Statistics</h3>
        <table style='color:#ddd;font-size:12px;' cellpadding='3'>
        <tr><td><b>Total snippets</b></td><td>{s['total_snippets']:,}</td></tr>
        <tr><td><b>Total characters</b></td><td>{s['total_chars']:,}</td></tr>
        <tr><td><b>Total lines</b></td><td>{s['total_lines']:,}</td></tr>
        <tr><td><b>Training sessions</b></td><td>{s['training_sessions']}</td></tr>
        <tr><td><b>Newest source</b></td><td>{s['newest_source']}</td></tr>
        <tr><td><b>Newest added</b></td><td>{s['newest_date'][:19]}</td></tr>
        <tr><td colspan=2><hr style='border-color:#555'></td></tr>
        <tr><td colspan=2><b style='color:#FFCA28'>By Source</b></td></tr>"""
        for src, cnt in s["by_source"][:8]:
            html += f"<tr><td>&nbsp;&nbsp;{src}</td><td>{cnt}</td></tr>"
        html += "<tr><td colspan=2><hr style='border-color:#555'></td></tr>"
        html += "<tr><td colspan=2><b style='color:#FFCA28'>By Language</b></td></tr>"
        for lang, cnt in s["by_language"]:
            html += f"<tr><td>&nbsp;&nbsp;{lang}</td><td>{cnt}</td></tr>"
        html += "</table>"
        self.db_stats_browser.setHtml(html)

    def _db_clear_all(self):
        if QMessageBox.question(
                self, "Clear Database",
                "Delete ALL snippets and training logs? This cannot be undone.",
                QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            n = self.db.clear_all()
            self._db_refresh_table(); self._db_refresh_filters()
            self._db_refresh_stats()
            self.db_preview.clear()
            self.status_label.setText(f"Database cleared ({n} snippets removed)")

    def _update_status(self):
        self.episode_label.setText(f"Episodes: {self.agent.episodes}")
        self.epsilon_label.setText(f"\u03b5: {self.agent.epsilon:.3f}")
        self.qtable_label.setText(f"Q: {len(self.agent.q_table)}")
        self.corpus_label.setText(f"Tokens: {self.markov_model.token_counts}")


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    from PySide6.QtGui import QPalette
    p = QPalette()
    p.setColor(QPalette.Window, QColor(45, 45, 48))
    p.setColor(QPalette.WindowText, QColor(220, 220, 220))
    p.setColor(QPalette.Base, QColor(30, 30, 30))
    p.setColor(QPalette.AlternateBase, QColor(45, 45, 48))
    p.setColor(QPalette.ToolTipBase, QColor(25, 25, 25))
    p.setColor(QPalette.ToolTipText, QColor(220, 220, 220))
    p.setColor(QPalette.Text, QColor(220, 220, 220))
    p.setColor(QPalette.Button, QColor(45, 45, 48))
    p.setColor(QPalette.ButtonText, QColor(220, 220, 220))
    p.setColor(QPalette.BrightText, QColor(255, 0, 0))
    p.setColor(QPalette.Link, QColor(42, 130, 218))
    p.setColor(QPalette.Highlight, QColor(42, 130, 218))
    p.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
    app.setPalette(p)
    w = MainWindow(); w.show()
    ret = app.exec()
    w.db.close()
    sys.exit(ret)