"""
NLI-based Judge
===============
Natural Language Inference based judge using cross-encoder models.

This is a lightweight, no-network alternative to LLM judges that runs
locally using sentence-transformers or transformers models.

Key benefits:
- No API costs
- Fast inference (~50ms per pair)
- Deterministic results
- Works offline

Usage:
    judge = NLIJudge()
    score = await judge.score_groundedness(answer, context)

Models supported:
- cross-encoder/nli-deberta-v3-base (default, balanced)
- cross-encoder/nli-deberta-v3-small (faster)
- cross-encoder/nli-deberta-v3-large (more accurate)
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Try to import transformers for NLI models
try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not installed. NLIJudge will use fallback.")


class NLIJudge:
    """
    NLI-based judge using cross-encoder models for entailment scoring.

    Maps NLI labels to scores:
    - entailment → high score (0.9-1.0)
    - neutral → medium score (0.4-0.6)
    - contradiction → low score (0.0-0.2)

    Handles long texts by chunking and averaging scores.
    """

    # NLI label mappings (index → score)
    # Most NLI models use: 0=contradiction, 1=neutral, 2=entailment
    LABEL_SCORES = {
        0: 0.1,  # contradiction
        1: 0.5,  # neutral
        2: 0.95,  # entailment
    }

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        device: str | None = None,
        max_length: int = 512,
        chunk_overlap: int = 100,
    ):
        """
        Initialize NLI Judge.

        Args:
            model_name: HuggingFace model name for NLI
            device: Device to use (cuda/cpu/mps, auto-detected if None)
            max_length: Maximum sequence length (will chunk longer texts)
            chunk_overlap: Overlap between chunks for context continuity
        """
        self.model_name = model_name
        self.max_length = max_length
        self.chunk_overlap = chunk_overlap

        # Usage tracking
        self._usage = {
            "pairs_evaluated": 0,
            "chunks_processed": 0,
        }

        if not TRANSFORMERS_AVAILABLE:
            logger.warning("transformers not available, using word overlap fallback")
            self.model = None
            self.tokenizer = None
            self.device = "cpu"
            return

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device

        try:
            logger.info(f"Loading NLI model: {model_name} on {device}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(device)
            self.model.eval()
            logger.info("NLI model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load NLI model: {e}")
            self.model = None
            self.tokenizer = None

    async def score_groundedness(self, answer: str, context: str) -> float:
        """
        Score groundedness using NLI entailment.

        For each sentence in the answer, checks if it's entailed by the context.
        Returns average entailment score.

        Args:
            answer: Generated answer text
            context: Retrieved context used for generation

        Returns:
            Score between 0.0 (not grounded) and 1.0 (fully grounded)
        """
        if not answer.strip() or not context.strip():
            return 0.0

        if self.model is None:
            return self._word_overlap_score(answer, context)

        # Split answer into sentences for fine-grained scoring
        answer_sentences = self._split_sentences(answer)
        if not answer_sentences:
            return 0.0

        # Chunk context if too long
        context_chunks = self._chunk_text(context)

        scores = []
        for sentence in answer_sentences:
            if len(sentence.strip()) < 10:  # Skip very short sentences
                continue

            # Check entailment against each context chunk, take max
            chunk_scores = []
            for chunk in context_chunks:
                score = self._compute_entailment(premise=chunk, hypothesis=sentence)
                chunk_scores.append(score)
                self._usage["chunks_processed"] += 1

            if chunk_scores:
                scores.append(max(chunk_scores))  # Take best chunk score

            self._usage["pairs_evaluated"] += 1

        if not scores:
            return 0.5  # Neutral if no sentences to score

        return sum(scores) / len(scores)

    async def score_relevance(self, answer: str, question: str) -> float:
        """
        Score relevance using NLI.

        Checks if the answer entails a semantic response to the question.

        Args:
            answer: Generated answer text
            question: Original user question

        Returns:
            Score between 0.0 (not relevant) and 1.0 (fully relevant)
        """
        if not answer.strip() or not question.strip():
            return 0.0

        if self.model is None:
            return self._word_overlap_score(answer, question)

        # Convert question to a premise statement
        # e.g., "What is X?" → "This text explains X"
        question_as_premise = self._question_to_premise(question)

        # Check if answer is relevant to question topic
        score = self._compute_entailment(premise=answer, hypothesis=question_as_premise)
        self._usage["pairs_evaluated"] += 1

        return score

    async def score_factuality(self, answer: str, facts: list[str]) -> tuple[float, list[str]]:
        """
        Score factual accuracy using NLI contradiction detection.

        Args:
            answer: Generated answer text
            facts: List of known facts to check against

        Returns:
            Tuple of (score, list of contradicting facts)
        """
        if not answer.strip() or not facts:
            return 1.0, []

        if self.model is None:
            return 0.5, []

        contradictions = []
        scores = []

        for fact in facts:
            score = self._compute_entailment(premise=answer, hypothesis=fact)
            scores.append(score)
            self._usage["pairs_evaluated"] += 1

            # If answer contradicts the fact (low entailment score)
            if score < 0.3:
                contradictions.append(fact)

        avg_score = sum(scores) / len(scores) if scores else 1.0
        return avg_score, contradictions

    def _compute_entailment(self, premise: str, hypothesis: str) -> float:
        """
        Compute entailment score between premise and hypothesis.

        Returns:
            Score between 0.0 (contradiction) and 1.0 (entailment)
        """
        if self.model is None or self.tokenizer is None:
            return self._word_overlap_score(hypothesis, premise)

        try:
            # Tokenize
            inputs = self.tokenizer(
                premise,
                hypothesis,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

            # Convert to probabilities
            probs = torch.softmax(logits, dim=-1)[0]

            # Weighted score based on label probabilities
            score = (
                probs[0].item() * self.LABEL_SCORES[0]  # contradiction
                + probs[1].item() * self.LABEL_SCORES[1]  # neutral
                + probs[2].item() * self.LABEL_SCORES[2]  # entailment
            )

            return score

        except Exception as e:
            logger.warning(f"NLI computation failed: {e}")
            return self._word_overlap_score(hypothesis, premise)

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        import re

        # Simple sentence splitting
        sentences = re.split(r"[.!?]+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _chunk_text(self, text: str) -> list[str]:
        """Chunk text into overlapping segments."""
        if self.tokenizer is None:
            # Word-based chunking fallback
            words = text.split()
            chunk_size = 400  # Approximate words per chunk
            chunks = []
            for i in range(0, len(words), chunk_size - self.chunk_overlap // 5):
                chunk = " ".join(words[i : i + chunk_size])
                if chunk:
                    chunks.append(chunk)
            return chunks if chunks else [text]

        # Token-based chunking
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunk_size = self.max_length - 50  # Reserve space for hypothesis + special tokens
        overlap = self.chunk_overlap

        chunks = []
        for i in range(0, len(tokens), chunk_size - overlap):
            chunk_tokens = tokens[i : i + chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            if chunk_text.strip():
                chunks.append(chunk_text)

        return chunks if chunks else [text]

    def _question_to_premise(self, question: str) -> str:
        """Convert question to a statement for relevance checking."""
        q_lower = question.lower().strip()

        # Simple templates for common question types
        if q_lower.startswith("what is"):
            return "This text explains " + question[8:].rstrip("?")
        elif q_lower.startswith("what are"):
            return "This text explains " + question[9:].rstrip("?")
        elif q_lower.startswith("how"):
            return "This text describes a method or process"
        elif q_lower.startswith("why"):
            return "This text explains reasons or causes"
        elif q_lower.startswith("when"):
            return "This text mentions time or dates"
        elif q_lower.startswith("where"):
            return "This text mentions locations"
        elif q_lower.startswith("who"):
            return "This text mentions people or entities"
        else:
            # Generic: "This text is related to: [question]"
            return f"This text addresses the topic: {question.rstrip('?')}"

    def _word_overlap_score(self, text1: str, text2: str) -> float:
        """Fallback word overlap heuristic."""
        if not text1 or not text2:
            return 0.0

        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def get_usage(self) -> dict[str, Any]:
        """Get usage statistics."""
        return self._usage.copy()

    def reset_usage(self) -> None:
        """Reset usage tracking."""
        self._usage = {
            "pairs_evaluated": 0,
            "chunks_processed": 0,
        }

    def __repr__(self) -> str:
        model_status = "loaded" if self.model is not None else "fallback"
        return f"NLIJudge(model={self.model_name!r}, status={model_status}, device={self.device})"
