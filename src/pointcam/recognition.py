"""
Recognition pipeline components for PointCam.

Tier 1 improvements:
- BibSetValidator: Validates predictions against known race bib numbers
- EnhancedTemporalVoting: Multi-frame voting with weighted consensus
- ConfidenceManager: Confidence thresholding and review queue

Tier 2 improvements (compute-efficient):
- CropQualityFilter: Reject blurry/partial crops before OCR (saves compute)
- PostOCRCleanup: Fix common OCR errors with minimal overhead
- CascadedOCR: Fast model first, slow model only if uncertain
"""

from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING
import json
import re
import time

# Optional imports for Tier 2 components (quality filter needs cv2/numpy)
# These are optional - Tier 1 works without them
try:
    import cv2
    import numpy as np
    from numpy import ndarray as NDArray

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

    # Dummy type for when numpy isn't available
    class NDArray:  # type: ignore
        pass


# ---------------------------------------------------------------------------
# Bib Set Validation
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Result of bib number validation against known set."""

    original: str  # Original OCR prediction
    validated: str  # Validated/corrected number
    is_valid: bool  # Exact match in bib set
    is_corrected: bool  # Fuzzy-matched to different number
    confidence_boost: float  # Confidence adjustment
    alternatives: List[str]  # Other close matches


class BibSetValidator:
    """
    Validates OCR predictions against known race bib numbers.

    Features:
    - Exact match validation
    - Fuzzy matching for single-digit errors
    - Digit count validation
    - Confidence boosting for valid numbers
    """

    def __init__(
        self,
        bib_set: Set[str],
        fuzzy_threshold: float = 0.75,
        max_edit_distance: int = 1,
    ):
        """
        Args:
            bib_set: Set of valid bib numbers for this race
            fuzzy_threshold: Minimum similarity ratio for fuzzy match
            max_edit_distance: Maximum edit distance for correction
        """
        self.bib_set = bib_set
        self.fuzzy_threshold = fuzzy_threshold
        self.max_edit_distance = max_edit_distance

        # Pre-compute bib lengths for quick filtering
        self.bib_lengths = {len(b) for b in bib_set}

        # Index by prefix for faster fuzzy matching
        self._prefix_index = self._build_prefix_index()

    def _build_prefix_index(self) -> Dict[str, List[str]]:
        """Build index of bibs by first 2 digits for faster lookup."""
        index: Dict[str, List[str]] = {}
        for bib in self.bib_set:
            if len(bib) >= 2:
                prefix = bib[:2]
                if prefix not in index:
                    index[prefix] = []
                index[prefix].append(bib)
        return index

    def validate(
        self,
        prediction: str,
        ocr_confidence: float,
    ) -> ValidationResult:
        """
        Validate an OCR prediction against the bib set.

        Args:
            prediction: OCR predicted bib number
            ocr_confidence: Original OCR confidence score

        Returns:
            ValidationResult with validation details
        """
        if not prediction:
            return ValidationResult(
                original="",
                validated="",
                is_valid=False,
                is_corrected=False,
                confidence_boost=0.0,
                alternatives=[],
            )

        # Exact match - best case
        if prediction in self.bib_set:
            return ValidationResult(
                original=prediction,
                validated=prediction,
                is_valid=True,
                is_corrected=False,
                confidence_boost=0.1,  # Boost confidence for valid numbers
                alternatives=[],
            )

        # No exact match - try fuzzy matching
        candidates = self._find_candidates(prediction)

        if candidates:
            best_match, similarity = candidates[0]
            alternatives = [c[0] for c in candidates[1:4]]

            # Only correct if similarity is high enough
            if similarity >= self.fuzzy_threshold:
                return ValidationResult(
                    original=prediction,
                    validated=best_match,
                    is_valid=False,
                    is_corrected=True,
                    confidence_boost=-0.1,  # Reduce confidence for corrections
                    alternatives=alternatives,
                )

        # No good match found
        return ValidationResult(
            original=prediction,
            validated=prediction,
            is_valid=False,
            is_corrected=False,
            confidence_boost=-0.2,  # Penalize unvalidated predictions
            alternatives=[],
        )

    def _find_candidates(
        self,
        prediction: str,
    ) -> List[Tuple[str, float]]:
        """
        Find bib numbers similar to the prediction.

        Returns list of (bib_number, similarity_score) sorted by similarity.
        """
        candidates = []

        # Priority 1: Leading-digit recovery — the most common OCR error
        # is truncating the first digit (camera angle clips left side of bib).
        # Try prepending each digit 1-9 and check for exact match.
        for d in "123456789":
            prepended = d + prediction
            if prepended in self.bib_set:
                # High similarity: all original digits are correct, just missing one
                sim = len(prediction) / (len(prediction) + 1)  # e.g. 3/4 = 0.75
                candidates.append((prepended, sim + 0.15))  # Boost for exact structural match

        # Priority 2: Trailing-digit recovery (less common but possible)
        for d in "0123456789":
            appended = prediction + d
            if appended in self.bib_set:
                sim = len(prediction) / (len(prediction) + 1)
                candidates.append((appended, sim + 0.10))

        # Priority 3: Check bibs with same prefix (faster)
        if len(prediction) >= 2:
            prefix = prediction[:2]
            prefix_bibs = self._prefix_index.get(prefix, [])
            for bib in prefix_bibs:
                sim = self._similarity(prediction, bib)
                if sim >= self.fuzzy_threshold:
                    candidates.append((bib, sim))

        # Priority 4: If no good matches yet, check all bibs with similar length
        if not candidates:
            for bib in self.bib_set:
                # Only compare similar lengths
                if abs(len(bib) - len(prediction)) <= 1:
                    sim = self._similarity(prediction, bib)
                    if sim >= self.fuzzy_threshold:
                        candidates.append((bib, sim))

        # Deduplicate (keep highest similarity per bib)
        best = {}
        for bib, sim in candidates:
            if bib not in best or sim > best[bib]:
                best[bib] = sim
        candidates = list(best.items())

        # Sort by similarity (highest first)
        candidates.sort(key=lambda x: -x[1])
        return candidates

    def _similarity(self, a: str, b: str) -> float:
        """Calculate similarity ratio between two strings."""
        return SequenceMatcher(None, a, b).ratio()

    def _edit_distance(self, a: str, b: str) -> int:
        """Calculate Levenshtein edit distance."""
        if len(a) < len(b):
            a, b = b, a
        if len(b) == 0:
            return len(a)

        prev_row = list(range(len(b) + 1))
        for i, ca in enumerate(a):
            curr_row = [i + 1]
            for j, cb in enumerate(b):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (ca != cb)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row
        return prev_row[-1]

    @classmethod
    def from_file(cls, filepath: str, **kwargs) -> "BibSetValidator":
        """
        Load bib set from a file (one bib number per line).

        Args:
            filepath: Path to bib set file
            **kwargs: Additional arguments for BibSetValidator

        Returns:
            BibSetValidator instance
        """
        with open(filepath, "r") as f:
            bib_set = {line.strip() for line in f if line.strip()}
        return cls(bib_set, **kwargs)

    @classmethod
    def from_range(cls, start: int, end: int, **kwargs) -> "BibSetValidator":
        """
        Create validator for a numeric range of bib numbers.

        Args:
            start: First bib number
            end: Last bib number (inclusive)
            **kwargs: Additional arguments for BibSetValidator

        Returns:
            BibSetValidator instance
        """
        bib_set = {str(i) for i in range(start, end + 1)}
        return cls(bib_set, **kwargs)


# ---------------------------------------------------------------------------
# Multi-Frame Voting
# ---------------------------------------------------------------------------


@dataclass
class VoteRecord:
    """Single OCR vote from a frame."""

    number: str
    confidence: float
    timestamp: float
    frame_idx: int


@dataclass
class ConsensusResult:
    """Result of multi-frame voting."""

    number: Optional[str]  # Consensus bib number
    confidence: float  # Combined confidence score
    vote_count: int  # Total frames with this number
    total_frames: int  # Total frames in window
    agreement_ratio: float  # vote_count / total_frames
    is_stable: bool  # Has reached stable consensus
    needs_review: bool  # Low confidence, needs human check


class EnhancedTemporalVoting:
    """
    Enhanced multi-frame voting with weighted consensus.

    Improvements over basic voting:
    - Recency weighting (newer frames count more)
    - Confidence weighting (high-conf predictions count more)
    - Stability detection (consistent reads over time)
    - Early exit when consensus is strong
    """

    def __init__(
        self,
        window_size: int = 15,
        min_votes: int = 3,
        stability_threshold: int = 5,
        confidence_threshold: float = 0.7,
        recency_decay: float = 0.95,
    ):
        """
        Args:
            window_size: Maximum frames to keep in history
            min_votes: Minimum votes before returning consensus
            stability_threshold: Consecutive same-number votes for "stable"
            confidence_threshold: Below this, mark as needs_review
            recency_decay: Weight decay for older frames (0.95 = 5% per frame)
        """
        self.window_size = window_size
        self.min_votes = min_votes
        self.stability_threshold = stability_threshold
        self.confidence_threshold = confidence_threshold
        self.recency_decay = recency_decay

        # track_id -> list of VoteRecords
        self.history: Dict[int, List[VoteRecord]] = defaultdict(list)

        # track_id -> (last_number, consecutive_count)
        self.stability_counter: Dict[int, Tuple[str, int]] = {}

    def update(
        self,
        track_id: int,
        number: Optional[str],
        confidence: float,
        frame_idx: int,
    ) -> None:
        """
        Add a new OCR result for a tracked bib.

        Args:
            track_id: Unique tracker ID
            number: OCR predicted number (None if no read)
            confidence: OCR confidence score
            frame_idx: Current frame index
        """
        if number is None:
            return

        # Add vote
        vote = VoteRecord(
            number=number,
            confidence=confidence,
            timestamp=time.time(),
            frame_idx=frame_idx,
        )
        self.history[track_id].append(vote)

        # Trim to window size
        if len(self.history[track_id]) > self.window_size:
            self.history[track_id] = self.history[track_id][-self.window_size :]

        # Update stability counter
        if track_id in self.stability_counter:
            last_number, count = self.stability_counter[track_id]
            if number == last_number:
                self.stability_counter[track_id] = (number, count + 1)
            else:
                self.stability_counter[track_id] = (number, 1)
        else:
            self.stability_counter[track_id] = (number, 1)

    def get_consensus(self, track_id: int) -> ConsensusResult:
        """
        Get weighted consensus from voting history.

        Returns:
            ConsensusResult with consensus details
        """
        if track_id not in self.history:
            return self._empty_result()

        votes = self.history[track_id]
        if len(votes) < self.min_votes:
            return self._empty_result()

        # Calculate weighted votes
        weighted_votes: Dict[str, float] = {}
        confidence_sums: Dict[str, float] = {}
        vote_counts: Dict[str, int] = {}

        for i, vote in enumerate(votes):
            # Recency weight: newer votes count more
            age = len(votes) - 1 - i
            recency_weight = self.recency_decay**age

            # Combined weight: recency * confidence
            weight = recency_weight * vote.confidence

            if vote.number not in weighted_votes:
                weighted_votes[vote.number] = 0.0
                confidence_sums[vote.number] = 0.0
                vote_counts[vote.number] = 0

            weighted_votes[vote.number] += weight
            confidence_sums[vote.number] += vote.confidence
            vote_counts[vote.number] += 1

        # Find winner
        winner = max(weighted_votes.keys(), key=lambda k: weighted_votes[k])
        winner_votes = vote_counts[winner]

        # Calculate metrics
        agreement_ratio = winner_votes / len(votes)
        avg_confidence = confidence_sums[winner] / winner_votes

        # Combined confidence: weight agreement ratio and avg OCR confidence
        combined_confidence = 0.5 * agreement_ratio + 0.5 * avg_confidence

        # Check stability
        is_stable = False
        if track_id in self.stability_counter:
            stable_number, stable_count = self.stability_counter[track_id]
            is_stable = stable_number == winner and stable_count >= self.stability_threshold

        # Determine if review needed
        needs_review = combined_confidence < self.confidence_threshold

        return ConsensusResult(
            number=winner,
            confidence=combined_confidence,
            vote_count=winner_votes,
            total_frames=len(votes),
            agreement_ratio=agreement_ratio,
            is_stable=is_stable,
            needs_review=needs_review,
        )

    def _empty_result(self) -> ConsensusResult:
        """Return empty/insufficient result."""
        return ConsensusResult(
            number=None,
            confidence=0.0,
            vote_count=0,
            total_frames=0,
            agreement_ratio=0.0,
            is_stable=False,
            needs_review=True,
        )

    def clear_track(self, track_id: int) -> None:
        """Clear voting history for a track."""
        if track_id in self.history:
            del self.history[track_id]
        if track_id in self.stability_counter:
            del self.stability_counter[track_id]

    def get_early_consensus(
        self,
        track_id: int,
        min_agreement: float = 0.9,
        min_confidence: float = 0.85,
    ) -> Optional[ConsensusResult]:
        """
        Get consensus early if agreement is very high.

        Useful for returning results before full window is filled.

        Returns:
            ConsensusResult if strong early consensus, None otherwise
        """
        if track_id not in self.history:
            return None

        votes = self.history[track_id]
        if len(votes) < self.min_votes:
            return None

        result = self.get_consensus(track_id)

        if result.agreement_ratio >= min_agreement and result.confidence >= min_confidence:
            return result

        return None


# ---------------------------------------------------------------------------
# Confidence Management
# ---------------------------------------------------------------------------


class ConfidenceLevel(Enum):
    """Confidence classification levels."""

    HIGH = "high"  # Confident, use directly
    MEDIUM = "medium"  # Acceptable, may want review
    LOW = "low"  # Uncertain, flag for review
    REJECT = "reject"  # Too low, don't use


@dataclass
class ClassifiedPrediction:
    """OCR prediction with confidence classification."""

    bib_number: str
    raw_confidence: float  # Original OCR confidence
    adjusted_confidence: float  # After validation adjustments
    level: ConfidenceLevel
    needs_review: bool
    validation_result: Optional[ValidationResult] = None
    voting_result: Optional[ConsensusResult] = None


@dataclass
class ReviewItem:
    """Item queued for human review."""

    id: str
    bib_number: str
    confidence: float
    timestamp: datetime
    frame_number: int
    evidence_path: Optional[str]
    alternatives: List[str]
    reason: str


class ConfidenceManager:
    """
    Manages confidence thresholds and review queue.

    Classifies predictions and maintains a queue of uncertain
    reads that may need human verification.
    """

    # Default thresholds
    DEFAULT_THRESHOLDS = {
        ConfidenceLevel.HIGH: 0.85,
        ConfidenceLevel.MEDIUM: 0.70,
        ConfidenceLevel.LOW: 0.50,
    }

    def __init__(
        self,
        thresholds: Optional[Dict[ConfidenceLevel, float]] = None,
        auto_accept_validated: bool = True,
        review_queue_size: int = 100,
    ):
        """
        Args:
            thresholds: Confidence level thresholds
            auto_accept_validated: Boost validated bibs to HIGH
            review_queue_size: Maximum items in review queue
        """
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS.copy()
        self.auto_accept_validated = auto_accept_validated
        self.review_queue_size = review_queue_size

        self.review_queue: List[ReviewItem] = []
        self._review_id_counter = 0

    def classify(
        self,
        bib_number: str,
        ocr_confidence: float,
        validation_result: Optional[ValidationResult] = None,
        voting_result: Optional[ConsensusResult] = None,
    ) -> ClassifiedPrediction:
        """
        Classify a prediction based on confidence and validation.

        Args:
            bib_number: Predicted bib number
            ocr_confidence: Raw OCR confidence
            validation_result: Result from BibSetValidator
            voting_result: Result from EnhancedTemporalVoting

        Returns:
            ClassifiedPrediction with confidence level
        """
        adjusted_confidence = ocr_confidence

        # Apply validation adjustment
        if validation_result:
            adjusted_confidence += validation_result.confidence_boost

            # If validated and auto-accept enabled, give additive boost
            # (not a floor — poor OCR with a valid number should not be forced HIGH)
            if self.auto_accept_validated and validation_result.is_valid:
                adjusted_confidence += 0.15

        # Apply voting adjustment
        if voting_result:
            # Voting consensus adds confidence via additive boosts
            # (not floors — a stable noisy reading is still noisy)
            if voting_result.is_stable:
                adjusted_confidence += 0.10
            elif voting_result.agreement_ratio >= 0.8:
                adjusted_confidence += 0.05

        # Clamp to [0, 1]
        adjusted_confidence = max(0.0, min(1.0, adjusted_confidence))

        # Determine level
        if adjusted_confidence >= self.thresholds[ConfidenceLevel.HIGH]:
            level = ConfidenceLevel.HIGH
        elif adjusted_confidence >= self.thresholds[ConfidenceLevel.MEDIUM]:
            level = ConfidenceLevel.MEDIUM
        elif adjusted_confidence >= self.thresholds[ConfidenceLevel.LOW]:
            level = ConfidenceLevel.LOW
        else:
            level = ConfidenceLevel.REJECT

        # Determine if review needed
        needs_review = level in (ConfidenceLevel.LOW, ConfidenceLevel.REJECT)

        # Also flag corrected predictions for review
        if validation_result and validation_result.is_corrected:
            needs_review = True

        return ClassifiedPrediction(
            bib_number=bib_number,
            raw_confidence=ocr_confidence,
            adjusted_confidence=adjusted_confidence,
            level=level,
            needs_review=needs_review,
            validation_result=validation_result,
            voting_result=voting_result,
        )

    def add_to_review_queue(
        self,
        prediction: ClassifiedPrediction,
        frame_number: int,
        evidence_path: Optional[str] = None,
        reason: str = "",
    ) -> ReviewItem:
        """
        Add a prediction to the human review queue.

        Args:
            prediction: Classified prediction to review
            frame_number: Frame where detection occurred
            evidence_path: Path to saved frame/crop
            reason: Why this needs review

        Returns:
            ReviewItem added to queue
        """
        self._review_id_counter += 1

        alternatives: List[str] = []
        if prediction.validation_result:
            alternatives = prediction.validation_result.alternatives

        item = ReviewItem(
            id=f"review_{self._review_id_counter}",
            bib_number=prediction.bib_number,
            confidence=prediction.adjusted_confidence,
            timestamp=datetime.now(),
            frame_number=frame_number,
            evidence_path=evidence_path,
            alternatives=alternatives,
            reason=reason or self._get_review_reason(prediction),
        )

        self.review_queue.append(item)

        # Trim queue if too large
        if len(self.review_queue) > self.review_queue_size:
            self.review_queue = self.review_queue[-self.review_queue_size :]

        return item

    def _get_review_reason(self, prediction: ClassifiedPrediction) -> str:
        """Generate reason string for review item."""
        reasons = []

        if prediction.level == ConfidenceLevel.REJECT:
            reasons.append("Very low confidence")
        elif prediction.level == ConfidenceLevel.LOW:
            reasons.append("Low confidence")

        if prediction.validation_result:
            if not prediction.validation_result.is_valid:
                reasons.append("Not in bib set")
            if prediction.validation_result.is_corrected:
                orig = prediction.validation_result.original
                corr = prediction.validation_result.validated
                reasons.append(f"Auto-corrected {orig}->{corr}")

        if prediction.voting_result:
            if not prediction.voting_result.is_stable:
                reasons.append("Unstable across frames")

        return "; ".join(reasons) if reasons else "Needs verification"

    def get_pending_reviews(self) -> List[ReviewItem]:
        """Get all items pending review."""
        return list(self.review_queue)

    def resolve_review(
        self,
        review_id: str,
        correct_number: Optional[str] = None,
    ) -> bool:
        """
        Resolve a review item.

        Args:
            review_id: ID of review item
            correct_number: Corrected number (None to accept original)

        Returns:
            True if item was found and resolved
        """
        for i, item in enumerate(self.review_queue):
            if item.id == review_id:
                self.review_queue.pop(i)
                return True
        return False

    def export_review_queue(self, filepath: str) -> None:
        """Export review queue to JSON file."""
        data = []
        for item in self.review_queue:
            data.append(
                {
                    "id": item.id,
                    "bib_number": item.bib_number,
                    "confidence": item.confidence,
                    "timestamp": item.timestamp.isoformat(),
                    "frame_number": item.frame_number,
                    "evidence_path": item.evidence_path,
                    "alternatives": item.alternatives,
                    "reason": item.reason,
                }
            )

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def clear_queue(self) -> None:
        """Clear the review queue."""
        self.review_queue = []
        self._review_id_counter = 0


# ---------------------------------------------------------------------------
# Tier 2: Crop Quality Filter (saves compute by rejecting bad crops)
# ---------------------------------------------------------------------------


@dataclass
class CropQuality:
    """Quality assessment of a bib crop."""

    is_acceptable: bool
    blur_score: float  # Higher = sharper (Laplacian variance)
    aspect_ratio: float
    pixel_count: int
    completeness_score: float = 1.0  # 0-1, lower means likely partial/obstructed
    rejection_reason: Optional[str] = None


class CropQualityFilter:
    """
    Filter out low-quality crops before OCR to save compute.

    Checks:
    - Blur detection (Laplacian variance)
    - Minimum size
    - Aspect ratio (bibs are roughly 2:1 to 4:1 width:height)
    - Minimum contrast
    - Completeness (detect partial/obstructed bibs)

    Requires: cv2, numpy
    """

    def __init__(
        self,
        min_blur_score: float = 50.0,
        min_width: int = 40,
        min_height: int = 15,
        min_aspect_ratio: float = 0.7,  # Crops can be nearly square; reject very tall/narrow
        max_aspect_ratio: float = 6.0,
        min_contrast: float = 20.0,
        min_completeness: float = 0.45,  # Combined score from multiple factors
        min_content_extent: float = 0.5,  # Content should span at least 50% of width
        check_completeness: bool = True,
    ):
        if not CV2_AVAILABLE:
            raise ImportError("CropQualityFilter requires cv2 and numpy")
        """
        Args:
            min_blur_score: Minimum Laplacian variance (higher = sharper)
            min_width: Minimum crop width in pixels
            min_height: Minimum crop height in pixels
            min_aspect_ratio: Minimum width/height ratio
            max_aspect_ratio: Maximum width/height ratio
            min_contrast: Minimum standard deviation of pixel values
            min_completeness: Minimum completeness score (0-1)
            check_completeness: Enable completeness checking
        """
        self.min_blur_score = min_blur_score
        self.min_width = min_width
        self.min_height = min_height
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_contrast = min_contrast
        self.min_completeness = min_completeness
        self.min_content_extent = min_content_extent
        self.check_completeness = check_completeness

    def assess(self, crop: Any) -> CropQuality:
        """
        Assess quality of a crop image.

        Args:
            crop: BGR image of cropped bib

        Returns:
            CropQuality with assessment details
        """
        if crop is None or crop.size == 0:
            return CropQuality(
                is_acceptable=False,
                blur_score=0.0,
                aspect_ratio=0.0,
                pixel_count=0,
                rejection_reason="Empty crop",
            )

        h, w = crop.shape[:2]
        pixel_count = h * w
        aspect_ratio = w / max(h, 1)

        # Size check
        if w < self.min_width:
            return CropQuality(
                is_acceptable=False,
                blur_score=0.0,
                aspect_ratio=aspect_ratio,
                pixel_count=pixel_count,
                rejection_reason=f"Too narrow: {w}px < {self.min_width}px",
            )

        if h < self.min_height:
            return CropQuality(
                is_acceptable=False,
                blur_score=0.0,
                aspect_ratio=aspect_ratio,
                pixel_count=pixel_count,
                rejection_reason=f"Too short: {h}px < {self.min_height}px",
            )

        # Aspect ratio check
        if aspect_ratio < self.min_aspect_ratio:
            return CropQuality(
                is_acceptable=False,
                blur_score=0.0,
                aspect_ratio=aspect_ratio,
                pixel_count=pixel_count,
                rejection_reason=f"Aspect ratio too low: {aspect_ratio:.2f}",
            )

        if aspect_ratio > self.max_aspect_ratio:
            return CropQuality(
                is_acceptable=False,
                blur_score=0.0,
                aspect_ratio=aspect_ratio,
                pixel_count=pixel_count,
                rejection_reason=f"Aspect ratio too high: {aspect_ratio:.2f}",
            )

        # Convert to grayscale for blur/contrast checks
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop

        # Blur detection using Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_score = float(laplacian.var())

        if blur_score < self.min_blur_score:
            return CropQuality(
                is_acceptable=False,
                blur_score=blur_score,
                aspect_ratio=aspect_ratio,
                pixel_count=pixel_count,
                rejection_reason=f"Too blurry: {blur_score:.1f} < {self.min_blur_score}",
            )

        # Contrast check
        contrast = float(gray.std())
        if contrast < self.min_contrast:
            return CropQuality(
                is_acceptable=False,
                blur_score=blur_score,
                aspect_ratio=aspect_ratio,
                pixel_count=pixel_count,
                rejection_reason=f"Low contrast: {contrast:.1f} < {self.min_contrast}",
            )

        # Completeness check (detect partial/obstructed bibs)
        completeness_score = 1.0
        content_extent = 1.0
        if self.check_completeness:
            completeness_score, content_extent = self._assess_completeness(gray)

            # Check content extent first (most reliable indicator)
            if content_extent < self.min_content_extent:
                return CropQuality(
                    is_acceptable=False,
                    blur_score=blur_score,
                    aspect_ratio=aspect_ratio,
                    pixel_count=pixel_count,
                    completeness_score=completeness_score,
                    rejection_reason=f"Partial bib - content spans only {content_extent:.0%} of width (need {self.min_content_extent:.0%})",
                )

            if completeness_score < self.min_completeness:
                return CropQuality(
                    is_acceptable=False,
                    blur_score=blur_score,
                    aspect_ratio=aspect_ratio,
                    pixel_count=pixel_count,
                    completeness_score=completeness_score,
                    rejection_reason=f"Likely partial/obstructed: score {completeness_score:.2f} < {self.min_completeness}",
                )

        return CropQuality(
            is_acceptable=True,
            blur_score=blur_score,
            aspect_ratio=aspect_ratio,
            pixel_count=pixel_count,
            completeness_score=completeness_score,
            rejection_reason=None,
        )

    def filter(self, crop: Any) -> bool:
        """
        Quick check if crop is acceptable for OCR.

        Args:
            crop: BGR image of cropped bib

        Returns:
            True if crop should be processed, False to skip
        """
        return self.assess(crop).is_acceptable

    def _assess_completeness(self, gray: Any) -> Tuple[float, float]:
        """
        Assess whether a bib crop appears complete vs partial/obstructed.

        This method uses multiple heuristics to detect partial bibs:
        1. Content extent - digits should span most of the crop width
        2. Content density - check for sufficient high-contrast regions
        3. Symmetry - partial crops often have lopsided content
        4. Edge analysis - content touching edges suggests truncation

        Args:
            gray: Grayscale image of crop

        Returns:
            (completeness_score, content_extent) tuple
            - completeness_score: 0.0 (definitely partial) to 1.0 (complete)
            - content_extent: fraction of width containing digit content
        """
        h, w = gray.shape[:2]

        # Binarize to find digit regions (adaptive threshold handles varying backgrounds)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10
        )

        # Find vertical projection (sum of white pixels in each column)
        col_sums = np.sum(binary, axis=0) / 255.0  # Normalize by height

        # Threshold to determine which columns have "content"
        content_threshold = h * 0.15  # At least 15% of column height should have content
        content_cols = col_sums > content_threshold

        # Find the extent of content (first and last column with content)
        content_indices = np.where(content_cols)[0]

        if len(content_indices) < 3:
            # Very little content found - likely garbage or very partial
            return 0.2, 0.0

        first_content = content_indices[0]
        last_content = content_indices[-1]
        content_span = last_content - first_content + 1
        content_extent = content_span / w

        # Check margins - complete bibs should have some margin on each side
        left_margin = first_content / w
        right_margin = (w - last_content - 1) / w

        # Calculate completeness score based on multiple factors
        scores = []

        # 1. Content extent score - content should span most of the width
        # For a 4-digit bib, digits should span ~70-90% of a proper crop
        extent_score = min(1.0, content_extent / self.min_content_extent)
        scores.append(extent_score)

        # 2. Margin balance - both sides should have similar margins
        # Lopsided margins suggest truncation on one side
        if left_margin > 0.02 and right_margin > 0.02:
            margin_ratio = min(left_margin, right_margin) / max(left_margin, right_margin)
            margin_score = margin_ratio  # 1.0 = perfectly balanced
        elif left_margin < 0.02 or right_margin < 0.02:
            # Content touching edge - penalize heavily
            margin_score = 0.3
        else:
            margin_score = 0.5
        scores.append(margin_score)

        # 3. Content density - how much of the content region is actually filled
        if content_span > 0:
            content_region_cols = content_cols[first_content:last_content + 1]
            density = np.mean(content_region_cols)
            # Digits have gaps between them, so expect 40-70% density
            if density < 0.3:
                density_score = density / 0.3  # Sparse content = likely partial
            elif density > 0.85:
                density_score = 0.8  # Too dense = might be obstruction
            else:
                density_score = 1.0
            scores.append(density_score)

        # 4. Content distribution - divide into sections and check each has content
        n_sections = 4
        section_width = w // n_sections
        sections_with_content = 0
        for i in range(n_sections):
            section_start = i * section_width
            section_end = min((i + 1) * section_width, w)
            section_content = np.any(content_cols[section_start:section_end])
            if section_content:
                sections_with_content += 1

        distribution_score = sections_with_content / n_sections
        scores.append(distribution_score)

        # 5. Vertical content check - digits should fill vertical space
        row_sums = np.sum(binary, axis=1) / 255.0
        row_content = row_sums > (w * 0.1)  # At least 10% of width
        row_indices = np.where(row_content)[0]
        if len(row_indices) > 0:
            vertical_extent = (row_indices[-1] - row_indices[0] + 1) / h
            vertical_score = min(1.0, vertical_extent / 0.5)  # Expect at least 50% height
            scores.append(vertical_score)

        # Combine scores with weights
        # Content extent is most important for catching partial bibs
        weights = [0.35, 0.2, 0.15, 0.2, 0.1]
        if len(scores) < len(weights):
            weights = weights[:len(scores)]

        weighted_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)

        return float(weighted_score), float(content_extent)


class BibCompletenessChecker:
    """
    Additional check for bib completeness based on detection box position.

    Use this with frame-level context to reject bibs that:
    - Are at frame edges (entering/exiting)
    - Have detection boxes touching boundaries
    """

    def __init__(
        self,
        edge_margin_ratio: float = 0.02,
        min_visible_ratio: float = 0.9,
    ):
        """
        Args:
            edge_margin_ratio: Margin from frame edge as ratio of frame size
            min_visible_ratio: Minimum portion of expected bib that should be visible
        """
        self.edge_margin_ratio = edge_margin_ratio
        self.min_visible_ratio = min_visible_ratio

    def is_fully_visible(
        self,
        bbox: Tuple[int, int, int, int],
        frame_width: int,
        frame_height: int,
    ) -> Tuple[bool, str]:
        """
        Check if detection box is fully within frame (not at edges).

        Args:
            bbox: (x1, y1, x2, y2) bounding box
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels

        Returns:
            (is_visible, reason) tuple
        """
        x1, y1, x2, y2 = bbox

        margin_x = int(frame_width * self.edge_margin_ratio)
        margin_y = int(frame_height * self.edge_margin_ratio)

        # Check if bbox touches frame edges
        at_left = x1 <= margin_x
        at_right = x2 >= frame_width - margin_x
        at_top = y1 <= margin_y
        at_bottom = y2 >= frame_height - margin_y

        if at_left:
            return False, "Bib at left edge (possibly entering frame)"
        if at_right:
            return False, "Bib at right edge (possibly exiting frame)"
        if at_top:
            return False, "Bib at top edge"
        if at_bottom:
            return False, "Bib at bottom edge"

        return True, "Fully visible"

    def check_with_tracking(
        self,
        bbox: Tuple[int, int, int, int],
        frame_width: int,
        frame_height: int,
        previous_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> Tuple[bool, float, str]:
        """
        Check visibility with tracking context.

        If we have previous bbox, we can estimate if the bib is
        fully in frame vs partially visible.

        Args:
            bbox: Current bounding box
            frame_width: Frame width
            frame_height: Frame height
            previous_bbox: Previous frame's bbox (if tracked)

        Returns:
            (is_complete, confidence, reason)
        """
        is_visible, reason = self.is_fully_visible(bbox, frame_width, frame_height)

        if not is_visible:
            return False, 0.0, reason

        # If we have previous bbox, check for sudden size changes
        # (might indicate partial visibility)
        if previous_bbox is not None:
            prev_w = previous_bbox[2] - previous_bbox[0]
            prev_h = previous_bbox[3] - previous_bbox[1]
            curr_w = bbox[2] - bbox[0]
            curr_h = bbox[3] - bbox[1]

            # Significant size decrease might mean partial occlusion
            width_ratio = curr_w / max(prev_w, 1)
            height_ratio = curr_h / max(prev_h, 1)

            if width_ratio < 0.7 or height_ratio < 0.7:
                return False, 0.5, f"Sudden size decrease (w:{width_ratio:.2f}, h:{height_ratio:.2f})"

        return True, 1.0, "Complete"


# ---------------------------------------------------------------------------
# Tier 2: Post-OCR Cleanup (fixes common OCR errors with minimal compute)
# ---------------------------------------------------------------------------


class PostOCRCleanup:
    """
    Clean up common OCR errors with minimal compute overhead.

    Fixes:
    - Letter-to-digit substitutions (O->0, I->1, S->5, etc.)
    - Leading zeros removal
    - Invalid digit count filtering
    - Duplicate digit collapse
    """

    # Common OCR letter-to-digit confusions
    LETTER_TO_DIGIT = {
        "O": "0",
        "o": "0",
        "Q": "0",
        "D": "0",
        "I": "1",
        "i": "1",
        "l": "1",
        "L": "1",
        "|": "1",
        "Z": "2",
        "z": "2",
        "E": "3",
        "S": "5",
        "s": "5",
        "G": "6",
        "b": "6",
        "T": "7",
        "B": "8",
        "g": "9",
        "q": "9",
    }

    def __init__(
        self,
        min_digits: int = 1,
        max_digits: int = 5,
        strip_leading_zeros: bool = True,
        fix_letter_confusion: bool = True,
        max_bib_value: Optional[int] = None,
    ):
        """
        Args:
            min_digits: Minimum valid digit count
            max_digits: Maximum valid digit count
            strip_leading_zeros: Remove leading zeros (0123 -> 123)
            fix_letter_confusion: Fix common letter-to-digit errors
            max_bib_value: Reject numbers above this value (e.g. 1700 for a race with bibs 1-1678)
        """
        self.min_digits = min_digits
        self.max_digits = max_digits
        self.strip_leading_zeros = strip_leading_zeros
        self.fix_letter_confusion = fix_letter_confusion
        self.max_bib_value = max_bib_value

    def clean(self, text: str) -> Tuple[str, bool]:
        """
        Clean OCR output.

        Args:
            text: Raw OCR text

        Returns:
            (cleaned_text, was_modified) tuple
        """
        if not text:
            return "", False

        original = text
        result = text

        # Fix letter-to-digit confusion
        if self.fix_letter_confusion:
            chars = list(result)
            for i, c in enumerate(chars):
                if c in self.LETTER_TO_DIGIT:
                    chars[i] = self.LETTER_TO_DIGIT[c]
            result = "".join(chars)

        # Extract only digits
        digits = "".join(c for c in result if c.isdigit())

        # Strip leading zeros (but keep at least one digit)
        if self.strip_leading_zeros and len(digits) > 1:
            digits = digits.lstrip("0") or "0"

        # Validate digit count
        if len(digits) < self.min_digits or len(digits) > self.max_digits:
            # Return truncated or empty based on situation
            if len(digits) > self.max_digits:
                digits = digits[: self.max_digits]
            elif len(digits) < self.min_digits:
                # Too few digits - might be garbage
                pass

        # Range check: reject numbers above known max bib value
        if self.max_bib_value and digits.isdigit() and int(digits) > self.max_bib_value:
            return "", True

        was_modified = digits != original
        return digits, was_modified

    def clean_batch(self, texts: List[str]) -> List[Tuple[str, bool]]:
        """Clean multiple OCR outputs."""
        return [self.clean(t) for t in texts]


# ---------------------------------------------------------------------------
# Tier 2: Cascaded OCR (fast model first, slow model if uncertain)
# ---------------------------------------------------------------------------


@dataclass
class CascadedOCRResult:
    """Result from cascaded OCR."""

    text: str
    confidence: float
    model_used: str  # "fast", "slow", or "both"
    fast_result: Optional[Tuple[str, float]] = None
    slow_result: Optional[Tuple[str, float]] = None


class CascadedOCR:
    """
    Cascaded OCR: run fast model first, only use slow model if uncertain.

    This gives near-slow-model accuracy at near-fast-model speed.

    Strategy:
    1. Run fast model (e.g., CRNN ONNX at 17ms)
    2. If confidence >= threshold, return fast result
    3. If confidence < threshold, run slow model (e.g., PARSeq at 124ms)
    4. Return slow model result

    Expected behavior:
    - ~70-80% of crops handled by fast model only
    - ~20-30% require slow model
    - Average latency: ~40-50ms instead of 124ms
    - Accuracy: within 1-2% of slow model alone
    """

    def __init__(
        self,
        fast_model: Callable[[Any], Tuple[str, float]],
        slow_model: Callable[[Any], Tuple[str, float]],
        cascade_threshold: float = 0.85,
        always_use_slow_for_short: bool = True,
        short_digit_count: int = 2,
    ):
        """
        Args:
            fast_model: Fast OCR function (crop -> (text, confidence))
            slow_model: Slow but accurate OCR function
            cascade_threshold: Confidence threshold to skip slow model
            always_use_slow_for_short: Always use slow model for short numbers
            short_digit_count: What counts as "short" (1-2 digits harder)
        """
        self.fast_model = fast_model
        self.slow_model = slow_model
        self.cascade_threshold = cascade_threshold
        self.always_use_slow_for_short = always_use_slow_for_short
        self.short_digit_count = short_digit_count

        # Stats tracking
        self.stats = {
            "total": 0,
            "fast_only": 0,
            "used_slow": 0,
        }

    def predict(self, crop: Any) -> CascadedOCRResult:
        """
        Run cascaded OCR on crop.

        Args:
            crop: BGR image of cropped bib

        Returns:
            CascadedOCRResult with prediction details
        """
        self.stats["total"] += 1

        # Run fast model
        fast_text, fast_conf = self.fast_model(crop)

        # Check if we need slow model
        need_slow = False

        if fast_conf < self.cascade_threshold:
            need_slow = True

        # Short numbers are harder - always verify with slow model
        if (
            self.always_use_slow_for_short
            and len(fast_text) <= self.short_digit_count
            and fast_text
        ):
            need_slow = True

        if not need_slow:
            self.stats["fast_only"] += 1
            return CascadedOCRResult(
                text=fast_text,
                confidence=fast_conf,
                model_used="fast",
                fast_result=(fast_text, fast_conf),
                slow_result=None,
            )

        # Run slow model
        self.stats["used_slow"] += 1
        slow_text, slow_conf = self.slow_model(crop)

        # Use slow model result (it's more accurate)
        return CascadedOCRResult(
            text=slow_text,
            confidence=slow_conf,
            model_used="slow",
            fast_result=(fast_text, fast_conf),
            slow_result=(slow_text, slow_conf),
        )

    def get_stats(self) -> Dict[str, float]:
        """Get cascade statistics."""
        total = max(self.stats["total"], 1)
        return {
            "total": self.stats["total"],
            "fast_only": self.stats["fast_only"],
            "fast_only_pct": 100 * self.stats["fast_only"] / total,
            "used_slow": self.stats["used_slow"],
            "used_slow_pct": 100 * self.stats["used_slow"] / total,
        }

    def reset_stats(self) -> None:
        """Reset statistics."""
        self.stats = {"total": 0, "fast_only": 0, "used_slow": 0}


# ---------------------------------------------------------------------------
# Tier 2: Digit Count Validator (catches truncation/insertion errors)
# ---------------------------------------------------------------------------


class DigitCountValidator:
    """
    Validate predictions based on expected digit counts.

    Many races have consistent bib number formats (e.g., all 4-digit).
    This catches truncation errors (1234 -> 123) and insertion errors (123 -> 1234).
    """

    def __init__(
        self,
        expected_counts: Optional[Set[int]] = None,
        penalize_unexpected: float = 0.15,
    ):
        """
        Args:
            expected_counts: Set of valid digit counts (e.g., {3, 4} for 3-4 digit bibs)
            penalize_unexpected: Confidence penalty for unexpected digit count
        """
        self.expected_counts = expected_counts or {1, 2, 3, 4}
        self.penalize_unexpected = penalize_unexpected

    def validate(self, text: str, confidence: float) -> Tuple[str, float, bool]:
        """
        Validate digit count and adjust confidence.

        Args:
            text: OCR prediction
            confidence: OCR confidence

        Returns:
            (text, adjusted_confidence, is_valid_count)
        """
        if not text:
            return text, confidence, False

        digit_count = len(text)

        if digit_count in self.expected_counts:
            return text, confidence, True
        else:
            # Penalize unexpected digit counts
            adjusted = max(0.0, confidence - self.penalize_unexpected)
            return text, adjusted, False

    @classmethod
    def from_bib_set(cls, bib_set: Set[str], **kwargs) -> "DigitCountValidator":
        """Create validator from bib set by inferring expected digit counts."""
        counts = {len(b) for b in bib_set if b}
        return cls(expected_counts=counts, **kwargs)


# ---------------------------------------------------------------------------
# Tier 2: Suspicious Prediction Filter (catches partial bib OCR errors)
# ---------------------------------------------------------------------------


@dataclass
class SuspiciousPredictionResult:
    """Result of suspicious prediction check."""

    is_suspicious: bool
    reason: Optional[str] = None
    adjusted_confidence: float = 0.0
    should_reject: bool = False


class SuspiciousPredictionFilter:
    """
    Post-OCR filter to catch suspicious predictions that indicate partial bibs.

    This filter catches cases where:
    1. Very short predictions (1-2 digits) with medium confidence
       - Likely a partial bib where only some digits are visible
    2. Low confidence predictions regardless of length
       - OCR is uncertain, possibly due to obstruction
    3. Unexpected digit counts combined with low confidence
       - e.g., expecting 4-digit bibs but got 1 digit with 0.28 confidence

    Use this AFTER OCR to catch partial bib errors that image analysis missed.
    """

    def __init__(
        self,
        min_confidence_short: float = 0.85,  # Require higher conf for 1-2 digit predictions
        min_confidence_medium: float = 0.7,  # Normal threshold for 3+ digits
        reject_confidence: float = 0.35,  # Below this, reject regardless
        expected_digit_counts: Optional[Set[int]] = None,  # e.g., {3, 4} for typical races
        strict_mode: bool = False,  # If True, reject suspicious; if False, just flag
    ):
        """
        Args:
            min_confidence_short: Min confidence for 1-2 digit predictions
            min_confidence_medium: Min confidence for 3+ digit predictions
            reject_confidence: Reject predictions below this confidence
            expected_digit_counts: Expected digit counts for this race
            strict_mode: If True, mark suspicious as should_reject
        """
        self.min_confidence_short = min_confidence_short
        self.min_confidence_medium = min_confidence_medium
        self.reject_confidence = reject_confidence
        self.expected_digit_counts = expected_digit_counts or {1, 2, 3, 4, 5}
        self.strict_mode = strict_mode

    def check(self, prediction: str, confidence: float) -> SuspiciousPredictionResult:
        """
        Check if a prediction is suspicious.

        Args:
            prediction: OCR predicted text (digits)
            confidence: OCR confidence score

        Returns:
            SuspiciousPredictionResult with assessment
        """
        if not prediction:
            return SuspiciousPredictionResult(
                is_suspicious=True,
                reason="Empty prediction",
                adjusted_confidence=0.0,
                should_reject=True,
            )

        digit_count = len(prediction)

        # Rule 1: Very low confidence - reject
        if confidence < self.reject_confidence:
            return SuspiciousPredictionResult(
                is_suspicious=True,
                reason=f"Very low confidence ({confidence:.2f} < {self.reject_confidence})",
                adjusted_confidence=confidence,
                should_reject=True,
            )

        # Rule 2: Short predictions need higher confidence
        if digit_count <= 2 and confidence < self.min_confidence_short:
            return SuspiciousPredictionResult(
                is_suspicious=True,
                reason=f"Short prediction ({digit_count} digits) with low confidence ({confidence:.2f})",
                adjusted_confidence=confidence * 0.7,  # Penalize
                should_reject=self.strict_mode,
            )

        # Rule 3: Unexpected digit count - always suspicious, reject if confidence not very high
        if digit_count not in self.expected_digit_counts:
            if confidence < 0.95:  # Need very high confidence for unexpected digit count
                return SuspiciousPredictionResult(
                    is_suspicious=True,
                    reason=f"Unexpected digit count ({digit_count}) - need >0.95 confidence, got {confidence:.2f}",
                    adjusted_confidence=confidence * 0.8,
                    should_reject=self.strict_mode or confidence < self.min_confidence_medium,
                )

        # Rule 4: Medium-length predictions with low confidence
        if confidence < self.min_confidence_medium:
            return SuspiciousPredictionResult(
                is_suspicious=True,
                reason=f"Low confidence ({confidence:.2f})",
                adjusted_confidence=confidence,
                should_reject=False,  # Flag but don't reject
            )

        # Passed all checks
        return SuspiciousPredictionResult(
            is_suspicious=False,
            reason=None,
            adjusted_confidence=confidence,
            should_reject=False,
        )

    def filter_batch(
        self, predictions: List[Tuple[str, float]]
    ) -> List[SuspiciousPredictionResult]:
        """Check multiple predictions."""
        return [self.check(pred, conf) for pred, conf in predictions]


# ---------------------------------------------------------------------------
# Combined Pipeline Helper
# ---------------------------------------------------------------------------


@dataclass
class OCRPipelineResult:
    """Complete result from OCR pipeline with all enhancements."""

    # Final output
    bib_number: str
    confidence: float
    level: ConfidenceLevel

    # Quality check
    quality: Optional[CropQuality] = None
    quality_passed: bool = True

    # OCR details
    ocr_raw: str = ""
    ocr_model_used: str = ""

    # Cleanup
    cleanup_applied: bool = False

    # Validation
    validation_result: Optional[ValidationResult] = None

    # Voting
    voting_result: Optional[ConsensusResult] = None

    # Flags
    needs_review: bool = False
    rejection_reason: Optional[str] = None


class EnhancedOCRPipeline:
    """
    Complete OCR pipeline with all Tier 1 and Tier 2 enhancements.

    Pipeline stages:
    1. Quality filter (skip bad crops)
    2. OCR (cascaded or single model)
    3. Post-cleanup (fix common errors)
    4. Digit count validation
    5. Bib set validation
    6. Confidence classification
    """

    def __init__(
        self,
        ocr_func: Callable[[Any], Tuple[str, float]],
        bib_validator: Optional[BibSetValidator] = None,
        quality_filter: Optional[CropQualityFilter] = None,
        post_cleanup: Optional[PostOCRCleanup] = None,
        digit_validator: Optional[DigitCountValidator] = None,
        confidence_mgr: Optional[ConfidenceManager] = None,
    ):
        """
        Args:
            ocr_func: OCR function (crop -> (text, confidence))
            bib_validator: Bib set validator (optional)
            quality_filter: Crop quality filter (optional, recommended)
            post_cleanup: Post-OCR cleanup (optional, recommended)
            digit_validator: Digit count validator (optional)
            confidence_mgr: Confidence manager (optional)
        """
        self.ocr_func = ocr_func
        self.bib_validator = bib_validator
        self.quality_filter = quality_filter or CropQualityFilter()
        self.post_cleanup = post_cleanup or PostOCRCleanup()
        self.digit_validator = digit_validator
        self.confidence_mgr = confidence_mgr or ConfidenceManager()

        # Stats
        self.stats = {
            "total": 0,
            "quality_rejected": 0,
            "cleanup_modified": 0,
            "validated": 0,
            "corrected": 0,
        }

    def process(self, crop: Any) -> OCRPipelineResult:
        """
        Process a single crop through the full pipeline.

        Args:
            crop: BGR image of cropped bib

        Returns:
            OCRPipelineResult with all details
        """
        self.stats["total"] += 1

        # Stage 1: Quality check
        quality = self.quality_filter.assess(crop)
        if not quality.is_acceptable:
            self.stats["quality_rejected"] += 1
            return OCRPipelineResult(
                bib_number="",
                confidence=0.0,
                level=ConfidenceLevel.REJECT,
                quality=quality,
                quality_passed=False,
                rejection_reason=quality.rejection_reason,
            )

        # Stage 2: OCR
        ocr_raw, ocr_conf = self.ocr_func(crop)
        ocr_model_used = "primary"

        if not ocr_raw:
            return OCRPipelineResult(
                bib_number="",
                confidence=0.0,
                level=ConfidenceLevel.REJECT,
                quality=quality,
                quality_passed=True,
                ocr_raw="",
                ocr_model_used=ocr_model_used,
                rejection_reason="OCR returned empty",
            )

        # Stage 3: Post-cleanup
        cleaned, was_modified = self.post_cleanup.clean(ocr_raw)
        if was_modified:
            self.stats["cleanup_modified"] += 1

        # Stage 4: Digit count validation
        if self.digit_validator:
            cleaned, ocr_conf, _ = self.digit_validator.validate(cleaned, ocr_conf)

        # Stage 5: Bib set validation
        validation_result = None
        if self.bib_validator and cleaned:
            validation_result = self.bib_validator.validate(cleaned, ocr_conf)
            if validation_result.is_valid:
                self.stats["validated"] += 1
            if validation_result.is_corrected:
                self.stats["corrected"] += 1
            cleaned = validation_result.validated

        # Stage 6: Confidence classification
        classified = self.confidence_mgr.classify(
            bib_number=cleaned,
            ocr_confidence=ocr_conf,
            validation_result=validation_result,
        )

        return OCRPipelineResult(
            bib_number=cleaned,
            confidence=classified.adjusted_confidence,
            level=classified.level,
            quality=quality,
            quality_passed=True,
            ocr_raw=ocr_raw,
            ocr_model_used=ocr_model_used,
            cleanup_applied=was_modified,
            validation_result=validation_result,
            needs_review=classified.needs_review,
        )

    def get_stats(self) -> Dict[str, float]:
        """Get pipeline statistics."""
        total = max(self.stats["total"], 1)
        return {
            "total": self.stats["total"],
            "quality_rejected": self.stats["quality_rejected"],
            "quality_rejected_pct": 100 * self.stats["quality_rejected"] / total,
            "cleanup_modified": self.stats["cleanup_modified"],
            "cleanup_modified_pct": 100 * self.stats["cleanup_modified"] / total,
            "validated": self.stats["validated"],
            "validated_pct": 100 * self.stats["validated"] / total,
            "corrected": self.stats["corrected"],
            "corrected_pct": 100 * self.stats["corrected"] / total,
        }

    def reset_stats(self) -> None:
        """Reset statistics."""
        self.stats = {
            "total": 0,
            "quality_rejected": 0,
            "cleanup_modified": 0,
            "validated": 0,
            "corrected": 0,
        }
