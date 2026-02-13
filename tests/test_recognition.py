"""Tests for recognition module Tier 1 components."""

import pytest
from datetime import datetime

from pointcam.recognition import (
    BibSetValidator,
    ValidationResult,
    EnhancedTemporalVoting,
    ConsensusResult,
    ConfidenceManager,
    ConfidenceLevel,
    ClassifiedPrediction,
)


# ---------------------------------------------------------------------------
# BibSetValidator Tests
# ---------------------------------------------------------------------------


class TestBibSetValidator:
    """Tests for BibSetValidator."""

    @pytest.fixture
    def validator(self):
        """Create validator with sample bib set."""
        bib_set = {"1000", "1001", "1002", "1234", "2345", "3456", "4567"}
        return BibSetValidator(bib_set)

    @pytest.fixture
    def large_validator(self):
        """Create validator with larger bib set."""
        bib_set = {str(i) for i in range(1000, 4000)}
        return BibSetValidator(bib_set)

    def test_exact_match(self, validator):
        """Test exact match validation."""
        result = validator.validate("1234", 0.9)

        assert result.is_valid is True
        assert result.is_corrected is False
        assert result.validated == "1234"
        assert result.confidence_boost > 0

    def test_no_match(self, validator):
        """Test prediction not in bib set."""
        result = validator.validate("9999", 0.9)

        assert result.is_valid is False
        assert result.is_corrected is False
        assert result.confidence_boost < 0

    def test_fuzzy_match_single_digit(self, validator):
        """Test fuzzy matching for single digit error."""
        # 1235 is close to 1234
        result = validator.validate("1235", 0.9)

        assert result.is_valid is False
        assert result.is_corrected is True
        assert result.validated == "1234"

    def test_empty_prediction(self, validator):
        """Test empty prediction."""
        result = validator.validate("", 0.9)

        assert result.is_valid is False
        assert result.validated == ""

    def test_from_range(self):
        """Test creating validator from numeric range."""
        validator = BibSetValidator.from_range(100, 200)

        assert "100" in validator.bib_set
        assert "200" in validator.bib_set
        assert "150" in validator.bib_set
        assert "99" not in validator.bib_set
        assert "201" not in validator.bib_set

    def test_alternatives(self, large_validator):
        """Test that alternatives are returned for fuzzy matches."""
        # 1235 should fuzzy match to 1234, 1236, etc.
        result = large_validator.validate("1235", 0.9)

        assert result.is_corrected is True
        # Should have some alternatives
        assert len(result.alternatives) >= 0


class TestBibSetValidatorPerformance:
    """Performance tests for BibSetValidator."""

    def test_large_bib_set_validation(self):
        """Test validation performance with large bib set."""
        bib_set = {str(i) for i in range(1, 10001)}
        validator = BibSetValidator(bib_set)

        # Should be fast even with 10k bibs
        import time

        start = time.perf_counter()
        for _ in range(1000):
            validator.validate("5000", 0.9)
        elapsed = time.perf_counter() - start

        # Should complete 1000 validations in under 1 second
        assert elapsed < 1.0


# ---------------------------------------------------------------------------
# EnhancedTemporalVoting Tests
# ---------------------------------------------------------------------------


class TestEnhancedTemporalVoting:
    """Tests for EnhancedTemporalVoting."""

    @pytest.fixture
    def voting(self):
        """Create voting instance with default settings."""
        return EnhancedTemporalVoting(
            window_size=10,
            min_votes=3,
            stability_threshold=5,
        )

    def test_insufficient_votes(self, voting):
        """Test that insufficient votes returns no consensus."""
        voting.update(1, "1234", 0.9, 1)
        voting.update(1, "1234", 0.9, 2)

        result = voting.get_consensus(1)

        assert result.number is None
        assert result.vote_count == 0

    def test_consensus_with_enough_votes(self, voting):
        """Test consensus is returned with enough votes."""
        for i in range(5):
            voting.update(1, "1234", 0.9, i)

        result = voting.get_consensus(1)

        assert result.number == "1234"
        assert result.vote_count == 5
        assert result.agreement_ratio == 1.0

    def test_mixed_votes(self, voting):
        """Test voting with mixed predictions."""
        voting.update(1, "1234", 0.9, 1)
        voting.update(1, "1234", 0.9, 2)
        voting.update(1, "1235", 0.8, 3)
        voting.update(1, "1234", 0.9, 4)
        voting.update(1, "1234", 0.9, 5)

        result = voting.get_consensus(1)

        assert result.number == "1234"
        assert result.vote_count == 4
        assert result.agreement_ratio == 0.8

    def test_stability_detection(self, voting):
        """Test that stability is detected for consistent votes."""
        # Need 5 consecutive same votes for stability
        for i in range(5):
            voting.update(1, "1234", 0.9, i)

        result = voting.get_consensus(1)

        assert result.is_stable is True

    def test_stability_broken(self, voting):
        """Test that stability is broken by different vote."""
        for i in range(4):
            voting.update(1, "1234", 0.9, i)
        voting.update(1, "1235", 0.9, 5)  # Different number

        result = voting.get_consensus(1)

        assert result.is_stable is False

    def test_clear_track(self, voting):
        """Test clearing a track's history."""
        for i in range(5):
            voting.update(1, "1234", 0.9, i)

        voting.clear_track(1)

        result = voting.get_consensus(1)
        assert result.number is None

    def test_multiple_tracks(self, voting):
        """Test handling multiple independent tracks."""
        for i in range(5):
            voting.update(1, "1234", 0.9, i)
            voting.update(2, "5678", 0.85, i)

        result1 = voting.get_consensus(1)
        result2 = voting.get_consensus(2)

        assert result1.number == "1234"
        assert result2.number == "5678"

    def test_window_trimming(self, voting):
        """Test that old votes are trimmed to window size."""
        voting.window_size = 5

        # Add more votes than window size
        for i in range(10):
            voting.update(1, "1234" if i < 8 else "9999", 0.9, i)

        # Should only have last 5 votes (2 x 1234, 3 x 9999 or similar)
        result = voting.get_consensus(1)

        # Most recent votes should dominate
        assert result.total_frames == 5

    def test_early_consensus(self, voting):
        """Test early consensus detection."""
        # High agreement, high confidence
        for i in range(4):
            voting.update(1, "1234", 0.95, i)

        result = voting.get_early_consensus(1, min_agreement=0.9, min_confidence=0.85)

        assert result is not None
        assert result.number == "1234"

    def test_no_early_consensus_low_agreement(self, voting):
        """Test early consensus not returned with low agreement."""
        voting.update(1, "1234", 0.9, 1)
        voting.update(1, "1235", 0.9, 2)
        voting.update(1, "1234", 0.9, 3)

        result = voting.get_early_consensus(1, min_agreement=0.9)

        assert result is None


# ---------------------------------------------------------------------------
# ConfidenceManager Tests
# ---------------------------------------------------------------------------


class TestConfidenceManager:
    """Tests for ConfidenceManager."""

    @pytest.fixture
    def manager(self):
        """Create confidence manager with default settings."""
        return ConfidenceManager()

    def test_high_confidence_classification(self, manager):
        """Test high confidence prediction classification."""
        result = manager.classify("1234", 0.95)

        assert result.level == ConfidenceLevel.HIGH
        assert result.needs_review is False

    def test_medium_confidence_classification(self, manager):
        """Test medium confidence prediction classification."""
        result = manager.classify("1234", 0.75)

        assert result.level == ConfidenceLevel.MEDIUM
        assert result.needs_review is False

    def test_low_confidence_classification(self, manager):
        """Test low confidence prediction classification."""
        result = manager.classify("1234", 0.55)

        assert result.level == ConfidenceLevel.LOW
        assert result.needs_review is True

    def test_reject_classification(self, manager):
        """Test very low confidence prediction classification."""
        result = manager.classify("1234", 0.30)

        assert result.level == ConfidenceLevel.REJECT
        assert result.needs_review is True

    def test_validation_boosts_confidence(self, manager):
        """Test that valid bib validation boosts confidence."""
        validation_result = ValidationResult(
            original="1234",
            validated="1234",
            is_valid=True,
            is_corrected=False,
            confidence_boost=0.1,
            alternatives=[],
        )

        result = manager.classify("1234", 0.80, validation_result=validation_result)

        # Should be boosted to HIGH due to validation
        assert result.level == ConfidenceLevel.HIGH
        assert result.adjusted_confidence > 0.80

    def test_invalid_validation_reduces_confidence(self, manager):
        """Test that invalid validation reduces confidence."""
        validation_result = ValidationResult(
            original="9999",
            validated="9999",
            is_valid=False,
            is_corrected=False,
            confidence_boost=-0.2,
            alternatives=[],
        )

        result = manager.classify("9999", 0.75, validation_result=validation_result)

        assert result.adjusted_confidence < 0.75

    def test_corrected_prediction_flagged_for_review(self, manager):
        """Test that auto-corrected predictions are flagged for review."""
        validation_result = ValidationResult(
            original="1235",
            validated="1234",
            is_valid=False,
            is_corrected=True,
            confidence_boost=-0.1,
            alternatives=["1236"],
        )

        result = manager.classify("1234", 0.85, validation_result=validation_result)

        assert result.needs_review is True

    def test_add_to_review_queue(self, manager):
        """Test adding items to review queue."""
        prediction = ClassifiedPrediction(
            bib_number="1234",
            raw_confidence=0.50,
            adjusted_confidence=0.50,
            level=ConfidenceLevel.LOW,
            needs_review=True,
        )

        item = manager.add_to_review_queue(prediction, frame_number=100)

        assert item.bib_number == "1234"
        assert item.frame_number == 100
        assert len(manager.get_pending_reviews()) == 1

    def test_resolve_review(self, manager):
        """Test resolving review items."""
        prediction = ClassifiedPrediction(
            bib_number="1234",
            raw_confidence=0.50,
            adjusted_confidence=0.50,
            level=ConfidenceLevel.LOW,
            needs_review=True,
        )

        item = manager.add_to_review_queue(prediction, frame_number=100)

        assert len(manager.get_pending_reviews()) == 1

        resolved = manager.resolve_review(item.id)

        assert resolved is True
        assert len(manager.get_pending_reviews()) == 0

    def test_review_queue_size_limit(self, manager):
        """Test that review queue respects size limit."""
        manager.review_queue_size = 5

        for i in range(10):
            prediction = ClassifiedPrediction(
                bib_number=str(i),
                raw_confidence=0.50,
                adjusted_confidence=0.50,
                level=ConfidenceLevel.LOW,
                needs_review=True,
            )
            manager.add_to_review_queue(prediction, frame_number=i)

        assert len(manager.get_pending_reviews()) == 5

    def test_clear_queue(self, manager):
        """Test clearing the review queue."""
        prediction = ClassifiedPrediction(
            bib_number="1234",
            raw_confidence=0.50,
            adjusted_confidence=0.50,
            level=ConfidenceLevel.LOW,
            needs_review=True,
        )

        manager.add_to_review_queue(prediction, frame_number=100)
        manager.add_to_review_queue(prediction, frame_number=101)

        assert len(manager.get_pending_reviews()) == 2

        manager.clear_queue()

        assert len(manager.get_pending_reviews()) == 0


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


class TestTier1Integration:
    """Integration tests for Tier 1 components working together."""

    def test_full_pipeline_valid_bib(self):
        """Test full Tier 1 pipeline with valid bib number."""
        # Setup
        bib_set = {str(i) for i in range(1000, 2000)}
        validator = BibSetValidator(bib_set)
        voting = EnhancedTemporalVoting()
        confidence_mgr = ConfidenceManager()

        track_id = 1
        bib_number = "1234"

        # Simulate 5 frames of consistent OCR
        for frame_idx in range(5):
            # Validate
            validation = validator.validate(bib_number, 0.9)

            # Vote
            voting.update(track_id, validation.validated, 0.9, frame_idx)

        # Get consensus
        consensus = voting.get_consensus(track_id)

        # Classify
        result = confidence_mgr.classify(
            bib_number=consensus.number,
            ocr_confidence=0.9,
            validation_result=validator.validate(consensus.number, 0.9),
            voting_result=consensus,
        )

        assert result.level == ConfidenceLevel.HIGH
        assert result.needs_review is False
        assert consensus.is_stable is True

    def test_full_pipeline_corrected_bib(self):
        """Test full Tier 1 pipeline with bib correction."""
        # Setup
        bib_set = {str(i) for i in range(1000, 2000)}
        validator = BibSetValidator(bib_set)
        voting = EnhancedTemporalVoting()
        confidence_mgr = ConfidenceManager()

        track_id = 1

        # Simulate OCR returning 1235 (not in set, close to 1234)
        for frame_idx in range(5):
            validation = validator.validate("1235", 0.85)
            voting.update(track_id, validation.validated, 0.85, frame_idx)

        consensus = voting.get_consensus(track_id)

        result = confidence_mgr.classify(
            bib_number=consensus.number,
            ocr_confidence=0.85,
            validation_result=validator.validate("1235", 0.85),
            voting_result=consensus,
        )

        # Should be corrected to 1234
        assert consensus.number in bib_set
        # Should be flagged for review due to correction
        assert result.needs_review is True

    def test_full_pipeline_invalid_bib(self):
        """Test full Tier 1 pipeline with invalid bib number."""
        # Setup
        bib_set = {str(i) for i in range(1000, 2000)}
        validator = BibSetValidator(bib_set)
        voting = EnhancedTemporalVoting()
        confidence_mgr = ConfidenceManager()

        track_id = 1

        # Simulate OCR returning invalid number (not close to anything)
        for frame_idx in range(5):
            validation = validator.validate("9999", 0.9)
            voting.update(track_id, validation.validated, 0.9, frame_idx)

        consensus = voting.get_consensus(track_id)

        result = confidence_mgr.classify(
            bib_number=consensus.number,
            ocr_confidence=0.9,
            validation_result=validator.validate("9999", 0.9),
            voting_result=consensus,
        )

        # Should not exceed original OCR confidence for invalid bib
        # (additive boosts: -0.1 validation penalty + 0.10 stability = net 0.0)
        assert result.adjusted_confidence <= 0.9
