import logging
from unittest.mock import Mock
import pytest

from ii_agent.llm.base import GeneralContentBlock, TextPrompt, TextResult
from ii_agent.llm.context_manager.amortized_forgetting import (
    AmortizedForgettingContextManager,
)
from ii_agent.llm.token_counter import TokenCounter


@pytest.fixture
def mock_logger():
    return Mock(spec=logging.Logger)


@pytest.fixture
def token_counter():
    return TokenCounter()


@pytest.fixture
def context_manager(token_counter, mock_logger):
    return AmortizedForgettingContextManager(
        token_counter=token_counter,
        logger=mock_logger,
        token_budget=1000,
        max_size=10,
        keep_first=2,
    )


def create_test_message_lists(count: int) -> list[list[GeneralContentBlock]]:
    """Create test message lists."""
    return [[TextPrompt(text=f"Message {i}")] for i in range(count)]


class TestAmortizedForgettingContextManager:
    """Test cases for AmortizedForgettingContextManager."""

    def test_init_valid_parameters(self, token_counter, mock_logger):
        """Test initialization with valid parameters."""
        context_manager = AmortizedForgettingContextManager(
            token_counter=token_counter,
            logger=mock_logger,
            token_budget=5000,
            max_size=20,
            keep_first=5,
        )

        assert context_manager.token_counter == token_counter
        assert context_manager.logger == mock_logger
        assert context_manager.token_budget == 5000
        assert context_manager.max_size == 20
        assert context_manager.keep_first == 5

    def test_init_default_values(self, token_counter, mock_logger):
        """Test initialization with default values."""
        context_manager = AmortizedForgettingContextManager(
            token_counter=token_counter, logger=mock_logger
        )

        assert context_manager.token_budget == 120_000
        assert context_manager.max_size == 100
        assert context_manager.keep_first == 1

    def test_init_invalid_keep_first_too_large(self, token_counter, mock_logger):
        """Test initialization fails when keep_first >= max_size // 2."""
        with pytest.raises(
            ValueError, match="keep_first .* must be less than half of max_size"
        ):
            AmortizedForgettingContextManager(
                token_counter=token_counter,
                logger=mock_logger,
                max_size=10,
                keep_first=5,  # 5 >= 10//2 (5)
            )

    def test_init_invalid_keep_first_negative(self, token_counter, mock_logger):
        """Test initialization fails when keep_first is negative."""
        with pytest.raises(ValueError, match="keep_first .* cannot be negative"):
            AmortizedForgettingContextManager(
                token_counter=token_counter, logger=mock_logger, keep_first=-1
            )

    def test_should_truncate_under_limit(self, context_manager):
        """Test should_truncate returns False when under max_size."""
        message_lists = create_test_message_lists(5)  # Less than max_size=10
        assert not context_manager.should_truncate(message_lists)

    def test_should_truncate_at_limit(self, context_manager):
        """Test should_truncate returns False when at max_size."""
        message_lists = create_test_message_lists(10)  # Equal to max_size=10
        assert not context_manager.should_truncate(message_lists)

    def test_should_truncate_over_limit(self, context_manager):
        """Test should_truncate returns True when over max_size."""
        message_lists = create_test_message_lists(15)  # Greater than max_size=10
        assert context_manager.should_truncate(message_lists)

    def test_apply_truncation_no_condensation_needed(self, context_manager):
        """Test apply_truncation when no condensation is needed."""
        message_lists = create_test_message_lists(8)  # Under max_size
        result = context_manager.apply_truncation_if_needed(message_lists)
        assert result == message_lists

    def test_apply_truncation_basic_condensation(self, context_manager):
        """Test basic condensation with head and tail preservation."""
        # Create 15 messages, max_size=10, keep_first=2
        # target_size = 10//2 = 5
        # head = first 2 messages
        # tail = last 3 messages (5 - 2 = 3)
        message_lists = create_test_message_lists(15)

        result = context_manager.apply_truncation_if_needed(message_lists)

        # Should have 5 messages total (target_size)
        assert len(result) == 5

        # First 2 should be the head (keep_first)
        assert result[0] == message_lists[0]
        assert result[1] == message_lists[1]

        # Last 3 should be the tail
        assert result[2] == message_lists[12]  # message_lists[-3]
        assert result[3] == message_lists[13]  # message_lists[-2]
        assert result[4] == message_lists[14]  # message_lists[-1]

    def test_apply_truncation_edge_case_small_list(self, context_manager):
        """Test truncation with very small message lists."""
        message_lists = create_test_message_lists(1)
        result = context_manager.apply_truncation_if_needed(message_lists)
        assert result == message_lists

    def test_apply_truncation_exact_target_size(self, context_manager):
        """Test truncation when list is exactly at target size after condensation."""
        # max_size=10, so target_size=5, keep_first=2
        # If we have 5 messages, no condensation should occur
        message_lists = create_test_message_lists(5)
        result = context_manager.apply_truncation_if_needed(message_lists)
        assert result == message_lists

    def test_apply_truncation_preserves_message_structure(self, context_manager):
        """Test that truncation preserves the message structure."""
        message_lists = [
            [TextPrompt(text="Message 1"), TextResult(text="Additional 1")],
            [TextPrompt(text="Message 2")],
            [TextResult(text="Message 3"), TextPrompt(text="Additional 3")],
        ] * 5  # 15 total messages

        result = context_manager.apply_truncation_if_needed(message_lists)

        # Should preserve structure of kept messages
        assert isinstance(result, list)
        for message_list in result:
            assert isinstance(message_list, list)
            for message in message_list:
                assert isinstance(message, GeneralContentBlock)

    def test_apply_truncation_zero_keep_first(self, token_counter, mock_logger):
        """Test truncation with keep_first=0."""
        context_manager = AmortizedForgettingContextManager(
            token_counter=token_counter, logger=mock_logger, max_size=10, keep_first=0
        )

        message_lists = create_test_message_lists(15)

        result = context_manager.apply_truncation(message_lists)

        # Should have 5 messages (target_size), all from tail
        assert len(result) == 5
        assert result == message_lists[-5:]

    def test_apply_truncation_calculates_forgotten_events_correctly(
        self, context_manager, mock_logger
    ):
        """Test that forgotten events calculation is correct."""
        # 15 messages total
        # keep_first = 2
        # target_size = 5, so events_from_tail = 3
        # forgotten should be messages 2-11 (indices 2 through 11, 10 messages)
        message_lists = create_test_message_lists(15)

        context_manager.apply_truncation(message_lists)

        # Check the log message for forgotten count
        mock_logger.info.assert_called_once()
        log_message = mock_logger.info.call_args[0][0]
        assert "forgetting 10 middle events" in log_message

    def test_apply_truncation_boundary_conditions(self, token_counter, mock_logger):
        """Test truncation at various boundary conditions."""
        # Test with max_size=2, keep_first=0 (target_size=1)
        context_manager = AmortizedForgettingContextManager(
            token_counter=token_counter, logger=mock_logger, max_size=2, keep_first=0
        )

        message_lists = create_test_message_lists(5)
        result = context_manager.apply_truncation(message_lists)

        # Should keep only the last message
        assert len(result) == 1
        assert result[0] == message_lists[-1]

    def test_apply_truncation_with_empty_tail(self, context_manager):
        """Test truncation when events_from_tail calculation results in edge cases."""
        # This tests the conditional logic in the tail calculation
        message_lists = create_test_message_lists(12)  # Slightly over max_size

        result = context_manager.apply_truncation(message_lists)

        # Should still work correctly
        assert len(result) == 5  # target_size
        assert result[:2] == message_lists[:2]  # head preserved
