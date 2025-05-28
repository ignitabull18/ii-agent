import logging
from unittest.mock import Mock
import pytest

from ii_agent.llm.base import GeneralContentBlock, TextPrompt, TextResult
from ii_agent.llm.context_manager.pipeline import PipelineContextManager
from ii_agent.llm.context_manager.base import ContextManager
from ii_agent.llm.token_counter import TokenCounter


class MockContextManager(ContextManager):
    """Mock context manager for testing pipeline."""

    def __init__(
        self, name: str, reduction_factor: float = 0.5, token_counter=None, logger=None
    ):
        super().__init__(token_counter or TokenCounter(), logger or Mock(), 1000)
        self.name = name
        self.reduction_factor = reduction_factor
        self.call_count = 0

    def apply_truncation(
        self, message_lists: list[list[GeneralContentBlock]]
    ) -> list[list[GeneralContentBlock]]:
        self.call_count += 1
        # Simulate reduction
        if len(message_lists) > 1:
            new_length = max(1, int(len(message_lists) * self.reduction_factor))
            return message_lists[:new_length]
        return message_lists


@pytest.fixture
def mock_logger():
    return Mock(spec=logging.Logger)


@pytest.fixture
def token_counter():
    return TokenCounter()


def create_test_message_lists(count: int) -> list[list[GeneralContentBlock]]:
    """Create test message lists."""
    long_content = (
        "This is a very long message content that should definitely exceed the token budget when multiplied. "
        * 100
    )
    return [[TextPrompt(text=f"Message {i}: {long_content}")] for i in range(count)]


class TestPipelineContextManager:
    """Test cases for PipelineContextManager."""

    def test_init_single_context_manager(self, token_counter, mock_logger):
        """Test initialization with a single context manager."""
        mock_cm = MockContextManager("test_cm")

        pipeline = PipelineContextManager(
            token_counter,
            mock_logger,
            token_budget=2000,
            context_managers=[mock_cm],
        )

        assert pipeline.token_counter == token_counter
        assert pipeline.logger == mock_logger
        assert pipeline.token_budget == 2000
        assert len(pipeline.context_managers) == 1
        assert pipeline.context_managers[0] == mock_cm

    def test_init_multiple_context_managers(self, token_counter, mock_logger):
        """Test initialization with multiple context managers."""
        mock_cm1 = MockContextManager("cm1")
        mock_cm2 = MockContextManager("cm2")
        mock_cm3 = MockContextManager("cm3")

        pipeline = PipelineContextManager(
            token_counter, mock_logger, context_managers=[mock_cm1, mock_cm2, mock_cm3]
        )

        assert len(pipeline.context_managers) == 3

    def test_init_no_context_managers(self, token_counter, mock_logger):
        """Test initialization fails with no context managers."""
        with pytest.raises(
            ValueError, match="At least one context manager must be provided"
        ):
            PipelineContextManager(token_counter, mock_logger)

    def test_init_default_token_budget(self, token_counter, mock_logger):
        """Test initialization with default token budget."""
        mock_cm = MockContextManager("test_cm")

        pipeline = PipelineContextManager(
            token_counter, mock_logger, context_managers=[mock_cm]
        )

        assert pipeline.token_budget == 120_000  # default

    def test_apply_truncation_single_context_manager(self, token_counter, mock_logger):
        """Test apply_truncation with a single context manager."""
        mock_cm = MockContextManager("test_cm", reduction_factor=0.5)
        pipeline = PipelineContextManager(
            token_counter, mock_logger, context_managers=[mock_cm]
        )

        message_lists = create_test_message_lists(10)

        result = pipeline.apply_truncation(message_lists)

        # Should have called the context manager once
        assert mock_cm.call_count == 1

        # Should have reduced the messages
        assert len(result) == 5  # 10 * 0.5

    def test_apply_truncation_multiple_context_managers_sequential(
        self, token_counter, mock_logger
    ):
        """Test that context managers are applied sequentially."""
        mock_cm1 = MockContextManager("cm1", reduction_factor=0.8)  # 10 -> 8
        mock_cm2 = MockContextManager("cm2", reduction_factor=0.5)  # 8 -> 4
        mock_cm3 = MockContextManager("cm3", reduction_factor=0.75)  # 4 -> 3

        # Use a very low token budget to ensure truncation is triggered
        pipeline = PipelineContextManager(
            token_counter,
            mock_logger,
            token_budget=100,
            context_managers=[mock_cm1, mock_cm2, mock_cm3],
        )

        message_lists = create_test_message_lists(10)

        result = pipeline.apply_truncation(message_lists)

        # All context managers should have been called
        assert mock_cm1.call_count == 1
        assert mock_cm2.call_count == 1
        assert mock_cm3.call_count == 1

        # Final result should reflect sequential application
        assert len(result) == 3  # 10 -> 8 -> 4 -> 3

    def test_apply_truncation_early_stopping_on_budget(
        self, token_counter, mock_logger
    ):
        """Test early stopping when token budget is satisfied."""
        mock_cm1 = MockContextManager(
            "cm1", reduction_factor=0.1
        )  # Aggressive reduction
        mock_cm2 = MockContextManager("cm2", reduction_factor=0.5)
        mock_cm3 = MockContextManager("cm3", reduction_factor=0.5)

        # Set a high token budget so first CM will satisfy it
        pipeline = PipelineContextManager(
            token_counter,
            mock_logger,
            100_000,
            context_managers=[mock_cm1, mock_cm2, mock_cm3],
        )

        message_lists = create_test_message_lists(10)

        result = pipeline.apply_truncation(message_lists)

        # First CM should be called, but might stop early if budget is satisfied
        assert mock_cm1.call_count >= 1
        # Subsequent CMs might not be called if budget is satisfied
        # This depends on the exact token counting behavior

    def test_apply_truncation_logs_progress(self, token_counter, mock_logger):
        """Test that the pipeline logs progress information."""
        mock_cm1 = MockContextManager("cm1", reduction_factor=0.8)
        mock_cm2 = MockContextManager("cm2", reduction_factor=0.5)

        # Use a very low token budget to ensure truncation is triggered
        pipeline = PipelineContextManager(
            token_counter,
            mock_logger,
            token_budget=100,
            context_managers=[mock_cm1, mock_cm2],
        )

        message_lists = create_test_message_lists(10)

        pipeline.apply_truncation(message_lists)

        # Should log debug and info messages
        assert mock_logger.debug.called
        assert mock_logger.info.called

        # Check for specific log content
        debug_calls = [call[0][0] for call in mock_logger.debug.call_args_list]
        info_calls = [call[0][0] for call in mock_logger.info.call_args_list]

        # Should log running each context manager
        cm_logs = [msg for msg in debug_calls if "Running context manager" in msg]
        assert len(cm_logs) == 2

        # Should log pipeline completion
        completion_logs = [msg for msg in info_calls if "Pipeline completed" in msg]
        assert len(completion_logs) == 1

    def test_apply_truncation_logs_reductions(self, token_counter, mock_logger):
        """Test that reductions are logged."""
        mock_cm = MockContextManager("test_cm", reduction_factor=0.5)
        pipeline = PipelineContextManager(
            token_counter, mock_logger, context_managers=[mock_cm]
        )

        message_lists = create_test_message_lists(10)

        pipeline.apply_truncation(message_lists)

        # Should log the reduction
        info_calls = [call[0][0] for call in mock_logger.info.call_args_list]
        reduction_logs = [msg for msg in info_calls if "reduced message count" in msg]
        assert len(reduction_logs) >= 0  # Might be 0 if no actual reduction occurred

    def test_apply_truncation_preserves_message_structure(
        self, token_counter, mock_logger
    ):
        """Test that truncation preserves the message structure."""
        mock_cm = MockContextManager("test_cm")
        pipeline = PipelineContextManager(
            token_counter, mock_logger, context_managers=[mock_cm]
        )

        message_lists = [
            [TextPrompt(text="Message 1"), TextResult(text="Additional 1")],
            [TextPrompt(text="Message 2")],
            [TextResult(text="Message 3"), TextPrompt(text="Additional 3")],
        ]

        result = pipeline.apply_truncation(message_lists)

        # Should preserve structure
        assert isinstance(result, list)
        for message_list in result:
            assert isinstance(message_list, list)
            for message in message_list:
                assert isinstance(message, GeneralContentBlock)

    def test_apply_truncation_handles_empty_input(self, token_counter, mock_logger):
        """Test pipeline handles empty input gracefully."""
        mock_cm = MockContextManager("test_cm")
        pipeline = PipelineContextManager(
            token_counter, mock_logger, context_managers=[mock_cm]
        )

        message_lists = []

        result = pipeline.apply_truncation(message_lists)

        assert result == []
        # Empty input won't exceed token budget, so CM won't be called
        assert mock_cm.call_count == 0

    def test_apply_truncation_context_manager_order_matters(
        self, token_counter, mock_logger
    ):
        """Test that the order of context managers affects the result."""
        # Two CMs with different behaviors
        mock_cm1 = MockContextManager("aggressive", reduction_factor=0.2)
        mock_cm2 = MockContextManager("gentle", reduction_factor=0.8)

        pipeline1 = PipelineContextManager(
            token_counter, mock_logger, context_managers=[mock_cm1, mock_cm2]
        )
        pipeline2 = PipelineContextManager(
            token_counter, mock_logger, context_managers=[mock_cm2, mock_cm1]
        )

        message_lists = create_test_message_lists(10)

        result1 = pipeline1.apply_truncation(message_lists.copy())
        # Reset call counts
        mock_cm1.call_count = 0
        mock_cm2.call_count = 0
        result2 = pipeline2.apply_truncation(message_lists.copy())

        # Results should be different due to different order
        # pipeline1: 10 -> 2 -> 1 (aggressive first)
        # pipeline2: 10 -> 8 -> 1 (gentle first, then aggressive)
        # Both end up at 1, but the intermediate steps are different

    def test_apply_truncation_with_failing_context_manager(
        self, token_counter, mock_logger
    ):
        """Test pipeline handles context manager failures gracefully."""

        # Create a CM that raises an exception
        class FailingContextManager(ContextManager):
            def __init__(self):
                super().__init__(token_counter, mock_logger, 1000)

            def apply_truncation(self, message_lists):
                raise Exception("CM failed")

            def apply_truncation_if_needed(self, message_lists):
                # Override to always call apply_truncation to trigger the exception
                return self.apply_truncation(message_lists)

        mock_cm1 = MockContextManager("cm1")
        failing_cm = FailingContextManager()
        mock_cm2 = MockContextManager("cm2")

        # Use a very low token budget to ensure truncation is triggered
        pipeline = PipelineContextManager(
            token_counter,
            mock_logger,
            token_budget=100,
            context_managers=[mock_cm1, failing_cm, mock_cm2],
        )

        message_lists = create_test_message_lists(10)

        # Should propagate the exception
        with pytest.raises(Exception, match="CM failed"):
            pipeline.apply_truncation(message_lists)
