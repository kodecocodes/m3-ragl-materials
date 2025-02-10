from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
  ContextualPrecisionMetric,
  ContextualRecallMetric,
  ContextualRelevancyMetric
)

contextual_precision = ContextualPrecisionMetric()
contextual_recall = ContextualRecallMetric()
contextual_relevancy = ContextualRelevancyMetric()

test_case = LLMTestCase(
  input="Which programmes were dropped from the 2024 Olympics?",
  actual_output="Four events were dropped from weightlifting for the 
    2024 Olympics. Additionally, in canoeing, two sprint events 
    were replaced with two slalom events. The overall event 
    total for canoeing remained at 16.",
  expected_output="Four events were dropped from weightlifting.",
  retrieval_context=[
    """Four events were dropped from weightlifting."""
 ]
)

evaluate(
  test_cases=[test_case],
  metrics=[contextual_precision, contextual_recall, contextual_relevancy]
)

from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from deepeval import evaluate

answer_relevancy = AnswerRelevancyMetric()
faithfulness = FaithfulnessMetric()

evaluate(
  test_cases=[test_case],
  metrics=[answer_relevancy, faithfulness]
)