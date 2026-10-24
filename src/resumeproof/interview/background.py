"""Background grading worker using ThreadPoolExecutor for CPU decoupling."""

from concurrent.futures import ThreadPoolExecutor, Future
import logging
from typing import Callable, Optional
from resumeproof.schemas.claims import Claim
from resumeproof.schemas.rubric import RubricItem
from resumeproof.schemas.interview import GradeResult
from resumeproof.interview.grader import AnswerGrader
from resumeproof.interview.caching import GradingCache

logger = logging.getLogger("resumeproof.interview.background")

class BackgroundGrader:
    """Decouples CPU-intensive 7B grading from conversational question flow."""

    def __init__(
        self,
        grader: AnswerGrader,
        cache: GradingCache,
        max_workers: int = 2,
    ):
        self.grader = grader
        self.cache = cache
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="resumeproof-grader")
        self.pending_futures: dict[str, Future[GradeResult]] = {}

    def grade_async(
        self,
        task_id: str,
        claim: Claim,
        question_text: str,
        answer_text: str,
        rubric: Optional[RubricItem] = None,
        on_complete: Optional[Callable[[GradeResult], None]] = None,
    ) -> Future[GradeResult]:
        """Dispatches grading job to background thread with caching check."""
        # Check cache immediately on caller thread
        cached_result = self.cache.get(
            claim_id=claim.id,
            question_text=question_text,
            answer_text=answer_text,
            model=self.grader.model,
        )

        def _execute_grade() -> GradeResult:
            if cached_result:
                grade = cached_result
            else:
                grade = self.grader.grade_answer(
                    claim=claim,
                    question_text=question_text,
                    answer_text=answer_text,
                    rubric=rubric,
                )
                self.cache.set(
                    claim_id=claim.id,
                    question_text=question_text,
                    answer_text=answer_text,
                    model=self.grader.model,
                    grade=grade,
                )

            if on_complete:
                try:
                    on_complete(grade)
                except Exception as e:
                    logger.error(f"Error in on_complete callback for task {task_id}: {e}")
            return grade

        future = self.executor.submit(_execute_grade)
        self.pending_futures[task_id] = future
        return future

    def grade_sync(
        self,
        claim: Claim,
        question_text: str,
        answer_text: str,
        rubric: Optional[RubricItem] = None,
    ) -> GradeResult:
        """Synchronous grading execution with caching."""
        cached = self.cache.get(
            claim_id=claim.id,
            question_text=question_text,
            answer_text=answer_text,
            model=self.grader.model,
        )
        if cached:
            return cached

        res = self.grader.grade_answer(
            claim=claim,
            question_text=question_text,
            answer_text=answer_text,
            rubric=rubric,
        )
        self.cache.set(
            claim_id=claim.id,
            question_text=question_text,
            answer_text=answer_text,
            model=self.grader.model,
            grade=res,
        )
        return res

    def wait_for(self, task_id: str, timeout: Optional[float] = 60.0) -> GradeResult:
        """Waits for a specific background grading task to finish."""
        if task_id in self.pending_futures:
            future = self.pending_futures.pop(task_id)
            return future.result(timeout=timeout)
        raise KeyError(f"No pending grading task with id {task_id}")

    def drain_all(self, timeout: Optional[float] = 120.0) -> dict[str, GradeResult]:
        """Waits for all pending background grading tasks to complete."""
        results = {}
        for task_id, future in list(self.pending_futures.items()):
            try:
                results[task_id] = future.result(timeout=timeout)
            except Exception as e:
                logger.error(f"Background grade task {task_id} failed: {e}")
            finally:
                self.pending_futures.pop(task_id, None)
        return results

    def shutdown(self, wait: bool = True) -> None:
        self.drain_all()
        self.executor.shutdown(wait=wait)
