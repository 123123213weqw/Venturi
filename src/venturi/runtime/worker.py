"""One explicit inference owner with condition-variable idle waiting.

No automatic startup/import-time device work. This worker resolves completion
futures; applications requiring streaming may own scheduler.tick directly.
"""
import math
import threading

from .continuous import ContinuousScheduler


class ContinuousWorker:
    def __init__(self, scheduler):
        if not isinstance(scheduler, ContinuousScheduler):
            raise TypeError("An explicitly prepared ContinuousScheduler is required")
        self.scheduler = scheduler
        self._condition = threading.Condition()
        self._thread = None
        self._stopping, self._closed, self._error = False, False, None

    def start(self):
        """Start the only tick owner; existing weights/buffers are not reloaded."""
        with self._condition:
            if self._stopping or self._closed:
                raise RuntimeError("Continuous worker closed")
            if self._thread is not None:
                if self._error is not None:
                    raise RuntimeError("Continuous worker failed") from self._error
                return self
            self._thread = threading.Thread(target=self._run, name="venturi-continuous", daemon=False)
            try:
                self._thread.start()
            except BaseException as exc:
                self._thread = None
                self._error = exc
                self._stopping = True
                raise
            return self

    def submit(self, prompt, *, max_new_tokens, eos_token_ids=()):
        with self._condition:
            if self._error is not None:
                raise RuntimeError("Continuous worker failed") from self._error
            if self._stopping or self._closed:
                raise RuntimeError("Continuous worker closed")
            if self._thread is None:
                raise RuntimeError("Start the continuous worker explicitly before submission")
            ticket = self.scheduler.submit(prompt, max_new_tokens=max_new_tokens, eos_token_ids=eos_token_ids)
            self._condition.notify()
            return ticket

    def cancel(self, ticket):
        result = self.scheduler.cancel(ticket)
        with self._condition:
            self._condition.notify()
        return result

    def _run(self):
        try:
            while True:
                with self._condition:
                    while not self._stopping and not self.scheduler.has_work:
                        self._condition.wait()
                    if self._stopping:
                        break
                self.scheduler.tick()
        except BaseException as exc:
            with self._condition:
                self._error = exc
        finally:
            try:
                self.scheduler.close()
            except BaseException as exc:
                with self._condition:
                    if self._error is None:
                        self._error = exc
            finally:
                with self._condition:
                    self._closed = True
                    self._condition.notify_all()

    def info(self):
        with self._condition:
            return dict(started=self._thread is not None,
                        alive=self._thread is not None and self._thread.is_alive(),
                        stopping=self._stopping, closed=self._closed,
                        error=repr(self._error) if self._error is not None else None,
                        scheduling=self.scheduler.info())

    def close(self, *, timeout=180):
        """Stop at a safe boundary, cancel/reset the lease, then join its owner.

        An observation timeout is not a stopped thread: the live owner is retained
        and callers must not close its underlying batch until it has really exited.
        """
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout < 0:
            raise ValueError("Explicit nonnegative join timeout required")
        with self._condition:
            if self._thread is threading.current_thread():
                raise RuntimeError("Cannot close/join the inference owner from its callback")
            self._stopping = True
            self._condition.notify_all()
            thread = self._thread
            if thread is None:
                close_unstarted = not self._closed
                self._closed = True
        if thread is None:
            # Reset may resolve user futures: never invoke their callbacks while
            # holding the worker's submission/notification condition.
            if close_unstarted:
                try:
                    self.scheduler.close()
                except BaseException as exc:
                    with self._condition:
                        if self._error is None:
                            self._error = exc
                    raise
            if self._error is not None:
                raise RuntimeError("Continuous worker startup/cleanup failed") from self._error
            return
        thread.join(timeout=timeout)
        if thread.is_alive():
            raise TimeoutError("Continuous owner is still in flight; do not free its batch")
        if self._error is not None:
            raise RuntimeError("Continuous worker terminated with an execution/cleanup failure") from self._error

    def __enter__(self):
        return self.start()

    def __exit__(self, kind, value, traceback):
        self.close()
