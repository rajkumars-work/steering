"""SIGALRM-based hard timeout helper.

SIGALRM is delivered to the main thread and interrupts any blocking syscall
(including stuck CUDA ops), unlike ThreadPoolExecutor timeouts which can't kill
the worker. Limitation: must be called from the main thread.
"""
from __future__ import annotations

import signal
from typing import Callable, TypeVar

T = TypeVar("T")


class CheckTimeout(Exception):
    """Raised when a function exceeds its wall-clock timeout."""


def _alarm_handler(signum, frame):
    raise CheckTimeout


def run_with_timeout(fn: Callable[..., T], *args, timeout: float = 30, **kwargs) -> T:
    """Run ``fn(*args, **kwargs)`` with a hard SIGALRM timeout.

    Raises ``CheckTimeout`` if the function does not return in ``timeout``
    seconds. Re-installs any pre-existing SIGALRM handler on exit.
    """
    old = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(int(timeout))
    try:
        return fn(*args, **kwargs)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
