"""Give the current macOS music worker the QoS of user-initiated work.

Python's worker threads start at default QoS even when their main thread has
a higher class. Live music has a deadline, so identify its dedicated model
thread to the scheduler. This does not change process priority or other
threads, and failure leaves inference available with its existing scheduling.
"""

import ctypes
import logging
import os
import sys

log = logging.getLogger(__name__)

_USER_INITIATED = 0x19
_QOS_NAMES = {
    0x15: "default",
    _USER_INITIATED: "user_initiated",
    0x21: "user_interactive",
}


def configure_worker_priority() -> str:
    """Configure only the calling thread; return its known QoS or policy.

    Call once from the dedicated model worker. ``MRT_WORKER_QOS=default``
    leaves scheduling untouched and does not require a native runtime.
    """
    requested = os.environ.get("MRT_WORKER_QOS", "user_initiated").strip().lower()
    if requested == "default":
        return "default"
    if requested != "user_initiated":
        log.warning(
            "unknown MRT_WORKER_QOS=%r; keeping default worker scheduling",
            requested,
        )
        return "default"
    if sys.platform != "darwin":
        return "unavailable"

    status = "unavailable"
    try:
        native = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        current_qos = native.qos_class_self
        current_qos.argtypes = []
        current_qos.restype = ctypes.c_uint
        qos = int(current_qos())
        status = _QOS_NAMES.get(qos, "unavailable")
        if qos >= _USER_INITIATED:
            return status

        set_qos = native.pthread_set_qos_class_self_np
        set_qos.argtypes = [ctypes.c_uint, ctypes.c_int]
        set_qos.restype = ctypes.c_int
        result = set_qos(_USER_INITIATED, 0)
        if result != 0:
            # This API returns an errno value directly; ctypes.get_errno()
            # would read unrelated thread-local state.
            log.warning(
                "music worker QoS request failed: %s (errno %d); retaining %s",
                os.strerror(result), result, status,
            )
            return status

        qos = int(current_qos())
        status = _QOS_NAMES.get(qos, "unavailable")
        if qos < _USER_INITIATED:
            log.warning("music worker QoS request left the thread at %s", status)
        else:
            log.info("music worker QoS: %s", status)
        return status
    except (OSError, AttributeError) as exc:
        log.warning("music worker QoS unavailable: %s; retaining %s", exc, status)
        return status
