"""Thread QoS configuration without native scheduling changes."""

import ctypes
import errno
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import worker_priority


class WorkerPriorityTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {"MRT_WORKER_QOS": "user_initiated"})
        platform = patch.object(worker_priority.sys, "platform", "darwin")
        env.start()
        platform.start()
        self.addCleanup(env.stop)
        self.addCleanup(platform.stop)

    @staticmethod
    def native(qos=(0x15, 0x19), result=0):
        return SimpleNamespace(
            qos_class_self=Mock(side_effect=qos),
            pthread_set_qos_class_self_np=Mock(return_value=result),
        )

    def test_sets_only_current_thread_with_exact_native_signature(self):
        native = self.native()
        with patch.object(worker_priority.ctypes, "CDLL", return_value=native) as load:
            self.assertEqual(worker_priority.configure_worker_priority(), "user_initiated")
        load.assert_called_once_with("/usr/lib/libSystem.B.dylib")
        native.pthread_set_qos_class_self_np.assert_called_once_with(0x19, 0)
        self.assertEqual(native.qos_class_self.call_count, 2)
        self.assertEqual(native.qos_class_self.argtypes, [])
        self.assertIs(native.qos_class_self.restype, ctypes.c_uint)
        self.assertEqual(native.pthread_set_qos_class_self_np.argtypes,
                         [ctypes.c_uint, ctypes.c_int])
        self.assertIs(native.pthread_set_qos_class_self_np.restype, ctypes.c_int)

    def test_preserves_equal_and_higher_priority(self):
        for qos, label in ((0x19, "user_initiated"), (0x21, "user_interactive")):
            with self.subTest(qos=qos):
                native = self.native(qos=(qos,))
                with patch.object(worker_priority.ctypes, "CDLL", return_value=native):
                    self.assertEqual(worker_priority.configure_worker_priority(), label)
                native.pthread_set_qos_class_self_np.assert_not_called()

    def test_non_darwin_never_loads_native_library(self):
        with patch.object(worker_priority.sys, "platform", "linux"), \
             patch.object(worker_priority.ctypes, "CDLL") as load:
            self.assertEqual(worker_priority.configure_worker_priority(), "unavailable")
        load.assert_not_called()

    def test_opt_out_needs_no_native_runtime(self):
        with patch.dict(os.environ, {"MRT_WORKER_QOS": "default"}), \
             patch.object(worker_priority.ctypes, "CDLL") as load:
            self.assertEqual(worker_priority.configure_worker_priority(), "default")
        load.assert_not_called()

    def test_unknown_setting_keeps_default_without_loading_native_runtime(self):
        with patch.dict(os.environ, {"MRT_WORKER_QOS": "realtime"}), \
             patch.object(worker_priority.ctypes, "CDLL") as load, \
             self.assertLogs("worker_priority", level="WARNING"):
            self.assertEqual(worker_priority.configure_worker_priority(), "default")
        load.assert_not_called()

    def test_failed_native_request_preserves_known_priority(self):
        native = self.native(qos=(0x15,), result=errno.EPERM)
        with patch.object(worker_priority.ctypes, "CDLL", return_value=native), \
             self.assertLogs("worker_priority", level="WARNING"):
            self.assertEqual(worker_priority.configure_worker_priority(), "default")
        native.pthread_set_qos_class_self_np.assert_called_once_with(0x19, 0)
        self.assertEqual(native.qos_class_self.call_count, 1)

    def test_missing_library_or_query_api_does_not_block_inference(self):
        for exception in (OSError("missing library"), AttributeError("missing API")):
            with self.subTest(exception=type(exception).__name__), \
                 patch.object(worker_priority.ctypes, "CDLL", side_effect=exception), \
                 self.assertLogs("worker_priority", level="WARNING"):
                self.assertEqual(worker_priority.configure_worker_priority(), "unavailable")

    def test_missing_setter_preserves_queried_priority(self):
        native = SimpleNamespace(qos_class_self=Mock(return_value=0x15))
        with patch.object(worker_priority.ctypes, "CDLL", return_value=native), \
             self.assertLogs("worker_priority", level="WARNING"):
            self.assertEqual(worker_priority.configure_worker_priority(), "default")

    def test_success_response_does_not_override_actual_priority(self):
        native = self.native(qos=(0x15, 0x15))
        with patch.object(worker_priority.ctypes, "CDLL", return_value=native), \
             self.assertLogs("worker_priority", level="WARNING"):
            self.assertEqual(worker_priority.configure_worker_priority(), "default")


if __name__ == "__main__":
    unittest.main(verbosity=2)
