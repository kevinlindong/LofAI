"""WebSocket admission and bounded-outbox invariants."""

import asyncio
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import server  # noqa: E402


class ServerTransportTests(unittest.TestCase):
    def test_default_browser_origins_are_allowed_and_foreign_origin_is_not(self):
        self.assertTrue(server._origin_allowed(None))
        self.assertTrue(server._origin_allowed("http://localhost:3000"))
        self.assertTrue(server._origin_allowed("http://127.0.0.1:3000"))
        self.assertFalse(server._origin_allowed("https://untrusted.example"))

    def test_full_outbox_never_silently_drops_pcm(self):
        queue = asyncio.Queue(maxsize=2)
        queue.put_nowait(b"first")
        queue.put_nowait(b"second")

        self.assertTrue(server._offer(queue, b"third"))
        self.assertEqual(list(queue._queue), [b"first", b"second"])

        self.assertTrue(server._offer(queue, {"type": "status"}))
        self.assertEqual(list(queue._queue), [b"second", {"type": "status"}])


if __name__ == "__main__":
    unittest.main(verbosity=2)
