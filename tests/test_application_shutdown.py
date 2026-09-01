"""Process-tree integration tests for start.sh and stop.sh."""

import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
START = ROOT / "start.sh"
STOP = ROOT / "stop.sh"


def wait_for(predicate, timeout=8.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def process_started_at(pid):
    result = subprocess.run(
        ["ps", "-o", "lstart=", "-p", str(pid)],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip()


def process_is_live(pid):
    result = subprocess.run(
        ["ps", "-o", "state=", "-p", str(pid)],
        text=True,
        capture_output=True,
        check=False,
    )
    state = result.stdout.strip()
    return bool(state) and not state.startswith("Z")


SERVICE_SOURCE = '''#!/usr/bin/env python3
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

state = Path(os.environ["FAKE_STATE"])
name = "__FAKE_NAME__"
ignore = os.environ.get("FAKE_IGNORE_TERM") == name
if ignore:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

child_code = """import os, signal, time
from pathlib import Path
if os.environ.get('FAKE_CHILD_IGNORE') == '1':
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path(os.environ['FAKE_CHILD_FILE']).write_text(str(os.getpid()))
while True:
    time.sleep(0.1)
"""
env = os.environ.copy()
env["FAKE_CHILD_IGNORE"] = "1" if ignore else "0"
env["FAKE_CHILD_FILE"] = str(state / f"{name}.child")
child = subprocess.Popen([sys.executable, "-c", child_code], env=env)
(state / f"{name}.parent").write_text(str(os.getpid()))
while True:
    time.sleep(0.1)
'''


class ApplicationShutdownTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.state = Path(self.temporary.name)

    def tearDown(self):
        env = os.environ.copy()
        env.update(
            LOFAI_STATE_DIR=str(self.state),
            LOFAI_STOP_TIMEOUT="0",
            LOFAI_KILL_TIMEOUT="1",
        )
        subprocess.run([str(STOP), "--quiet"], env=env, check=False)
        self.temporary.cleanup()

    def make_service(self, name):
        path = self.state / f"{name}.py"
        path.write_text(SERVICE_SOURCE.replace("__FAKE_NAME__", name))
        path.chmod(0o755)
        return path

    def test_terminating_supervisor_stops_all_descendants_and_forces_resistant_one(self):
        backend = self.make_service("backend")
        frontend = self.make_service("frontend")
        env = os.environ.copy()
        env.update(
            LOFAI_STATE_DIR=str(self.state),
            LOFAI_BACKEND_SCRIPT=str(backend),
            LOFAI_FRONTEND_SCRIPT=str(frontend),
            LOFAI_STARTUP_DELAY="0",
            LOFAI_STOP_TIMEOUT="1",
            LOFAI_KILL_TIMEOUT="1",
            FAKE_STATE=str(self.state),
            FAKE_IGNORE_TERM="frontend",
        )

        supervisor = subprocess.Popen(
            [str(START)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        files = [
            self.state / "backend.parent",
            self.state / "backend.child",
            self.state / "frontend.parent",
            self.state / "frontend.child",
            self.state / "backend.pid",
            self.state / "frontend.pid",
            self.state / "logs.pid",
        ]
        self.assertTrue(wait_for(lambda: all(path.exists() for path in files)))
        service_pids = [int(path.read_text()) for path in files[:4]]

        stopped = subprocess.run([str(STOP)], env=env, text=True, capture_output=True)
        self.assertEqual(stopped.returncode, 0, stopped.stdout + stopped.stderr)
        output, _ = supervisor.communicate(timeout=8)
        self.assertIn("lofAI is running", output)
        self.assertEqual(supervisor.returncode, 143)

        self.assertTrue(
            wait_for(lambda: not any(process_is_live(pid) for pid in service_pids)),
            f"live service descendants: {[pid for pid in service_pids if process_is_live(pid)]}",
        )
        for name in (".lofai.lock", "logs.pid", "frontend.pid", "backend.pid"):
            self.assertFalse((self.state / name).exists(), name)

        repeated = subprocess.run([str(STOP)], env=env, text=True, capture_output=True)
        self.assertEqual(repeated.returncode, 0)
        self.assertIn("already stopped", repeated.stdout)

    def test_external_stop_interrupts_startup_wait_and_its_process(self):
        backend = self.make_service("backend")
        frontend = self.make_service("frontend")
        env = os.environ.copy()
        env.update(
            LOFAI_STATE_DIR=str(self.state),
            LOFAI_BACKEND_SCRIPT=str(backend),
            LOFAI_FRONTEND_SCRIPT=str(frontend),
            LOFAI_STARTUP_DELAY="60",
            LOFAI_STOP_TIMEOUT="1",
            LOFAI_KILL_TIMEOUT="1",
            FAKE_STATE=str(self.state),
            FAKE_IGNORE_TERM="",
        )
        supervisor = subprocess.Popen(
            [str(START)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self.assertTrue(wait_for(lambda: (self.state / "backend.child").exists()))
        backend_pids = [
            int((self.state / "backend.parent").read_text()),
            int((self.state / "backend.child").read_text()),
        ]
        child_listing = subprocess.run(
            ["ps", "-axo", "pid=,ppid="], text=True, capture_output=True, check=True
        ).stdout.splitlines()
        supervisor_children = [
            int(line.split()[0])
            for line in child_listing
            if len(line.split()) == 2 and int(line.split()[1]) == supervisor.pid
        ]

        started = time.monotonic()
        stopped = subprocess.run([str(STOP)], env=env, text=True, capture_output=True)
        elapsed = time.monotonic() - started
        output, _ = supervisor.communicate(timeout=5)

        self.assertEqual(stopped.returncode, 0, stopped.stdout + stopped.stderr)
        self.assertLess(elapsed, 5, "stop waited for the 60-second startup delay")
        self.assertIn("Starting backend", output)
        all_children = backend_pids + supervisor_children
        self.assertTrue(wait_for(lambda: not any(process_is_live(pid) for pid in all_children)))
        self.assertFalse((self.state / ".lofai.lock").exists())

    def test_second_launcher_serially_replaces_the_first_run(self):
        backend = self.make_service("backend")
        frontend = self.make_service("frontend")
        env = os.environ.copy()
        env.update(
            LOFAI_STATE_DIR=str(self.state),
            LOFAI_BACKEND_SCRIPT=str(backend),
            LOFAI_FRONTEND_SCRIPT=str(frontend),
            LOFAI_STARTUP_DELAY="0",
            LOFAI_STOP_TIMEOUT="1",
            LOFAI_KILL_TIMEOUT="1",
            FAKE_STATE=str(self.state),
            FAKE_IGNORE_TERM="",
        )

        first = subprocess.Popen(
            [str(START)], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        marker_files = [
            self.state / "backend.parent",
            self.state / "backend.child",
            self.state / "frontend.parent",
            self.state / "frontend.child",
            self.state / "logs.pid",
        ]
        self.assertTrue(wait_for(lambda: all(path.exists() for path in marker_files)))
        first_service_pids = [int(path.read_text()) for path in marker_files[:4]]

        second = subprocess.Popen(
            [str(START)], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        first.communicate(timeout=6)

        def second_is_ready():
            lock = self.state / ".lofai.lock"
            service_metadata = [self.state / "backend.pid", self.state / "frontend.pid"]
            if (
                not lock.exists()
                or any(not path.exists() for path in marker_files)
                or any(not path.exists() for path in service_metadata)
            ):
                return False
            try:
                lock_pid = int(lock.read_text().split("|")[1])
                current = [int(path.read_text()) for path in marker_files[:4]]
                tracked = [int(path.read_text().split("|")[1]) for path in service_metadata]
            except (IndexError, ValueError):
                return False
            return (
                lock_pid == second.pid
                and current != first_service_pids
                and tracked == [current[0], current[2]]
            )

        self.assertTrue(wait_for(second_is_ready))
        second_service_pids = [int(path.read_text()) for path in marker_files[:4]]
        self.assertTrue(
            wait_for(lambda: not any(process_is_live(pid) for pid in first_service_pids))
        )

        stopped = subprocess.run([str(STOP)], env=env, text=True, capture_output=True)
        second.communicate(timeout=6)
        self.assertEqual(stopped.returncode, 0, stopped.stdout + stopped.stderr)
        self.assertTrue(
            wait_for(lambda: not any(process_is_live(pid) for pid in second_service_pids))
        )
        self.assertFalse((self.state / ".lofai.lock").exists())

    def test_stop_kills_group_descendant_after_its_leader_has_exited(self):
        child_file = self.state / "orphan.child"
        release_file = self.state / "release"
        code = """import os, subprocess, sys, time
from pathlib import Path
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
Path(sys.argv[1]).write_text(str(child.pid))
while not Path(sys.argv[2]).exists():
    time.sleep(0.02)
"""
        leader = subprocess.Popen(
            [os.sys.executable, "-c", code, str(child_file), str(release_file)],
            start_new_session=True,
        )
        self.assertTrue(wait_for(child_file.exists))
        child_pid = int(child_file.read_text())
        started = process_started_at(leader.pid)
        (self.state / "frontend.pid").write_text(
            f"group|{leader.pid}|{started}|{ROOT}\n"
        )

        release_file.touch()
        leader.wait(timeout=2)
        self.assertTrue(process_is_live(child_pid))

        env = os.environ.copy()
        env.update(
            LOFAI_STATE_DIR=str(self.state),
            LOFAI_STOP_TIMEOUT="1",
            LOFAI_KILL_TIMEOUT="1",
        )
        result = subprocess.run([str(STOP)], env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(wait_for(lambda: not process_is_live(child_pid)))
        self.assertFalse((self.state / "frontend.pid").exists())

    def test_stale_metadata_does_not_kill_a_reused_unrelated_pid(self):
        sentinel = subprocess.Popen(["sleep", "60"])
        try:
            (self.state / "backend.pid").write_text(
                f"pid|{sentinel.pid}|definitely not its start time|{ROOT}\n"
            )
            env = os.environ.copy()
            env["LOFAI_STATE_DIR"] = str(self.state)
            result = subprocess.run([str(STOP)], env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0)
            self.assertTrue(process_is_live(sentinel.pid))
            self.assertFalse((self.state / "backend.pid").exists())
        finally:
            sentinel.terminate()
            sentinel.wait(timeout=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
