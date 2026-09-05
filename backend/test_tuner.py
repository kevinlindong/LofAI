# the tuner's control law, without needing a slow machine to see it
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E
import session as S
import session_manager as M

class FakeConfig:
    num_codebooks = 12
    num_active_codebooks = None

def fresh(target=1.15):
    e = E.MRTEngine()
    e.target_rtf = target
    e._depth_config = FakeConfig()
    e.max_codebooks = 12
    e.codebooks = 12
    e._fast = True
    return e

ok = []
def check(name, cond, detail=""):
    ok.append(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {name}{'  — ' + detail if detail else ''}")

# a listener-reported gap spends a codebook straight away
e = fresh()
e.note_gap()
check("a reported gap costs a codebook", e.codebooks == 11, f"now {e.codebooks}")

# ...and cannot dig below the floor
for _ in range(20): e.note_gap()
check("gaps stop at the floor", e.codebooks == E.MIN_CODEBOOKS, f"now {e.codebooks}")
check("floor is applied to the live config", e._depth_config.num_active_codebooks == E.MIN_CODEBOOKS)

# slow renders walk it down, one step per dwell
e = fresh()
slow = 0.050  # 50ms/frame -> 0.8x
steps = []
t = 0.0
for i in range(60):
    e._costs.append(slow)
    e._retune(t)
    steps.append(e.codebooks)
    t += 1.0
check("sustained slowness walks down to the floor", e.codebooks == E.MIN_CODEBOOKS, f"path {steps[0]}->{e.codebooks}")
changes = sum(1 for a, b in zip(steps, steps[1:]) if a != b)
check("it steps rather than lurches", changes == 12 - E.MIN_CODEBOOKS, f"{changes} changes over 60s")

# fast renders give detail back, but only with real headroom
e = fresh(); e.codebooks = 8; e._depth_config.num_active_codebooks = 8
t = 0.0
for i in range(80):
    e._costs.append(0.040 / 1.50)   # 1.50x - comfortably over 1.15*1.3
    e._retune(t); t += 1.0
check("real headroom restores detail", e.codebooks == 12, f"now {e.codebooks}")

# a factor that merely clears the target must NOT climb (that would oscillate)
e = fresh(); e.codebooks = 8; e._depth_config.num_active_codebooks = 8
t = 0.0
for i in range(80):
    e._costs.append(0.040 / 1.20)   # 1.20x: above target, below the raise line
    e._retune(t); t += 1.0
check("a bare pass does not climb", e.codebooks == 8, f"now {e.codebooks}")

# pinned means pinned
e = fresh(); e.pinned_codebooks = 10; e.set_codebooks(10)
e.note_gap()
for i in range(40):
    e._costs.append(0.060)
    e._retune(float(i))
check("a pinned count never moves", e.codebooks == 10, f"now {e.codebooks}")

# low reservoir pressure acts before a gap, but respects the dwell
e = fresh(); e._last_tune = 100.0
e._seed_cost(0.040 / 1.0)
from unittest.mock import patch
with patch.object(E.time, "monotonic", return_value=107.0):
    e.note_pressure()
    e.note_pressure()
check("low reservoir proactively costs one codebook", e.codebooks == 11, f"now {e.codebooks}")
e = fresh(); e._last_tune = 100.0; e._seed_cost(0.020)
with patch.object(E.time, "monotonic", return_value=107.0):
    e.note_pressure()
check("socket pressure cannot degrade a fast renderer", e.codebooks == 12)
e = fresh(); e._last_tune = 100.0; e._seed_cost(0.040 / 1.19)
with patch.object(E.time, "monotonic", return_value=107.0):
    e.note_pressure()
check("normal startup sawtooth keeps measured headroom", e.codebooks == 12)

# Retuning clears old-depth samples, but the UI must not briefly see 0x speed.
e = fresh(); e._seed_cost(0.040 / 1.19)
e._clear_costs()
check(
    "quality changes retain the last displayed speed",
    abs(e.realtime_factor() - 1.19) < 1e-9 and not e.throughput_ready(),
    f"now {e.realtime_factor():.2f}x",
)

# An explicit lower floor remains possible for controlled quality comparisons.
e = fresh(); e.min_codebooks = E.ABSOLUTE_MIN_CODEBOOKS
for _ in range(20): e.note_gap()
check("an explicit experimental floor is honored",
      e.codebooks == E.ABSOLUTE_MIN_CODEBOOKS, f"now {e.codebooks}")

# the smoother ignores a single bad chunk
e = fresh()
for _ in range(30): e.note_render(50, 50 * 0.030)   # steady 1.33x
before = e.realtime_factor()
e.note_render(50, 50 * 0.200)                        # one six-times-slow chunk
after = e.realtime_factor()
check("one slow chunk does not move the estimate",
      abs(after - before) < 1e-9, f"{before:.2f}x -> {after:.2f}x")

# ...but a machine that is genuinely slow still gets noticed
e = fresh()
for _ in range(30): e.note_render(50, 50 * 0.030)
for _ in range(6): e.note_render(50, 50 * 0.055)     # six slow chunks running
check("a sustained slowdown is noticed",
      e.realtime_factor() < 0.8, f"now {e.realtime_factor():.2f}x")

# an outlier every other chunk still must not drag it under
e = fresh()
for i in range(40): e.note_render(50, 50 * (0.200 if i % 3 == 0 else 0.030))
check("one bad chunk in three is still judged on the good ones",
      e.realtime_factor() > 1.25, f"now {e.realtime_factor():.2f}x")

# pause keeps the exact amount of already-generated audio, while a disconnect
# correctly discards that client-side lead
s = S.Session("test", "neutral", "guitar")
with patch.object(S.time, "monotonic", side_effect=[100.0, 100.0, 100.5, 200.0]):
    s.start_clock()
    s.note_generated(2.0)
    s.stop_clock(preserve_audio=True)
    s.start_clock()
check("pause preserves generated lead", abs(s.playhead - 201.5) < 1e-9,
      f"playhead {s.playhead:.1f}")
with patch.object(S.time, "monotonic", return_value=201.0):
    s.stop_clock(preserve_audio=False)
with patch.object(S.time, "monotonic", return_value=300.0):
    s.start_clock()
check("disconnect discards client-only lead", s.playhead == 300.0,
      f"playhead {s.playhead:.1f}")

# feedback raised on the web thread is coalesced and only applied by the worker
class FakeEngine:
    def __init__(self): self.gap = 0; self.pressure = 0
    def note_gap(self): self.gap += 1
    def note_pressure(self): self.pressure += 1

m = M.SessionManager(); m.engine = FakeEngine(); m._running = True
m.report_pressure(); m.report_pressure(); m._apply_feedback()
check("pressure reports coalesce on worker", m.engine.pressure == 1)
m._sessions[s.id] = s
m.report_pressure(); m.report_gap(s); m._apply_feedback()
check("gap supersedes simultaneous pressure", m.engine.gap == 1 and m.engine.pressure == 1)

# Aggregate model speed must be divided among concurrent listeners when the
# quality controller decides whether there is real-time headroom.
e = fresh(); e._seed_cost(0.040 / 1.8); e._active_streams = 2
per_listener = e.effective_realtime_factor()
e._retune(7.0)
check("two listeners tune against per-listener speed", e.codebooks == 11,
      f"effective {per_listener:.2f}x")

# connected pauses remain valid beyond the detached-session TTL
paused = S.Session("paused", "neutral", "guitar")
paused.status = S.SUSPENDED; paused.last_seen = 0.0; paused.sink = lambda _pcm: None
m = M.SessionManager(); m._sessions[paused.id] = paused
with patch.object(M.time, "monotonic", return_value=M.SESSION_TTL + 10.0):
    m._evict_stale()
check("connected pause is not reaped", paused.id in m._sessions)
paused.sink = None
with patch.object(M.time, "monotonic", return_value=M.SESSION_TTL + 10.0):
    m._evict_stale()
check("detached stale session is reaped", paused.id not in m._sessions)

# style slices use midpoint conditioning and never cross the ramp endpoint
class FakeStyleEngine:
    def embed(self, prompt):
        return __import__("numpy").array([0.0 if "guitar" in prompt else 1.0], dtype="float32")

sty = S.Session("style", "neutral", "guitar"); fake = FakeStyleEngine()
sty.style_plan(fake, 1)
sty.request_style("neutral", "piano")
plan = sty.style_plan(fake, 25)
values = [float(style[0]) for style, _key, _frames in plan]
check(
    "style ramp uses segment midpoints",
    0.0 < values[0] < values[-2] < values[-1]
    and abs(values[-1] - 1.0) < 1e-6
    and values == sorted(values),
    f"values {values}",
)
check("style plan preserves frame count", sum(frames for _s, _k, frames in plan) == 25)

print(f"\n{sum(ok)}/{len(ok)} passed")
if __name__ == "__main__":
    sys.exit(0 if all(ok) else 1)
if not all(ok):
    raise AssertionError("standalone tuner checks failed during discovery")
