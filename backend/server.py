# fastapi server for lofai - one magenta realtime 2 stream per listener

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

import engine as engine_mod
import session_manager as manager_mod
import styles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("lofai")

# how much audio we hold for a client that has stopped draining. the lookahead
# cap means a healthy session never banks more than a few seconds here, so this
# is insurance against a wedged socket rather than a working buffer.
OUTBOX_LIMIT = max(4, int(30.0 / manager_mod.CHUNK_SECONDS))

manager = manager_mod.SessionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    manager.start()
    try:
        yield
    finally:
        manager.stop()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _offer(queue: asyncio.Queue, item):
    # Never block the worker thread. Return True if queueing would lose PCM so
    # the socket owner can reconnect instead of concealing a sequence hole.
    # Control messages are small and worth delivering; report if one evicts
    # audio for the same reason.
    if not queue.full():
        queue.put_nowait(item)
        return False

    if isinstance(item, bytes):
        return True

    try:
        removed = queue.get_nowait()
    except asyncio.QueueEmpty:
        removed = None
    queue.put_nowait(item)
    return isinstance(removed, bytes)


async def _drain(websocket: WebSocket, queue: asyncio.Queue):
    # the one writer for this socket - starlette websockets cannot be sent on
    # from two tasks at once, so control messages queue behind audio here
    try:
        while True:
            item = await queue.get()
            if isinstance(item, bytes):
                await websocket.send_bytes(item)
            else:
                await websocket.send_json(item)
    except (WebSocketDisconnect, RuntimeError):
        pass


@app.websocket("/ws/session")
async def session_socket(websocket: WebSocket):
    # one socket per listener: json for control, binary frames for pcm audio
    await websocket.accept()
    loop = asyncio.get_running_loop()
    outbox: asyncio.Queue = asyncio.Queue(maxsize=OUTBOX_LIMIT)
    closing_for_overflow = False
    closed = False
    helper_tasks: set[asyncio.Task] = set()

    async def close_for_overflow():
        try:
            await websocket.close(code=1013, reason="audio backlog")
        except RuntimeError:
            pass

    def deliver(item):
        nonlocal closing_for_overflow
        if closed or closing_for_overflow:
            return
        if _offer(outbox, item):
            # Never resume after silently dropping a middle PCM packet: that
            # would splice unrelated samples and click. Reconnect/reset is a
            # clean, recoverable discontinuity for a wedged client.
            closing_for_overflow = True
            task = asyncio.create_task(close_for_overflow())
            helper_tasks.add(task)

    try:
        hello = await asyncio.wait_for(websocket.receive_json(), timeout=10.0)
    except (asyncio.TimeoutError, WebSocketDisconnect, ValueError):
        await websocket.close(code=1002)
        return

    if not isinstance(hello, dict):
        await websocket.close(code=1002)
        return

    try:
        session, resumed = manager.attach(
            hello.get("sessionId"),
            hello.get("mood", styles.DEFAULT_MOOD),
            hello.get("instrument", styles.DEFAULT_INSTRUMENT),
            sink=lambda pcm: loop.call_soon_threadsafe(deliver, pcm),
            on_status=lambda payload: loop.call_soon_threadsafe(deliver, payload),
        )
    except RuntimeError:
        await websocket.close(code=1013, reason="backend unavailable")
        return
    log.info("%s session %s", "resumed" if resumed else "opened", session.id[:8])

    _offer(
        outbox,
        {
            "type": "hello",
            "sessionId": session.id,
            "resumed": resumed,
            "sampleRate": engine_mod.SAMPLE_RATE,
            "channels": engine_mod.CHANNELS,
            "chunkSeconds": manager_mod.CHUNK_SECONDS,
            "mood": session.mood,
            "instrument": session.instrument,
        },
    )
    _offer(outbox, manager.status_for(session))

    drain = asyncio.create_task(_drain(websocket, outbox))
    try:
        while True:
            message = await websocket.receive_json()
            if not isinstance(message, dict):
                continue

            kind = message.get("type")
            session.touch()

            if kind == "style":
                mood, instrument = session.request_style(
                    message.get("mood", session.mood),
                    message.get("instrument", session.instrument),
                )
                _offer(outbox, {"type": "style", "mood": mood, "instrument": instrument})

            elif kind == "pause":
                manager.suspend(session, preserve_audio=True)

            elif kind == "resume":
                manager.resume(session)

            elif kind == "gap":
                # the client ran its reservoir dry and had to refill. only it
                # can tell us this - the server has no idea what was audible -
                # so it is also the one honest input the quality tuner gets.
                manager.report_gap(session)

            elif kind == "pressure":
                manager.report_pressure()

            elif kind == "ping":
                _offer(outbox, {"type": "pong"})

    except (WebSocketDisconnect, ValueError, RuntimeError):
        pass
    finally:
        closed = True
        drain.cancel()
        session.sink = None
        session.on_status = None
        # the state stays warm for MRT_SESSION_TTL so a reconnect or an unpause
        # picks the same music back up
        manager.suspend(session)
        tasks = (drain, *helper_tasks)
        for task in tasks:
            task.cancel()
        # gather retrieves every exception as well as waiting for cancellation;
        # no helper task can outlive its socket or become an unobserved error.
        await asyncio.gather(*tasks, return_exceptions=True)
        log.info("detached session %s", session.id[:8])


@app.get("/health")
async def health_check():
    return {"status": "ok", **manager.stats()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, ws_per_message_deflate=False)
