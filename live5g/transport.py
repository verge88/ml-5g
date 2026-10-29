"""Read-only HTTP/2 client restricted to fixed local lab ports."""
import asyncio
import json
import time
from h2.config import H2Configuration
from h2.connection import H2Connection
from h2.events import DataReceived, ResponseReceived, StreamEnded, StreamReset


def validate_target(port, path, host="127.0.0.1"):
    if host != "127.0.0.1" or port not in (7777, 7778, 7779):
        raise ValueError("only the fixed loopback lab is supported")
    if not path.startswith("/") or "\r" in path or "\n" in path:
        raise ValueError("invalid path")


async def request(port, path, timeout=0.4, host="127.0.0.1"):
    validate_target(port, path, host)
    started = time.monotonic_ns()

    async def once():
        reader, writer = await asyncio.open_connection(host, port)
        h = H2Connection(H2Configuration(client_side=True, header_encoding="utf-8"))
        try:
            h.initiate_connection()
            h.send_headers(1, [(":method", "GET"), (":scheme", "http"),
                              (":authority", f"{host}:{port}"), (":path", path),
                              ("accept", "application/json")], end_stream=True)
            writer.write(h.data_to_send())
            await writer.drain()
            body, status, nbytes = bytearray(), None, 0
            while True:
                data = await reader.read(65535)
                if not data:
                    raise ConnectionError("incomplete response")
                nbytes += len(data)
                for e in h.receive_data(data):
                    if isinstance(e, ResponseReceived):
                        status = int(dict(e.headers)[":status"])
                    elif isinstance(e, DataReceived):
                        body.extend(e.data)
                        h.acknowledge_received_data(e.flow_controlled_length, e.stream_id)
                    elif isinstance(e, StreamReset):
                        raise ConnectionError("stream reset")
                    elif isinstance(e, StreamEnded):
                        try:
                            doc = json.loads(body) if body else None
                        except (ValueError, UnicodeDecodeError):
                            doc = bytes(body).decode("utf-8", errors="replace")
                        return {"ok": True, "status": status, "body": doc, "received_bytes": nbytes}
                writer.write(h.data_to_send())
                await writer.drain()
        finally:
            writer.close()
            try:
                await asyncio.wait_for(writer.wait_closed(), 0.1)
            except (OSError, asyncio.TimeoutError):
                pass
    try:
        result = await asyncio.wait_for(once(), timeout)
    except Exception as exc:
        result = {"ok": False, "status": None, "body": None, "error": type(exc).__name__, "received_bytes": 0}
    result["elapsed_ms"] = (time.monotonic_ns() - started) / 1e6
    return result
