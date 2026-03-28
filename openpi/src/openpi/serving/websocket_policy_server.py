import asyncio
import logging
import traceback

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server
import websockets.frames
import jax
import numpy as np

class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently implements the `load`, `infer`, and `get_prefix_rep` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection):
        logging.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        while True:
            try:
                message = msgpack_numpy.unpackb(await websocket.recv())
                method = message.get("method", "infer")  # default to infer for backward compatibility
                obs = message.get("obs", message)  # if no method specified, assume old format
                
                if method == "infer":
                    noise = obs.pop("noise", None)
                    result = self._policy.infer(obs, noise)
                elif method == "get_prefix_rep":
                    result = self._policy.get_prefix_rep(obs)
                else:
                    raise ValueError(f"Unknown method: {method}")
                # convert result to numpy array
                result = jax.tree.map(lambda x: np.asarray(x).astype(np.float32), result)
                await websocket.send(packer.pack(result))
            except websockets.ConnectionClosed:
                logging.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise