"""CPU simulator retaining HW8-10 backend(manifest, root)/warmup/generate contract."""
import asyncio
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import time
from release import canonical, digest, verify_artifact
from previous_service import StubBackend


class SimulatedBackend(StubBackend):
    def __init__(self, manifest, root, config):
        verify_artifact(manifest, root)
        if digest(canonical(manifest["runtime"])) != manifest["runtime_sha256"]:
            raise ValueError("Runtime fingerprint mismatch")
        super().__init__(manifest, root)
        self.manifest = manifest
        self.config = config
        self.cache = OrderedDict()

    async def compute(self, seconds):
        # Wall-clock bounded CPU work, yielding every ~4ms for HTTP/probes.
        end = time.perf_counter() + seconds
        block = b'cpu-simulator' * 512
        while time.perf_counter() < end:
            until = min(end, time.perf_counter() + .004)
            while time.perf_counter() < until:
                block = hashlib.sha256(block).digest() * 256
            await asyncio.sleep(0)

    async def prepare(self, history):
        # Explicit simulated local prefix cache. History remains client-owned.
        key = hashlib.sha256(history[0]['content'].encode()).hexdigest()
        hit = key in self.cache
        await self.compute(self.config['hit_seconds'] if hit else self.config['miss_seconds'])
        self.cache[key] = True
        self.cache.move_to_end(key)
        while len(self.cache) > self.config['cache_capacity']:
            self.cache.popitem(last=False)
        return hit

    def generate(self, prompt, limit):
        self.calls += 1
        return {'answer': f'SIMULATED {self.loaded_id}: {prompt}', 'output_chunks': limit,
                'backend_mode': 'simulated'}
