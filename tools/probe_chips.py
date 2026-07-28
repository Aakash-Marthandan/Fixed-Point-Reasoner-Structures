# Chip-pinning shakedown: under the shard_run.sh env recipe each process must
# see EXACTLY ONE TPU device and be able to run a computation on it.
import os
import jax
import jax.numpy as jnp

n = jax.device_count()
x = float(jnp.sum(jnp.ones((256, 256)) @ jnp.ones((256, 256))))
print(f"chip={os.environ.get('TPU_VISIBLE_CHIPS', '?')} devices={n} "
      f"compute={'OK' if x == 256.0 * 256 * 256 else x} "
      f"({jax.devices()[0].platform})", flush=True)
assert n == 1, f"pinning failed: {n} devices visible"
