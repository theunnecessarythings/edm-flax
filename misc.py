import numpy as np
import jax.numpy as jnp
import torch
from typing import Any


#----------------------------------------------------------------------------
# Cached construction of constant tensors. Avoids CPU=>GPU copy when the
# same constant is used multiple times.

_constant_cache = dict()

class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]


def constant(value, shape=None, dtype=None):
    value = np.asarray(value)
    if shape is not None:
        shape = tuple(shape)
    if dtype is None:
        dtype = jnp.float32
    
    key = (value.shape, value.dtype, value.tobytes(), shape, dtype)
    array = _constant_cache.get(key, None)
    if array is None:
        array = jnp.asarray(value.copy(), dtype=dtype)
        if shape is not None:
            array = jnp.broadcast_to(array, shape)
        _constant_cache[key] = array
    return array


# Sampler for torch.utils.data.DataLoader that loops over the dataset
# indefinitely, shuffling items as it goes.

class InfiniteSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, rank=0, num_replicas=1, shuffle=True, seed=0, window_size=0.5):
        assert len(dataset) > 0
        assert num_replicas > 0
        assert 0 <= rank < num_replicas
        assert 0 <= window_size <= 1
        super().__init__(dataset)
        self.dataset = dataset
        self.rank = rank
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.window_size = window_size

    def __iter__(self):
        order = np.arange(len(self.dataset))
        rnd = None
        window = 0
        if self.shuffle:
            rnd = np.random.RandomState(self.seed)
            rnd.shuffle(order)
            window = int(np.rint(order.size * self.window_size))

        idx = 0
        while True:
            i = idx % order.size
            if idx % self.num_replicas == self.rank:
                yield order[i]
            if window >= 2:
                j = (i - rnd.randint(window)) % order.size
                order[i], order[j] = order[j], order[i]
            idx += 1


def linspace_from_neg_one(grid, num_steps, align_corners):
    if num_steps <= 1:
        return 0
    range = jnp.linspace(-1, 1, num_steps)
    if not align_corners:
        range = range * (num_steps - 1) / num_steps
    return range

def make_base_grid_4D(theta, N, C, H, W, align_corners):
    base_grid = jnp.empty((N, H, W, 3))
    base_grid = base_grid.at[..., 0].set(linspace_from_neg_one(theta, W, align_corners))
    base_grid = base_grid.at[..., 1].set(jnp.expand_dims(linspace_from_neg_one(theta, H, align_corners), -1))
    base_grid = base_grid.at[..., 2].set(jnp.full_like(base_grid[..., 2], 1))
    return base_grid

def make_base_grid_5D(theta, N, C, D, H, W, align_corners):
    base_grid = jnp.empty((N, D, H, W, 4))
    base_grid = base_grid.at[..., 0].set(linspace_from_neg_one(theta, W, align_corners))
    base_grid = base_grid.at[..., 1].set(jnp.expand_dims(linspace_from_neg_one(theta, H, align_corners), -1))
    base_grid = base_grid.at[..., 2].set(jnp.expand_dims(jnp.expand_dims(linspace_from_neg_one(theta, D, align_corners), -1), -1))
    base_grid = base_grid.at[..., 3].set(jnp.full_like(base_grid[..., 3], 1))
    return base_grid

def affine_grid_generator_4D(theta, N, C, H, W, align_corners):
    base_grid = make_base_grid_4D(theta, N, C, H, W, align_corners)
    grid = jnp.matmul(jnp.reshape(base_grid, (N, H * W, 3)), jnp.transpose(theta, (0, 2, 1)))
    return jnp.reshape(grid, (N, H, W, 2))

def affine_grid_generator_5D(theta, N, C, D, H, W, align_corners):
    base_grid = make_base_grid_5D(theta, N, C, D, H, W, align_corners)
    grid = jnp.matmul(jnp.reshape(base_grid, (N, D * H * W, 4)), jnp.transpose(theta, (0, 2, 1, 3)))
    return jnp.reshape(grid, (N, D, H, W, 3))

def affine_grid_generator(theta, size, align_corners):
    print(size)
    if len(size) != 4 and len(size) != 5:
        raise ValueError('AffineGridGenerator needs 4d (spatial) or 5d (volumetric) inputs')
    if len(size) == 4:
        return affine_grid_generator_4D(theta, size[0], size[1], size[2], size[3], align_corners)
    else:
        return affine_grid_generator_5D(theta, size[0], size[1], size[2], size[3], size[4], align_corners)
        