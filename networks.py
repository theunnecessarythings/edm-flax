import jax 
import jax.numpy as jnp
import flax.linen as nn

from typing import Optional, Any, List, Union, Tuple, Sequence
from dataclasses import field
import numpy as np
from jax.lax import ConvGeneralDilatedDimensionNumbers, conv_dimension_numbers, conv_general_dilated, index_in_dim

from collections import OrderedDict

Array = Any
PrecisionType = Any
PrecisionLike = Union[None, PrecisionType, Tuple[PrecisionType, PrecisionType]]

def _flip_axes(x, axes):
    """Flip ndarray 'x' along each axis specified in axes tuple."""
    for axis in axes:
        x = np.flip(x, axis)
    return x

def _unstack(x, axis=0):
    return [index_in_dim(x, i, axis, keepdims=False) for i in range(x.shape[axis])]

def _deconv_output_length(input_length,
                            filter_size,
                            padding,
                            output_padding=None,
                            stride=0,
                            dilation=1):
    """Determines the output length of a transposed convolution given the input length.
    Function modified from Keras.
    Arguments:
        input_length: Integer.
        filter_size: Integer.
        padding: one of `"SAME"`, `"VALID"`, or a 2-integer tuple.
        output_padding: Integer, amount of padding along the output dimension. Can
            be set to `None` in which case the output length is inferred.
        stride: Integer.
        dilation: Integer.
    Returns:
        The output length (integer).
    """
    if input_length is None:
        return None

    # Get the dilated kernel size
    filter_size = filter_size + (filter_size - 1) * (dilation - 1)

    # Infer length if output padding is None, else compute the exact length
    if output_padding is None:
        if padding == 'VALID':
            length = input_length * stride + max(filter_size - stride, 0)
        elif padding == 'SAME':
            length = input_length * stride
        else:
            length = ((input_length - 1) * stride + filter_size
                        - padding[0] - padding[1])

    else:
        if padding == 'SAME':
            pad = filter_size // 2
            total_pad = pad * 2
        elif padding == 'VALID':
            total_pad = 0
        else:
            total_pad = padding[0] + padding[1]

        length = ((input_length - 1) * stride + filter_size - total_pad +
                    output_padding)

    return length


def _compute_adjusted_padding(
        input_size: int,
        output_size: int,
        kernel_size: int,
        stride: int,
        padding: Union[str, Tuple[int, int]],
        dilation: int = 1,
) -> Tuple[int, int]:
    """Computes adjusted padding for desired ConvTranspose `output_size`.
    Ported from DeepMind Haiku.
    """
    kernel_size = (kernel_size - 1) * dilation + 1
    if padding == "VALID":
        expected_input_size = (output_size - kernel_size + stride) // stride
        if input_size != expected_input_size:
            raise ValueError(f"The expected input size with the current set of input "
                                f"parameters is {expected_input_size} which doesn't "
                                f"match the actual input size {input_size}.")
        padding_before = 0
    elif padding == "SAME":
        expected_input_size = (output_size + stride - 1) // stride
        if input_size != expected_input_size:
            raise ValueError(f"The expected input size with the current set of input "
                                f"parameters is {expected_input_size} which doesn't "
                                f"match the actual input size {input_size}.")
        padding_needed = max(0,
                                                 (input_size - 1) * stride + kernel_size - output_size)
        padding_before = padding_needed // 2
    else:
        padding_before = padding[0]    # type: ignore[assignment]

    expanded_input_size = (input_size - 1) * stride + 1
    padded_out_size = output_size + kernel_size - 1
    pad_before = kernel_size - 1 - padding_before
    pad_after = padded_out_size - expanded_input_size - pad_before
    return (pad_before, pad_after)


def gradient_based_conv_transpose(lhs: Array, rhs: Array, strides: Sequence[int],
                                    padding: Union[str, Sequence[Tuple[int, int]]],
                                    output_padding: Optional[Sequence[int]] = None,
                                    output_shape: Optional[Sequence[int]] = None,
                                    dilation: Optional[Sequence[int]] = None,
                                    dimension_numbers: ConvGeneralDilatedDimensionNumbers = None,
                                    transpose_kernel: bool = True,
                                    feature_group_count:int = 1,
                                    precision: PrecisionLike = None) -> Array:
    """Convenience wrapper for calculating the N-d transposed convolution.
    Much like `conv_transpose`, this function calculates transposed convolutions
    via fractionally strided convolution rather than calculating the gradient
    (transpose) of a forward convolution. However, the latter is more common
    among deep learning frameworks, such as TensorFlow, PyTorch, and Keras.
    This function provides the same set of APIs to help reproduce results in these frameworks.
    Args:
        lhs: a rank `n+2` dimensional input array.
        rhs: a rank `n+2` dimensional array of kernel weights.
        strides: sequence of `n` integers, amounts to strides of the corresponding forward convolution.
        padding: `"SAME"`, `"VALID"`, or a sequence of `n` integer 2-tuples that controls
            the before-and-after padding for each `n` spatial dimension of
            the corresponding forward convolution.
        output_padding: A sequence of integers specifying the amount of padding along
            each spacial dimension of the output tensor, used to disambiguate the output shape of
            transposed convolutions when the stride is larger than 1.
            (see a detailed description at
            1https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html)
            The amount of output padding along a given dimension must
            be lower than the stride along that same dimension.
            If set to `None` (default), the output shape is inferred.
            If both `output_padding` and `output_shape` are specified, they have to be mutually compatible.
        output_shape: Output shape of the spatial dimensions of a transpose
            convolution. Can be `None` or an iterable of `n` integers. If a `None` value is given (default),
            the shape is automatically calculated.
            Similar to `output_padding`, `output_shape` is also for disambiguating the output shape
            when stride > 1 (see also
            https://www.tensorflow.org/api_docs/python/tf/nn/conv2d_transpose)
            If both `output_padding` and `output_shape` are specified, they have to be mutually compatible.
        dilation: `None`, or a sequence of `n` integers, giving the
            dilation factor to apply in each spatial dimension of `rhs`. Dilated convolution
            is also known as atrous convolution.
        dimension_numbers: tuple of dimension descriptors as in
            lax.conv_general_dilated. Defaults to tensorflow convention.
        transpose_kernel: if `True` flips spatial axes and swaps the input/output
            channel axes of the kernel. This makes the output of this function identical
            to the gradient-derived functions like keras.layers.Conv2DTranspose and
            torch.nn.ConvTranspose2d applied to the same kernel.
            Although for typical use in neural nets this is unnecessary
            and makes input/output channel specification confusing, you need to set this to `True`
            in order to match the behavior in many deep learning frameworks, such as TensorFlow, Keras, and PyTorch.
        precision: Optional. Either ``None``, which means the default precision for
            the backend, a ``lax.Precision`` enum value (``Precision.DEFAULT``,
            ``Precision.HIGH`` or ``Precision.HIGHEST``) or a tuple of two
            ``lax.Precision`` enums indicating precision of ``lhs``` and ``rhs``.
    Returns:
        Transposed N-d convolution.
    """
    assert len(lhs.shape) == len(rhs.shape) and len(lhs.shape) >= 2
    ndims = len(lhs.shape)
    one = (1,) * (ndims - 2)
    # Set dimensional layout defaults if not specified.
    if dimension_numbers is None:
        if ndims == 2:
            dimension_numbers = ('NC', 'IO', 'NC')
        elif ndims == 3:
            dimension_numbers = ('NHC', 'HIO', 'NHC')
        elif ndims == 4:
            dimension_numbers = ('NHWC', 'HWIO', 'NHWC')
        elif ndims == 5:
            dimension_numbers = ('NHWDC', 'HWDIO', 'NHWDC')
        else:
            raise ValueError('No 4+ dimensional dimension_number defaults.')
    dn = conv_dimension_numbers(lhs.shape, rhs.shape, dimension_numbers)
    k_shape = np.take(rhs.shape, dn.rhs_spec)
    k_sdims = k_shape[2:]    # type: ignore[index]
    i_shape = np.take(lhs.shape, dn.lhs_spec)
    i_sdims = i_shape[2:]    # type: ignore[index]

    # Calculate correct output shape given padding and strides.
    if dilation is None:
        dilation = (1,) * (rhs.ndim - 2)

    if output_padding is None:
        output_padding = [None] * (rhs.ndim - 2)    # type: ignore[list-item]

    if isinstance(padding, str):
        if padding in {'SAME', 'VALID'}:
            padding = [padding] * (rhs.ndim - 2)    # type: ignore[list-item]
        else:
            raise ValueError(f"`padding` must be 'VALID' or 'SAME'. Passed: {padding}.")

    inferred_output_shape = tuple(map(_deconv_output_length, i_sdims, k_sdims,
                                        padding, output_padding, strides, dilation))
    if output_shape is None:
        output_shape = inferred_output_shape    # type: ignore[assignment]
    else:
        if not output_shape == inferred_output_shape:
            raise ValueError(f"`output_padding` and `output_shape` are not compatible."
                    f"Inferred output shape from `output_padding`: {inferred_output_shape}, "
                    f"but got `output_shape` {output_shape}")

    pads = tuple(map(_compute_adjusted_padding, i_sdims, output_shape,
                                    k_sdims, strides, padding, dilation))

    if transpose_kernel:
        # flip spatial dims and swap input / output channel axes
        rhs = _flip_axes(rhs, np.array(dn.rhs_spec)[2:])
        rhs = np.swapaxes(rhs, dn.rhs_spec[0], dn.rhs_spec[1])
    return conv_general_dilated(lhs, rhs, one, pads, strides, dilation, dn, feature_group_count, precision=precision)

#----------------------------------------------------------------------------
# Attention weight computation, i.e., softmax(Q^T * K).
# Performs all computation using FP32, but uses the original datatype for
# inputs/outputs/gradients to conserve memory.
def attention_op(q, k):
    w = nn.softmax(jnp.einsum('ncq,nck->nqk', q.astype(jnp.float32), (k/jnp.sqrt(k.shape[1])).astype(jnp.float32)), axis=2).astype(q.dtype)
    return w

def weight_init():
    def init(key, shape, mode, fan_in, fan_out, weight, dtype):
        if mode == 'xavier_uniform':
            return jnp.sqrt(6 / (fan_in + fan_out)) * (jax.random.uniform(key, shape, dtype)*2 - 1) * weight
        if mode == 'xavier_normal':
            return jnp.sqrt(2 / (fan_in + fan_out)) * (jax.random.normal(key, shape, dtype)) * weight
        if mode == 'kaiming_uniform':
            return jnp.sqrt(3 / fan_in) * (jax.random.uniform(key, shape, dtype) * 2 - 1) * weight
        if mode == 'kaiming_normal':
            return jnp.sqrt(1 / fan_in) * jax.random.normal(key, shape, dtype) * weight
        raise ValueError(f'Invalid init mode "{mode}"')
    return init


class Linear(nn.Module):
    features: int 
    use_bias: bool = True
    init_mode: str = 'kaiming_normal'
    init_weight: float = 1
    init_bias: float = 0
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs):
        in_features = jnp.shape(inputs)[-1]
        weight = self.param('weight', weight_init(), (in_features, self.features), self.init_mode, in_features, self.features, self.init_weight, self.dtype)
        if self.use_bias:
            bias = self.param('bias', weight_init(), (self.features,), self.init_mode, in_features, self.features, self.init_bias, self.dtype) if self.use_bias else None

        x = jnp.matmul(inputs, weight)
        if self.use_bias:
            x = x + bias
        return x


class Conv2d(nn.Module):
    channels: int
    kernel: int
    use_bias: bool = True
    up: bool = False
    down: bool = False
    resample_filter: List[float] = None
    fused_resample: bool = False
    init_mode: str = 'kaiming_normal'
    init_weight: float = 1
    init_bias: float = 0
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        in_channels = jnp.shape(inputs)[-1]
        dn = ('NHWC', 'HWIO', 'NHWC')
        fan_in = in_channels*self.kernel*self.kernel
        fan_out = self.channels*self.kernel*self.kernel
        weight = self.param('weight', weight_init(), [self.kernel, self.kernel, in_channels, self.channels], self.init_mode, fan_in, fan_out, self.init_weight, self.dtype) if self.kernel else None

        bias = self.param('bias', weight_init(), [self.channels], self.init_mode, fan_in, fan_out, self.init_bias, self.dtype) if self.kernel and self.use_bias else None

        f = jnp.asarray(self.resample_filter, self.dtype)
        f = jnp.expand_dims(jnp.expand_dims(jnp.outer(f, f), -1), -1) / (f.sum() ** 2)
        f = f if self.up or self.down else None

        self.sow('params', 'resample_filter', f if self.up or self.down else None)

        w_pad = weight.shape[1] // 2 if weight is not None else 0
        f_pad = (f.shape[1] - 1) // 2 if f is not None else 0
        repeat = [1, 1, 1, in_channels]
    
        if self.fused_resample and self.up and weight is not None:
            p = max(f_pad - w_pad, 0)
            x = gradient_based_conv_transpose(x, jnp.tile(f.__mul__(4), repeat), strides=(2, 2), padding=((p, p), (p, p)), feature_group_count=in_channels, transpose_kernel=False)
            p = w_pad + f_pad
            x = conv_general_dilated(x, weight, padding=((p, p), (p, p)), window_strides=(1, 1), dimension_numbers=dn)
        elif self.fused_resample and self.down and weight is not None:
            p = w_pad + f_pad
            x = conv_general_dilated(x, weight, padding=((p,p), (p,p)), window_strides=(1, 1), dimension_numbers=dn)
            x = conv_general_dilated(x, jnp.tile(f, [1, 1, 1, self.channels]), window_strides=(2, 2), feature_group_count=self.channels, padding='SAME', dimension_numbers=dn)
        else:
            if self.up:
                x = gradient_based_conv_transpose(x, jnp.tile(f.__mul__(4), repeat), strides=(2, 2), padding=((f_pad, f_pad), (f_pad, f_pad)), feature_group_count=in_channels, transpose_kernel=False)
            if self.down:
                x = conv_general_dilated(x, jnp.tile(f, repeat), window_strides=(2, 2), padding=((f_pad, f_pad), (f_pad, f_pad)), feature_group_count=in_channels, dimension_numbers=dn)
            if weight is not None:
                x = conv_general_dilated(x, weight, padding=((w_pad, w_pad), (w_pad, w_pad)), window_strides=(1,1), dimension_numbers=dn)
        if bias is not None:
            x += (bias.reshape(1, 1, 1, -1))
        
        return x

def group_norm(num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5, dtype=jnp.float32, name=None):
    num_groups = min(num_groups, num_channels // min_channels_per_group)
    return nn.GroupNorm(num_groups=num_groups, epsilon=eps, dtype=dtype, name=name)


#----------------------------------------------------------------------------
# Unified U-Net block with optional up/downsampling and self-attention.
# Represents the union of all features employed by the DDPM++, NCSN++, and
# ADM architectures.

class UNetBlock(nn.Module):
    channels: int
    emb_channels: int 
    up: bool = False
    down: bool = False
    attention: bool = False
    num_heads: int = None
    channels_per_head: int = 64
    dropout: float = 0
    skip_scale: int = 1
    eps: float = 1e-5
    resample_filter: List[int] = field(default_factory=lambda: [1, 1])
    resample_proj: bool = False
    adaptive_scale: bool = True
    init: dict = None
    init_zero: dict = None
    init_attn: Any = None
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs, emb, train=True):
        num_heads = 0 if not self.attention else self.num_heads if self.num_heads is not None else self.channels // self.channels_per_head

        in_channels = jnp.shape(inputs)[-1]
        
        x = inputs
        x = group_norm(num_channels=in_channels, eps=self.eps, name='norm0')(x)
        x = nn.silu(x)

        x = Conv2d(channels=self.channels, kernel=3, up=self.up, down=self.down, resample_filter=self.resample_filter, dtype=self.dtype, name='conv0', **self.init)(x)
        

        params = Linear(features=self.channels*(2 if self.adaptive_scale else 1), dtype=self.dtype, name='affine', **self.init)(emb)
        params = jnp.expand_dims(jnp.expand_dims(params, 1), 2)

        if self.adaptive_scale:
            scale, shift = jnp.split(params, 2, 3)
            x = group_norm(num_channels=self.channels, eps=self.eps, dtype=self.dtype, name='norm1')(x)
            x = nn.silu(shift + x * (scale + 1))
        else:
            x += params
            x = group_norm(num_channels=self.channels, eps=self.eps, dtype=self.dtype, name='norm1')(x)
            x = nn.silu(x)
        
        x = nn.Dropout(self.dropout)(x, deterministic=not train)
        x = Conv2d(channels=self.channels, kernel=3, dtype=self.dtype, name='conv1', **self.init_zero)(x)

        skip = None
        if self.channels != in_channels or self.up or self.down:
            # Skip
            kernel = 1 if self.resample_proj or self.channels != in_channels else 0
            
            skip = Conv2d(channels=self.channels, kernel=kernel, up=self.up, down=self.down, resample_filter=self.resample_filter, dtype=self.dtype, name='skip', **self.init)(inputs)
        if skip is not None:
            x += skip
        else:
            x += inputs
        x *= self.skip_scale

        if num_heads:
            g = group_norm(num_channels=self.channels, eps=self.eps, dtype=self.dtype, name='norm2')(x)
            g = Conv2d(channels=self.channels * 3, kernel=1, name='qkv', **(self.init_attn if self.init_attn is not None else self.init))(g)
            g = jnp.transpose(g, (0, 3, 1, 2))
            qkv = jnp.reshape(g, (x.shape[0] * num_heads, x.shape[-1] // num_heads, 3, -1))
            q, k, v = _unstack(qkv, axis=2)
            w = attention_op(q, k)
            a = jnp.einsum('nqk,nck->ncq', w, v)
            a = jnp.reshape(a, (x.shape[0], x.shape[3], x.shape[1], x.shape[2]))
            a = jnp.transpose(a, (0, 2, 3, 1))
            x += Conv2d(channels=self.channels, kernel=1, dtype=self.dtype, name='proj', **self.init_zero)(a)
            x *= self.skip_scale
        return x

# Timestep embedding used in the DDPM++ and ADM architectures.
class PositionalEmbedding(nn.Module):
    num_channels: int
    max_positions: int = 10000
    endpoint: bool = False

    @nn.compact
    def __call__(self, inputs):
        freqs = jnp.arange(start=0, stop=self.num_channels//2, dtype=jnp.float32)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = jnp.outer(inputs, freqs.astype(inputs.dtype))
        x = jnp.concatenate([jnp.cos(x), jnp.sin(x)], axis=1)
        return x

# Timestep embedding used in the NCSN++ architecture.
class FourierEmbedding(nn.Module):
    num_channels: int
    scale: int = 16
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs):
        key = self.make_rng('params')
        freqs = jax.random.normal(key, (self.num_channels // 2,), dtype=self.dtype) * self.scale

        x = jnp.outer(inputs, 2 * jnp.pi * freqs).astype(inputs.dtype)
        x = jnp.concatenate([jnp.cos(x), jnp.sin(x)], axis=1)
        return x


#----------------------------------------------------------------------------
# Reimplementation of the DDPM++ and NCSN++ architectures from the paper
# "Score-Based Generative Modeling through Stochastic Differential
# Equations". Equivalent to the original implementation by Song et al.,
# available at https://github.com/yang-song/score_sde_pytorch

class SongUNet(nn.Module):
    img_resolution: int
    channels: int 
    label_dim: int = 0 
    augment_dim: int = 0
    model_channels: int = 128
    channel_mult = [1, 2, 2, 2]
    channel_mult_emb: int = 4
    num_blocks: int = 4
    attn_resolutions = [16]
    dropout: float = 0.10
    label_dropout: float = 0
    embedding_type: str = 'positional'
    channel_mult_noise: int = 1
    encoder_type: str = 'standard'
    decoder_type: str = 'standard'
    resample_filter = [1, 1]
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs, noise_labels, class_labels, augment_labels=None, train=True):
        assert self.embedding_type in ['fourier', 'positional']
        assert self.encoder_type in ['standard', 'skip', 'residual']
        assert self.decoder_type in ['standard', 'skip']

        emb_channels = self.model_channels * self.channel_mult_emb
        noise_channels = self.model_channels * self.channel_mult_noise
        init = dict(init_mode='xavier_uniform')
        init_zero = dict(init_mode='xavier_uniform', init_weight=1e-5)
        init_attn = dict(init_mode='xavier_uniform', init_weight=jnp.sqrt(0.2))
        block_kwargs = dict(emb_channels=emb_channels, num_heads=1, dropout=self.dropout, skip_scale=jnp.sqrt(0.5), eps=1e-6, resample_filter=self.resample_filter, resample_proj=True, adaptive_scale=False, init=init, init_zero=init_zero, init_attn=init_attn)

        # Mapping
        emb = PositionalEmbedding(num_channels=noise_channels, endpoint=True)(noise_labels)
        emb = jnp.reshape(jnp.flip(jnp.reshape(emb, (emb.shape[0], 2, -1)), 1), emb.shape)

        if self.label_dim:
            tmp = class_labels
            if train and self.label_dropout:
                dropout_key = self.make_rng('dropout')
                tmp = tmp * (jax.random.normal(dropout_key, [x.shape[0], 1]) >= self.label_dropout).astype(tmp.dtype)
            emb = emb + Linear(features=noise_channels, dtype=self.dtype, name='map_label', **init)(tmp * jnp.sqrt(self.label_dim))
        
        if self.augment_dim and augment_labels is not None:
            emb = emb + Linear(features=noise_channels, use_bias=False, dtype=self.dtype, name='map_augment', **init)(augment_labels)
        emb = Linear(features=emb_channels, dtype=self.dtype, name='map_layer0', **init)(emb)
        emb = nn.silu(emb)
        emb = Linear(features=emb_channels, dtype=self.dtype, name='map_layer1', **init)(emb)
        emb = nn.silu(emb)


        # Encoder
        enc = OrderedDict()
        cout = caux = jnp.shape(inputs)[-1]
        skip_channels = []
        for level, mult in enumerate(self.channel_mult):
            res = self.img_resolution >> level
            if level == 0:
                cout = self.model_channels
                enc[f'{res}x{res}_conv'] = Conv2d(channels=cout, kernel=3, dtype=self.dtype, name=f'enc_{res}x{res}_conv', **init)
                skip_channels.append(cout)
            else:
                enc[f'{res}x{res}_down'] = UNetBlock(channels=cout, down=True, dtype=self.dtype, name=f'enc_{res}x{res}_down', **block_kwargs)
                skip_channels.append(cout)
                if self.encoder_type == 'skip':
                    enc[f'{res}x{res}_aux_down'] = Conv2d(channels=caux, kernel=0, down=True, resample_filter=self.resample_filter, dtype=self.dtype, name=f'enc_{res}x{res}_aux_down')
                    enc[f'{res}x{res}_aux_skip'] = Conv2d(channels=cout, kernel=1, dtype=self.dtype, name=f'enc_{res}x{res}_aux_skip', **init)
                if self.encoder_type == 'residual':
                    enc[f'{res}x{res}_aux_residual'] = Conv2d(channels=cout, kernel=3, down=True, resample_filter=self.resample_filter, fused_resample=True, dtype=self.dtype, name=f'enc_{res}x{res}_aux_residual', **init)
                    caux = cout
            for idx in range(self.num_blocks):
                cout = self.model_channels * mult
                attn = (res in self.attn_resolutions)
                enc[f'{res}x{res}_block{idx}'] = UNetBlock(channels=cout, attention=attn, dtype=self.dtype, name=f'enc_{res}x{res}_block{idx}', **block_kwargs)
                skip_channels.append(cout)
            

        skips = []
        aux = x = inputs
        for name, block in enc.items():
            if 'aux_down' in name:
                aux = block(aux)
            elif 'aux_skip' in name:
                x = skips[-1] = x + block(aux)
            elif 'aux_residual' in name:
                x = skips[-1] = aux = (x + block(aux)) / jnp.sqrt(2)
            else:
                x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
                skips.append(x)


        # Decoder
        dec = OrderedDict()
        for level, mult in reversed(list(enumerate(self.channel_mult))):
            res = self.img_resolution >> level
            if level == len(self.channel_mult) - 1:
                dec[f'{res}x{res}_in0'] = UNetBlock(channels=cout, attention=True, dtype=self.dtype, name=f'dec_{res}x{res}_in0', **block_kwargs)
                dec[f'{res}x{res}_in1'] = UNetBlock(channels=cout, dtype=self.dtype, name=f'dec_{res}x{res}_in1' , **block_kwargs)
            else:
                dec[f'{res}x{res}_up'] = UNetBlock(channels=cout, up=True, dtype=self.dtype, name=f'dec_{res}x{res}_up', **block_kwargs)
            for idx in range(self.num_blocks + 1):
                cin = cout + skip_channels.pop()
                cout = self.model_channels * mult
                attn = (idx == self.num_blocks and res in self.attn_resolutions)
                dec[f'{res}x{res}_block{idx}'] = UNetBlock(channels=cout, attention=attn, dtype=self.dtype, name=f'dec_{res}x{res}_block{idx}', **block_kwargs)
            if self.decoder_type == 'skip' or level == 0:
                if self.decoder_type == 'skip' and level < len(self.channel_mult) - 1:
                    dec[f'{res}x{res}_aux_up'] = Conv2d(channels=self.channels, kernel=0, up=True, resample_filter=self.resample_filter, dtype=self.dtype, name=f'dec_{res}x{res}_aux_up')
                dec[f'{res}x{res}_aux_norm'] = group_norm(num_channels=cout, eps=1e-6, name=f'dec_{res}x{res}_aux_norm')
                dec[f'{res}x{res}_aux_conv'] = Conv2d(channels=self.channels, kernel=3, dtype=self.dtype, name=f'dec_{res}x{res}_aux_conv', **init_zero)
        
        aux = None
        tmp = None
        for name, block in dec.items():
            if 'aux_up' in name:
                aux = block(aux)
            elif 'aux_norm' in name:
                tmp = block(x)
            elif 'aux_conv' in name:
                tmp = block(nn.silu(tmp))
                aux = tmp if aux is None else tmp + aux
            else:
                if 'block' in name:
                    x = jnp.concatenate([x, skips.pop()], axis=-1)
                x = block(x, emb)
        return aux



#----------------------------------------------------------------------------
# Reimplementation of the ADM architecture from the paper
# "Diffusion Models Beat GANS on Image Synthesis". Equivalent to the
# original implementation by Dhariwal and Nichol, available at
# https://github.com/openai/guided-diffusion

class DhariwalUNet(nn.Module):
    img_resolution: int 
    channels: int
    label_dim: int = 0
    augment_dim: int = 0
    model_channels: int = 192
    channel_mult = [1, 2, 3, 4]
    channel_mult_emb: int = 4
    num_blocks: int = 3
    attn_resolutions = [32, 16, 8]
    dropout: float = 0.10
    label_dropout: float = 0
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs, noise_labels, class_labels, augment_labels=None, train=True):
        emb_channels = self.model_channels * self.channel_mult_emb
        init = dict(init_mode='kaiming_uniform', init_weight=np.sqrt(1/3), init_bias=np.sqrt(1/3))
        init_zero = dict(init_mode='kaiming_uniform', init_weight=0, init_bias=0)
        block_kwargs = dict(emb_channels=emb_channels, channels_per_head=64, dropout=self.dropout, init=init, init_zero=init_zero)

        # Mapping
        emb = PositionalEmbedding(num_channels=self.model_channels, name='map_noise')(noise_labels)
        if self.augment_dim and augment_labels is not None:
            emb += Linear(features=self.model_channels, use_bias=False, dtype=self.dtype, name='map_augment', **init_zero)(augment_labels)
        
        emb = Linear(features=emb_channels, dtype=self.dtype, name='map_layer0', **init)(emb)
        emb = nn.silu(emb)
        emb = Linear(features=emb_channels, dtype=self.dtype, name='map_layer1', **init)(emb)

        if self.label_dim:
            tmp = class_labels
            if train and self.label_dropout:
                dropout_key = self.make_rng('dropout')
                tmp = tmp * (jax.random.normal(dropout_key, [x.shape[0], 1]) >= self.label_dropout).astype(tmp.dtype)
            emb += Linear(features=emb_channels, use_bias=False, init_mode='kaiming_normal', init_weight=np.sqrt(self.label_dim), dtype=self.dtype, name='map_label')(tmp)
        emb = nn.silu(emb)

        # Encoder
        enc = OrderedDict()
        skip_channels = []
        cout = jnp.shape(inputs)[-1]
        for level, mult in enumerate(self.channel_mult):
            res = self.img_resolution >> level
            if level == 0:
                cout = self.model_channels * mult
                enc[f'{res}x{res}_conv'] = Conv2d(channels=cout, kernel=3, dtype=self.dtype, name=f'enc_{res}x{res}_conv', **init)
                skip_channels.append(cout)
            else:
                enc[f'{res}x{res}_down'] = UNetBlock(channels=cout, down=True, dtype=self.dtype, name=f'enc_{res}x{res}_down', **block_kwargs)
                skip_channels.append(cout)
            for idx in range(self.num_blocks):
                cout = self.model_channels * mult
                enc[f'{res}x{res}_block{idx}'] = UNetBlock(channels=cout, attention=(res in self.attn_resolutions), dtype=self.dtype, name=f'enc_{res}x{res}_block{idx}', **block_kwargs)
                skip_channels.append(cout)
            
        skips = []
        x = inputs
        for block in enc.values():
            x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
            skips.append(x)

            
        # Decoder
        dec = OrderedDict()
        for level, mult in reversed(list(enumerate(self.channel_mult))):
            res = self.img_resolution >> level
            if level == len(self.channel_mult) - 1:
                dec[f'{res}x{res}_in0'] = UNetBlock(channels=cout, attention=True, dtype=self.dtype, name=f'dec_{res}x{res}_in0', **block_kwargs)
                dec[f'{res}x{res}_in1'] = UNetBlock(channels=cout, dtype=self.dtype, name=f'dec_{res}x{res}_in1', **block_kwargs)
            else:
                dec[f'{res}x{res}_up'] = UNetBlock(channels=cout, up=True, dtype=self.dtype, name=f'dec_{res}x{res}_up', **block_kwargs)
            for idx in range(self.num_blocks + 1):
                cin = cout + skip_channels.pop()
                cout = self.model_channels * mult
                dec[f'{res}x{res}_block{idx}'] = UNetBlock(channels=cout, attention=(res in self.attn_resolutions), dtype=self.dtype, name=f'dec_{res}x{res}_block{idx}', **block_kwargs)
        
        for name, block in dec.items():
            if 'block' in name:
                x = jnp.concatenate([x, skips.pop()], axis=-1)
            x = block(x, emb)
        x = nn.silu(group_norm(num_channels=cout, dtype=self.dtype, name='out_norm')(x))
        x = Conv2d(channels=self.channels, kernel=3, dtype=self.dtype, name='out_conv',  **init_zero)(x)

        return x



#----------------------------------------------------------------------------
# Preconditioning corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

class VPPrecond(nn.Module):
    img_resolution: int
    img_channels: int
    label_dim: int = 0
    use_fp16: bool = False
    beta_d: float = 19.9
    beta_min: float = 0.1
    M: int = 1000
    epsilon_t: float = 1e-5
    model_type: str = 'SongUNet'
    model_kwargs: Any = None

    @nn.compact
    def __call__(self, inputs, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        sigma_min = float(self.sigma(self.epsilon_t))
        sigma_max = float(self.sigma(1))

        model = globals()[self.model_type](img_resolution=self.img_resolution, channels=self.img_channels, label_dim=self.label_dim, **model_kwargs)

        x = inputs.astype(jnp.float32)
        sigma = jnp.reshape(sigma.astype(jnp.float32), [-1, 1, 1, 1])
        class_labels = None if self.label_dim == 0 else jnp.zeros([1, self.label_dim]) if class_labels is None else jnp.reshape(class_labels.astype(jnp.float32), (-1, self.label_dim))
        dtype = jnp.float16 if (self.use_fp16 and not force_fp32) else jnp.float32

        c_skip = 1
        c_out = -sigma
        c_in = 1 / jnp.sqrt(sigma**2 + 1)
        c_noise = (self.M - 1) * self.sigma_inv(sigma)

        F_x = model((c_in * x).astype(dtype), jnp.ravel(c_noise), class_labels=class_labels, **model_kwargs)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.astype(jnp.float32)

        return D_x
    

    def sigma(self, t):
        return jnp.sqrt(jnp.exp((0.5 * self.beta_d * (t ** 2) + self.beta_min * t)) - 1)


    def sigma_inv(self, sigma): 
        return jnp.sqrt((self.beta_min ** 2 + 2 * self.beta_d * jnp.log(1 + sigma ** 2)))

    def round_sigma(self, sigma):
        return jnp.asarray(sigma)


#----------------------------------------------------------------------------
# Preconditioning corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

class VEPrecond(nn.Module):
    img_resolution: int
    img_channels: int 
    label_dim: int = 0
    use_fp16: bool = False
    sigma_min: float = 0.02
    sigma_max: float = 100
    model_type: str = 'SongUNet'
    model_kwargs: Any = field(default_factory=lambda: {})

    @nn.compact
    def __call__(self, inputs, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        model = globals()[self.model_type](img_resolution=self.img_resolution, channels=self.img_channels, label_dim=self.label_dim, **self.model_kwargs)

        x = inputs.astype(jnp.float32)
        sigma = jnp.reshape(sigma.astype(jnp.float32), (-1, 1, 1, 1))
        class_labels = None if self.label_dim == 0 else jnp.zeros([1, self.label_dim]) if class_labels is None else jnp.reshape(class_labels.astype(jnp.float32), (-1, self.label_dim))
        dtype = jnp.float16 if (self.use_fp16 and not force_fp32) else jnp.float32

        c_skip = 1
        c_out = sigma
        c_in = 1
        c_noise = jnp.log(0.5 * sigma)

        F_x = model((c_in * x).astype(dtype), jnp.ravel(c_noise), class_labels=class_labels, **model_kwargs)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.astype(jnp.float32)
        return D_x

    def round_sigma(self, sigma):
        return jnp.asarray(sigma)



#----------------------------------------------------------------------------
# Preconditioning corresponding to improved DDPM (iDDPM) formulation from
# the paper "Improved Denoising Diffusion Probabilistic Models".

class iDDPMPrecond(nn.Module):
    img_resolution: int
    img_channels: int
    label_dim: int = 0
    use_fp16: bool = False
    C_1: float = 0.001
    C_2: float = 0.008
    M: int = 1000
    model_type: str = 'DhariwalUNet'
    model_kwargs: Any = field(default_factory=lambda: {})

    @nn.compact
    def __call__(self, inputs, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        model = globals()[self.model_type](img_resolution=self.img_resolution, channels=self.img_channels*2, label_dim=self.label_dim, **self.model_kwargs)

        u = np.zeros(self.M + 1)
        for j in range(self.M, 0, -1):
            u[j - 1] = (u[j] ** 2 + 1) / np.sqrt(np.clip(self.alpha_bar(j-1) / self.alpha_bar(j), a_min=self.C_1, a_max=None) - 1)
        u = jnp.asarray(u)
        sigma_min = float(u[self.M - 1])
        sigma_max = float(u[0])
        self.variable('buffer', 'u', lambda : u)

        x = inputs.astype(jnp.float32)
        sigma= jnp.reshape(sigma.astype(jnp.float32), (-1, 1, 1, 1))
        class_labels = None if self.label_dim == 0 else jnp.zeros([1, self.label_dim]) if class_labels is None else jnp.reshape(class_labels.astype(jnp.float32), (-1, self.label_dim))
        dtype = jnp.float16 if (self.use_fp16 and not force_fp32) else jnp.float32

        c_skip = 1
        c_out = -sigma
        c_in = jnp.sqrt(1/ (sigma**2 + 1))
        c_noise = self.M - 1 - self.round_sigma(sigma, return_index=True).astype(jnp.float32)

        F_x = model((c_in * x).astype(dtype), jnp.ravel(c_noise), class_labels=class_labels, **model_kwargs)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x[:, :self.img_channels].astype(jnp.float32)

        return D_x

    def alpha_bar(self, j):
        return jnp.sin(0.5 * np.pi * j / self.M / (self.C_2 + 1)) ** 2
    
    def round_sigma(self, sigma, return_index=False):
        cdist =  lambda x,y : jnp.sqrt(jnp.sum((x[:,None,:]-y[None,:,:])**2, axis=-1))
        sigma = jnp.asarray(sigma)
        print(sigma.shape, self.variables['buffer']['u'].shape)
        a = jnp.reshape(sigma.astype(jnp.float32), (1, -1, 1))
        u = self.variables['buffer']['u']
        b= jnp.reshape(u, (1, -1, 1))
        index = cdist(a, b).argmin(2)
        result = index if return_index else self.u[jnp.ravel(index)].astype(sigma)
        return jnp.reshape(result, sigma.shape)


#----------------------------------------------------------------------------
# Improved preconditioning proposed in the paper "Elucidating the Design
# Space of Diffusion-Based Generative Models" (EDM).

class EDMPrecond(nn.Module):
    img_resolution: int
    img_channels: int 
    label_dim: int = 0
    use_fp16: bool = False
    sigma_min: float = 0
    sigma_max: float = float('inf')
    sigma_data: float = 0.5
    model_type: str = 'DhariwalUNet'
    model_kwargs: Any = field(default_factory=lambda: {})

    @nn.compact
    def __call__(self, inputs, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        model = globals()[self.model_type](img_resolution=self.img_resolution, channels=self.img_channels, label_dim=self.label_dim, **self.model_kwargs)
        x = inputs.astype(jnp.float32)
        sigma = jnp.reshape(sigma.astype(jnp.float32), (-1, 1, 1, 1))
        class_labels = None if self.label_dim == 0 else jnp.zeros([1, self.label_dim]) if class_labels is None else jnp.reshape(class_labels.astype(jnp.float32), (-1, self.label_dim))
        dtype = jnp.float16 if (self.use_fp16 and not force_fp32) else jnp.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / jnp.sqrt(sigma ** 2 + self.sigma_data ** 2)
        c_in = 1 / jnp.sqrt(self.sigma_data ** 2 + sigma ** 2)
        c_noise = jnp.log(sigma) / 4

        F_x = model((c_in * x).astype(dtype), jnp.ravel(c_noise), class_labels=class_labels, **model_kwargs)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.astype(jnp.float32)
        
        return D_x

    def round_sigma(self, sigma):
        return jnp.asarray(sigma)


#----------------------------------------------------------------------------

import temp
import torch
from flax.traverse_util import flatten_dict, unflatten_dict

key1 = jax.random.PRNGKey(0)
key2 = jax.random.PRNGKey(1)
key3 = jax.random.PRNGKey(11)
key4 = jax.random.PRNGKey(112)
# emb = jax.random.normal(key2, (2, 128))


x = jax.random.normal(key1, (2, 224, 224, 3))
noise = jax.random.normal(key2, (2,))
label = jax.random.normal(key3, (2, 128))
aug = jax.random.normal(key4, (2, 128))
in_torch = torch.from_numpy(np.array(x)).permute(0, 3, 1, 2)
noise_torch = torch.from_numpy(np.array(noise))
label_torch = torch.from_numpy(np.array(label))
aug_torch = torch.from_numpy(np.array(aug))


net_torch = temp.iDDPMPrecond(224, 3, 128).eval()

t_out = net_torch(in_torch, noise_torch, label_torch, aug_torch).permute(0, 2, 3, 1).detach().numpy()
print(t_out.shape)


params = net_torch.state_dict()
def convert_state_dict_to_params(params):
    updated_params = OrderedDict()
    for key, val in params.items():
        new_val = jnp.asarray(params[key])
        if 'enc' in key or 'dec' in key:
            new_key = key.split('.')
            new_key = [new_key[0] + '_' + new_key[1],] + new_key[2:]
        else:
            new_key = key.split('.')
        if 'norm' in new_key[-2] and 'weight' == new_key[-1]:
            new_key[-1] = 'scale'
        elif ('conv' in new_key[-2] or 'skip' in new_key[-2] or 'qkv' in new_key[-2] or 'proj' in new_key[-2]) and new_key[-1] == 'weight':
            new_val = jnp.transpose(new_val, (2, 3, 1, 0))
        elif ('affine' in new_key[-2] or 'map' in new_key[-2]) and new_key[-1] == 'weight':
            new_val = jnp.transpose(new_val, (1, 0))
        new_key.insert(0, 'params')

        updated_params[tuple(new_key)] = new_val
    updated_params = unflatten_dict(updated_params)
    return updated_params

# updated_params = convert_state_dict_to_params(params)

net_flax = iDDPMPrecond(224, 3, 128)
j_out, params = net_flax.init_with_output({'params':jax.random.PRNGKey(4), 'dropout': jax.random.PRNGKey(5)}, x, noise, label, aug)
# j_out= net_flax.apply(updated_params, x, noise, label, aug, False, rngs={'params':jax.random.PRNGKey(4), 'dropout': jax.random.PRNGKey(5)})


# print(j_out.shape, t_out.shape)


# np.testing.assert_almost_equal(j_out, t_out, decimal=7)

# import misc
# net = net_torch
# batch_gpu = 2
# device='cpu'
# images = torch.zeros([batch_gpu, 3, 224, 224], device=device)
# sigma = torch.ones([batch_gpu], device=device)
# labels = torch.zeros([batch_gpu, 128], device=device)
# misc.print_module_summary(net, [images, sigma, labels], max_nesting=2)


# net_flax = DhariwalUNet(img_resolution=224, channels=3, label_dim=128, augment_dim=128)
# tab = net_flax.tabulate({'params':jax.random.PRNGKey(4), 'dropout': jax.random.PRNGKey(5)}, x, noise, label, aug, False,)

# print(tab)