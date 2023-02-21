import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from jax.random import normal, uniform, randint
import misc
#----------------------------------------------------------------------------
# Coefficients of various wavelet decomposition low-pass filters.

wavelets = {
    'haar': [0.7071067811865476, 0.7071067811865476],
    'db1':  [0.7071067811865476, 0.7071067811865476],
    'db2':  [-0.12940952255092145, 0.22414386804185735, 0.836516303737469, 0.48296291314469025],
    'db3':  [0.035226291882100656, -0.08544127388224149, -0.13501102001039084, 0.4598775021193313, 0.8068915093133388, 0.3326705529509569],
    'db4':  [-0.010597401784997278, 0.032883011666982945, 0.030841381835986965, -0.18703481171888114, -0.02798376941698385, 0.6308807679295904, 0.7148465705525415, 0.23037781330885523],
    'db5':  [0.003335725285001549, -0.012580751999015526, -0.006241490213011705, 0.07757149384006515, -0.03224486958502952, -0.24229488706619015, 0.13842814590110342, 0.7243085284385744, 0.6038292697974729, 0.160102397974125],
    'db6':  [-0.00107730108499558, 0.004777257511010651, 0.0005538422009938016, -0.031582039318031156, 0.02752286553001629, 0.09750160558707936, -0.12976686756709563, -0.22626469396516913, 0.3152503517092432, 0.7511339080215775, 0.4946238903983854, 0.11154074335008017],
    'db7':  [0.0003537138000010399, -0.0018016407039998328, 0.00042957797300470274, 0.012550998556013784, -0.01657454163101562, -0.03802993693503463, 0.0806126091510659, 0.07130921926705004, -0.22403618499416572, -0.14390600392910627, 0.4697822874053586, 0.7291320908465551, 0.39653931948230575, 0.07785205408506236],
    'db8':  [-0.00011747678400228192, 0.0006754494059985568, -0.0003917403729959771, -0.00487035299301066, 0.008746094047015655, 0.013981027917015516, -0.04408825393106472, -0.01736930100202211, 0.128747426620186, 0.00047248457399797254, -0.2840155429624281, -0.015829105256023893, 0.5853546836548691, 0.6756307362980128, 0.3128715909144659, 0.05441584224308161],
    'sym2': [-0.12940952255092145, 0.22414386804185735, 0.836516303737469, 0.48296291314469025],
    'sym3': [0.035226291882100656, -0.08544127388224149, -0.13501102001039084, 0.4598775021193313, 0.8068915093133388, 0.3326705529509569],
    'sym4': [-0.07576571478927333, -0.02963552764599851, 0.49761866763201545, 0.8037387518059161, 0.29785779560527736, -0.09921954357684722, -0.012603967262037833, 0.0322231006040427],
    'sym5': [0.027333068345077982, 0.029519490925774643, -0.039134249302383094, 0.1993975339773936, 0.7234076904024206, 0.6339789634582119, 0.01660210576452232, -0.17532808990845047, -0.021101834024758855, 0.019538882735286728],
    'sym6': [0.015404109327027373, 0.0034907120842174702, -0.11799011114819057, -0.048311742585633, 0.4910559419267466, 0.787641141030194, 0.3379294217276218, -0.07263752278646252, -0.021060292512300564, 0.04472490177066578, 0.0017677118642428036, -0.007800708325034148],
    'sym7': [0.002681814568257878, -0.0010473848886829163, -0.01263630340325193, 0.03051551316596357, 0.0678926935013727, -0.049552834937127255, 0.017441255086855827, 0.5361019170917628, 0.767764317003164, 0.2886296317515146, -0.14004724044296152, -0.10780823770381774, 0.004010244871533663, 0.010268176708511255],
    'sym8': [-0.0033824159510061256, -0.0005421323317911481, 0.03169508781149298, 0.007607487324917605, -0.1432942383508097, -0.061273359067658524, 0.4813596512583722, 0.7771857517005235, 0.3644418948353314, -0.05194583810770904, -0.027219029917056003, 0.049137179673607506, 0.003808752013890615, -0.01495225833704823, -0.0003029205147213668, 0.0018899503327594609],
}


#----------------------------------------------------------------------------
# Helpers for constructing transformation matrices.
# TODO : Fix this fn
def matrix(*rows):
    assert all(len(row) == len(rows[0]) for row in rows)
    elems = [x for row in rows for x in row]
    ref = [x for x in elems if isinstance(x, jax.Array)]

    if len(ref) == 0:
        return misc.constant(np.asarray(rows))
    elems = [x if isinstance(x, jax.Array) else misc.constant(x, shape=ref[0].shape) for x in elems]
    return jnp.stack(elems, axis=-1).reshape(ref[0].shape + (len(rows), -1))



def translate2d(tx, ty, **kwargs):
    return matrix(
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1],
        **kwargs)

def translate3d(tx, ty, tz, **kwargs):
    return matrix(
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1],
        **kwargs)

def scale2d(sx, sy, **kwargs):
    return matrix(
        [sx, 0,  0],
        [0,  sy, 0],
        [0,  0,  1],
        **kwargs)

def scale3d(sx, sy, sz, **kwargs):
    return matrix(
        [sx, 0,  0,  0],
        [0,  sy, 0,  0],
        [0,  0,  sz, 0],
        [0,  0,  0,  1],
        **kwargs)

def rotate2d(theta, **kwargs):
    return matrix(
        [jnp.cos(theta), jnp.sin(-theta), 0],
        [jnp.sin(theta), jnp.cos(theta),  0],
        [0,                0,                 1],
        **kwargs)

def rotate3d(v, theta, **kwargs):
    vx = v[..., 0]; vy = v[..., 1]; vz = v[..., 2]
    s = jnp.sin(theta); c = jnp.cos(theta); cc = 1 - c
    return matrix(
        [vx*vx*cc+c,    vx*vy*cc-vz*s, vx*vz*cc+vy*s, 0],
        [vy*vx*cc+vz*s, vy*vy*cc+c,    vy*vz*cc-vx*s, 0],
        [vz*vx*cc-vy*s, vz*vy*cc+vx*s, vz*vz*cc+c,    0],
        [0,             0,             0,             1],
        **kwargs)

def translate2d_inv(tx, ty, **kwargs):
    return translate2d(-tx, -ty, **kwargs)

def scale2d_inv(sx, sy, **kwargs):
    return scale2d(1 / sx, 1 / sy, **kwargs)

def rotate2d_inv(theta, **kwargs):
    return rotate2d(-theta, **kwargs)


#----------------------------------------------------------------------------
# Augmentation pipeline main class.
# All augmentations are disabled by default; individual augmentations can
# be enabled by setting their probability multipliers to 1.

class AugmentPipe(nn.Module):
    p: float = 1
    xflip:float = 0
    yflip:float = 0
    rotate_int:float = 0
    translate_int:float = 0
    translate_int_max:float = 0.125
    scale:float = 0 
    rotate_frac:float = 0
    aniso:float = 0
    translate_frac:float = 0
    scale_std:float = 0.2
    rotate_frac_max:float = 1
    aniso_std:float = 0.2
    aniso_rotate_prob:float = 0.5
    translate_frac_std:float = 0.125
    brightness:float = 0
    contrast:float = 0
    lumaflip:float = 0
    hue:float = 0
    saturation:float = 0
    brightness_std:float = 0.2
    contrast_std:float = 0.5
    hue_max:float = 1
    saturation_std:float = 1
    
    @nn.compact
    def __call__(self, images):

        N, H, W, C = images.shape
        labels = [jnp.zeros([images.shape[0], 0])]

        make_key = lambda : self.make_rng('augment')

        # ---------------
        # Pixel blitting.
        # ---------------

        if self.xflip > 0:
            w = randint(make_key(), [N, 1, 1, 1], minval=0, maxval=2)
            w = jnp.where(uniform(make_key(), [N, 1, 1, 1]) < self.xflip * self.p, w, jnp.zeros_like(w))
            images = jnp.where(w == 1, jnp.flip(images, 2), images) # TODO : axis probably wrong
            labels += [w]
        
        if self.yflip > 0:
            w = randint(make_key(), [N, 1, 1, 1], minval=0, maxval=2)
            w = jnp.where(uniform(make_key(), [N, 1, 1, 1]) < self.yflip * self.p, w, jnp.zeros_like(w))
            images = jnp.where(w == 1, jnp.flip(images, 1), images) # TODO : axis probably wrong
            labels += [w]
        
        if self.rotate_int > 0:
            w = randint(make_key(), [N, 1, 1, 1], minval=0, maxval=4)
            w = jnp.where(uniform(make_key(), [N, 1, 1, 1]) < self.rotate_int * self.p, w, jnp.zeros_like(w))
            images = jnp.where((w == 1) | (w == 2), jnp.flip(images, 2), images)
            images = jnp.where((w == 2) | (w == 3), jnp.flip(images, 1), images)
            images = jnp.where((w == 1) | (w == 3), jnp.transpose(images, (0, 2, 1, 3)), images)
            labels += [(w == 1) | (w == 2), (w == 2) | (w == 3)]

        if self.translate_int > 0:
            w = uniform(make_key(), [2, N, 1, 1, 1]) * 2 - 1
            w = jnp.where(uniform(make_key(), [1, N, 1, 1, 1]) < self.translate_int * self.p, w, jnp.zeros_like(w))
            tx = jnp.round(w[0] * W * self.translate_int_max).astype(jnp.int32)
            ty = jnp.round(w[1] * H * self.translate_int_max).astype(jnp.int32)
            b, y, x, c = jnp.meshgrid(*(jnp.arange(x) for x in images.shape), indexing='ij')
            x = W - 1 - jnp.abs(W - 1 - (x - tx) % (W * 2 - 2))
            y = H - 1 - jnp.abs(H - 1 - (y + ty) % (H * 2 - 2))
            images = jnp.ravel(images)[(((b * H) + y) * W + x) * C + c]
            labels += [jnp.divide(tx, W * self.translate_int_max), jnp.divide(ty, H * self.translate_int_max)]


        # ------------------------------------------------
        # Select parameters for geometric transformations.
        # ------------------------------------------------

        I_3 = jnp.eye(3)
        G_inv = I_3

        if self.scale > 0:
            w = normal(make_key(), [N])
            w = jnp.where(uniform(make_key(), [N]) < self.scale * self.p, w, jnp.zeros_like(w))
            s = jnp.exp2(w * self.scale_std)
            G_inv = G_inv @ scale2d_inv(s, s)
            labels += [w]

        
        if self.rotate_frac > 0:
            w = (uniform(make_key(), [N]) * 2 - 1) * (jnp.pi * self.rotate_frac_max)
            w = jnp.where(uniform(make_key(), [N]) < self.rotate_frac * self.p, w, jnp.zeros_like(w))
            G_inv = G_inv @ rotate2d_inv(-w)
            labels += [jnp.cos(w) - 1, jnp.sin(w)]
        

        if self.aniso > 0:
            w = normal(make_key(), [N])
            r = (uniform(make_key(), [N]) * 2 - 1) * jnp.pi
            w = jnp.where(uniform(make_key(), [N]) < self.aniso * self.p, w, jnp.zeros_like(w))
            r = jnp.where(uniform(make_key(), [N]) < self.aniso_rotate_prob, r, jnp.zeros_like(r))
            s = jnp.exp2(w * self.aniso_std)
            G_inv = G_inv @ rotate2d_inv(r) @ scale2d_inv(s, 1/s) @ rotate2d_inv(-r)
            labels += [w * jnp.cos(r), w * jnp.sin(r)]

        if self.translate_frac > 0:
            w = normal(make_key(), [2, N])
            w = jnp.where(uniform(make_key(), [N]) < self.translate_frac * self.p, w, jnp.zeros_like(w))
            G_inv = G_inv @ translate2d_inv(w[0] * W * self.translate_frac_std, w[1] * H * self.translate_frac_std)
            labels += [w[0], w[1]]
        

        # ----------------------------------
        # Execute geometric transformations.
        # ----------------------------------

        if G_inv is not I_3:
            cx = (W - 1) / 2
            cy = (H - 1) / 2
            cp = matrix([-cx, -cy, 1], [cx, -cy, 1], [cx, cy, 1], [-cx, cy, 1]) # [idx, xyz]
            cp = G_inv @ cp.T
            Hz = np.asarray(wavelets['sym6'], dtype=np.float32)
            Hz_pad = len(Hz) // 4
            margin = jnp.reshape(jnp.transpose(cp[:, :2, :], (1, 0, 2)), (-1, cp.shape[0] * cp.shape[-1]))
            margin = jnp.max(jnp.concatenate([-margin, margin]), axis=1)
            margin = margin + misc.constant([Hz_pad * 2 - cx, Hz_pad * 2 - cy] * 2)
            margin = jnp.maximum(margin, misc.constant([0, 0] * 2))
            margin = jnp.minimum(margin, misc.constant([W - 1, H - 1] * 2))
            mx0, my0, mx1, my1 = jnp.ceil(margin).astype(jnp.int32)


            # Pad image and adjust origin
            images = jnp.pad(images, pad_width=[mx0, mx1, my0, my1], mode='reflect')
            G_inv = translate2d((mx0 - mx1) / 2, (my0 - my1) / 2) @ G_inv

            # Upsample
            conv_weight = jnp.tile(misc.constant(Hz[None, None, ::-1], dtype=
            images.dtype), [images.shape[1], 1, 1])
            conv_pad = (len(Hz) + 1) // 2
            images = jnp.reshape(jnp.stack([images, jnp.zeros_like(images)], axis=4), (N, C, images.shape[2], -1))[:, :, :, :-1]
            images = jax.lax.conv_general_dilated(images, jnp.expand_dims(conv_weight, 2), feature_group_count=images.shape[1], padding=[0, conv_pad])
            images = jnp.reshape(jnp.stack([images, jnp.zeros_like(images)], axis=3), (N, C, -1, images.shape[3]))[:, :, :-1, :]
            images = jax.lax.conv_general_dilated(images, jnp.expand_dims(conv_weight, 3), feature_group_count=images.shape[1], padding=[conv_pad, 0])
            G_inv = scale2d(2, 2) @ G_inv @ scale2d_inv(2, 2)
            G_inv = translate2d(-0.5, -0.5) @ G_inv @ translate2d_inv(-0.5, -0.5)

            # Execute Transformation
            shape = [N, C, (H + Hz_pad * 2) * 2, (W + Hz_pad * 2) * 2]
            G_inv = scale2d(2 / images.shape[3], 2 / images.shape[2]) @ G_inv @ scale2d_inv(2 / shape[3], 2 / shape[2])
            grid = misc.affine_grid_generator(theta=G_inv[:,:2,:], size=shape, align_corners=False)
            # grid sample

            # Downsample and crop
            conv_weight = jnp.tile(misc.constant(Hz[None, None, :], dtype=images.dtype), [images.shape[1], 1, 1])
            conv_pad = (len(H) - 1) // 2
            images = jax.lax.conv_general_dilated(images, jnp.expand_dims(conv_weight, 2), feature_group_count=images.shape[1], window_strides=[1,2], padding=[0, conv_pad])[:, :, :, Hz_pad: -Hz_pad]
            images = jax.lax.conv_general_dilated(images, jnp.expand_dims(conv_weight, 3), feature_group_count=images.shape[1], window_strides=[2, 1], padding=[conv_pad, 0])[:, :, Hz_pad:-Hz_pad, :]


        # --------------------------------------------
        # Select parameters for color transformations.
        # --------------------------------------------

        I_4 = jnp.eye(4)
        M = I_4
        luma_axis = misc.constant(np.asarray([1, 1, 1, 0]) / np.sqrt(3))

        if self.brightness > 0:
            w = normal(make_key(), [N])
            w = jnp.where(uniform(make_key(), [N]) < self.brightness * self.p, w, jnp.zeros_like(w))
            b = w * self.brightness_std
            M = translate3d(b, b, b) @ M
            labels += [w]
        
        if self.contrast > 0:
            w = normal(make_key(), [N])
            w = jnp.where(uniform(make_key(), [N]) < self.contrast * self.p, w, jnp.zeros_like(w))
            c = jnp.exp2(w * self.contrast_std)
            M = scale3d(c, c, c) @ M
            labels += [w]

        if self.lumaflip > 0:
            w = randint(make_key(), [N, 1, 1], minval=0, maxval=2)
            w = jnp.where(uniform(make_key(), [N, 1, 1]) < self.lumaflip * self.p, w, jnp.zeros_like(w))
            M = (I_4 - 2 * jnp.outer(luma_axis, luma_axis) * w) @ M
            labels += [w]

        if self.hue > 0:
            w = (uniform(make_key(), [N]) * 2 - 1) * (jnp.pi * self.hue_max)
            w = jnp.where(uniform(make_key(), [N]) < self.hue * self.p, w, jnp.zeros_like(w))
            M = rotate3d(luma_axis, w) @ M
            labels += [jnp.cos(w) - 1, jnp.sin(w)]

        if self.saturation > 0:
            w = normal(make_key(), [N, 1, 1])
            w = jnp.where(uniform(make_key(), [N, 1, 1]) < self.saturation * self.p, w, jnp.zeros_like(w))
            M = (jnp.outer(luma_axis, luma_axis) + (I_4 - jnp.outer(luma_axis, luma_axis)) * jnp.exp2(w * self.saturation_std)) @ M
            labels += [w]

        # ------------------------------
        # Execute color transformations.
        # ------------------------------

        if M is not I_4:
            images = jnp.reshape(images, [N, H * W, C])
            if C == 3:
                images = images @ jnp.transpose(M[:, :3, :3], (0, 2, 1)) + M[:, :3, 3]
            elif C == 1:
                M = jnp.mean(M[:, :3, :], axis=1, keepdims=True)
                images = images * jnp.sum(M[:, :, :3], axis=2, keepdims=True) + M[:, :, 3:]
            else:
                raise ValueError('Image must be RGB (3 channels) or L (1 channel)')
            images = jnp.reshape(images, [N, H, W, C])
        
        # labels = jnp.concatenate([])
        return images, []