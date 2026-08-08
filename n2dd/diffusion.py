"""DDPM over the 30-dim standardized feature vector, with property guidance.

Fixes carried over from the repaired notebook:

* `schedule` is an object, not a dict -- the SI code's `schedule['alpha'][t]` is a TypeError.
* x0 estimate is clipped; 1/sqrt(alpha_bar) blows up at large t and, with the SI's
  lambda_cond=10, drives every sample onto a single alloy.
* Property targeting happens at *sampling* time via classifier guidance, scaled by
  sqrt(1 - alpha_bar_t) so guidance strength tracks the noise level.
"""
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .config import Config


class DiffusionSchedule:
    def __init__(self, cfg: Config):
        self.timesteps = cfg.timesteps
        self.betas = np.linspace(cfg.beta_start, cfg.beta_end, cfg.timesteps, dtype=np.float32)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        # tf copies for use inside @tf.function
        self.tf_alphas_cumprod = tf.constant(self.alphas_cumprod, tf.float32)
        self.tf_sqrt_ac = tf.constant(self.sqrt_alphas_cumprod, tf.float32)
        self.tf_sqrt_1mac = tf.constant(self.sqrt_one_minus_alphas_cumprod, tf.float32)


def sinusoidal_embedding(t, embedding_dim):
    half = embedding_dim // 2
    emb = np.log(10000) / (half - 1)
    emb = np.exp(np.arange(half) * -emb)
    emb = tf.cast(t[:, None], tf.float32) * emb[None, :]
    return tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)


def build_denoiser(cfg: Config) -> keras.Model:
    x_in = keras.Input(shape=(cfg.vector_size,))
    t_in = keras.Input(shape=(), dtype=tf.int32)
    t_emb = layers.Lambda(lambda t: sinusoidal_embedding(t, cfg.embedding_dim))(t_in)
    t_emb = layers.Dense(128, activation="swish")(t_emb)
    t_emb = layers.Dense(128, activation="swish")(t_emb)
    x = layers.Dense(128, activation="swish")(x_in)
    x = layers.Concatenate()([x, t_emb])
    x = layers.Dense(256, activation="swish")(x)
    x = layers.Dense(256, activation="swish")(x)
    x = layers.Dense(cfg.vector_size)(x)
    return keras.Model([x_in, t_in], x, name="denoiser")


class DiffusionModel:
    def __init__(self, cfg: Config, surrogate: keras.Model):
        self.cfg = cfg
        self.schedule = DiffusionSchedule(cfg)
        self.model = build_denoiser(cfg)
        self.surrogate = surrogate
        self.optimizer = keras.optimizers.Adam(cfg.learning_rate)
        self.mse = keras.losses.MeanSquaredError()
        self._build_steps()

    def _build_steps(self):
        cfg, sch, sur = self.cfg, self.schedule, self.surrogate
        target_const = tf.constant(cfg.target_value, tf.float32)

        @tf.function
        def train_step(x0):
            b = tf.shape(x0)[0]
            t = tf.random.uniform((b,), 0, cfg.timesteps, dtype=tf.int32)
            noise = tf.random.normal(tf.shape(x0))
            sqrt_ac = tf.gather(sch.tf_sqrt_ac, t)[:, None]
            sqrt_1mac = tf.gather(sch.tf_sqrt_1mac, t)[:, None]
            noisy = sqrt_ac * x0 + sqrt_1mac * noise

            with tf.GradientTape() as tape:
                eps = self.model([noisy, t], training=True)
                loss_denoise = self.mse(noise, eps)
                if cfg.lambda_cond > 0.0:
                    ab = tf.gather(sch.tf_alphas_cumprod, t)[:, None]
                    x0_pred = tf.clip_by_value(
                        (noisy - tf.sqrt(1.0 - ab) * eps) / tf.sqrt(ab),
                        -cfg.x0_clip, cfg.x0_clip)
                    pred = sur(x0_pred, training=False)
                    # weight by alpha_bar: only trust x0_pred where the sample is nearly clean
                    loss_cond = (tf.reduce_sum(ab * tf.square(pred - target_const))
                                 / (tf.reduce_sum(ab) + 1e-8))
                else:
                    loss_cond = tf.constant(0.0)
                loss = loss_denoise + cfg.lambda_cond * loss_cond

            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            return loss, loss_denoise, loss_cond

        @tf.function
        def grad_condition(x, tgt):
            with tf.GradientTape() as tape:
                tape.watch(x)
                l = tf.reduce_sum(tf.square(sur(x, training=False) - tgt))
            return tf.clip_by_norm(tape.gradient(l, x), cfg.grad_clip_norm, axes=[1])

        self.train_step = train_step
        self.grad_condition = grad_condition

    # ---------------------------------------------------------------- training
    def fit(self, X: np.ndarray, log_every: int = 200, verbose: bool = True):
        cfg = self.cfg
        ds = tf.data.Dataset.from_tensor_slices(X).shuffle(1000).batch(cfg.batch_size)
        ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
        mgr = tf.train.CheckpointManager(ckpt, str(cfg.ckpt_dir), max_to_keep=3)

        history = {"loss": [], "denoise": [], "cond": []}
        t0, nb = time.time(), 0
        for epoch in range(cfg.epochs):
            tot = den = con = 0.0
            nb = 0
            for xb in ds:
                l, ld, lc = self.train_step(xb)
                tot += float(l); den += float(ld); con += float(lc); nb += 1
            history["loss"].append(tot / nb)
            history["denoise"].append(den / nb)
            history["cond"].append(con / nb)
            if verbose and ((epoch + 1) % log_every == 0 or epoch == 0):
                mgr.save()
                print(f"  epoch {epoch+1:>5} | loss {tot/nb:.6f} | denoise {den/nb:.6f}")
        mgr.save()
        self.history = history
        self.train_seconds = time.time() - t0
        self.gradient_steps = cfg.epochs * nb
        return history

    def restore(self):
        ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
        mgr = tf.train.CheckpointManager(ckpt, str(self.cfg.ckpt_dir), max_to_keep=3)
        if not mgr.latest_checkpoint:
            raise FileNotFoundError(f"no checkpoint in {self.cfg.ckpt_dir}; train first")
        ckpt.restore(mgr.latest_checkpoint).expect_partial()
        return mgr.latest_checkpoint

    # ---------------------------------------------------------------- sampling
    def sample(self, n, target=None, guidance_scale=None, x_init=None, seed=None):
        """Ancestral DDPM sampling. `x_init` lets you start from a perturbed known
        catalyst instead of pure noise (the SI README's 'starting point is free to change')."""
        cfg, sch = self.cfg, self.schedule
        target = cfg.target_value if target is None else target
        gs = cfg.guidance_scale if guidance_scale is None else guidance_scale
        if seed is not None:
            tf.random.set_seed(seed)
        x = tf.random.normal((n, cfg.vector_size)) if x_init is None else tf.cast(x_init, tf.float32)
        tgt = tf.constant(target, tf.float32)

        for ti in range(cfg.timesteps - 1, -1, -1):
            t = tf.constant([ti] * n, dtype=tf.int32)
            eps = self.model([x, t], training=False)
            if gs > 0.0:
                eps += gs * sch.sqrt_one_minus_alphas_cumprod[ti] * self.grad_condition(x, tgt)
            a, ac, b = sch.alphas[ti], sch.alphas_cumprod[ti], sch.betas[ti]
            z = tf.random.normal((n, cfg.vector_size)) if ti > 0 else tf.zeros_like(x)
            x = (1 / np.sqrt(a)) * (x - (b / np.sqrt(1 - ac)) * eps) + np.sqrt(b) * z
        return x.numpy()
