import jax
from jax import numpy as jnp
from optax._src import base, combine, transform
from typing import NamedTuple, Optional


def ldl2tridiag(Lsub, D):
    n = D.shape[0]
    Xd = jnp.concatenate((D[0:1], D[1:] + Lsub * Lsub * D[: n - 1]))
    Xe = Lsub * D[: n - 1]

    return Xd, Xe


def tridiag(Sd, Se):
    n = Sd.shape[0]
    psi = Se / Sd[1:]
    cond_cov = jnp.concatenate((Sd[: n - 1] - psi * Se, Sd[n - 1 :]))
    psi = jnp.where(cond_cov[:-1] <= 0, 0, psi)
    cond_cov = jnp.concatenate((Sd[: n - 1] - psi * Se, Sd[n - 1 :]))
    D = 1 / cond_cov
    Lsub = -psi
    Xd, Xe = ldl2tridiag(Lsub, D)

    return Xd, Xe


def tridiag_mult(Xd, Xe, vecv):
    a = Xd * vecv
    a = jnp.concatenate((a[0:1], a[1:] + Xe * vecv[:-1]))
    a = jnp.concatenate((a[:-1] + Xe * vecv[1:], a[-1:]))

    return a


class SONewState(NamedTuple):
    step: jax.Array
    exp_avg: base.Updates
    exp_avg_sq: base.Updates
    sub_exp_avg_sq: base.Updates


def scale_by_sonew(*, beta1, beta2, eps):
    def init_fn(params: base.Params):
        return SONewState(
            step=jnp.array(1, dtype=jnp.int64),
            exp_avg=jax.tree.map(lambda x: jnp.zeros_like(x, shape=(x.size,)), params),
            exp_avg_sq=jax.tree.map(
                lambda x: jnp.zeros_like(x, shape=(x.size,)), params
            ),
            sub_exp_avg_sq=jax.tree.map(
                lambda x: jnp.zeros_like(x, shape=(x.size,)), params
            ),
        )

    def update_fn(
        updates: base.Updates, state: SONewState, params: Optional[base.Params] = None
    ):
        del params

        exp_avg = jax.tree.map(
            lambda avg, grad: beta1 * avg + (1 - beta1) * grad.reshape(-1),
            state.exp_avg,
            updates,
        )
        exp_avg_sq = jax.tree.map(
            lambda avg_sq, grad: beta2 * avg_sq
            + (1 - beta2) * grad.reshape(-1) * grad.reshape(-1),
            state.exp_avg_sq,
            updates,
        )
        sub_exp_avg_sq = jax.tree.map(
            lambda sub_avg_sq, grad: jnp.concatenate(
                (
                    beta2 * sub_avg_sq[:-1]
                    + (1 - beta2) * grad.reshape(-1)[:-1] * grad.reshape(-1)[1:],
                    sub_avg_sq[-1:],
                )
            ),
            state.sub_exp_avg_sq,
            updates,
        )

        bias_correction1 = 1 - beta1**state.step
        bias_correction2 = 1 - beta2**state.step

        denom_diag = jax.tree.map(
            lambda avg_sq: avg_sq / bias_correction2 + eps, exp_avg_sq
        )
        denom_sub_diag = jax.tree.map(
            lambda sub_avg_sq: sub_avg_sq / bias_correction2, sub_exp_avg_sq
        )

        inverse = jax.tree.map(
            lambda denom, denom_sub: tridiag(denom, denom_sub[:-1]),
            denom_diag,
            denom_sub_diag,
        )
        moment = jax.tree.map(lambda avg: avg / bias_correction1, exp_avg)
        sonew_update = jax.tree.map(
            lambda mom, inv: tridiag_mult(inv[0], inv[1], mom), moment, inverse
        )
        adam_update = jax.tree.map(
            lambda mom, avg_sq: mom / (jnp.sqrt(avg_sq) + eps), moment, exp_avg_sq
        )
        sonew_update = jax.tree.map(
            lambda sonew, adam: sonew
            * (jnp.linalg.norm(adam) / jnp.linalg.norm(sonew)),
            sonew_update,
            adam_update,
        )
        sonew_update = jax.tree.map(
            lambda sonew, grad: sonew.reshape(grad.shape), sonew_update, updates
        )

        return sonew_update, SONewState(
            step=state.step + 1,
            exp_avg=exp_avg,
            exp_avg_sq=exp_avg_sq,
            sub_exp_avg_sq=sub_exp_avg_sq,
        )

    return base.GradientTransformation(init_fn, update_fn)


def sonew(
    learning_rate=1e-3, *, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0, mask=None
):
    return combine.chain(
        scale_by_sonew(beta1=beta1, beta2=beta2, eps=eps),
        transform.add_decayed_weights(weight_decay=weight_decay, mask=mask),
        transform.scale_by_learning_rate(learning_rate=learning_rate),
    )
