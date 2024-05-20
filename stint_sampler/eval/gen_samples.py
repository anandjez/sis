from jax import random
import jax.numpy as jnp
from stint_sampler.stint.linearInterpolants import linear
from stint_sampler.stint.interpolants import linearInterpolant
from stint_sampler.models.energyNets import DISnet

def generateSamples_fullsis(params,d,target,interpolant_type,Ns,Nt,return_traj=False):
    traj_points = 100
    intrplnt = linearInterpolant(*linear(interpolant_type))

    g = intrplnt.g
    dg = intrplnt.dg
    r = intrplnt.r
    dr = intrplnt.dr

    k1,k2 = random.split(random.PRNGKey(0))
    X = random.normal(k1, (Ns, d)) * jnp.sqrt(1.0)
    if return_traj:
        samples = jnp.zeros((traj_points,Ns,d))
        sample_intrvl = Nt//traj_points
    delT = (self.T0 - self.eps0) / Nt

    for i in range(0, Nt):
        t = (i * delT) + self.eps0
        tv = jnp.ones((Ns, 1)) * t
        Z = self.velocityFn(self.params_ema['params0'], tv, g(t) * X / r(t))
        b = (dg(t) * r(t) - g(t) * dr(t)) * Z + (dr(t) / r(t)) * X
        s = self.sigSampling(t)
        score = g(t) * Z / r(t) - X / (r(t) ** 2)
        # Z = s * score
        mu = b + (s ** 2) * score / 2

        k1, self.k = random.split(self.k)
        W = random.normal(k1, X.shape)
        X += (mu * delT + s * W * jnp.sqrt(delT))
    # @jit
    # def step0(i,X,delT,k):
    #     t = (i * delT) + self.eps0
    #     tv = jnp.ones((Ns, 1)) * t
    #     Z = self.velocityFn(self.params_ema['params0'], tv, g(t) * X / r(t))
    #     b = (dg(t) * r(t) - g(t) * dr(t)) * Z + (dr(t) / r(t)) * X
    #     s = self.sigSampling(t)
    #     score = g(t) * Z / r(t) - X / (r(t) ** 2)
    #     # Z = s * score
    #     mu = b + (s ** 2) * score / 2
    #
    #     W = random.normal(k, X.shape)
    #     X += (mu * delT + s * W * jnp.sqrt(delT))
    #     return X
    #
    # for i in range(0,Nt):
    #     k1, self.k = random.split(self.k)
    #     X = step0(i,X,delT,k1)

    delT = (self.T - self.T0 - self.eps1) / Nt

    for i in range(0, Nt):
        t = (i * delT) + self.T0
        tv = jnp.ones((Ns, 1)) * t
        Z = self.scoreFn(self.params_ema['params1'], tv, X)
        b = (dg(t) * r(t) / g(t) - dr(t)) * r(t) * Z + (dg(t) / g(t)) * X
        s = self.sigSampling(t)
        score = Z
        # Z = s * score
        mu = b + (s ** 2) * score / 2

        k1, self.k = random.split(self.k)
        W = random.normal(k1, X.shape)
        X += (mu * delT + s * W * jnp.sqrt(delT))

    # @jit
    # def step1(i,X,delT,k):
    #     t = (i * delT) + self.T0
    #     tv = jnp.ones((Ns, 1)) * t
    #     Z = self.scoreFn(self.params_ema['params1'], tv, X)
    #     b = (dg(t) * r(t) / g(t) - dr(t)) * r(t) * Z + (dg(t) / g(t)) * X
    #     s = self.sigSampling(t)
    #     score = Z
    #     # Z = s * score
    #     mu = b + (s ** 2) * score / 2
    #
    #     W = random.normal(k, X.shape)
    #     X += (mu * delT + s * W * jnp.sqrt(delT))
    #     return X
    #
    # for i in range(0, Nt):
    #     k1, self.k = random.split(self.k)
    #     X = step1(i,X,delT,k1)

    imp_wts = jnp.exp(
        self.target(X) - self.hjb_solver_score.velocityPot(self.params_ema['params1'], jnp.ones((Ns, 1)) * self.T, X)[:,
                         0])
    normalization = jnp.mean(imp_wts)
    logging.info("Mean importance weight: %s", normalization)

    # for i in range(0,Nt):
    #
    #     t = (i * delT) + self.T0
    #     tv = jnp.ones((Ns, 1)) * t
    #     s = self.sig0(t)
    #     Z = s * self.scoreFn(self.params['params1'], tv, X)
    #     mu = (dg(t) / g(t)) * X + s * Z
    #     k1, self.k = random.split(self.k)
    #     W = random.normal(k1, X.shape)
    #     X += (mu * delT + s*W * jnp.sqrt(delT))

    return X, imp_wts / normalization