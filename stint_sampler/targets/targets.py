"""Distributions and datasets for sampler debugging.
"""
import distrax

import jax
import jax.numpy as jnp

from jax.scipy.special import logsumexp
from jax.scipy.stats import multivariate_normal
from jax.scipy.stats import norm
from scipy.stats import multivariate_t

import numpy as np
import pathlib
from stint_sampler.targets.lgcp import LogGaussianCoxPines

homePath = pathlib.Path.home()

def student(d=2,nu=2.5,var=0.3):
    log_density = lambda x:jnp.clip(-0.5*(nu+d)*jnp.log(1+jnp.sum(x**2)/(nu*var)),-1e4,1e4)
    def sample(nsamples):
      return 0
    return jax.vmap(log_density),sample

def mos(d=100,k=10,var=1.0):
    nu = 2.0
    mean = float(k)
    mean_vec = (jax.random.uniform(jax.random.PRNGKey(0), (d, k)) - 0.5) * 2 * mean
    log_density_comp = lambda x: -0.5*(nu+d)*jnp.log(1+jnp.sum((jnp.outer(x,jnp.ones((mean_vec.shape[1],)))-mean_vec)**2,axis=0)/(nu*var)) + jax.scipy.special.gammaln(0.5*(nu+d))- jax.scipy.special.gammaln(nu/2)-jnp.log(k)-jnp.log(nu*jnp.pi**var)*d/2
    log_density = lambda x:jnp.clip(jax.scipy.special.logsumexp(log_density_comp(x)),-1e8,1e8)
    def sample(nsamples):
        rv = multivariate_t(loc = np.zeros((d,)),shape=np.identity(d),df=2.0)
        return jnp.array(rv.rvs(size=(nsamples,))) + mean_vec.T[np.random.randint(0, mean_vec.shape[1], nsamples)]
    return jax.vmap(log_density),sample
def lgcp(d=1600):
    LGCP = LogGaussianCoxPines()
    def sample(nsamples):
      return 0

    return LGCP.evaluate_log_density, sample
def GSG(d=10,beta=0.0,lbda=0.5,h=0):
    # beta = 0.99-lbda
    filename = "python/sis/stint_sampler/targets/params/GSG_matrix"+str(d)+".npy"
    try:
        A = np.load(homePath / filename)
        A = jnp.array(A)
    except:
        A = np.random.randn(d,d)
        np.save(homePath / filename,A)
        A = jnp.array(A)
    def log_density(x):
        H = -(1/jnp.sqrt(2*d))*jnp.dot(x,A@x)-h*jnp.sum(x)
        log_density = -beta*H-((beta**2)/(4*d))*((jnp.sum(x**2))**2)+(lbda/2)*jnp.sum(x**2)-(1/2)*jnp.sum(x**2)-d*jnp.log(2*jnp.pi)/2
        return jnp.clip(log_density,-1e4,1e4)
    def sample(nsamples):
      return 0

    return jax.vmap(log_density), sample

def SK_continuous(d=10,beta=10,gamma=1.0,delta=1.0):
    filename = "python/sis/stint_sampler/targets/params/SK_matrix"+str(d)+".npy"
    filenameh = "python/sis/stint_sampler/targets/params/SK_field"+str(d)+".npy"
    try:
        A = np.load(homePath / filename)
        h = np.load(homePath / filenameh)
        A = jnp.array(A)
        h = jnp.array(h)
    except:
        A = np.random.randn(d,d)
        h = np.random.randn(d)*0.1
        A = A+np.diag(-np.diag(A))
        np.save(homePath / filename,A)
        np.save(homePath / filenameh,h)
        A = jnp.array(A)
        h = jnp.array(h)

    log_density = lambda x: jnp.clip(-beta*(gamma*jnp.sum((x**2-delta)**2)+(1/jnp.sqrt(d))*jnp.dot(x,A@x)+jnp.dot(x,h)),-1e4,1e4)
    def sample(nsamples):
      return 0

    return jax.vmap(log_density), sample

def double_well(d=10,w=3,delta=2):
    log_density = lambda x: jnp.clip(-jnp.sum((x[:w]**2-delta)**2)-0.5*jnp.sum(x[w:]**2),-1e4,1e4)
    def sample(nsamples):
      return 0

    return jax.vmap(log_density), sample
def gmm(d=2,mean=5.0,var=0.3):
    mean_vec = jnp.array([[mean * (i - 1), mean * (j - 1)] for i in range(3) for j in range(3)]).T
    log_density_comp = lambda x: -jnp.sum((jnp.outer(x,jnp.ones((mean_vec.shape[1],)))-mean_vec)**2,axis=0)/(2*var)-jnp.log(mean_vec.shape[1]*2*jnp.pi*var)
    log_density = lambda x:jnp.clip(jax.scipy.special.logsumexp(log_density_comp(x)),-1e4,1e4)
    def sample(nsamples):
      return jnp.array(np.random.randn(nsamples,d))+mean_vec.T[np.random.randint(0,mean_vec.shape[1],nsamples)]
    return jax.vmap(log_density),sample

def mog(d=100,k=40,var=1.0):
    mean=float(k)
    # mean_vec = jnp.array([[mean * (i - 1), mean * (j - 1)] for i in range(3) for j in range(3)]).T
    mean_vec = (jax.random.uniform(jax.random.PRNGKey(0),(d,k))-0.5)*2*mean
    log_density_comp = lambda x: -jnp.sum((jnp.outer(x,jnp.ones((mean_vec.shape[1],)))-mean_vec)**2,axis=0)/(2*var)-jnp.log(2*jnp.pi*var)*d/2-jnp.log(k)
    log_density = lambda x:jnp.clip(jax.scipy.special.logsumexp(log_density_comp(x)),-1e8,1e8)
    def sample(nsamples):
      return jnp.array(np.random.randn(nsamples,d))+mean_vec.T[np.random.randint(0,mean_vec.shape[1],nsamples)]
    return jax.vmap(log_density),sample

def funnel(d=10, sig=3, clip_y=11):
  """Funnel distribution for testing. Returns energy and sample functions."""

  def neg_energy(x):
    def unbatched(x):
      v = x[0]
      log_density_v = norm.logpdf(v,
                                  loc=0.,
                                  scale=3.)
      variance_other = jnp.exp(v)
      other_dim = d - 1
      cov_other = jnp.eye(other_dim) * variance_other
      mean_other = jnp.zeros(other_dim)
      log_density_other = multivariate_normal.logpdf(x[1:],
                                                     mean=mean_other,
                                                     cov=cov_other)
      return jnp.clip(log_density_v + log_density_other,-1e4,1e4)
    output = jax.vmap(unbatched)(x)
    return output

  def sample_data(n_samples):
    # sample from Nd funnel distribution
    y = (sig * jnp.array(np.random.randn(n_samples, 1))).clip(-clip_y, clip_y)
    x = jnp.array(np.random.randn(n_samples, d - 1)) * jnp.exp(y / 2)
    return jnp.concatenate((y, x), axis=1)

  return neg_energy, sample_data

# def rings(d=2,n_comp=4, std=0.02, radius=1.0):
#     radius_comp = jnp


def toy_rings(d=2,n_comp=4, std=0.05, radius=0.5):
  """Mixture of rings distribution. Returns energy and sample functions."""

  weights = np.ones(n_comp) / n_comp

  def neg_energy(x):
    r = jnp.sqrt((x[:, 1] ** 2) + (x[:, 0] ** 2))[:, None]
    means = (jnp.arange(1, n_comp + 1) * radius)[None, :]
    c = jnp.log(n_comp * 2 * np.pi * std**2)
    f = -jax.nn.logsumexp(-0.5 * jnp.square((r - means) / std), axis=1) + c
    return -f

  def sample(n_samples):
    toy_sample = np.zeros(0).reshape((0, 2, 1, 1))
    sample_group_sz = np.random.multinomial(n_samples, weights)
    for i in range(n_comp):
      sample_radii = radius*(i+1) + std * np.random.randn(sample_group_sz[i])
      sample_thetas = 2 * np.pi * np.random.random(sample_group_sz[i])
      sample_x = sample_radii.reshape(-1, 1) * np.cos(sample_thetas).reshape(
          -1, 1)
      sample_y = sample_radii.reshape(-1, 1) * np.sin(sample_thetas).reshape(
          -1, 1)
      sample_group = np.concatenate((sample_x, sample_y), axis=1)
      toy_sample = np.concatenate(
          (toy_sample, sample_group.reshape((-1, 2, 1, 1))), axis=0)
    return toy_sample[:, :, 0, 0]

  return neg_energy, sample