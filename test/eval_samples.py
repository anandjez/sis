from stint_sampler.targets.targets import gmm,funnel,double_well
import jax.numpy as jnp

target,sampler = gmm()
X = sampler(10000)
print("GMM:","Mean Absolute=",jnp.mean(jnp.sum(jnp.abs(X),axis=-1)),"Mean Squared=",jnp.mean(jnp.sum(X**2,axis=-1)))

target,sampler = funnel()
X = sampler(10000)
print("Funnel:","Mean Absolute=",jnp.mean(jnp.sum(jnp.abs(X),axis=-1)),"Mean Squared=",jnp.mean(jnp.sum(X**2,axis=-1)))