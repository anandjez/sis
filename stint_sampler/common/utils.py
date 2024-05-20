import jax.numpy as jnp
from jax import grad,vmap
def target_init(t,x):
    pass

def generateIntrplntFn(f_scalar):
    def f(t):
        # if type(t) == int or type(t)==float:
        if hasattr(t,"ndim"):
            axes = t.ndim
            if axes == 1:
                return vmap(f_scalar)(t)
            elif axes==2:
                return vmap(f_scalar)(t[:, 0])[:, None]
            else:
                return f_scalar(t)
        else:
            return f_scalar(t)

    return f