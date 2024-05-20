import jax.numpy as jnp

def linear(type):
    if type=='trig_const':
        g = lambda t: jnp.sin(jnp.pi * t / 2)
        r = lambda t: 1.0#jnp.cos(jnp.pi * t / 2)
    elif type=='trig_lin':
        g = lambda t: jnp.sin(jnp.pi * t / 2)/2
        r = lambda t: 1.0-t/2#jnp.cos(jnp.pi * t / 2)
    elif type=='lin':
        g = lambda t: t/2
        r = lambda t: 1.0-t/2#jnp.cos(jnp.pi * t / 2)\
    elif type=='lin_inc':
        g = lambda t: t/2
        r = lambda t: 1.0+t/3#jnp.cos(jnp.pi * t / 2)
    elif type=='trig_inc':
        g = lambda t: jnp.sin(jnp.pi * t / 2)
        r = lambda t: 1.0+t/3#jnp.cos(jnp.pi * t / 2)
    else:
        g = lambda t: jnp.sin(jnp.pi * t / 2)
        r = lambda t: 1.0#jnp.cos(jnp.pi * t / 2)
    return g,r