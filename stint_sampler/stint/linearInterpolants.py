import jax.numpy as jnp

def linear(type):
    if type=='trig':
        g = lambda t: jnp.sin(jnp.pi * t / 2)
        r = lambda t: jnp.cos(jnp.pi * t / 2)
    elif type=='trig2':
        g = lambda t: jnp.sin(jnp.pi * t / 2)**2
        r = lambda t: jnp.cos(jnp.pi * t / 2)**2
    elif type=='lin':
        g = lambda t: t
        r = lambda t: 1.0-t#jnp.cos(jnp.pi * t / 2)
    elif type=='enc_dec':
        g = lambda t: (jnp.sin(jnp.pi * t)**2)*(t<0.5)+1.0*(t>=0.5)
        r = lambda t: (jnp.sin(jnp.pi * t)**2)*(t>=0.5)+1.0*(t<0.5)
    elif type=='enc_sin':
        g = lambda t: jnp.sin(jnp.pi * t/2)
        r = lambda t: (jnp.sin(jnp.pi * t))*(t>=0.5)+1.0*(t<0.5)
    elif type=='enc_sin_inc':
        g = lambda t: jnp.sin(jnp.pi * t/2)
        r = lambda t: (1.0+t/4)*(jnp.sin(jnp.pi * t))*(t>=0.5)+(1.0+t/4)*(t<0.5)
    elif type=='trig_const':
        g = lambda t: jnp.sin(jnp.pi * t / 2)
        r = lambda t: 1.0*t/t#jnp.cos(jnp.pi * t / 2)
    elif type=='half_lin':
        g = lambda t: t
        r = lambda t: 1.0-t/2#jnp.cos(jnp.pi * t / 2)\
    elif type=='half_trig':
        g = lambda t: jnp.sin(jnp.pi * t / 2)
        r = lambda t: 1.0-t/2#jnp.cos(jnp.pi * t / 2)\
    elif type=='lin_inc':
        g = lambda t: t
        r = lambda t: 1.0+t/8#jnp.cos(jnp.pi * t / 2)
    else:
        g = lambda t: jnp.sin(jnp.pi * t / 2)
        r = lambda t: 1.0#jnp.cos(jnp.pi * t / 2)
    return g,r