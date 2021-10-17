import pytest
from Trial import Reactor
import chex
import jax.numpy as jnp

@pytest.fixture
def reactor():
    reactor = Reactor()
    return reactor


def test_pfr(reactor):
    F_i0 = jnp.array([10727.0, 23684.2, 756.7, 9586.5, 108.8, 4333.1,
                      8072.0]) / 3600 * 1000 / reactor.MW_i  # mol/s, molar flow of species/tube
    T = jnp.array([530.0])
    var = jnp.concatenate((F_i0, T))
    z = jnp.array([2.0])
    output = reactor.pfr(var, z)
    chex.assert_shape(output, (8,))
