import jax
from jax import grad
import jax.numpy as jnp
from jax.experimental.ode import odeint as jodeint
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from functools import partial


#@partial(jax.jit, static_argnums=(0, 3))
#def func_to_be_jitted(self, array1, array2, string):
#    return jnp.sum(array1) + jnp.sum(array2)

def heat_of_formation(T):
    heat_of_formation_298 = jnp.array([-110.525, -393.509, -200.660, 0.0, -241.818])
    A = jnp.array([3.376, 5.457, 2.211, 3.249, 3.47, 1.702, 3.28, 3.639])
    B = jnp.array([0.557, 1.045, 12.216, 0.422, 1.45, 9.081, 0.593, 0.506])*10**3
    C = jnp.array([0.0, 0.0, -3.45, 0.0, 0.0, -2.164, 0.0, 0.0])*10**6
    D = jnp.array([-0.031, -1.157, 0.0, 0.083, 0.121, 0.0, 0.04, -0.227])*10**-5
    T_0 = 298   # K
    R = 8.314
    delta_constants_i = jnp.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, -1],
                                   [0, 0, 0, -2, 0, 0, 0, -0.5],
                                   [0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, -1, 0, 0, 0, -0.5]])
    delta_A = delta_constants_i @ jnp.transpose(A)
    delta_B = delta_constants_i @ jnp.transpose(B)
    delta_C = delta_constants_i @ jnp.transpose(C)
    delta_D = delta_constants_i @ jnp.transpose(D)
    delta_Cp = delta_A + delta_B / 2 * T_0 * (T / T_0 + 1) + delta_C / 3 * T_0 ** 2 * (
                (T / T_0) ** 2 + T / T_0 + 1) + delta_D / T / T_0
    heat_of_formation = heat_of_formation_298 + delta_Cp*R*(T - T_0)
    return heat_of_formation

def kinetic_model(T):
    R = 8.314

    k_constants = jnp.array([[1.07, 36.696], [1.22 * 10 ** 10, -94.765]])
    K_constants = jnp.array([[3453.38, 0.0], [0.499, 17.197], [6.62 * 10 ** -11, 124.119]])  # K by A/B
    Keq_constants = jnp.array([[3066.0, 10.592], [-2073.0, -2.029]])  # K by A/B

    k = k_constants[:, 0] * jnp.exp(k_constants[:, 1] / R / T)
    K = K_constants[:, 0] * jnp.exp(K_constants[:, 1] / R / T)
    Keq = 10.0 ** (Keq_constants[:, 0] / T - Keq_constants[:, 1])

    return jnp.concatenate((k, K, Keq))

def reactor(var0):
    # constant parameters
    A = 3.0  # cross-sectional area of reactor tube
    e_cat = 1.0  # void fraction of catalyst bed
    p_cat = 1.0  # density of catalyst bed
    a = 1.0  # activity of catalyst
    D_i = 1.0  # internal diameter of reactor tube
    MW_i = jnp.array([28.01, 44.01, 32.04, 1.0079, 18.015, 16.04, 28.0134])  # molecular weights (CO, CO2, CH3OH, H2, H20, CH4, N2)
    U_shell = 850  # overall heat transfer coefficient from reactor tube-side to shell-side
    z = 7 # length of PFR
    volume_t = jnp.pi*D_i**2/4*z
    n_i = 1

    Ct = 1  # total molar concentration
    T_shell = 1  # temperature of boiling water on the shell side

    m_i0 = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])   # initial mass flow rate of species into reactor
    total_mass_t = sum(m_i0)    # assume total mass per tube stays constant
    T_0 = 300.0     # initial temperature of stream

    Z = jnp.arange(0.0, z, 0.1)

    sol = odeint(pfr, var_0, Z)
    return sol[-1, 2]

def pfr(var, z):
    x = var[0:7]
    T = var[7]
    # need  r_i - need it in terms of a limiting reagent thing
    m_i = x*total_mass_t
    mt = sum(m_i)
    Cm = total_mass_concentration(m_i)
    Ct = total_molar_concentration(m_i)
    y_i = molar_fraction(m_i)
    p = partial_pressure(m_i, P)
    r_i = rate_expression(T, p)
    Cp = specific_heat(T, y_i) ###### fix
    Hf = heat_of_formation(T)

    dxdz = n * r_i * p_cat * a / e_cat / Cm * MW_i * A * e_cat * Cm / mt
    dTdz = jnp.array([p_cat*a/e_cat/Ct/Cp * (sum(n_i*r_i[0:5]*-Hf)) + jnp.pi*D_i/A/e_cat/Ct/Cp*U_shell*(T_shell-T)])

    return jnp.concatenate((dxdz, dTdz))



# before obj func added
import jax
from jax import grad
import jax.numpy as jnp
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from functools import partial

plt.interactive(False)

class species:
    heat_of_formation_298 = jnp.array([-110.525, -393.509, -200.660, 0.0, -241.818])
    A = jnp.array([3.376, 5.457, 2.211, 3.249, 3.47, 1.702, 3.28, 3.639])
    B = jnp.array([0.557, 1.045, 12.216, 0.422, 1.45, 9.081, 0.593, 0.506]) * 10 ** 3
    C = jnp.array([0.0, 0.0, -3.45, 0.0, 0.0, -2.164, 0.0, 0.0]) * 10 ** 6
    D = jnp.array([-0.031, -1.157, 0.0, 0.083, 0.121, 0.0, 0.04, -0.227]) * 10 ** -5
    T_0 = 298  # K
    R = 8.314

    def heat_of_formation(self, T):
        T_0 = species.T_0
        delta_constants_i = jnp.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0, -1],
                                       [0, 0, 0, -2, 0, 0, 0, -0.5],
                                       [0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, -1, 0, 0, 0, -0.5]])
        delta_A = delta_constants_i @ jnp.transpose(species.A)
        delta_B = delta_constants_i @ jnp.transpose(species.B)
        delta_C = delta_constants_i @ jnp.transpose(species.C)
        delta_D = delta_constants_i @ jnp.transpose(species.D)
        delta_Cp = delta_A + delta_B / 2 * T_0 * (T / T_0 + 1) + delta_C / 3 * T_0 ** 2 * ((T / T_0) ** 2 + T / T_0 + 1) + delta_D / T / T_0
        heat_of_formation = species.heat_of_formation_298 + delta_Cp * species.R * (T - T_0)
        return heat_of_formation

    def specific_heat(self, T, y_i):
        A = species.A[0:7]
        B = species.B[0:7]
        C = species.C[0:7]
        D = species.D[0:7]
        Cp_i = (A + B / T + C * T ** 2 + D / T ** 2) * species.R
        Cp = sum(y_i * Cp_i)
        return Cp

# species: CO, CO2, CH3OH, H2, H20, CH4, N2
class reactor:
    # constant parameters
    species = species()
    R = 8.314   # J/mol.K
    P = 51.7*100    # bar
    e_cat = 0.38  # void fraction of catalyst bed
    p_cat = 1100  # kg/m3, density of catalyst bed
    a = 1.0  # activity of catalyst,
    D_i = 0.033  # m, internal diameter of reactor tube
    MW_i = jnp.array(
        [28.01, 44.01, 32.04, 1.0079, 18.015, 16.04, 28.0134])  # molecular weights (CO, CO2, CH3OH, H2, H20, CH4, N2)
    U_shell = 631  # W/m2K, overall heat transfer coefficient from reactor tube-side to shell-side
    z = 7  # m, length of PFR
    volume_t = jnp.pi * D_i ** 2 / 4 * z
    A = jnp.pi*D_i**2/4  # cross-sectional area of reactor tube
    n_i = 1
    T_shell = 511  # K, temperature of boiling water on the shell side

    def __init__(self, initial_temperature, initial_mass_flowrate_t):
        self.T0 = initial_temperature
        self.m_i0 = initial_mass_flowrate_t
        self.total_mass_t = sum(initial_mass_flowrate_t)    # assume mass flowrate remains constant

    def kinetic_model(self, T):
        k_constants = jnp.array([[1.07, 36.696], [1.22 * 10 ** 10, -94.765]])
        K_constants = jnp.array([[3453.38, 0.0], [0.499, 17.197], [6.62 * 10 ** -11, 124.119]])  # K by A/B
        Keq_constants = jnp.array([[3066.0, 10.592], [-2073.0, -2.029]])  # K by A/B

        k = k_constants[:, 0] * jnp.exp(k_constants[:, 1] / reactor.R / T)
        K = K_constants[:, 0] * jnp.exp(K_constants[:, 1] / reactor.R / T)
        Keq = 10.0 ** (Keq_constants[:, 0] / T - Keq_constants[:, 1])
        return jnp.concatenate((k, K, Keq))

    def rate_expression(self, T, m_i):
        p = self.partial_pressure(m_i)
        p_CO = p[0]
        p_CO2 = p[1]
        p_CH3OH = p[2]
        p_H2 = p[3]
        p_H20 = p[4]

        reaction_equilibrium_constants = self.kinetic_model(T)  # kMeOH, kRWGS, K1, K2, K3, K1eq, K2eq
        k_CH3OH = reaction_equilibrium_constants[0]
        k_RWGS = reaction_equilibrium_constants[1]
        K_1 = reaction_equilibrium_constants[2]
        K_2 = reaction_equilibrium_constants[3]
        K_3 = reaction_equilibrium_constants[4]
        K_1eq = reaction_equilibrium_constants[5]
        K_2eq = reaction_equilibrium_constants[6]

        rate_CH3OH = k_CH3OH * p_CO2 * p_H2 * (1 - (1 / K_1eq) * p_H20 * p_CH3OH / (p_H2 * p_CO2)) / (
                1 + K_1 * (p_H20 / p_H2) + K_2 * (p_H2) ** 0.5 + K_3 * p_H20) ** 3
        rate_RWGS = k_RWGS * p_CO2 * (1 - K_2eq * (p_H20 * p_CO) / (p_CO2 * p_H2)) / (
                1 + K_1 * (p_H20 / p_H2) + K_2 * (p_H2) ** 0.5 + K_3 * p_H20)

        stoich_coefficient_CH3OH = jnp.array([-1.0, 0.0, 1.0, -2.0, 0.0, 0.0, 0.0])
        stoich_coefficient_RWGS = jnp.array([1.0, -1.0, 0.0, -1.0, 1.0, 0.0, 0.0])
        r_i = stoich_coefficient_CH3OH*rate_CH3OH + stoich_coefficient_RWGS*rate_RWGS
        return r_i

    def total_mass_concentration(self, m_i):   # per tube
        C_m = sum(m_i) / reactor.volume_t
        return C_m

    def molar_concentration(self, m_i):   # per tube
        C_i = m_i/reactor.MW_i / reactor.volume_t
        return C_i

    def total_molar_concentration(self, m_i):
        C_t = sum(m_i/reactor.MW_i) / reactor.volume_t
        return C_t

    def mass_fraction(self, m_i):
        x_i = m_i / sum(m_i)
        return x_i

    def molar_fraction(self, m_i):
        moles = m_i / reactor.MW_i
        y_i = moles / sum(moles)
        return y_i

    def partial_pressure(self, m_i):
        y_i = (m_i / reactor.MW_i) / sum((m_i / reactor.MW_i))
        p = y_i*reactor.P
        return p

    def var_0(self):
        x_i = self.mass_fraction(self.m_i0)
        T_0 = jnp.array([self.T0])
        return jnp.concatenate((x_i, T_0))

    @partial(jax.jit, static_argnums=(0))
    def pfr(self, var, z):
        x = var[0:7]
        T = var[7]
        # need  r_i - need it in terms of a limiting reagent thing

        mt = self.total_mass_t
        m_i = x * mt
        Cm = self.total_mass_concentration(m_i)
        Ct = self.total_molar_concentration(m_i)
        y_i = self.molar_fraction(m_i)
        r_i = self.rate_expression(T, m_i)
        Cp = self.species.specific_heat(T, y_i)
        Hf = self.species.heat_of_formation(T)

        dxdz = reactor.n_i * r_i * reactor.p_cat * reactor.a / reactor.e_cat / Cm * reactor.MW_i * reactor.A * reactor.e_cat * Cm / mt
        dTdz = jnp.array([reactor.p_cat * reactor.a / reactor.e_cat / Ct / Cp * (jnpsum(reactor.n_i * r_i[0:5] * -Hf)) + jnp.pi * reactor.D_i / reactor.A / reactor.e_cat / Ct / Cp * reactor.U_shell * (reactor.T_shell - T)])

        return jnp.concatenate((dxdz, dTdz))

    def mass_energy_balance(self):
        Z = jnp.arange(0.0, self.z, 0.1)
        var_0 = self.var_0()
        sol = jodeint(self.pfr, var_0, Z)
        return sol

learning_rate = 1e-3
optimizer = adam(learning_rate)

def step(dept, opt_state):
    grads = jax.grad(objective_function)(dept)
    updates, opt_state = optimizer.update(grads, opt_state)
    dept = optax.apply_updates(dept,updates)
    return dept
step_jit = jit(step)

def loop_jit(dept):
  deptvalues = []
  opt_state = optimizer.init(dept)

  for _ in range(100):
    dept = step_jit(dept, opt_state)
    deptvalues.append(dept)
  return deptvalues

if __name__ == '__main__':
    initial_mass_flowrate_t = jnp.array([0.285, 0.032, 0, 0.675, 0, 0.006, 0.003])*2125/6560  # kg/h, mass flow rate/tube
    initial_temperature = 523   # K
    reactor = reactor(initial_temperature, initial_mass_flowrate_t)
    sol = reactor.mass_energy_balance()
    Z = jnp.arange(0.0, 7.0, 0.1)
    plt.show()
    plt.plot(Z, sol[:,7])
    print(sol[:,-1])

#plt.show()
#plt.plot([1, 2, 3])

class Separator:
    species = Species()

    def __init__(self, variable):
        self.variable = variable

    def rachford_rice_call(self, F, y_i, T, P):
        Ki = species.K_values(T, P)

    def rachford_rice_equation(self, V):

        RR = y_i * (Ki - 1) / (1 + )


    #@partial(jax.jit, static_argnums=(0))
    def step(self, dept, opt_state):
        grads = jax.grad(self.objective_function)(dept)
        updates, opt_state = reactor.optimizer.update(grads, opt_state)
        dept = optax.apply_updates(dept, updates)
        return dept

    def loop(self, dept):
        deptvalues = []
        opt_state = reactor.optimizer.init(dept)
        for _ in range(self.no_optim_steps):
            dept = self.step(dept, opt_state)
            deptvalues.append(dept)
        return deptvalues

# above pfr
#@partial(jax.jit, static_argnums=(0))




    def rachford_rice_call(self, P, T):
        V0 = 0.5 * self.F  # initial guess at V
        # option 1
        minimise_func = lambda V: self.rachford_rice_equation(P, T, V)

        ## this lambda function is equivalent to
        def minimise_func(V):
            return self.rachford_rice_equation(P, T, V)  # grab P and T from outside of function scope

        # option 2, I think this is the most clean
        from functools import partial
        minimise_func = partial(self.rachford_rice_equation, P, T)
        V_value = minimize(minimise_func, jnp.array([V0]), P, method='bfgs')[0]

        return V_value
