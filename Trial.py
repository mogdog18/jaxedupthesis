import jax
from jax import grad, vmap
import jax.numpy as jnp
from jax.experimental.ode import odeint as jodeint
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from functools import partial
import optax
from optax import adam
from scipy.optimize import differential_evolution, dual_annealing, minimize
from jax.config import config
config.update("jax_enable_x64", True)
jax.config.update('jax_disable_jit', True)

class Species:
    heat_of_formation_298 = jnp.array([-110.525, -393.509, -200.660, 0.0, -241.818]) * 1000  # J
    A = jnp.array([3.376, 5.457, 2.211, 3.249, 3.47, 1.702, 3.28, 3.639])
    B = jnp.array([0.557, 1.045, 12.216, 0.422, 1.45, 9.081, 0.593, 0.506]) * 10 ** -3
    C = jnp.array([0.0, 0.0, -3.45, 0.0, 0.0, -2.164, 0.0, 0.0]) * 10 ** -6
    D = jnp.array([-0.031, -1.157, 0.0, 0.083, 0.121, 0.0, 0.04, -0.227]) * 10 ** 5
    T_0 = 298  # K
    R = 8.314  # J/mol.K

    def heat_of_formation(self, T):
        T_0 = Species.T_0
        delta_constants_i = jnp.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0, -1],
                                       [0, 0, 0, -2, 0, 0, 0, -0.5],
                                       [0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, -1, 0, 0, 0, -0.5]])
        delta_A = delta_constants_i @ jnp.transpose(Species.A)
        delta_B = delta_constants_i @ jnp.transpose(Species.B)
        delta_C = delta_constants_i @ jnp.transpose(Species.C)
        delta_D = delta_constants_i @ jnp.transpose(Species.D)
        delta_Cp = delta_A + delta_B / 2 * T_0 * (T / T_0 + 1) + delta_C / 3 * T_0 ** 2 * (
                    (T / T_0) ** 2 + T / T_0 + 1) + delta_D / T / T_0
        heat_of_formation = Species.heat_of_formation_298 + delta_Cp * Species.R * (T - T_0)
        return heat_of_formation

    def specific_heat(self, T, y_i):
        A = Species.A[0:7]
        B = Species.B[0:7]
        C = Species.C[0:7]
        D = Species.D[0:7]
        Cp_i = (A + B * T + C * T ** 2 + D / (T ** 2)) * Species.R
        Cp = sum(y_i * Cp_i)
        return Cp

    def K_values(self, T, P):
        A = jnp.array([6.72527, 7.58828, 18.5875, 6.14858, 8.05573, 6.84566, 6.72531])
        B = jnp.array([295.228, 861.82, 3625.55, 80.948, 1723.64, 435.621, 285.573])
        C = jnp.array([269.243, 271.883, -34.29, 277.532, 233.076, 271.361, 270.087])
        Psat_i = 10 ** (A - B / (C + T)) * 133.322 * 10 ** -5  # bar
        K_values = Psat_i / P
        return K_values

# species: CO, CO2, CH3OH, H2, H20, CH4, N2
class Reactor:
    # constant parameters
    species = Species()

    R = 8.314472  # m3/bar/K/mol
    P = 69.7  # bar
    e_cat = 0.39  # void fraction of catalyst bed
    p_cat = 1100.0  # kg/m3, density of catalyst bed
    a = 1.0  # activity of catalyst,
    D_i = 0.038  # m, internal diameter of reactor tube
    MW_i = jnp.array(
        [28.01, 44.01, 32.04, 1.0079, 18.015, 16.04, 28.0134])  # molecular weights (CO, CO2, CH3OH, H2, H20, CH4, N2)
    U_shell = 631.0  # W/m2K, overall heat transfer coefficient from reactor tube-side to shell-side
    z = 7.0  # m, length of PFR
    volume_t = jnp.pi * D_i ** 2 / 4 * z
    A = jnp.pi * D_i ** 2 / 4  # cross-sectional area of reactor tube
    n_i = 1.0
    T_shell = 511.0  # K, temperature of boiling water on the shell side
    F_i0 = jnp.array([10727.0, 23684.2, 756.7, 9586.5, 108.8, 4333.1,
                      8072.0]) / 3600 * 1000 / MW_i  # mol/s, molar flow of species/tube
    Nt = 6560.0
    learning_rate = 1e-2
    optimizer = adam(learning_rate)

    def __init__(self, no_ode_steps=1000, no_optim_steps=1000):
        self.no_ode_steps = no_ode_steps
        self.no_optim_steps = no_optim_steps

    def kinetic_model(self, T):
        k_constants = jnp.array([[1.07, 36696], [1.22 * 10 ** 10, -94765]])
        K_constants = jnp.array([[3453.38, 0.0], [0.499, 17197], [6.62 * 10 ** -11, 124119]])  # K by A/B
        Keq_constants = jnp.array([[3066.0, 10.592], [-2073.0, -2.029]])  # K by A/B

        k = k_constants[:, 0] * jnp.exp(k_constants[:, 1] / reactor.R / T)
        K = K_constants[:, 0] * jnp.exp(K_constants[:, 1] / reactor.R / T)
        Keq = 10.0 ** (Keq_constants[:, 0] / (T) - Keq_constants[:, 1])
        return jnp.concatenate((k, K, Keq))

    def rate_expression(self, T, F_i):
        p = self.partial_pressure(F_i)
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

        rate_CH3OH = k_CH3OH * p_CO2 * p_H2 * (1 - (1 / K_1eq) * p_H20 * p_CH3OH / (p_H2 ** 3 * p_CO2)) / (
                1 + K_1 * (p_H20 / p_H2) + K_2 * (p_H2) ** 0.5 + K_3 * p_H20) ** 3
        rate_RWGS = k_RWGS * p_CO2 * (1 - 1 / K_2eq * (p_H20 * p_CO) / (p_CO2 * p_H2)) / (
                1 + K_1 * (p_H20 / p_H2) + K_2 * (p_H2) ** 0.5 + K_3 * p_H20)

        stoich_coefficient_CH3OH = jnp.array([0.0, -1.0, 1.0, -3.0, 1.0, 0.0, 0.0])
        stoich_coefficient_RWGS = jnp.array([1.0, -1.0, 0.0, -1.0, 1.0, 0.0, 0.0])
        r_i = (stoich_coefficient_CH3OH * rate_CH3OH + stoich_coefficient_RWGS * rate_RWGS)
        return r_i

    def molar_fraction(self, F_i):
        y_i = F_i / jnp.sum(F_i)
        return y_i

    def partial_pressure(self, F_i):
        p = F_i/jnp.sum(F_i) * reactor.P
        return p

    def total_molar_t(self, F_i):
        return jnp.sum(F_i)

    def var_0(self, T_0):
        F_i0 = reactor.F_i0
        return jnp.concatenate((F_i0, T_0))

    @partial(jax.jit, static_argnums=(0))
    def pfr(self, var, z):
        F_i = var[0:7]
        T = var[7]
        ft = self.total_molar_t(F_i)
        y_i = self.molar_fraction(F_i)
        r_i = self.rate_expression(T, F_i)
        Cp = self.species.specific_heat(T, y_i)
        Hf = self.species.heat_of_formation(T)
        Ct = reactor.P / (reactor.R * 10 ** -5) / T

        dFdz = reactor.Nt * reactor.A * reactor.p_cat * (1 - reactor.e_cat) * r_i
        dTdz = jnp.array([(reactor.p_cat * reactor.a / reactor.e_cat / Ct / Cp * (jnp.sum(reactor.n_i * r_i[
                                                                                                        0:5] * -Hf)) + 0*jnp.pi * reactor.D_i / reactor.A / reactor.e_cat / Ct / Cp * reactor.U_shell * (
                                       reactor.T_shell - T)) * reactor.A * reactor.e_cat * Ct / ft])
        return jnp.concatenate((dFdz, dTdz))

    def mass_energy_balance(self, T_0):
        Z = jnp.linspace(0.0, self.z, self.no_ode_steps)
        var_0 = self.var_0(T_0)
        sol = jodeint(self.pfr, var_0, Z)
        return sol

    def objective_function(self, T_0):
        final_methanol_flowrate = self.mass_energy_balance(T_0)[-1, 2]  # mol/s, total methanol out reactor
        return -final_methanol_flowrate

    @partial(jax.jit, static_argnums=(0))
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


class Separator:
    species = Species()
    T = 600.0  # temperature of separator, assumed constant

    def __init__(self, F, y_i):
        self.F = F  # total flowrate to separator
        self.y_i = y_i  # composition of feed to separtor

    def rachford_rice_call(self, P, T):
        V0 = 0.5 * self.F  # initial guess at V
        minimise_func = partial(self.rachford_rice_equation, T, P)
        V_value = minimize(minimise_func, jnp.array([V0]), method='bfgs')[0]
        return V_value

    def rachford_rice_equation(self, T, P, V):
        Ki = self.species.K_values(T, P)
        RR = jnp.sum(self.y_i * (Ki - 1) / (1 + V / self.F * (1 + Ki)))
        return jnp.abs(RR)

if __name__ == '__main__':
    initial_temperature = jnp.array([550.0])  # K
    reactor = Reactor()
    sol = reactor.mass_energy_balance(initial_temperature)
    Z = jnp.linspace(0.0, reactor.z, reactor.no_ode_steps)

    # Temperature vs length plots
    plt.plot(Z, sol[:, 7], 'ro', markersize=1 * 0.5)
    plt.xlabel("Length of reactor (m)")
    plt.ylabel("Temperature (K)")
    plt.savefig("T vs length")

    # Methanol vs length plots
    # plt.plot(Z, sol[:,2])
    # plt.xlabel("reactor length (m)")
    # plt.ylabel("Methanol molar flow (mol/s)")
    # plt.savefig("CH3OH mole flow vs length")

    # # Rate plots
    temperature_range = jnp.linspace(298,600,100)
    # r_i = []
    # # Cp = []
    # # Hf = []
    # F_i0 = jnp.array([10727.0, 23684.2, 756.7, 9586.5, 108.8, 4333.1,
    #                    8072.0]) / 3600 * 1000 / reactor.MW_i  # mol/s, molar flow of species/tube
    # y_i = reactor.molar_fraction(F_i0)
    # species = Species()
    # for temp in temperature_range:
    #      rates = reactor.rate_expression(temp, F_i0)
    # #     Cps = species.specific_heat(temp, y_i)
    # #     Hfs = species.heat_of_formation(temp)
    #      r_i.append(rates)
    # #     Cp.append(Cps)
    # #     Hf.append(Hfs)
    # # Hf = jnp.asarray(Hf)
    # plt.plot(temperature_range, r_i)
    # plt.ylabel("Heat of formation (J/mol)")
    # plt.xlabel("temperature (K)")
    # plt.savefig("rates 2")

    # optimization
    # start_point = jnp.array([540.0])
    # T_values = reactor.loop(start_point)
    # methanol_flowrate = vmap(reactor.objective_function)(jnp.array(T_values)) * -1
    # methanol_flowrate = jnp.reshape(methanol_flowrate, [1, reactor.no_optim_steps])*-1

    # fig, axs = plt.subplots(2)
    # fig.suptitle('Flow/T vs length')
    # axs[0].plot(jnp.linspace(0,reactor.no_optim_steps, reactor.no_optim_steps),methanol_flowrate)
    # axs[0].set(ylabel = "Methanol flowrate out (mol/S)")
    # axs[1].plot(jnp.linspace(0,reactor.no_optim_steps, reactor.no_optim_steps),T_values)
    # plt.xlabel("iteration")
    # axs[1].set(ylabel = "Temperature in (K)")
    # plt.savefig("flowrate vs iteration")

    # values = []
    #
    #
    # def callback(xk, convergence=0.1):
    #     values.append(xk)
    #
    #
    # bounds = [(200, 900)]
    # result_DE = differential_evolution(reactor.objective_function, bounds, callback=callback)
    # print(result_DE)
    # plt.plot(values, 'ro')
    # plt.savefig("DE")
    # result_DA = dual_annealing(reactor.objective_function, bounds)
    # print(result_DA)

    # Separator
    # F = jnp.array([30.0])
    # y_i = jnp.array([0.1, 0.7, 0.1, 0.1, 0.0, 0.0, 0.0])
    # separator = Separator(F, y_i)
    # print(separator.rachford_rice_call(jnp.array([30.0]), jnp.array([500.0])))


