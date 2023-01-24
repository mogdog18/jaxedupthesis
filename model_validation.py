import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from models import Reactor, Separator, Species, Process, MixSplit
from cyipopt import minimize_ipopt

reactor = Reactor()
process = Process()
species = Species()
separator = Separator()

if __name__ == '__main__':
    # # Temperature vs length plots
    # P = 68.2, T = 485
    Z = jnp.linspace(0.0, reactor.z, reactor.no_ode_steps)
    plt.figure()
    initial_temperature = jnp.array([498.0])
    ratio_CO_CO2 = jnp.array([0.81])
    F1 = jnp.array([process.feed_carbon_content * ratio_CO_CO2[0], process.feed_carbon_content * (1 - ratio_CO_CO2)[0],
                    process.F_i0[2], process.F_i0[3], process.F_i0[4], process.F_i0[5], process.F_i0[6]])
    sol = reactor.mass_energy_balance(initial_temperature, process.F_i0)
    sol = reactor.mass_energy_balance(initial_temperature, F1)
    output_reactor_massflows = sol[-1, 0:7]*process.MW_i*3600/1000
    plt.plot(Z, sol[:, 7], markersize=1 * 0.5, color='#E182FD')
    plt.xlabel("Reactor length (m)", fontsize=12)
    plt.ylabel("Temperature (K)", fontsize=12)
    plt.savefig("T_vs_length")
    # Methanol vs length plots
    fig, ax = plt.subplots()
    colour_collection = ['#E182FD', '#6E98F8', '#4BA89B', '#9424B3', "#3855C9", "#27695C", "#EFC0FD"]
    plt.plot(Z, sol[:, 0], color=colour_collection[0])
    plt.plot(Z, sol[:, 1], color=colour_collection[1])
    plt.plot(Z, sol[:, 2], color=colour_collection[2])
    # plt.plot(Z, sol[:, 3], color=colour_collection[3])
    plt.plot(Z, sol[:, 4], color=colour_collection[4])
    plt.plot(Z, sol[:, 5], color=colour_collection[5])
    plt.plot(Z, sol[:, 6], color=colour_collection[6])
    plt.xlabel("Reactor length (m)", fontsize=12)
    plt.ylabel("Molar flowrate (mol/s)", fontsize=12)
    plt.ylim([0, 180])
    plt.legend(["CO", "CO2", "CH3OH", "H2O", "CH4", "N2"], prop={'size': 9})
    plt.savefig("smallreactants_vs_length")

    fig, ax = plt.subplots()
    colour_collection = ['#E182FD', '#6E98F8', '#4BA89B', '#9424B3', "#3855C9", "#27695C", "#EFC0FD"]
    plt.plot(Z, sol[:, 0], color=colour_collection[0])
    plt.plot(Z, sol[:, 1], color=colour_collection[1])
    plt.plot(Z, sol[:, 2], color=colour_collection[2])
    plt.plot(Z, sol[:, 4], color=colour_collection[3])
    plt.plot(Z, sol[:,3], color=colour_collection[4])
    plt.xlabel("Reactor length (m)", fontsize=12)
    plt.ylabel("Molar flowrate (mol/s)", fontsize=12)
    plt.legend(["CO", "CO2", "CH3OH", "H2O", "H2"], prop={'size': 9}, loc=7)
    plt.savefig("species_vs_length")

    fig, ax = plt.subplots()
    for T in jnp.linspace(430, 600, 10):
        initial_temperature = jnp.array([T])  # K
        sol = reactor.mass_energy_balance(initial_temperature, process.F_i0)
        plt.plot(Z, sol[:, 7], markersize=1 * 0.5)
    plt.xlabel("Reactor length (m)")
    plt.ylabel("Temperature (K)")
    plt.title("Temperature profile for various feed temperatures")
    plt.savefig("multi_T_vs_length")

    plt.figure()
    for T in jnp.linspace(430, 600, 10):
        initial_temperature = jnp.array([T])  # K
        sol = reactor.mass_energy_balance(initial_temperature, process.F_i0)
        plt.plot(Z, sol[:, 2], markersize=1 * 0.5)
    plt.xlabel("Length of reactor (m)")
    plt.ylabel("Methanol molar flow (mol/s)")
    plt.title("Temperature profile for various feed temperatures")
    plt.savefig("multi_CH3OH_vs_length")

    # K values plots
    K_COCO_varyT = jnp.array([[227.332, 4.83554,  0.00748350, 313.541,  0.00196173, 52.1751, 123.230],
        [214.178, 5.07306,  0.00943050, 289.159, 0.00254384, 50.6072, 118.608],
        [202.270, 5.31375, 0.0117836, 267.523, 0.00326869, 49.1658, 114.368],
        [191.514, 5.55838, 0.0146061, 248.310, 0.00416361, 47.8505, 110.505],
                        [181.829, 5.80799, 0.0179677, 231.247, 0.00525944, 46.6621, 107.015]])
    T_COCO = jnp.array([310.0, 315.0, 320.0, 325.0, 330.0])
    K_COCO_varyP = jnp.array([[100, 2, 3, 4, 5, 6, 7],  # species K values at T1
                              [200, 2, 3, 4, 5, 6, 7],
                              [300, 2, 3, 4, 5, 6, 7],
                              [100, 2, 3, 4, 6, 7, 8],
                              [10, 2, 3, 4, 5, 2, 12]])
    P_COCO = jnp.array([60.0, 61.0, 62.0, 63.0, 64.0])
    y_i = jnp.array([0.020771939, 0.0595115259, 0.01268075, 0.77971855, 0.0013109827, 0.060903278, 0.065102974])
    x_i = jnp.array([0.00011514431, 0.010469182, 0.72283337, 0.0033709576, 0.26127321, 0.0013214924, 0.00061663844])
    P = jnp.array([60.0])
    Ki_s = []
    T_range = jnp.linspace(310, 330, 30)
    F_i = jnp.array([23.098533, 106.55632, 132.92926, 1026.2555, 44.717035, 75.103268, 80.121109])
    for T in T_range:
        Ki, V, info = separator.separator_loop(F_i, T, P)
        Ki_s.append(Ki)
    Ki_s = jnp.asarray(Ki_s)
    plt.figure()
    plt.ylabel("K value", fontsize=11)
    plt.xlabel("Temperature (K)", fontsize=11)
    plt.ylim([0, 400])
    colour_collection = ['#E182FD', '#6E98F8','#4BA89B', '#9424B3', "#3855C9", "#27695C", "#EFC0FD"]
    for i in range(7):
             plt.plot(T_range, Ki_s[:, i], color=colour_collection[i])
    plt.legend(["CO", "CO2", "CH3OH", "H2", "H2O", "CH4", "N2"], prop={'size': 6})
    for i in range(7):
             plt.plot(T_COCO, K_COCO_varyT[:, i], "o", color=colour_collection[i])
    plt.savefig("Ki_vs_T.svg", format='svg', dpi=1200)

    Ki_s = []
    P_range = jnp.linspace(60, 65, 50)
    T = jnp.array([320.0])
    for P in P_range:
        Ki = species.K_values(T, P, y_i, x_i)
        Ki_s.append(Ki)
    Ki_s = jnp.asarray(Ki_s)
    plt.figure()
    plt.ylabel("K value")
    plt.xlabel("Pressure (bar)")
    for i in range(7):
             plt.plot(P_range, Ki_s[:, i], color=colour_collection[i])
    plt.legend(["CO", "CO2", "CH3OH", "H2", "H2O", "CH4", "N2"], prop={'size': 6})
    for i in range(7):
             plt.plot(P_COCO, K_COCO_varyP[:, i], "o", color=colour_collection[i])
    plt.title("Effect of pressure on K values")
    plt.savefig("Ki_vs_P")




