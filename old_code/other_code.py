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
    process = Process(epsilon=10 **-8)
    reactor = Reactor()
    separator = Separator()
    species = Species()
    mixsplit = MixSplit()

    # Base Case Initial Guess - taken from BaseCase_and_UpperBound code
    x = jnp.array(
        [1.51638438e+02, 2.32481901e+02, 1.28277974e+01, 2.32651356e+03, 2.42826475e+00, 1.49902708e+02, 1.60015287e+02,
         9.05145757e+01, 1.68114085e+02, 1.38319476e+02, 2.01116239e+03, 6.67960808e+01,
         1.49902708e+02, 1.60015287e+02, 9.04954308e+01, 1.65979990e+02,
         1.25352771e+01, 2.01082532e+03, 1.50124123e+00, 1.49745027e+02,
         1.59950438e+02, 4.52477154e+01, 8.29899952e+01, 6.26763854e+00,
         1.00541266e+03, 7.50620617e-01, 7.48725133e+01, 7.99752192e+01,
         4.52477154e+01, 8.29899952e+01, 6.26763854e+00, 1.00541266e+03,
         7.50620617e-01, 7.48725133e+01, 7.99752192e+01, 1.91448682e-02,
         2.13409470e+00, 1.25784199e+02, 3.37069196e-01, 6.52948396e+01,
         1.57681585e-01, 6.48489600e-02, 500, 308, 60.0, 5.47420626e+02, 0.529, 0.5])
    bons = [(0, 5.12095147e+02), (0, 6.71121273e+02), (0, 5.48100450e+01), (0, 9.67574347e+03),
            (0, 8.86013966e+00), (0, 7.48512296e+02), (0, 7.99687990e+02), (0, 3.74186230e+02),
            (0, 4.68736806e+02), (0, 3.95103428e+02), (0, 8.79277224e+03), (0, 2.11244607e+02),
            (0, 7.48512296e+02), (0, 7.99687990e+02), (0, 3.74142127e+02), (0, 4.65171826e+02),
            (0, 5.21121591e+01), (0, 8.79192709e+03), (0, 6.88106424e+00), (0, 7.48064883e+02),
            (0, 7.99509818e+02), (0, 7.48284253e+01), (0, 9.30343651e+01), (0, 1.04224318e+01),
            (0, 1.75838542e+03), (0, 1.37621285e+00), (0, 1.49612977e+02), (0, 1.59901964e+02),
            (0, 2.99313701e+02), (0, 3.72137461e+02), (0, 4.16897273e+01), (0, 7.03354167e+03),
            (0, 5.50485139e+00), (0, 5.98451906e+02), (0, 6.39607854e+02), (0, 4.41037707e-02),
            (0, 3.56498002e+00), (0, 3.42991269e+02), (0, 8.45154629e-01), (0, 2.04363542e+02),
            (0, 4.47413281e-01), (0, 1.78172716e-01),
            (500, 583), (308, 323), (50, 75), (500, 583), (0.0, 1.0), (0.01, 0.8)]
    # Ipopt - methanol production
    cons_ipopt = [{'type': 'eq', 'fun': process.equality_constraints, 'jac': process.num_grad_equality_constraints}]
    cons_ipopt_jac = [{'type': 'eq', 'fun': process.equality_constraints, 'jac': process.grad_equality_constraints}]
    result_ipopt_jac = minimize_ipopt(process.objective_function, jac=process.grad_objective_function, x0=x,
                                      bounds=bons, constraints=cons_ipopt_jac, options={'disp': 5, "output_file":
            "Ipopt JAX File Print", "file_print_level": 8})
    result_ipopt = minimize_ipopt(process.objective_function, jac=process.num_grad_objective_function, x0=x,
                                  bounds=bons, constraints=cons_ipopt,
                                  options={'disp': 5, "output_file": "Ipopt File Print", "file_print_level": 8})
    S_table_ipopt, OC_table_ipopt, OP_table_ipopt = process.stream_information(result_ipopt, "ipopt")
    S_table_ipopt_jac, OC_table_ipopt_jac, OP_table_ipopt_jac = process.stream_information(result_ipopt_jac,
                                                                                           "ipopt with jac")
    print(result_ipopt)
    print(result_ipopt_jac)
    print(process.F4_conversion(result_ipopt_jac["x"]))
    print(process.F6_conversion(result_ipopt_jac["x"]))
    # Ipopt - cost
    result_ipopt_jac = minimize_ipopt(process.objective_function_cost, jac=process.grad_objective_function_cost, x0=x,
                                      bounds=bons, constraints=cons_ipopt_jac, options={'disp': 5, "output_file":
            "Ipopt JAX File Print", "file_print_level": 8})
    result_ipopt = minimize_ipopt(process.objective_function_cost, jac=process.num_grad_objective_function_cost, x0=x,
                                  bounds=bons, constraints=cons_ipopt,
                                  options={'disp': 5, "output_file": "Ipopt File Print", "file_print_level": 8})
    S_table_ipopt, OC_table_ipopt, OP_table_ipopt = process.stream_information(result_ipopt, "ipopt")
    S_table_ipopt_jac, OC_table_ipopt_jac, OP_table_ipopt_jac = process.stream_information(result_ipopt_jac,
                                                                                           "ipopt with jac")
    print(result_ipopt)
    print(result_ipopt_jac)
    print(process.F6_conversion(result_ipopt_jac["x"]))
    print(process.F4_conversion(result_ipopt_jac["x"]))

    S_table_DE, OC_table_DE, OP_table_DE = process.stream_information(result_DE, "differential evolution")
    history_DE = jnp.asarray(callback.hist)
    plt.figure()
    plt.plot(history_DE[:, 0:3], 'o', markersize=1 * 0.5)
    plt.xlabel("Iteration")
    plt.ylabel("Temperature (K)/ Pressure (bar)")
    plt.legend(["T reactor", "T separator", "P separator"])
    plt.savefig("DE")

    plt.figure()
    plt.plot(history_DE[:, 0], 'o', markersize=1 * 0.5)
    plt.xlabel("Iteration")
    plt.ylabel("Temperature (K)")
    plt.savefig("DE T reactor")

    # # reactor = Reactor()
    # # initial_temperature = jnp.array([523.0])
    # # flow_reactor_out = reactor.mass_energy_balance(initial_temperature, process.F_i0)[-1, 0:7]
    # separator = Separator()
    T_separator = jnp.array([330.0])
    P_separator = jnp.array([60.0])

    flow_coco = jnp.array([386.45788, 1279.0561, 831.93083, 4372.4646, 133.64492, 423.92416, 129.28562])
    print(grad(separator.rachford_rice_equation)(T_separator, P_separator,flow_coco, jnp.array([1398])))
    print(jnp.sum(flow_coco))
    v_param = separator.rachford_rice_call(flow_coco, T_separator, P_separator)
    print(v_param)
    # print(v_param)
    print(separator.mass_balance(flow_coco, T_separator, P_separator))
    print(separator.rachford_rice_equation(T_separator, P_separator, flow_coco, v_param))
    #
    # initial_temperature = jnp.array([523.0])  # K
    reactor = Reactor()
    sol = reactor.mass_energy_balance(initial_temperature, process.F_i0)
    Z = jnp.linspace(0.0, reactor.z, reactor.no_ode_steps)

    # Temperature vs length plots
    plt.figure()
    plt.plot(Z, sol[:, 7], 'ro', markersize=1 * 0.5)
    plt.xlabel("Length of reactor (m)")
    plt.ylabel("Temperature (K)")
    plt.savefig("T vs length")

    # Methanol vs length plots
    plt.figure()
    plt.plot(Z, sol[:, 2])
    plt.xlabel("reactor length (m)")
    plt.ylabel("Methanol molar flow (mol/s)")
    plt.savefig("CH3OH mole flow vs length")

    #optimization
    start_point = jnp.array([540.0])
    T_values = reactor.loop(start_point)
    methanol_flowrate = vmap(reactor.objective_function)(jnp.array(T_values)) * -1
    methanol_flowrate = jnp.reshape(methanol_flowrate, [1, reactor.no_optim_steps])*-1

    fig, axs = plt.subplots(2)
    fig.suptitle('Flow/T vs length')
    axs[0].plot(jnp.linspace(0,reactor.no_optim_steps, reactor.no_optim_steps),methanol_flowrate)
    axs[0].set(ylabel = "Methanol flowrate out (mol/S)")
    axs[1].plot(jnp.linspace(0,reactor.no_optim_steps, reactor.no_optim_steps),T_values)
    plt.xlabel("iteration")
    axs[1].set(ylabel = "Temperature in (K)")
    plt.savefig("flowrate vs iteration")

    F_i = jnp.array([51.019131, 148.22169, 177.31822, 1914.9441, 56.058626, 149.78939, 159.95742])
    T = jnp.array([330.0])
    P = jnp.array([60.0])
    y_i = jnp.array([0.020771939, 0.0595115259, 0.01268075, 0.77971855, 0.0013109827, 0.060903278, 0.065102974])
    x_i = jnp.array([0.00011514431, 0.010469182, 0.72283337, 0.0033709576, 0.26127321, 0.0013214924, 0.00061663844])
    x = jnp.array([520.0, 330.0, 60, 0.7, 0.3, 3.9295303, 15.028284, 3.1656873, 174.12167, 0.28198171, 13.202387, 14.106927])    F_i = jnp.array([51.019131, 148.22169, 177.31822, 1914.9441, 56.058626, 149.78939, 159.95742])
    T = jnp.array([330.0])
    P = jnp.array([60.0])
    y_i = jnp.array([0.020771939, 0.0595115259, 0.01268075, 0.77971855, 0.0013109827, 0.060903278, 0.065102974])
    x_i = jnp.array([0.00011514431, 0.010469182, 0.72283337, 0.0033709576, 0.26127321, 0.0013214924, 0.00061663844])
    x = jnp.array([520.0, 330.0, 60, 0.7, 0.3, 3.9295303, 15.028284, 3.1656873, 174.12167, 0.28198171, 13.202387, 14.106927])