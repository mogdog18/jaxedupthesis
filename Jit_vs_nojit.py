from JAXedThesisMainCode import Reactor, Separator, Species
import jax.numpy as jnp
import time
import jax
import matplotlib.pyplot as plt
import numpy as np

# code to time how long model takes to solve with and without jit compilation
if __name__ == '__main__':
    reactor = Reactor()
    separator = Separator()
    species = Species()
    T_reactor = jnp.array([510.0])
    F_i_reactor = jnp.array([270.2685764, 49.40806719, 29.19175308, 3070.869085, 2.189968503, 226.8649092, 242.7711833])
    T_separator = jnp.array([320.0])
    P_separator = jnp.array([65.0])
    F_i_separator = jnp.array([61.74455216, 34.09243863, 253.0314059, 2607.874151, 17.50559706, 226.8649092, 242.7711833])

    def jit_compile_and_run(T_reactor, F_i_reactor, F_i_separator, T_separator, P_separator, no_times):
        start = time.time()
        for i in range(no_times):
            jax.jit(reactor.mass_energy_balance)(T_reactor, F_i_reactor)
        time_taken_ODE = time.time() - start

        start = time.time()
        for i in range(no_times):
            jax.jit(separator.mass_balance)(F_i_separator, T_separator, P_separator)
        time_taken_separator = time.time() - start
        return time_taken_ODE, time_taken_separator

    def jit_run_only(T_reactor, F_i_reactor, F_i_separator, T_separator, P_separator, no_times):
        jax.jit(reactor.mass_energy_balance)(T_reactor, F_i_reactor)
        start = time.time()
        for i in range(no_times):
            jax.jit(reactor.mass_energy_balance)(T_reactor, F_i_reactor)
        time_taken_ODE = time.time() - start

        jax.jit(separator.mass_balance)(F_i_separator, T_separator, P_separator)
        start = time.time()
        for i in range(no_times):
            jax.jit(separator.mass_balance)(F_i_separator, T_separator, P_separator)
        time_taken_separator = time.time() - start
        return time_taken_ODE, time_taken_separator

    def no_jit_time(T_reactor, F_i_reactor, F_i_separator, T_separator, P_separator, no_times):
        start = time.time()
        for i in range(no_times):
            reactor.mass_energy_balance(T_reactor, F_i_reactor)
        time_taken_ODE = time.time() - start

        start = time.time()
        for i in range(no_times):
            separator.mass_balance(F_i_separator, T_separator, P_separator)
        time_taken_separator = time.time() - start
        return time_taken_ODE, time_taken_separator

    no_times_run = [1]
    list_of_times_run_reactor = []
    list_of_times_nojit_reactor = []
    list_of_times_compile_and_run_reactor = []
    list_of_times_run_separator = []
    list_of_times_nojit_separator = []
    list_of_times_compile_and_run_separator = []

    no_times_test = 100
    for i in range(no_times_test):
        for j in no_times_run:
            run_and_compile_reactor, run_and_compile_separator = jit_compile_and_run(T_reactor, F_i_reactor, F_i_separator, T_separator, P_separator, j)
            list_of_times_compile_and_run_reactor.append(run_and_compile_reactor)
            list_of_times_compile_and_run_separator.append(run_and_compile_separator)

            run_reactor, run_separator = jit_run_only(T_reactor, F_i_reactor, F_i_separator, T_separator, P_separator, j)
            list_of_times_run_reactor.append(run_reactor)
            list_of_times_run_separator.append(run_separator)

            nojit_reactor, nojit_separator = no_jit_time(T_reactor, F_i_reactor, F_i_separator, T_separator, P_separator, j)
            list_of_times_nojit_reactor.append(nojit_reactor)
            list_of_times_nojit_separator.append(nojit_separator)

    # reactor
    list_of_times_compile_and_run_reactor = jnp.asarray(list_of_times_compile_and_run_reactor)
    avg_time_compile_and_run_reactor = np.mean(list_of_times_compile_and_run_reactor)
    std_time_compile_and_run_reactor = np.std(list_of_times_compile_and_run_reactor)
    list_of_times_run_reactor = jnp.asarray(list_of_times_run_reactor)
    avg_time_run_reactor = np.mean(list_of_times_run_reactor)
    std_time_run_reactor = np.std(list_of_times_run_reactor)
    list_of_times_nojit_reactor = jnp.asarray(list_of_times_nojit_reactor)
    avg_time_nojit_reactor = np.mean(list_of_times_nojit_reactor)
    std_nojit_reactor = np.std(list_of_times_nojit_reactor)
    # separator
    list_of_times_compile_and_run_separator = jnp.asarray(list_of_times_compile_and_run_separator)
    avg_time_compile_and_run_separator = np.mean(list_of_times_compile_and_run_separator)
    std_time_compile_and_run_separator = np.std(list_of_times_compile_and_run_separator)
    list_of_times_run_separator = jnp.asarray(list_of_times_run_separator)
    avg_time_run_separator = np.mean(list_of_times_run_separator)
    std_time_run_separator = np.std(list_of_times_run_separator)
    list_of_times_nojit_separator = jnp.asarray(list_of_times_nojit_separator)
    avg_time_nojit_separator = np.mean(list_of_times_nojit_separator)
    std_nojit_separator = np.std(list_of_times_nojit_separator)

    fig, ax = plt.subplots()
    plt.plot(range(no_times_test), list_of_times_nojit_reactor, '-o', alpha=0.5, color='#E182FD')
    plt.plot(range(no_times_test), list_of_times_compile_and_run_reactor, '-o', alpha=0.5, color='#6E98F8')
    plt.plot(range(no_times_test), list_of_times_run_reactor, '-o', alpha=0.5, color='#4BA89B')
    ax.set_yscale('log')
    plt.ylabel("Computational time (s)", fontsize=11)
    plt.legend(["run time (no compilation)", "compile and run time", "run time (with compilation)"], loc="best")
    plt.xlabel("Run number", fontsize=11)
    plt.savefig("reactor.svg", format='svg', dpi=1200)

    fig, ax = plt.subplots()
    plt.plot(range(no_times_test), list_of_times_nojit_separator, '-o', alpha=0.5, color='#E182FD')
    plt.plot(range(no_times_test), list_of_times_compile_and_run_separator, '-o', alpha=0.5, color='#6E98F8')
    plt.plot(range(no_times_test), list_of_times_run_separator, '-o', alpha=0.5, color='#4BA89B')
    ax.set_yscale('log')
    plt.ylabel("Computational time (s)", fontsize=11)
    plt.legend(["run time (no compilation)", "compile and run time", "run time (with compilation)"], loc="best")
    plt.xlabel("Run number", fontsize=11)
    plt.savefig("separator.svg", format='svg', dpi=1200)

    # jit_times_taken_ODE = []
    # jit_times_taken_separator = []
    # nojit_times_taken_ODE = []
    # nojit_times_taken_separator = []
    # no_times_test = 100
    #
    # for i in range(no_times_test):
    #     jit_time_taken_ODE, jit_time_taken_separator = jit_time(T_reactor, F_i_reactor, F_i_separator, T_separator,
    #                                                             P_separator, no_times)
    #     nojit_time_taken_ODE, nojit_time_taken_separator = no_jit_time(T_reactor, F_i_reactor, F_i_separator,
    #                                                                    T_separator, P_separator, no_times)
    #     jit_times_taken_ODE.append(jit_time_taken_ODE)
    #     jit_times_taken_separator.append(jit_time_taken_separator)
    #     nojit_times_taken_ODE.append(nojit_time_taken_ODE)
    #     nojit_times_taken_separator.append(nojit_time_taken_separator)

    # jit_times_taken_ODE = jnp.asarray(jit_times_taken_ODE)
    # jit_times_taken_separator = jnp.asarray(jit_times_taken_separator)
    # nojit_times_taken_ODE = jnp.asarray(nojit_times_taken_ODE)
    # nojit_times_taken_separator = jnp.asarray(nojit_times_taken_separator)
    # # average times taken
    # avg_jit_time_ODE = np.mean(jit_times_taken_ODE)
    # avg_jit_time_separator = np.mean(jit_times_taken_separator)
    # avg_nojit_time_ODE = np.mean(nojit_times_taken_ODE)
    # avg_nojit_time_separator = np.mean(nojit_times_taken_separator)
    # # standard deviation of times taken
    # std_jit_time_ODE = np.std(jit_times_taken_ODE)
    # std_jit_time_separator = np.std(jit_times_taken_separator)
    # std_nojit_time_ODE = np.std(nojit_times_taken_ODE)
    # std_nojit_time_separator = np.std(nojit_times_taken_separator)

    # labels, heights, error bar heights
    labels = ["reactor MB", "separator MB"]
    x_pos = jnp.arange(len(labels))
    #
    r_CTEs = [avg_time_run_reactor, avg_time_run_separator]
    r_error = [std_time_run_reactor, std_time_run_reactor]
    cr_CTEs = [avg_time_compile_and_run_reactor, avg_time_compile_and_run_reactor]
    cr_error = [std_time_compile_and_run_reactor, std_time_compile_and_run_separator]
    nj_CTEs = [avg_time_nojit_reactor, avg_time_nojit_separator]
    nj_error = [std_nojit_reactor, std_nojit_separator]
    fig, ax = plt.subplots()
    width = 0.25
    ax.bar(x_pos, nj_CTEs, width, yerr=nj_error, align='center', alpha=0.5, ecolor='black', capsize=10,
           color='#E182FD')
    ax.bar(x_pos + width, cr_CTEs, width, yerr=cr_error, align='center', alpha=0.5, ecolor='black', capsize=10,
           color='#6E98F8')
    ax.bar(x_pos + 2*width, r_CTEs, width, yerr=r_error, align='center', alpha=0.5, ecolor='black', capsize=10,
           color='#4BA89B')
    ax.set_xticks(x_pos+width/2)
    labels = ["reactor MB", "separator MB"]
    ax.set_xticklabels(labels)
    plt.legend(["run time (no compilation)", "compile and run time", "run time (with compilation)"], loc="best")
    ax.set_yscale('log')
    ax.set_ylabel("Computational time (s)", fontsize=11)
    plt.ylim([0, 10])
    plt.savefig("jitbargraph.svg", format='svg', dpi=1200)
    #
    # print(avg_jit_time_ODE/avg_nojit_time_ODE)
    # print(avg_jit_time_separator / avg_nojit_time_separator)






