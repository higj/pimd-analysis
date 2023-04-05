from helper import LammpsAnalyzer, plt

lmp_path = ""  # Path to LAMMPS output folder
energy_unit = 'K'

sim = LammpsAnalyzer(path=lmp_path)
equil_step = sim.min_equilibration_step()

ke, ke_list = sim.calculate_primitive_estimator(step_start=equil_step, out_unit=energy_unit)
pe, pe_list = sim.potential_energy_estimator(step_start=equil_step, out_unit=energy_unit)

print("Primitive kinetic energy:", ke / sim.num_of_particles)
print("Potential energy:", pe / sim.num_of_particles)

plt.plot(ke_list / sim.num_of_particles)
plt.xlabel("Timestep")
plt.ylabel(fr"$\left\langle E_{{\mathrm{{kin}}}}\right\rangle / N$ [{energy_unit}]")
plt.title(r"$ \frac{Pd}{2\beta} + \frac{1}{N} \left \langle V_B^{(N)}"
          r" + \beta \frac{\partial V_B^{(N)}}{\partial \beta} \right \rangle$")

plt.show()