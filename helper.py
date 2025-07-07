from pathlib import Path
from io import StringIO
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from collections.abc import Generator
from typing import Union
from pint import UnitRegistry
import emcee

import sys


class LammpsAnalyzer:
    """A helper class for post-processing LAMMPS results."""

    def __init__(self, num_of_beads=None, path="output/", log_files_name="log.lammps"):
        self.path = Path(path)
        self.log_files_name = log_files_name

        self.num_of_beads = num_of_beads

        if num_of_beads is None:
            self.num_of_beads = self.detect_num_beads()

        self.u_reg = UnitRegistry()
        self.u_reg.default_format = '~'  # Use abbreviated unit names
        self.Q_ = self.u_reg.Quantity
        self.u_reg.setup_matplotlib(True)

        # Various LAMMPS unit systems
        self.lammps_units = {
            'time': {'real': 'fs', 'metal': 'ps', 'si': 'sec',
                     'cgs': 'sec', 'electron': 'fs', 'micro': 'μs', 'nano': 'ns'},
            'energy': {'real': 'kcal/mol', 'metal': 'eV', 'si': 'Joules',
                       'cgs': 'ergs', 'electron': 'Hartrees'},
        }

        # LAMMPS units in pint format
        self.lu_to_pint = {
            'time': {'real': 'fs', 'metal': 'ps', 'si': 'second',
                     'cgs': 'sec', 'electron': 'fs', 'micro': 'μs', 'nano': 'ns'},
            'energy': {'real': 'kcal/mol', 'metal': 'eV', 'si': 'joule',
                       'cgs': 'erg', 'electron': 'hartree'},
            'temperature': {'real': 'kelvin', 'metal': 'kelvin', 'si': 'kelvin',
                            'cgs': 'kelvin', 'electron': 'kelvin'}
        }

        # Using the first log file (log.0), extract the simulation data
        data = self.get_params()
        self.units = data['units']  # Type of units used in this LAMMPS simulation
        self.dim = data['dimension']  # Dimension of the system (in LAMMPS, it is either 2 or 3)
        self.num_of_particles = data['particles']
        self.temperature = self.Q_(data['temperature'], self.lu_to_pint['temperature'][self.units])
        self.timestep = self.Q_(data['timestep'], self.lu_to_pint['time'][self.units])
        self.log_table_start = data['log_table_start']
        self.rows = data['rows']  # Total number of recorded observables

        # Physical observables (expectation values)
        self.energies = []
        self.mean_energy = 0

        # Thermodynamic beta
        self.beta = 1.0 / self.temperature.to(self.lu_to_pint['energy'][self.units], 'boltzmann')

    def get_log_filename(self, i) -> Path:
        """Return the full path to the ith log file (corresponding to the ith bead),
        where i=0,...,P-1 and P is the number of beads.

        :param i: Bead index.
        :return: Path to the corresponding log file.
        """
        return self.path / (self.log_files_name + '.' + str(i))

    def log_filename_generator(self) -> Generator[Path, None, None]:
        """Return a generator of paths to all the log files."""

        for i in range(self.num_of_beads):
            yield self.get_log_filename(i)

    def get_results_table(self, filename: Path) -> pd.DataFrame:
        """Parse LAMMPS log files and return a nicely formatted table.

        :param filename: The full filename of the log file to be parsed.
        :return: Returns a Pandas table
        """

        return pd.read_csv(filename, sep=r"\s+", skiprows=self.log_table_start, nrows=self.rows)

    def detect_num_beads(self) -> int:
        """Extracts the number of beads based on the number of log files."""

        # Extract all the file names present in the provided path
        file_names_list = [f.name for f in self.path.iterdir() if f.is_file()]

        # Based on the file names, determine all the existing beads
        prefix = self.log_files_name + '.'
        existing_beads = sorted([int(name.replace(prefix, '')) for name in file_names_list if prefix in name])

        if not existing_beads:
            raise ValueError("The provided path does not contain any bead files!")

        bead_num = existing_beads[-1] + 1
        correct_seq = list(range(bead_num))

        # Check for missing beads
        missing_beads = list(set(correct_seq) - set(existing_beads))

        if missing_beads:
            raise ValueError(f"""The are missing log files in the provided path.
            Couldn't find the log files associated with beads {', '.join(map(str, missing_beads))}.""")

        return bead_num

    def last_row(self, row_num):
        """Checks if the specified last row is set correctly.

        :param row_num: The specified row number (can be None).
        :return: Returns the row number if it is valid. If row_num is 'None', returns the last row in the table.
        """
        if row_num is None:
            return self.rows

        if row_num > self.rows:
            raise ValueError("The last row cannot exceed the total number of rows in the table!")

        return row_num

    def read_col(self, col_names, bead_idx=0, step_start=0, step_end=None) -> Union[dict, np.array]:
        """Extract specific column(s) from a specific log file (bead).

        :param col_names: A list containing the names of the columns to be summed across all beads.
        :param bead_idx: The index of the bead (i.e., the number of the log file) to read.
        :param step_start: The first row from which the summation begins (typically, we always skip some rows
                            at the beginning, in order to exclude the effects of thermalization.)
        :param step_end: The last row included in the summation.
        :return: Returns a dictionary containing the specified columns,
                 or the column itself if only one column name was provided.
        """
        if not isinstance(col_names, list) and not isinstance(col_names, str):
            raise ValueError("The parameter 'col_names' must be either a list of column names or a string "
                             "containing a single column name.")

        col_names_lst = col_names

        # If col_names is a string, interpret it as a single column name.
        if isinstance(col_names, str):
            col_names_lst = [col_names]

        # Dictionary that will hold all the required columns.
        col_vals = dict.fromkeys(col_names_lst, 0)

        f = self.get_results_table(self.get_log_filename(bead_idx))
        f = f.iloc[step_start:self.last_row(step_end), :]

        # Copy each of the requested columns into the dictionary.
        for col_name in col_names_lst:
            # TODO: Use pint-pandas instead of performing NumPy conversion
            col_vals[col_name] = pd.to_numeric(f[col_name]).to_numpy(dtype="float64")

        # If only one column name was provided, return the resulting column instead of a dictionary
        if len(col_names_lst) == 1:
            return col_vals[col_names_lst[0]]

        return col_vals

    def sum_col(self, col_names, step_start=0, step_end=None) -> Union[dict, np.array]:
        """Sum the values of a specific column(s) (using all the log files) across all beads.

       For example, if there are P log files (where P is the number of beads), and if each
       such log file contains the column 'A', then read_col('A') would return a column whose ith
       row is obtained by adding up all the values in the ith row of the column 'A' in each log file.

        :param col_names: A list containing the names of the columns to be summed across all beads.
        :param step_start: The first row from which the summation begins.
        :param step_end: The last row which is included in the summation.
        :return: Returns a dictionary containing the specified columns,
                 or the column itself if only one column name was provided.
        """

        if not isinstance(col_names, list) and not isinstance(col_names, str):
            raise ValueError("The parameter 'col_names' must be either a list of column names or a string "
                             "containing a single column name.")

        col_names_lst = col_names

        # If col_names is a string, interpret it as a single column name.
        if isinstance(col_names, str):
            col_names_lst = [col_names]

        # Dictionary that will hold all the required columns.
        col_vals = dict.fromkeys(col_names_lst, 0)

        for filename in self.log_filename_generator():
            f = self.get_results_table(filename)
            f = f.iloc[step_start:self.last_row(step_end), :]

            # Accumulate values from all the files (i.e., from all the beads).
            for col_name in col_names_lst:
                # TODO: Use pint-pandas instead of performing NumPy conversion
                col_vals[col_name] += pd.to_numeric(f[col_name]).to_numpy()

        # If only one column name was provided, return the resulting column instead of a dictionary
        if len(col_names_lst) == 1:
            return col_vals[col_names_lst[0]]

        return col_vals

    def potential_estimator_col(self, col_name='PotEng', step_start=0, step_end=None, out_unit='meV') -> np.array:
        """Returns the potential estimator values at all the time steps in the chosen range of steps.

        :param col_name: The name of the column corresponding to the potential estimator.
        :param step_start: The first row from which the summation begins.
        :param step_end: The last row included in the summation across all beads.
        :param out_unit: Specifies which units the result should be converted to. (default: meV)
        :return: Returns a column of the potential estimator of the system.
        """
        return self.Q_(self.sum_col(col_name, step_start, step_end) / self.num_of_beads,
                       self.lu_to_pint['energy'][self.units]).to(out_unit, 'boltzmann')

    pot_est_col = potential_estimator_col  # Alias

    def virial_estimator_col(self, col_name='v_newvir', step_start=0, step_end=None, out_unit='meV') -> np.array:
        """Returns the virial estimator values at all the time steps in the chosen range of steps.

        Note: There is no need to divide the column by the number of beads because the variable
        'virial' (which stores the values of the virial estimator) in fix_pimdb.cpp already
        accounts for that. Indeed, the forces which appear in the expression for the virial
        estimator (f[atomnum][i], where f is atom->f, i.e. all the forces acting on a particular
        bead of a particular atom excluding the spring forces.), are divided by the number of beads;
        This happens in the method FixPIMDB::post_force, where atom->f[i][j] /= np, where np is
        the number of beads.

        :param col_name: The name of the column corresponding to the virial estimator.
        :param step_start: The first row from which the summation begins.
        :param step_end: The last row included in the summation across all beads.
        :param out_unit: Specifies which units the result should be converted to. (default: meV)
        :return: Returns a column of the virial estimator of the system.
        """
        return self.Q_(self.sum_col(col_name, step_start, step_end),
                       self.lu_to_pint['energy'][self.units]).to(out_unit, 'boltzmann')

    vir_est_col = virial_estimator_col  # Alias

    def kinetic_energy_estimator(self, virial_col_name='v_newvir',
                                 step_start=0, step_end=None, out_unit='meV') -> (float, np.array):
        """Returns both the mean and the instantaneous kinetic energies of the system (using the virial estimator).
        Use only for bound systems that do not possess translational invariance.

        :param virial_col_name: The name of the virial estimator column in the log files.
        :param step_start: The first row from which the summation begins.
        :param step_end: The last row which is included in the summation across all beads.
        :param out_unit: Specifies which units the result should be converted to. (default: meV)
        :return: Mean kinetic energy and an array of instantaneous kinetic energies in the specified energy units.
        """

        virial_col = self.virial_estimator_col(virial_col_name, step_start=step_start,
                                               step_end=step_end, out_unit=out_unit)

        return np.mean(virial_col), virial_col

    def potential_energy_estimator(self, step_start=0, step_end=None, out_unit='meV') -> (float, np.array):
        """Returns both the mean and the instantaneous potential energies of the system (due to external forces).

        :param step_start: The first row from which the summation begins.
        :param step_end: The last row included in the summation across all beads.
        :param out_unit: Specifies which units the result should be converted to. (default: meV)
        :return: Mean potential energy and an array of instantaneous potential energies in the specified energy units.
        """

        # pot_col = self.sum_col('PotEng', step_start=step_start, step_end=step_end) / self.num_of_beads
        pot_col = self.potential_estimator_col(step_start=step_start, step_end=step_end, out_unit=out_unit)

        return np.mean(pot_col), pot_col

    def calculate_primitive_estimator(self, step_start=0, step_end=None, out_unit='meV') -> (float, np.array):
        """Returns the kinetic energy of the system, by calculating the primitive estimator
        from the pimdb.log file.

        :param step_start: The first row from which the summation begins.
        :param step_end: The last row included in the summation across all beads.
        :param out_unit: Specifies which units the result should be converted to. (default: meV)
        :return: Mean kinetic energy in the specified energy units.
        """
        primitive_est = []

        with open(self.path / "pimdb.log") as file:
            # Each line corresponds to an MD timestep
            for line in file:
                energies = [float(x) for x in line.split()]
                E_kn_size = int(self.num_of_particles * (self.num_of_particles + 1) / 2)
                E_kn = energies[:E_kn_size]
                V = energies[E_kn_size:E_kn_size + self.num_of_particles + 1]

                est = np.zeros(self.num_of_particles + 1)

                for m in range(1, self.num_of_particles + 1):
                    sig = 0.0

                    m_count = int(m * (m + 1) / 2)

                    ################################
                    # Numerical stability.
                    ################################

                    # Xiong-Xiong method (arXiv.2206.08341)
                    e_tilde = sys.float_info.max
                    for k in range(m, 0, -1):
                        e_tilde = min(e_tilde, E_kn[m_count - k] + V[m - k])

                    # Hirshberg-Rizzi-Parrinello method (pnas.1913365116)
                    # e_tilde = min(E_kn[m_count - 1] + V[m - 1], E_kn[m_count - m] + V[0])

                    ################################
                    # Estimator evaluation.
                    ################################

                    for k in range(m, 0, -1):
                        E_kn_val = E_kn[m_count - k]

                        sig += (est[m - k] - E_kn_val) * np.exp(-self.beta.m * (E_kn_val + V[m - k] - e_tilde))

                    sig_denom_m = m * np.exp(-self.beta.m * (V[m] - e_tilde))

                    est[m] = sig / sig_denom_m

                primitive_est.append(est[self.num_of_particles])

        factor = 0.5 * self.dim * self.num_of_beads * self.num_of_particles / self.beta.m

        kinetic_energy = self.Q_(factor + np.array(primitive_est[step_start:]),
                                 self.lu_to_pint['energy'][self.units]).to(out_unit, 'boltzmann')

        # DEBUG
        print("1st factor:", self.Q_(factor, self.lu_to_pint['energy'][self.units]).to(out_unit, 'boltzmann'))
        print("2nd factor:", self.Q_(np.mean(np.array(primitive_est[step_start:])),
                                     self.lu_to_pint['energy'][self.units]).to(out_unit, 'boltzmann'))

        return np.mean(kinetic_energy), kinetic_energy

    def total_energy_estimator(self, virial_col_name='v_newvir', step_start=0, step_end=None, out_unit='meV') -> float:
        """Returns the value of the total energy estimator (in the desired units). The total energy
        estimator is given by the sum of the potential and virial estimators.

           :param virial_col_name: The name of the column corresponding to the virial estimator.
           :param step_start: The first row from which the summation begins.
           :param step_end: The last row which is included in the summation across all beads.
           :param out_unit: Specifies which units the result should be converted to.
           :return: Returns the overall mean energy of the quantum system.
        """

        cols = self.sum_col(['PotEng', virial_col_name], step_start=step_start, step_end=step_end)

        # Save the total quantum energy at the specified time steps to an array
        # Note 1: To extract the magnitude and the units separately, use '<quantity>.magnitude' and '<quantity>.units'
        # Note 2: The context is set to 'boltzmann' for energy/temperature conversion (e.g. eV to K)
        self.energies = self.Q_(cols['PotEng'] / self.num_of_beads + cols[virial_col_name],
                                self.lu_to_pint['energy'][self.units]).to(out_unit, 'boltzmann')

        self.mean_energy = np.mean(self.energies)

        return self.mean_energy

    tot_energy = total_energy_estimator  # Alias

    def classical_energies(self, step_start=0, step_end=None) -> dict:
        """Returns columns of classical energies (kinetic, potential, spring and total).
        Useful for timestep calibration.

        Note: LAMMPS calculates 'KinEng' (kinetic energy) using Thermo::compute_ke() (see src/thermo.cpp),
              which is based on the equipartition theorem: 0.5*k_B*T*(DOF). For a 3D simulation with N particles,
              DOF=3N-3, and for a 2D simulation DOF=2N-2 (see https://docs.lammps.org/compute_modify.html).
        :param step_start: The first row from which the summation begins.
        :param step_end: The last row which is included in the summation.
        :return: Returns a column of the total classical energies of the system in the chosen range of steps.
        """

        cols = self.sum_col(['KinEng', 'PotEng', 'v_springE'], step_start=step_start, step_end=step_end)

        return {'kinetic': cols['KinEng'] / self.num_of_beads,
                'potential': cols['PotEng'] / self.num_of_beads,
                'spring': cols['v_springE'],
                # In older versions of LAMMPS, fix_pimd.cpp contains a bug where the spring_energy is negative
                # (and is also missing a spring constant). Here, we assume the bug has been fixed.
                # See: https://github.com/lammps/lammps/pull/3612
                'total': (cols['KinEng'] + cols['PotEng']) / self.num_of_beads + cols['v_springE']}

    def get_params(self) -> dict:
        """Fetch the simulation parameters (units, timestep, number of particles, temperature, dimension,
        the frequency at which the observables are being recorded) as well as technical information
        (the number of rows in the log tables, the position of the log tables etc.), from the first log file (log.0)."""

        filename = next(self.log_filename_generator())

        # Initialize default values
        data = {'dimension': 3, 'units': 'metal', 'timestep': 1.0, 'particles': 1,
                'temperature': 1.0, 'save_freq': 1, 'log_table_start': 0, 'rows': 0
                }

        # For simple "<keyword> <value>" commands
        params = {'units': ('units', str),
                  'dimension': ('dimension', int),
                  'timestep': ('timestep', float),
                  # TODO: Add support for non-numeric values of 'thermo' (e.g., an equal-style variable)
                  'save_freq': ('thermo ', int)
                  }

        # For slightly more complicated cases
        regexes = {'temperature': (r"fix[a-zA-Z\s0-9\.]+temp\s+(\d+\.?\d*)\s+nhc", float),
                   'particles': (r"Loop time.+steps with (\d+) atoms", int)
                   }

        # Open the log file
        with open(filename, 'r') as f:
            # Iterate through each line
            for line_num, line in enumerate(f, 1):
                for param, (command, to_type) in params.items():
                    if line.startswith(command):
                        data[param] = to_type(line.split()[1])
                        break
                else:
                    for param, (regex, to_type) in regexes.items():
                        match = re.match(regex, line)
                        if match:
                            data[param] = to_type(match.group(1))
                            break

                if line.startswith("Per MPI rank"):
                    data['log_table_start'] = line_num
                    continue
                elif line.startswith("Loop time"):
                    data['rows'] = line_num - data['log_table_start'] - 2
                    break

        return data

    def min_equilibration_step(self, threshold=0.1) -> int:
        """Determines the row number at which one can start recording observables, based on
        the thermalization threshold (set by default to 10%).

        :param threshold: The threshold for thermalization (default value 0.1=10%).
        :return: The first row that comes after the system has reached thermal equilibrium.
        """

        return int(self.read_col('Step', step_start=0).size * threshold)

    equil_step = min_equilibration_step  # Alias

    def get_integrated_autocorrelation_time(self, step_size=5, tolerance=50) -> float:
        """Find the integrated autocorrelation time for a given set of energy measurements.

        :param step_size: The step size for the window search. (default 5)
        :param tolerance: The minimum number of autocorrelation times needed to trust the estimate. (default: 50)
        :return: Integrated autocorrelation time.
        """
        if not self.energies:
            raise ValueError("Cannot calculate the IAT because there are no energy measurements!")

        return emcee.autocorr.integrated_time(self.energies, c=step_size, tol=tolerance, quiet=False)

    get_iat = get_integrated_autocorrelation_time  # Alias

    def plot_classical_energy_conservation(self, ptype='relative', dt_unit='fs', title=None, step_start=0,
                                           step_end=None) -> (np.array, np.array):
        """Plot the energy fluctuations of the classical system (regular PIMD with a disabled thermostat),
        as a function of time. The fluctuations (E-Avg[E])/Avg[E] are shown as percents. The goal is to
        choose the MD timestep such that the fluctuations will not exceed 0.1%.

        :param ptype: The type of plot to return. At the moment, there are three options:
                       1) 'relative' (Default) plots (E-Avg[E])/Avg[E] only. Aliases: 'rel'
                       2) 'components' plots the classical energy balance of
                           the system (KE, PE, spring and total energies). Aliases: 'comp'
                       3) 'combined' is a combined plot of 'relative' and 'components'. Aliases: 'comb'
        :param dt_unit: The units in which the timestep will be shown (femto-seconds by default).
        :param title: Custom title text for the plot.
        :param step_start: The first row from which the summation begins.
        :param step_end: The last row which is included in the summation.
        :return: Returns an array of times and the corresponding energies
        """

        times = self.read_col('Time', step_start=step_start, step_end=step_end)

        _type = self.ClassicPlotTypes(ptype)

        # To extract the magnitude and the units separately, use 'dt_label.magnitude' and 'dt_label.units'
        dt_label = self.timestep.to(dt_unit)

        if title is None:
            if _type.RELATIVE:
                _title = fr"Conservation of classical energy ($P = {self.num_of_beads}$, $T= {self.temperature}$)"
            else:
                _title = fr"Conservation of classical energy ($P = {self.num_of_beads}$, $T= {self.temperature}$, " \
                         fr"$\Delta t$={dt_label.m:.2f} {dt_label.u})"
        else:
            _title = title

        energies = self.classical_energies(step_start=step_start, step_end=step_end)
        total = energies['total']

        if _type.COMBINED:
            fig, ax = plt.subplots(2, figsize=[9, 8])
            fig.suptitle(_title)
            comp_ax = ax[0]
            rel_ax = ax[1]
        else:
            fig, ax = plt.subplots(figsize=[9, 4.8])
            ax.set_title(_title)
            comp_ax = ax
            rel_ax = ax

        if _type.COMPONENTS or _type.COMBINED:
            kinetic = energies['kinetic']
            potential = energies['potential']
            spring = energies['spring']
            comp_ax.plot(times, kinetic, label="Kinetic", markersize=5)
            comp_ax.plot(times, potential, label="Potential", markersize=5)
            comp_ax.plot(times, spring, label="Spring", markersize=5)
            comp_ax.plot(times, total, label="Total", markersize=5)
            # ax.axhline(y=0, color='#4ac6ff', linestyle='--')  # Previously, color='r'
            comp_ax.set_ylabel(f"Energies [{self.lammps_units['energy'][self.units]}]", fontsize=15)
            comp_ax.legend(loc='upper right')
            comp_ax.ticklabel_format(useMathText=True, scilimits=(0, 0))

            if _type.COMPONENTS:
                comp_ax.set_xlabel(f"Time [{dt_unit}]", fontsize=15)
                plt.show()
                return times, energies

        # (E-Avg[E])/Avg[E]
        rel_energies = (total / np.mean(total) - 1)

        if _type.RELATIVE:
            rel_ax.plot(times, rel_energies, label=fr"$\Delta t$={dt_label.m:.2f} {dt_label.u}", color='black',
                        markersize=5)
            rel_ax.legend(loc='upper right')
        else:
            rel_ax.plot(times, rel_energies, color='black', markersize=5)

        rel_ax.axhline(y=0, color='#4ac6ff', linestyle='--')
        rel_ax.ticklabel_format(useMathText=True, scilimits=(0, 0))
        rel_ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=3, is_latex=True))

        # As long as the fluctuations do not exceed 0.1%, show the graph between -0.1% and 0.1%,
        # even if the amplitude of the fluctuations is much smaller.
        tolerance = 0.001

        if np.amax(rel_energies) < tolerance:
            rel_ax.set_ylim(-tolerance, tolerance)

        rel_ax.set_xlabel(f"Time [{dt_unit}]", fontsize=15)
        rel_ax.set_ylabel(r"$\Delta E / \bar{E}$", fontsize=15)

        plt.show()

        return times, total

    class ClassicPlotTypes:
        """Types for classical plots."""

        def __init__(self, ptype: str):
            self.COMBINED = False
            self.RELATIVE = False
            self.COMPONENTS = False

            if ptype in ['combined', 'comb']:
                self.COMBINED = True
            elif ptype in ['components', 'comp']:
                self.COMPONENTS = True
            elif ptype in ['relative', 'rel']:
                self.RELATIVE = True
            else:
                raise ValueError("Invalid classical energies plot type.")
