import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import re

def read_aoa(cfg_file="pravaha.cfg"):
    with open(cfg_file, "r") as f:
        for line in f:

            # remove comments
            line = line.split('#')[0]

            if "R_BETA" in line.upper():

                nums = re.findall(r'[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?', line)

                if nums:
                    return float(nums[0])

    raise ValueError("AOA not found in pravaha.cfg")

# ==========================================================
# Read forces_moment.out
# ==========================================================
def read_forces_moment(filename):

    columns = [
        "Iteration",
        "F_inv_X",
        "F_inv_Y",
        "F_inv_Z",
        "M_inv_X",
        "M_inv_Y",
        "M_inv_Z",
        "F_visc_X",
        "F_visc_Y",
        "F_visc_Z",
        "M_visc_X",
        "M_visc_Y",
        "M_visc_Z"
    ]

    data = []

    with open(filename, "r") as f:

        for line in f:

            line = line.strip()

            if (
                not line
                or line.startswith("#")
            ):
                continue

            parts = line.split()

            if len(parts) != 13:
                continue

            try:
                row = [float(x) for x in parts]
                data.append(row)

            except ValueError:
                continue

    df = pd.DataFrame(data, columns=columns)

    return df

def extract_and_plot_solver_data(input_file, output_name, format='csv'):
    data = []

    # Read and parse the log file
    with open(input_file, 'r') as f:
        for line in f:
            parts = line.strip().split()

            # Expected format check
            if len(parts) == 14 and parts[12] == "Time:":
                try:
                    iteration = int(parts[0])

                    row_vals = (
                        [iteration]
                        + [float(x) for x in parts[1:12]]
                        + [float(parts[13])]
                    )

                    data.append(row_vals)

                except ValueError:
                    continue

    if not data:
        print("No valid data found!")
        return

    # Column headers
    columns = [
        "Iteration",
        "mass",
        "momentum",
        "energy",
        "turb_1",
        "turb_2",
        "CFx",
        "CFy",
        "CFz",
        "CMx",
        "CMy",
        "CMz",
        "Time"
    ]

    # Create DataFrame
    df = pd.DataFrame(data, columns=columns)


    # Replace inf values
    df = df.replace([np.inf, -np.inf], np.nan)

    # -----------------------------
    # SAVE DATA
    # -----------------------------
    if format.lower() == 'csv':
        df.to_csv(f"{output_name}.csv", index=False)
        print(f"Data saved to {output_name}.csv")

    elif format.lower() == 'excel':
        df.to_excel(f"{output_name}.xlsx", index=False)
        print(f"Data saved to {output_name}.xlsx")

    # -----------------------------
    # RESIDUALS PLOT
    # -----------------------------
    plt.figure(figsize=(10, 6))

    residuals = ["mass", "momentum", "energy", "turb_1", "turb_2"]
    # Global maximum across all residual columns
    global_max = df[residuals].max().max()

    #for res in residuals:
    #    plt.plot(df['Iteration'], df[res] / global_max, label=res)
    for res in residuals:
    	initial = abs(df[res].iloc[0])

    	if initial > 0:
        	plt.plot(
            	df['Iteration'],
            	df[res] - initial,
            	label=res
        )
    plt.xlabel('Iteration')
    plt.ylabel('Normalized Residual')
    plt.title('Normalized Residuals vs Iteration')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig('residuals.png')
    plt.show()

    # -----------------------------
    # COEFFICIENTS PLOT
    # -----------------------------
    plt.figure(figsize=(10, 6))

    coeffs = ["CFx", "CFy", "CFz", "CMx", "CMy", "CMz"]

    for coeff in coeffs:
        plt.plot(df['Iteration'], df[coeff], label=coeff)

    plt.xlabel('Iteration')
    plt.ylabel('Coefficient')
    plt.title('Force and Moment Coefficients vs Iteration')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig('coefficients_plot.png')
    plt.show()


# -----------------------------
# MAIN
# -----------------------------
import sys

if __name__ == "__main__":

    input_files = sys.argv[1:]

    combined_log = "combined.log"

    with open(combined_log, "w") as outfile:
        for fname in input_files:
            with open(fname, "r") as infile:
                outfile.write(infile.read())

    extract_and_plot_solver_data(
        input_file=combined_log,
        output_name="solver_output",
        format="csv"
    )

# ==========================================================
# Main
# ==========================================================
def main():

    forces_file = "force_moments_convergence.out"
    cfg_file = "pravaha.cfg"

    aoa = read_aoa(cfg_file)

    alpha = -np.radians(aoa)

    print(f"\nAOA = {aoa:.4f} deg")

    df = read_forces_moment(forces_file)

    # ------------------------------------------------------
    # Pressure / Inviscid contributions
    # ------------------------------------------------------
    df["CDP"] = (
        df["F_inv_X"] * np.cos(alpha)
        + df["F_inv_Z"] * np.sin(alpha)
    )

    df["CLP"] = (
        -df["F_inv_X"] * np.sin(alpha)
        + df["F_inv_Z"] * np.cos(alpha)
    )

    # ------------------------------------------------------
    # Viscous contributions
    # ------------------------------------------------------
    df["CDV"] = (
        df["F_visc_X"] * np.cos(alpha)
        + df["F_visc_Z"] * np.sin(alpha)
    )

    df["CLV"] = (
        -df["F_visc_X"] * np.sin(alpha)
        + df["F_visc_Z"] * np.cos(alpha)
    )

    # ------------------------------------------------------
    # Total coefficients
    # ------------------------------------------------------
    df["CD"] = df["CDP"] + df["CDV"]
    df["CL"] = df["CLP"] + df["CLV"]

    # ------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------
    output_csv = "forces_coefficients.csv"

    df.to_csv(output_csv, index=False)

    print(f"\nSaved: {output_csv}")

    # ------------------------------------------------------
    # Print final values
    # ------------------------------------------------------
    last = df.iloc[-1]

    print("\n===================================")
    print("FINAL VALUES")
    print("===================================")

    print(f"CL  = {last['CL']:.6f}")
    print(f"CD  = {last['CD']:.6f}")

    print()

    print(f"CLP = {last['CLP']:.6f}")
    print(f"CLV = {last['CLV']:.6f}")

    print()

    print(f"CDP = {last['CDP']:.6f}")
    print(f"CDV = {last['CDV']:.6f}")

    print("===================================\n")

    # ------------------------------------------------------
    # Plot Lift / Drag history
    # ------------------------------------------------------
    plt.figure(figsize=(10, 6))

    plt.plot(
        df["Iteration"],
        df["CL"],
        label="CL"
    )

    plt.plot(
        df["Iteration"],
        df["CD"],
        label="CD"
    )


    plt.xlabel("Iteration")
    plt.ylabel("Coefficient")
    plt.title("Aerodynamic Coefficients")

    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "aero_coefficients.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved: aero_coefficients.png")


if __name__ == "__main__":
    main()
