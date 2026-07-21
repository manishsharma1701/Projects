
import numpy as np
import sys

# ============================================================
# CHECK INPUT
# ============================================================

if len(sys.argv) != 2:
    print("Usage: python fix_su2.py <mesh.su2>")
    sys.exit(1)

input_file = sys.argv[1]

# overwrite same file
output_file = input_file

# ============================================================
# READ FILE
# ============================================================

with open(input_file, "r") as f:
    lines = f.readlines()

# ============================================================
# READ POINTS
# ============================================================

points = {}

for i, line in enumerate(lines):

    if line.startswith("NPOIN"):

        npoints = int(line.split("=")[1])

        start = i + 1

        for j in range(npoints):

            vals = lines[start + j].split()

            x = float(vals[0])
            y = float(vals[1])

            points[j] = (x, y)

        break

# ============================================================
# FIX ELEMENT ORIENTATION
# ============================================================

new_lines = []

inside_elem = False

nelem = 0
elem_count = 0

tri_count = 0
quad_count = 0

tri_fixed = 0
quad_fixed = 0

for i, line in enumerate(lines):

    # --------------------------------------------------------
    # Start element section
    # --------------------------------------------------------

    if line.startswith("NELEM"):

        inside_elem = True

        nelem = int(line.split("=")[1])

        elem_count = 0

        new_lines.append(line)

        continue

    # --------------------------------------------------------
    # Process elements
    # --------------------------------------------------------

    if inside_elem and elem_count < nelem:

        vals = line.split()

        etype = int(vals[0])

        # ====================================================
        # TRIANGLES
        # SU2 type = 5
        # ====================================================

        if etype == 5:

            tri_count += 1

            n1, n2, n3 = map(int, vals[1:4])

            p1 = np.array(points[n1])
            p2 = np.array(points[n2])
            p3 = np.array(points[n3])

            # Signed area
            area = 0.5 * (
                p1[0]*(p2[1]-p3[1]) +
                p2[0]*(p3[1]-p1[1]) +
                p3[0]*(p1[1]-p2[1])
            )

            # Fix clockwise orientation
            if area < 0:

                vals[1:4] = [
                    str(n1),
                    str(n3),
                    str(n2)
                ]

                tri_fixed += 1

        # ====================================================
        # QUADS
        # SU2 type = 9
        # ====================================================

        elif etype == 9:

            quad_count += 1

            n1, n2, n3, n4 = map(int, vals[1:5])

            p1 = np.array(points[n1])
            p2 = np.array(points[n2])
            p3 = np.array(points[n3])
            p4 = np.array(points[n4])

            # Polygon signed area
            area = 0.5 * (
                p1[0]*p2[1] - p2[0]*p1[1] +
                p2[0]*p3[1] - p3[0]*p2[1] +
                p3[0]*p4[1] - p4[0]*p3[1] +
                p4[0]*p1[1] - p1[0]*p4[1]
            )

            # Fix orientation
            if area < 0:

                vals[1:5] = [
                    str(n1),
                    str(n4),
                    str(n3),
                    str(n2)
                ]

                quad_fixed += 1

        new_lines.append(" ".join(vals) + "\n")

        elem_count += 1

        continue

    # --------------------------------------------------------
    # Copy remaining lines
    # --------------------------------------------------------

    new_lines.append(line)

# ============================================================
# WRITE FILE
# ============================================================

with open(output_file, "w") as f:
    f.writelines(new_lines)

# ============================================================
# SUMMARY
# ============================================================

print("\nFinished fixing mesh.\n")

print("Total elements  : {}".format(nelem))

print("Triangles       : {}".format(tri_count))
print("Triangles fixed : {}".format(tri_fixed))

print("Quads           : {}".format(quad_count))
print("Quads fixed     : {}".format(quad_fixed))


with open(input_file, "r") as f:
    lines = f.readlines()

new_lines = []

i = 0

while i < len(lines):

    line = lines[i]

    # --------------------------------------------------------
    # Modify dimension
    # --------------------------------------------------------

    if line.startswith("NDIME="):
        new_lines.append("NDIME= 2\n")
        i += 1
        continue

    # --------------------------------------------------------
    # Modify point coordinates
    # --------------------------------------------------------

    if line.startswith("NPOIN="):

        new_lines.append(line)

        npoints = int(line.split("=")[1])

        i += 1

        for _ in range(npoints):

            parts = lines[i].split()

            # SU2 format:
            # x y z id
            x = parts[0]
            y = parts[1]

            # keep point index if present
            if len(parts) == 4:
                idx = parts[3]
                new_lines.append(f"{x} {y} {idx}\n")
            else:
                new_lines.append(f"{x} {y}\n")

            i += 1

        continue

    # --------------------------------------------------------
    # Copy everything else
    # --------------------------------------------------------

    new_lines.append(line)
    i += 1

with open(output_file, "w") as f:
    f.writelines(new_lines)

print(f"2D mesh written to: {output_file}")
