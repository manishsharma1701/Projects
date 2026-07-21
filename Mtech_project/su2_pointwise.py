import sys 

def process_su2(input_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    out_lines = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i].strip()

        # --- Handle NELEM ---
        if line.startswith("NELEM"):
            out_lines.append(lines[i])
            num_elem = int(line.split('=')[1])
            i += 1

            for idx in range(num_elem):
                parts = lines[i].strip()
                new_line = f"{parts} {idx}\n"
                out_lines.append(new_line)
                i += 1

        # --- Handle NPOIN ---
        elif line.startswith("NPOIN"):
            out_lines.append(lines[i])
            num_points = int(line.split('=')[1])
            i += 1

            for idx in range(num_points):
                parts = lines[i].strip()
                new_line = f"{parts} {idx}\n"
                out_lines.append(new_line)
                i += 1

        else:
            out_lines.append(lines[i])
            i += 1

    with open(input_file, 'w') as f:
        f.writelines(out_lines)


# ---- Usage ----
process_su2(sys.argv[1])
print("Done.")
