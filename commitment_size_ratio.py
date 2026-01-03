import csv

CELL_SIZE = 2048           # bytes
COMMITMENT_UNIT_SIZE = 48  # bytes

n_values = list(range(8, 257, 8))
fragment_sizes = [1, 8, 16, 32]

with open("results/commitment_size/commitment_benchmark.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "n",
        "ratio_f1",
        "ratio_f8",
        "ratio_f16",
        "ratio_f32"
    ])

    for n in n_values:
        block_size = (n * n) * CELL_SIZE

        ratios = []
        for fsize in fragment_sizes:
            commitment_size = n * fsize * COMMITMENT_UNIT_SIZE
            ratios.append(commitment_size / block_size)

        writer.writerow([n] + ratios)

print("Wrote commitment_benchmark.csv")