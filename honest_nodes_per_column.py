import csv
from grid_core import *

if __name__ == "__main__":
    cols = 100
    n_total = 2500
    n_init = 20
    n_warmup = n_total - n_init
    steps = 50000
    churn = 50
    row_range = [1, 5, 10, 25]

    all_data = []
    
    results_dict = {}
    for rows in row_range:
        params = ProtocolParameters(k1=rows, k2=cols, delta_sub=1, m=100)
        schedule = generate_schedule(n_init=n_init, n_warmup=n_warmup, churn=churn, steps=steps)
        statistics = simulate_protocol_run(schedule, params)
        results_dict[rows] = statistics.honest_nodes_columns_graph

    with open('protocol_results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ['step'] + [f'rows_{r}' for r in row_range]
        writer.writerow(header)
        for t in range(0, steps, 100):
            row_data = [t] + [results_dict[r][t] for r in row_range]
            writer.writerow(row_data)

    print("Đã xuất dữ liệu thành công ra file protocol_results.csv")