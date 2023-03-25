import os
import glob
import numpy as np
import argparse
import concurrent.futures

def process_data_file(file_path):
    with open(file_path, 'r') as f:
        data = [
            [float(eval), float(eval_next), int(winner), int(white_elo), int(black_elo), int(time_control), int(time_taken)]
            for _, eval, eval_next, winner, white_elo, black_elo, time_control, time_taken in (line.strip().split(',') for line in f)
        ]
    return data

def process_batch(sampled_files):
    data_sample = [entry for file_path in sampled_files for entry in process_data_file(file_path)]

    data_array = np.array(data_sample)
    mean_values = np.mean(data_array, axis=0)
    variance_values = np.var(data_array, axis=0)

    return mean_values, variance_values, len(data_array)

def sample_data_files(data_files, num_workers, sample_size):
    np.random.shuffle(data_files)
    file_subsets = np.array_split(data_files, num_workers)
    sampled_files = [np.random.choice(subset, sample_size // num_workers, replace=False) for subset in file_subsets]
    return sampled_files

def calculate_mean_std_error(folder_path, sample_size=100, num_batches=1, num_workers=1):
    data_files = glob.glob(os.path.join(folder_path, '*.data'))

    means = []
    variances = []
    sizes = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for _ in range(num_batches):
            sampled_files = sample_data_files(data_files, num_workers, sample_size)
            batch_futures = [executor.submit(process_batch, file_subset) for file_subset in sampled_files]

            for future in concurrent.futures.as_completed(batch_futures):
                mean_values, variance_values, size = future.result()
                means.append(mean_values)
                variances.append(variance_values)
                sizes.append(size)

    mean_values = np.mean(means, axis=0)

    total_size = sum(sizes)
    pooled_variance = np.sum([(size - 1) * variance for size, variance in zip(sizes, variances)], axis=0) / (total_size - num_batches)
    pooled_std = np.sqrt(pooled_variance)

    # Calculate the standard error of the mean
    sem = pooled_std / np.sqrt(total_size)

    return mean_values, pooled_std, sem

def main():
    parser = argparse.ArgumentParser(description='Compute mean and pooled standard deviation of .data files.')
    parser.add_argument('folder_path', type=str, help='Path to folder containing .data files')
    parser.add_argument('-s', '--sample_size', type=int, default=100, help='Number of data files to sample in each batch (default: 100)')
    parser.add_argument('-n', '--num_batches', type=int, default=1, help='Number of batches to process (default: 1)')
    parser.add_argument('-w', '--num_workers', type=int, default=1, help='Number of worker processes to use for multiprocessing (default: 1)')

    args = parser.parse_args()

    mean_values, pooled_std_values, sem_values = calculate_mean_std_error(args.folder_path, args.sample_size, args.num_batches, args.num_workers)

    column_labels = ['eval', 'eval_next', 'winner', 'white_elo', 'black_elo', 'time_control', 'time_taken']

    print("Mean values (+/- standard error):")
    for label, value, sem in zip(column_labels, mean_values, sem_values):
        print(f"{label:<12}: {value:>10.4f} +/- {sem:>6.4f}")

    print("\nPooled standard deviations:")
    for label, value in zip(column_labels, pooled_std_values):
        print(f"{label:<12}: {value:>10.4f}")

if __name__ == '__main__':
    main()
