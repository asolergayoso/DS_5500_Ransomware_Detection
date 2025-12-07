
import pandas as pd
import numpy as np
import os
import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
import multiprocessing
import time
import traceback
import gc
import psutil
import argparse
import logging
import sys

os.environ['LOG_PATH'] = './logs'
os.environ['SLURM_CPUS_PER_TASK'] = '6'
    
# Setup logging
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_DIR = Path(os.environ.get('LOG_PATH'))
LOG_DIR.mkdir(parents=True, exist_ok=True)


log_file = LOG_DIR / f'feature_engineering_{timestamp}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(processName)s] - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB


def get_datetime_min(time1, time2):
    # try:
    time1_s, time1_ns = time1.unix_time_s, time1.unix_time_ns
    time2_s, time2_ns = time2.unix_time_s, time2.unix_time_ns
    
    if time1_s < time2_s:
        return time1
    elif time2_s < time1_s:
        return time2
    else:
        if time1_ns <= time2_ns:
            return time1
        return time2
    # except Exception as e:
    #     logger.error(f"Failed at get_datetime_min: {e}")
    #     return None


def compute_relative_time(df, start_sec, start_nsec):
    """
    Compute relative time (in seconds and nanoseconds) from a start point.
    Keeps full nanosecond precision.
    """
    # try:
    # Convert everything to total nanoseconds from the start point
    total_ns = (df['unix_time_s'] - start_sec) * 1_000_000_000 + (df['unix_time_ns'] - start_nsec)

    df['relative_time'] = pd.to_timedelta(total_ns, unit='ns')
    return df
    # except Exception as e:
    #     logger.error(f"Failed at compute_relative_time: {e}")
    #     return None

def timedelta_to_ns(t):
    # try:
    if type(t) == pd.Timedelta:
        return t.total_seconds() * 1e9 + t.nanoseconds
    else:
        return t.total_seconds() * 1e9
    # except Exception as e:
    #     logger.error(f"Failed at timedelta_to_ns: {e}")
    #     return None
    
def generate_combined_log_file(dataframes):

    # strg_read, strg_write, mem_read, mem_write, mem_read_write, mem_exec = read_access_file(operation_path)
    strg_read, strg_write, mem_read, mem_write, mem_read_write, mem_exec = dataframes['storage_read'], dataframes['storage_write'], dataframes['memory_read'], dataframes['memory_write'], dataframes['memory_readwrite'], dataframes['memory_exec']
    # print("Got indiv action dfs")
    
    strg_read['action'] = 'ata_read'
    strg_write['action'] = 'ata_write'
    mem_read['action'] = 'mem_read'
    mem_write['action'] = 'mem_write'
    mem_read_write['action'] = 'mem_read_write'
    mem_exec['action'] = 'mem_exec'
    
    # print("Added action type")
    
    start_point = get_datetime_min(
                    get_datetime_min(
                        get_datetime_min(
                            get_datetime_min(
                                get_datetime_min(strg_read.loc[0, ['unix_time_s', 'unix_time_ns']], strg_write.loc[0, ['unix_time_s', 'unix_time_ns']]), 
                                mem_read.loc[0, ['unix_time_s', 'unix_time_ns']]), 
                            mem_write.loc[0, ['unix_time_s', 'unix_time_ns']]), 
                        mem_read_write.loc[0, ['unix_time_s', 'unix_time_ns']]), 
                    mem_exec.loc[0, ['unix_time_s', 'unix_time_ns']])

    # print("Got Minimum Datetime")
    strg_read = compute_relative_time(strg_read, start_point.unix_time_s, start_point.unix_time_ns)
    strg_write = compute_relative_time(strg_write, start_point.unix_time_s, start_point.unix_time_ns)
    mem_read = compute_relative_time(mem_read, start_point.unix_time_s, start_point.unix_time_ns)
    mem_write = compute_relative_time(mem_write, start_point.unix_time_s, start_point.unix_time_ns)
    mem_read_write = compute_relative_time(mem_read_write, start_point.unix_time_s, start_point.unix_time_ns)
    mem_exec = compute_relative_time(mem_exec, start_point.unix_time_s, start_point.unix_time_ns)
    
    # print("Converted time to real time")
    combined_logs = pd.concat([strg_read, strg_write, mem_read, mem_write, mem_read_write, mem_exec], ignore_index=True)
    combined_logs.drop(['unix_time_s', 'unix_time_ns'], axis=1, inplace=True)
    
    return combined_logs

# Temporal Evolution Features 
def temporal_evolution_features(strg_write_window_data, mem_write_window_data, mem_read_write_window_data, window_start, entropy_spike_threshold=0.9, consecutive_entropy_threshold=0.8):
    
    def compute_acceleration(signal):
        """Compute 2nd derivative (acceleration)"""
        if len(signal) < 3:
            return 0 
        first_derivative = np.gradient(signal)
        second_derivative = np.gradient(first_derivative)
        return np.mean(second_derivative)

    def count_consecutive_above_threshold(values, threshold):
        above = (values > threshold).astype(int)
        max_consecutive = 0
        current = 0
        
        for val in above:
            if val == 1:
                current += 1
                max_consecutive = max(max_consecutive, current)
            else:
                current = 0
        
        return max_consecutive

    def compute_time_to_threshold(window_data, feature, window_start, threshold):
        """Time to first event above threshold in window"""
        high_entropy = window_data[window_data[feature] > threshold]
        if len(high_entropy) == 0:
            return np.nan
        return timedelta_to_ns(high_entropy['relative_time'].iloc[0] - window_start)
    
    def get_temporal_features_for_action(window_data, feature, window_start, entropy_spike_threshold, consecutive_entropy_threshold, prefix=''):
        features = {
            f'{prefix}_high_entropy_count': (window_data[feature] > entropy_spike_threshold).sum(),
            f'{prefix}_max_entropy_jump': np.diff(window_data[feature].values).max() if len(window_data) > 1 else 0,
            f'{prefix}_time_to_high_entropy': compute_time_to_threshold(window_data, feature, window_start, entropy_spike_threshold),
            
            f'{prefix}_sustained_high_entropy_events': count_consecutive_above_threshold(
                window_data[feature].values, threshold=consecutive_entropy_threshold
            ),
            f'{prefix}_entropy_acceleration': compute_acceleration(window_data[feature].values),
        }
        return features
    
    features = {}
    
    features = features | get_temporal_features_for_action(strg_write_window_data[['relative_time', 'storage_entropy']], 'storage_entropy', window_start, 
                                                            entropy_spike_threshold, consecutive_entropy_threshold, 'strg_write')

    features = features | get_temporal_features_for_action(mem_write_window_data[['relative_time', 'mem_entropy']], 'mem_entropy', window_start, 
                                                            entropy_spike_threshold, consecutive_entropy_threshold, 'mem_write')
    
    features = features | get_temporal_features_for_action(mem_read_write_window_data[['relative_time', 'mem_entropy']], 'mem_entropy', window_start, 
                                                            entropy_spike_threshold, consecutive_entropy_threshold, 'mem_read_write')
    
    return features

# Burst Detection Features
def burst_features(strg_read_window_data, strg_write_window_data, mem_read_window_data, mem_write_window_data, mem_read_write_window_data, mem_exec_window_data, window_size_ns):
    
    def compute_instant_throughput(df, timestamp_col='relative_time', size_col='size'):
        df2 = df.copy()
        if len(df2) < 1:
            df2['instant_throughput'] = []
            return df2

        df2 = df2.sort_values(timestamp_col)
        
        start_time = df2.iloc[0][timestamp_col]
        if start_time == pd.Timedelta(0):
            start_time = pd.Timedelta(microseconds=0.001) # set start_time to 1ns

        grouped = df2.groupby(timestamp_col, sort=True)[size_col].sum().reset_index().rename(columns={size_col: 'size_sum'})

        grouped['time_diff_ns'] = grouped[timestamp_col].diff().apply(lambda x: timedelta_to_ns(start_time) if pd.isnull(x) else timedelta_to_ns(x))
        grouped['throughput'] = grouped['size_sum'] / grouped['time_diff_ns']

        mapping = dict(zip(grouped[timestamp_col], grouped['throughput']))
        df2["instant_throughput"] = df2[timestamp_col].map(mapping)
        
        return df2
    
    def compute_burst_clustering(window_data, quantile_threshold = 0.75):
        """Measure temporal clustering of high-throughput events"""
        if len(window_data) < 2:
            return 0
        
        threshold = window_data['instant_throughput'].quantile(quantile_threshold)
        burst_times = window_data[window_data['instant_throughput'] > threshold]['relative_time'].values
        
        if len(burst_times) < 2:
            return 0
        
        inter_burst_intervals = np.diff(burst_times)
        
        return np.mean(inter_burst_intervals).astype(int)
    
    def get_burst_features_for_action(window_data, prefix, size_col, window_size_ns):
        window_data_w_throughput = compute_instant_throughput(window_data, size_col=size_col)
        features = {
            f'{prefix}_peak_instant_throughput': window_data_w_throughput['instant_throughput'].max(),
                        
            f'{prefix}_burst_intensity': (window_data_w_throughput['instant_throughput'].max() / (window_data_w_throughput['instant_throughput'].median() + 1e-10)),
            
            f'{prefix}_micro_burst_count': (window_data_w_throughput['instant_throughput'] > 
                                window_data_w_throughput['instant_throughput'].mean() + 2 * window_data_w_throughput['instant_throughput'].std()).sum(),
            
            f'{prefix}_burst_clustering': compute_burst_clustering(window_data_w_throughput),
            
            f'{prefix}_event_rate': len(window_data_w_throughput) / window_size_ns,
        }
        
        return features
    
    features = {}
    
    features = features | get_burst_features_for_action(strg_read_window_data, 'strg_read', 'storage_size', window_size_ns)
    features = features | get_burst_features_for_action(strg_write_window_data, 'strg_write', 'storage_size', window_size_ns)
    features = features | get_burst_features_for_action(mem_read_window_data, 'mem_read', 'mem_size', window_size_ns)
    features = features | get_burst_features_for_action(mem_write_window_data, 'mem_write', 'mem_size', window_size_ns)
    features = features | get_burst_features_for_action(mem_read_write_window_data, 'mem_read_write', 'mem_size', window_size_ns)
    features = features | get_burst_features_for_action(mem_exec_window_data, 'mem_exec', 'mem_size', window_size_ns)
    
    return features

# Address Spatial Access Features
def spatial_access_patterns(strg_read_window_data, strg_write_window_data, mem_read_window_data, mem_write_window_data, mem_read_write_window_data, mem_exec_window_data):
    """
    Analyze LBA and GPA access sequences from raw events
    """
    
    def compute_sequential_ratio(addr_sequence):
        """Fraction of address accesses that are sequential"""
        if len(addr_sequence) < 2:
            return 0
        
        diffs = np.abs(np.diff(addr_sequence))
        sequential = (diffs == 1).sum()
        return sequential / len(diffs)

    def compute_access_direction_ratio(addr_sequence):
        if len(addr_sequence) < 2:
            return 1.0
        
        diffs = np.diff(addr_sequence)
        forward = (diffs > 0).sum()
        backward = (diffs < 0).sum()
        
        return forward / (backward + 1e-10)

    def get_spatial_access_features_for_action(addr_sequence, prefix):

        if prefix in ['strg_read', 'strg_write']:
            addr_name = 'lba'
        elif prefix in ['mem_read', 'mem_write', 'mem_read_write', 'mem_exec']:
            addr_name = 'gpa'
        else:
            addr_name = 'addr'
        features = { 
            f'{prefix}_seq_addr_access_ratio': compute_sequential_ratio(addr_sequence),
            
            f'{prefix}_avg_{addr_name}_jump': np.mean(np.abs(np.diff(addr_sequence))) if len(addr_sequence) > 1 else 0,
            f'{prefix}_max_{addr_name}_jump': np.max(np.abs(np.diff(addr_sequence))) if len(addr_sequence) > 1 else 0,
            
            f'{prefix}_access_locality': len(np.unique(addr_sequence)) / len(addr_sequence) if len(addr_sequence) > 0 else 0,
            
            f'{prefix}_frwrd_bckwrd_ratio': compute_access_direction_ratio(addr_sequence),
            
            f'{prefix}_{addr_name}_range': addr_sequence.max() - addr_sequence.min() if len(addr_sequence) > 0 else 0,
        }
        return features
    
    
    strg_read_addr_sequence = strg_read_window_data['lba'].values
    strg_write_addr_sequence = strg_write_window_data['lba'].values
    mem_read_addr_sequence = mem_read_window_data['gpa'].values
    mem_write_addr_sequence = mem_write_window_data['gpa'].values
    mem_read_write_addr_sequence = mem_read_write_window_data['gpa'].values
    mem_exec_addr_sequence = mem_exec_window_data['gpa'].values
    
    features = {}
    features = features | get_spatial_access_features_for_action(strg_read_addr_sequence, 'strg_read')
    features = features | get_spatial_access_features_for_action(strg_write_addr_sequence, 'strg_write')
    features = features | get_spatial_access_features_for_action(mem_read_addr_sequence, 'mem_read')
    features = features | get_spatial_access_features_for_action(mem_write_addr_sequence, 'mem_write')
    features = features | get_spatial_access_features_for_action(mem_read_write_addr_sequence, 'mem_read_write')
    features = features | get_spatial_access_features_for_action(mem_exec_addr_sequence, 'mem_exec')
        
    return features
    
def extract_features(df: pd.DataFrame, is_ransomware: bool, operation: str, window_s: int, epsilon_s: int = 1) -> pd.DataFrame:
    '''
    window_s: Window size in seconds
    epsilon_s: Step size in seconds
    '''
    
    data_rows = []
    window_delta = datetime.timedelta(seconds=window_s)
    epsilon_delta = datetime.timedelta(seconds=epsilon_s)
    
    last_start_possible = max(df['relative_time']) - window_delta
        
    start_time = min(df['relative_time'])
    
    strg_read = df[df['action'] == 'ata_read'][["relative_time", "lba", "storage_size"]]
    strg_write = df[df['action'] == 'ata_write'][["relative_time", "lba", "storage_size", "storage_entropy"]]
    mem_read = df[df['action'] == 'mem_read'][["relative_time", "gpa", 'mem_size', 'mem_entropy', 'mem_access_type']]
    mem_write = df[df['action'] == 'mem_write'][["relative_time", "gpa", 'mem_size', 'mem_entropy', 'mem_access_type']]
    mem_read_write = df[df['action'] == 'mem_read_write'][["relative_time", "gpa", 'mem_size', 'mem_entropy', 'mem_access_type']]
    mem_exec = df[df['action'] == 'mem_exec'][["relative_time", "gpa", 'mem_size', 'mem_entropy', 'mem_access_type']]
    
    while start_time <= last_start_possible:
        # try:
        end_time = start_time + window_delta
        
        d_ata_read = strg_read[(start_time <= strg_read['relative_time']) & (strg_read['relative_time'] <= end_time)]
        d_ata_write = strg_write[(start_time <= strg_write['relative_time']) & (strg_write['relative_time'] <= end_time)]
        d_mem_read = mem_read[(start_time <= mem_read['relative_time']) & (mem_read['relative_time'] <= end_time)]
        d_mem_write = mem_write[(start_time <= mem_write['relative_time']) & (mem_write['relative_time'] <= end_time)]
        d_mem_read_write = mem_read_write[(start_time <= mem_read_write['relative_time']) & (mem_read_write['relative_time'] <= end_time)]
        d_mem_exec = mem_exec[(start_time <= mem_exec['relative_time']) & (mem_exec['relative_time'] <= end_time)]

        features = {}
        
        features = features | {"start": start_time, "end": end_time}
        
        features = features | {
            "avg_ata_read": d_ata_read['storage_size'].sum() / window_s,
            "avg_ata_write": d_ata_write['storage_size'].sum() / window_s,
            "lba_read_var": d_ata_read['lba'].var(),
            "lba_write_var": d_ata_write['lba'].var(),
            "avg_storage_entropy": d_ata_write['storage_entropy'].mean()
        }
        
        features = features | {
            "avg_entropy_mem_write": d_mem_write['mem_entropy'].mean(),
            "avg_entropy_mem_read_write": d_mem_read_write['mem_entropy'].mean()
        }

        features = features | {
            "num_4KB_pages_mem_read": len(d_mem_read[d_mem_read['mem_access_type'] == 1]),
            "num_4KB_pages_mem_write": len(d_mem_write[d_mem_write['mem_access_type'] == 1]),
            "num_4KB_pages_mem_read_write": len(d_mem_read_write[d_mem_read_write['mem_access_type'] == 1]),
            "num_4KB_pages_mem_exec": len(d_mem_exec[d_mem_exec['mem_access_type'] == 1]),
            
            "num_2MB_pages_mem_read": len(d_mem_read[d_mem_read['mem_access_type'] == 2]),
            "num_2MB_pages_mem_write": len(d_mem_write[d_mem_write['mem_access_type'] == 2]),
            "num_2MB_pages_mem_read_write": len(d_mem_read_write[d_mem_read_write['mem_access_type'] == 2]),
            "num_2MB_pages_mem_exec": len(d_mem_exec[d_mem_exec['mem_access_type'] == 2]),
            
            "num_MMIO_pages_mem_read": len(d_mem_read[d_mem_read['mem_access_type'] == 4]),
            "num_MMIO_pages_mem_write": len(d_mem_write[d_mem_write['mem_access_type'] == 4]),
            "num_MMIO_pages_mem_read_write": len(d_mem_read_write[d_mem_read_write['mem_access_type'] == 4]),
            "num_MMIO_pages_mem_exec": len(d_mem_exec[d_mem_exec['mem_access_type'] == 4]),
        }
        
        features = features | {
            "gpa_mem_read_var": d_mem_read['gpa'].var(),
            "gpa_mem_write_var": d_mem_write['gpa'].var(),
            "gpa_mem_read_write_var": d_mem_read_write['gpa'].var(),
            "gpa_mem_exec_var": d_mem_exec['gpa'].var()
        }
        
        features = features | temporal_evolution_features(d_ata_write, d_mem_write, d_mem_read_write, start_time)
        features = features | burst_features(d_ata_read, d_ata_write, d_mem_read, d_mem_write, d_mem_read_write, d_mem_exec, window_s*1e9)
        features = features | spatial_access_patterns(d_ata_read, d_ata_write, d_mem_read, d_mem_write, d_mem_read_write, d_mem_exec)
        

        data_rows.append(features)
        
        start_time += epsilon_delta
    
    preprocessed_df = pd.DataFrame(data_rows)
    preprocessed_df['is_ransomware'] = is_ransomware
    preprocessed_df['operation'] = operation
    return preprocessed_df


def process_single_execution(args):
    """
    Process one execution
    Returns DataFrame or None on failure
    """
    # print(args)
    execution_id, execution_cnt, app_name, app_path, window_size, step_size, ransomware = args
    # app_executions[exec_ind], exec_ind, app_name, app_path, window_size, step_size, ransomware
    
    # setup_worker_logger(execution_id, log_dir)
    # logger = logging.LoggerAdapter(logging.getLogger(), {"exec_id": execution_id, "app": app_name})
    
    logger.info(f"[{app_name}, {execution_id}] Worker started")
    
    try:
        # exec_dir = self.data_dir / app_path / f"{execution_id}"
        # exec_dir = os.path.join(app_path, execution_id)
        # print(f"Processing {execution_id} - Memory: {get_memory_usage():.1f} MB")
        
        exec_dir = app_path.joinpath(execution_id)
        
        csv_files = {
            'storage_read': 'ata_read.csv',
            'storage_write': 'ata_write.csv',
            'memory_read': 'mem_read.csv',
            'memory_write': 'mem_write.csv',
            'memory_readwrite': 'mem_readwrite.csv',
            'memory_exec': 'mem_exec.csv'
        }
        
        dataframes = {}
        for name, filename in csv_files.items():
            # filepath = exec_dir / filename
            # filepath = os.path.join(exec_dir, filename)
            filepath = exec_dir.joinpath(filename)
            if not filepath.exists():
                # print(f"X {filepath} not found")
                logger.error(f"[{app_name}, {execution_id}] [Worker exec_id = {execution_id}] Filename {filepath} doesnt exist")
                return None
            
            # Read based on file type
            if 'storage' in name:
                # cols = ['ts', 'tns', 'lba', 'size', 'entropy']
                cols = ['unix_time_s', 'unix_time_ns', 'lba', 'storage_size', 'storage_entropy', 'tmp']
                usecols=[0,1,2,3,4]
            else:
                # cols = ['ts', 'tns', 'gpa', 'size', 'entropy', 'type']
                cols = ['unix_time_s', 'unix_time_ns', 'gpa', 'mem_size', 'mem_entropy', 'mem_access_type']
                usecols=[0,1,2,3,4,5]
            
            #
            dataframes[name] = pd.read_csv(filepath, names=cols, usecols=usecols)
        
        # print(f"  Loaded CSVs - Memory: {get_memory_usage():.1f} MB")
        # print("Trying to combine 6 log DFs")
        # for k, v in dataframes.items():
            # print(f"{k}: {type(v)} - {len(v)}")
            
        combined_df = generate_combined_log_file(dataframes)
        
        del dataframes
        gc.collect()
        
        logger.info(f"[{app_name}, {execution_id}] Worker combined 6 log file dataframes")
        
        # print(f"  Combined - Memory: {get_memory_usage():.1f} MB")
        # print("Combined 6 log DFs. Length: ", len(combined_df))
        # Extract features (your custom logic)
        is_ransomware = app_name in ransomware
        execution_df = extract_features(combined_df, is_ransomware, app_name, window_size, step_size)
        # execution_df = pd.DataFrame()
        
        del combined_df
        gc.collect()
        
        # print(f"  Extracted features - Memory: {get_memory_usage():.1f} MB")
        logger.info(f"[{app_name}, {execution_id}] Worker Extracted Features. Execution DF shape: {execution_df.shape}")
        
        # Add metadata
        execution_df['execution_id'] = execution_id
        execution_df['execution_log_cnt'] = execution_cnt
        
        return execution_df
    
    except MemoryError as e:
        # print(f"✗ MEMORY ERROR in {execution_id}: {e}")
        logger.error(f"[Worker exec_id = {execution_id}] Worker Memory Error: {e}")
        gc.collect()
        raise
    
    except Exception as e:
        # print(f"Error in {app_path} exec {execution_id}: {type(e).__name__}:  {e}")
        logger.error(f"[Worker exec_id = {execution_id}] Worker Unhandled Exception: {e}")
        # traceback.print_exc()
        raise
        

def process_application(app_name, app_parent_path, sys_config, window_size, step_size, ransomware, output_dir, max_workers=2):
    """
    Process all executions of one application in parallel
    """
    logger.info(f"--- Starting process_application for {app_name} of {sys_config} with window size {window_size}, step size {step_size} ---")
    
    start_time = time.time()
    results = []
    errors = []
    
    app_path = app_parent_path.joinpath(app_name)
    app_executions = os.listdir(app_path)
    num_app_executions = len(app_executions)
    
    logger.info(f"({sys_config} : {app_name}) - Found {num_app_executions} executions")
    # logger.info(f"Using {max_workers} parallel workers")
    
    worker_args = [
        (app_executions[exec_ind], exec_ind, app_name, app_path, window_size, step_size, ransomware)
        for exec_ind in range(num_app_executions)
    ]
    
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_execution, args): args[0] # args[0] is execution_id
            for args in worker_args
        }
        
        completed = 0
        for future in as_completed(futures):
            exec_id = futures[future]
            completed += 1
            try:
                result = future.result(timeout=600)
                if result is not None:
                    results.append(result)
                    logger.info(f"[OK] {completed}/{num_app_executions} Completed {exec_id}")
                else:
                    errors.append(exec_id)
                    logger.error(f"[FAIL] [{completed}/{num_app_executions}] Failed {exec_id} returned None")
                    
            except TimeoutError:
                errors.append(exec_id)
                logger.error(f"[TIMEOUT] [{completed}/{num_app_executions}] TIMEOUT {exec_id}")
            except Exception as e:
                errors.append(exec_id)
                logger.exception(f"[ERROR] [{completed}/{num_app_executions}] ERROR {exec_id}: {type(e).__name__}: {e}")
                # pbar.update(1)

    
    elapsed = time.time() - start_time
    # logger = logging.LoggerAdapter(logger, {"exec_id": None})
    # print(f"\n{'-'*70}")
    # print(f"Summary for {app_name}")
    # print(f"{'-'*70}")
    # print(f"Successful: {len(results)}/{num_app_executions}")
    # print(f"Failed: {len(errors)}/{num_app_executions}")
    # print(f"Time: {elapsed:.2f}s")
    logger.info(f"({sys_config} : {app_name}) - Processing Time: {elapsed}")
    logger.info(f"({sys_config} : {app_name}) - Results length: {len(results)}")
    
    if errors:
        # print(f"\nFailed executions: {errors}")
        logger.error(f"({sys_config} : {app_name}) - Executions with Processing Errors: {errors}")
        
    # Combine and save
    if results:
        combined_df = pd.concat(results, ignore_index=True)
        combined_df['sys_config'] = sys_config
        
        logger.info(f"({sys_config} : {app_name}) - Combined DF Shape: {combined_df.shape}")
        
        # Save
        output_parent_path = output_dir.joinpath(sys_config)
        output_parent_path.mkdir(parents=True, exist_ok=True)
        output_file_path = os.path.join(output_parent_path, f"{app_name}.csv")
        
        logger.info(f"({sys_config} : {app_name}) - Saving Combined DF to: {output_file_path}")
        
        # output_file_path = self.output_dir.joinpath(sys_config).joinpath(f"{app_name}.csv")
        # output_file_path.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(output_file_path, index=False)
        
        logger.info(f"({sys_config} : {app_name}) - Saved {len(results)}/{num_app_executions} executions")
        
        # print(f"✓ {app_name}: {len(results)}/{num_app_executions} executions")
        
        # # print(f"  Time: {elapsed:.2f}s")
        # print(f"  Saved: {output_file_path}")
        # print(f"  Shape: {combined_df.shape}")
        # print(f"{'-'*70}")
        return combined_df
    else:
        logger.error(f"({sys_config} : {app_name}) - No Results. All executions had errors")
        # print(f"✗ {app_name}: No successful executions")
        # print(f"{'-'*70}")
        return None

def process_config(data_dir: Path, output_dir: Path, window_size, step_size, max_workers):
    
    ransomware = set(["WannaCry", "Ryuk", "REvil", "LockBit", "Darkside", "Conti"])
    
    collecs = ['original', 'extra']

    for collec in collecs:
        collec_dir = data_dir.joinpath(collec)
        cpus = os.listdir(collec_dir)
        for cpu in cpus:
            cpu_path = collec_dir.joinpath(cpu)
            ram_configs = os.listdir(cpu_path)
            for ram_config in ram_configs:
                config_path = cpu_path.joinpath(ram_config)
                all_config_name = f"{collec}_{cpu}_{ram_config}"
                
                apps = os.listdir(config_path)
                
                for app in apps[:2]:
                    process_application(app, config_path, all_config_name, window_size, step_size, ransomware, output_dir, max_workers)


def main():
    
    cpus_allocated = os.getenv("SLURM_CPUS_PER_TASK")
    
    if cpus_allocated is not None:
        num_cores = int(cpus_allocated)
    else:
        num_cores = multiprocessing.cpu_count()
    
    
    max_workers = max(1, num_cores-2)

    parser = argparse.ArgumentParser(description="Process ransomware configs")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--window-size", type=int, required=True)
    parser.add_argument("--step-size", type=int, required=True)
    
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    window_size = args.window_size
    step_size = args.step_size

    logger.info(f"Starting pipeline with {max_workers} workers")
    
    process_config(data_dir=data_dir, output_dir=output_dir, window_size=window_size, step_size=step_size, max_workers=max_workers)

    logger.info("Finished pipeline")

if __name__ == "__main__":
    main()