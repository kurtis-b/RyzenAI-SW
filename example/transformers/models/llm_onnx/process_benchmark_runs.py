import ijson
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os


NUM_BENCHMARK_RUNS = 10
NUM_FORWARD_PASSES = 11
NUM_PROMPTS = 10
BENCHMARK_RUNS_DIR = "benchmark_runs"
NPU_ANALYSIS_DIR = "npu_analysis"


def compare_json_incremental(cpu_file, npu_file, output_file_name):
    result = []
    mismatch_count = 0
    cpu_mat_mul_count = 0

    # The following keys show up in the nodes' attributes
    NAME_KEY = 'name'
    CPU_MATMUL_KEY = 'MatMul_Q4_fence_before'
    NPU_MATMUL_KEY_1 = 'VitisAIExecutionProvider'
    NPU_MATMUL_KEY_2 = 'fence_before'

    with open(cpu_file, 'r') as f_cpu, open(npu_file, 'r') as f_npu:
        # Use ijson to stream each item in the array of objects
        cpu_parser = ijson.items(f_cpu, 'item')  # Process CPU data items
        npu_parser = ijson.items(f_npu, 'item')  # Process NPU data items
        cpu_obj = next(cpu_parser, None)
        npu_obj = next(npu_parser, None)
        
        while cpu_obj is not None and npu_obj is not None:
            cpu_name = cpu_obj[NAME_KEY]
            npu_name = npu_obj[NAME_KEY]

            # Keep track of the number of times mat mul is executed on the CPU and NPU
            if CPU_MATMUL_KEY in cpu_name:
                cpu_mat_mul_count = cpu_mat_mul_count + 1
            if NPU_MATMUL_KEY_1 in npu_name and NPU_MATMUL_KEY_2 in npu_name:
                mismatch_count = mismatch_count + 1

            if cpu_name == npu_name:
                # If names match, move to the next line in both files
                cpu_obj = next(cpu_parser, None)
                npu_obj = next(npu_parser, None)
            else:
                # If names don't match, store CPU names until a match is found in NPU
                mismatch_data = {'nodes': [],
                                'mismatches': []}
                header = npu_name.split("_before")[0]
                mismatch_data['nodes'].append(npu_name)
                for _ in range(2):
                    npu_obj = next(npu_parser, None)
                    npu_name = npu_obj[NAME_KEY]
                    mismatch_data['nodes'].append(npu_name)
                next_npu_name = npu_name

                # Collect mismatched CPU lines until names match again
                npu_obj = next(npu_parser, None)
                if npu_obj is not None:
                    next_npu_name = npu_obj[NAME_KEY]
                while cpu_obj is not None and cpu_name != next_npu_name:
                    mismatch_data['mismatches'].append(cpu_name)
                    cpu_obj = next(cpu_parser, None)
                    if cpu_obj is not None:
                        cpu_name = cpu_obj[NAME_KEY]

                result.append({header: mismatch_data})

                # Move to the next matching NPU line or break if CPU data ends
                if cpu_obj is None:
                    break
    
    print(f'Found {mismatch_count} mismatches')
    cpu_mat_mul_count = cpu_mat_mul_count
    result.insert(0, {'Mismatches found': mismatch_count,
                      'CPU Mat Mul executions': cpu_mat_mul_count,
                      'Mat Mul executions not offloaded to NPU': cpu_mat_mul_count - mismatch_count})
    # Save mismatched data to an output JSON file
    save_data_to_json(result, output_file_name, indent=4)


def find_execution_distribution(cpu_file, npu_file, output_file_name):
    parsers = {}
    NAME_KEY = 'name'
    ARGS_KEY = 'args'
    OP_NAME_KEY = 'op_name'
    TIME_START_PARSER_KEY = 'ts'
    DURATION_PARSER_KEY = 'dur'
    EXEC_TIME_PREFIX_1 = 'First'
    EXEC_TIME_PREFIX_2 = 'Subsequent'

    with open(cpu_file, 'r') as f_cpu, open(npu_file, 'r') as f_npu:
        # Use ijson to stream each item in the array of objects
        parsers[output_file_name +'_cpu'] = ijson.items(f_cpu, 'item')  # Process CPU data items
        parsers[output_file_name +'_npu'] = ijson.items(f_npu, 'item')  # Process NPU data items

        for file_name, parser in parsers.items():
            result = [{ f"Task {x+1}": {
            "SequentialExecutor::Execute": {
                f'{EXEC_TIME_PREFIX_1} token time': -1,
                f'{EXEC_TIME_PREFIX_2} token times': [],
                f'{EXEC_TIME_PREFIX_2} token average': -1,
                f'{EXEC_TIME_PREFIX_2} token maximum': -1,
                f'{EXEC_TIME_PREFIX_2} token minimum': -1,
            },
            "model_run": {
                f'{EXEC_TIME_PREFIX_1} token time': -1,
                f'{EXEC_TIME_PREFIX_2} token times': [],
                f'{EXEC_TIME_PREFIX_2} token average': -1,
                f'{EXEC_TIME_PREFIX_2} token maximum': -1,
                f'{EXEC_TIME_PREFIX_2} token minimum': -1,
            }
            }} for x in range(NUM_PROMPTS)]
            parser_obj = next(parser, None)

            counter = 1
            run = 0
            run_key = f"Task {run+1}"
            while parser_obj is not None:
                name = parser_obj[NAME_KEY]
                
                if name == "model_loading_uri":
                    dict_to_append = {name: {'Duration': parser_obj[DURATION_PARSER_KEY],
                                            'Time start': parser_obj[TIME_START_PARSER_KEY]}}
                    result.append(dict_to_append)
                if name == "session_initialization":
                    dict_to_append = {name: {'Duration': parser_obj[DURATION_PARSER_KEY],
                                            'Time start': parser_obj[TIME_START_PARSER_KEY]}}
                    result.append(dict_to_append)
                elif name == "SequentialExecutor::Execute":
                    if counter == 1:
                        result[run][run_key][name][f'{EXEC_TIME_PREFIX_1} token time'] = parser_obj[DURATION_PARSER_KEY]
                    else:
                        result[run][run_key][name][f'{EXEC_TIME_PREFIX_2} token times'].append(parser_obj[DURATION_PARSER_KEY])
                elif name == "model_run":
                    if counter == 1:
                        result[run][run_key][name][f'{EXEC_TIME_PREFIX_1} token time'] = parser_obj[DURATION_PARSER_KEY]
                    else:
                        result[run][run_key][name][f'{EXEC_TIME_PREFIX_2} token times'].append(parser_obj[DURATION_PARSER_KEY])
                    counter = (counter + 1) % NUM_FORWARD_PASSES
                    if counter == 1:
                        run = run + 1
                        run_key = f"Task {run+1}"
                else:
                    if "kernel_time" in name:
                        if "VitisAIExecutionProvider" in name:
                            if counter == 1:
                                if 'vitisaiep' not in result[run][run_key]:
                                    result[run][run_key]['vitisaiep'] = {f'{EXEC_TIME_PREFIX_1} token times': [parser_obj[DURATION_PARSER_KEY]],
                                                                f'{EXEC_TIME_PREFIX_2} token times': []}
                                else:
                                    result[run][run_key]['vitisaiep'][f'{EXEC_TIME_PREFIX_1} token times'].append(parser_obj[DURATION_PARSER_KEY])
                            else:
                                result[run][run_key]['vitisaiep'][f'{EXEC_TIME_PREFIX_2} token times'].append(parser_obj[DURATION_PARSER_KEY])
                        else:
                            if counter == 1:
                                if parser_obj[ARGS_KEY][OP_NAME_KEY] not in result[run][run_key]:
                                    result[run][run_key][parser_obj[ARGS_KEY][OP_NAME_KEY]] = {f'{EXEC_TIME_PREFIX_1} token times': [parser_obj[DURATION_PARSER_KEY]],
                                                                                f'{EXEC_TIME_PREFIX_2} token times': []}
                                else:
                                    result[run][run_key][parser_obj[ARGS_KEY][OP_NAME_KEY]][f'{EXEC_TIME_PREFIX_1} token times'].append(parser_obj[DURATION_PARSER_KEY])
                            else:
                                result[run][run_key][parser_obj[ARGS_KEY][OP_NAME_KEY]][f'{EXEC_TIME_PREFIX_2} token times'].append(parser_obj[DURATION_PARSER_KEY])
                parser_obj = next(parser, None)

            for run in range(NUM_PROMPTS):
                run_key = f"Task {run+1}"
                for exec_name, data in result[run][run_key].items():
                    if exec_name == 'model_loading_uri' or exec_name == 'session_initialization': 
                        continue
                    elif exec_name == "SequentialExecutor::Execute" or exec_name == "model_run":
                        key_prefixes = [EXEC_TIME_PREFIX_2]
                        for prefix in key_prefixes:
                            microseconds_data = []  # Need to convert the time values to microseconds since they're in a particular format
                            for times in data[f'{prefix} token times']:
                                microseconds_data.append(time_to_microseconds(times))
                            result[run][run_key][exec_name][f'{prefix} token average'] = microseconds_to_time_format(round(np.average(microseconds_data)))
                            result[run][run_key][exec_name][f'{prefix} token maximum'] = microseconds_to_time_format(round(np.max(microseconds_data)))
                            result[run][run_key][exec_name][f'{prefix} token minimum'] = microseconds_to_time_format(round(np.min(microseconds_data)))
                    else:
                        key_prefixes = [EXEC_TIME_PREFIX_1, EXEC_TIME_PREFIX_2]
                        for prefix in key_prefixes:
                            microseconds_data = []  # Need to convert the time values to microseconds since they're in a particular format
                            for times in data[f'{prefix} token times']:
                                microseconds_data.append(time_to_microseconds(times))
                            result[run][run_key][exec_name][f'{prefix} token average'] = microseconds_to_time_format(round(np.average(microseconds_data)))
                            result[run][run_key][exec_name][f'{prefix} token maximum'] = microseconds_to_time_format(round(np.max(microseconds_data)))
                            result[run][run_key][exec_name][f'{prefix} token minimum'] = microseconds_to_time_format(round(np.min(microseconds_data)))
                    
            # Save data to an output JSON file
            save_data_to_json(result, f'{file_name}.json', indent=1)


def save_data_to_json(result, output_file, indent):
    with open(output_file, 'w') as f_out:
        json.dump(result, f_out, indent=indent)


def microseconds_to_time_format(total_microseconds):
    """
    Convert a total time in microseconds into the custom format with 
    minutes, seconds, milliseconds, and microseconds.
    
    Args:
    total_microseconds (int): Total time in microseconds.
    
    Returns:
    int: Time formatted as microseconds (3 digits), milliseconds (3 digits), 
         seconds (2 digits), and minutes (1 digit).
    """
    # Calculate minutes
    minutes = total_microseconds // (60 * 1_000_000)
    remaining_microseconds = total_microseconds % (60 * 1_000_000)
    
    # Calculate seconds
    seconds = remaining_microseconds // 1_000_000
    remaining_microseconds = remaining_microseconds % 1_000_000
    
    # Calculate milliseconds
    milliseconds = remaining_microseconds // 1_000
    microseconds = remaining_microseconds % 1_000
    
    # Format the result: mssmmuuu (minutes, seconds, milliseconds, microseconds)
    formatted_time = f"{minutes}{seconds:02}{milliseconds:03}{microseconds:03}"
    
    return int(formatted_time)


def time_to_microseconds(time_val):
    """
    Convert a custom time format (formatted from right to left) into microseconds.
    
    Args:
    time_val (int): Time formatted as microseconds (3 digits), milliseconds (3 digits), 
                    seconds (2 digits), and minutes (1 digit).
    
    Returns:
    int: Total time in microseconds.
    """
    time_str = str(time_val).zfill(9)  # Pad the number to ensure it's 9 digits long
    
    # Extract microseconds, milliseconds, seconds, and minutes
    microseconds = int(time_str[-3:])         # Last 3 digits are microseconds
    milliseconds = int(time_str[-6:-3])       # 3 digits before the last are milliseconds
    seconds = int(time_str[-8:-6])            # 2 digits before that are seconds
    minutes = int(time_str[-9:-8])            # The first digit is the minute
    
    # Convert everything to microseconds
    total_microseconds = (minutes * 60 * 1_000_000) + (seconds * 1_000_000) + (milliseconds * 1_000) + microseconds
    
    return total_microseconds


def calculate_ratio(time1, time2):
    """
    Calculate the ratio of two time values.
    
    Args:
    time1 (int): First time value.
    time2 (int): Second time value.
    
    Returns:
    float: Ratio of time1 to time2.
    """
    # time1_microseconds = time_to_microseconds(time1)
    # time2_microseconds = time_to_microseconds(time2)
    
    return time1 / time2

def generate_bar_chart(data, title, save_name=None, large_set=False, order_values=False):
    # Generate session data bar chart
    width = 0.5  # the width of the bars
    multiplier = 0

    if large_set:
        fig, ax = plt.subplots(figsize=(16,8))
    else:
        fig, ax = plt.subplots(figsize=(8, 6))

    chart_data = {'x_labels': data['x_labels'], 'npu:cpu': data['npu']}
    if len(data['cpu']) == len(data['npu']):
        for idx in range(len(data['cpu'])):
            chart_data['npu:cpu'][idx] = round(calculate_ratio(chart_data['npu:cpu'][idx], data['cpu'][idx]), 2)
        if order_values:
            new_order = np.array(chart_data['npu:cpu']).argsort()[::-1]  # Order in descending order
            for key, values in chart_data.items():
                values = [values[i] for i in new_order]
                chart_data[key] = values
    x = np.arange(len(chart_data['x_labels']))  # the label locations

    threshold = 1
    hatches = ['x' if chart_data['x_labels'][i] == 'MatMulNBits' else '' for i in range(len(chart_data['x_labels']))]
    labels = ['Offloaded to NPU' if chart_data['x_labels'][i] == 'MatMulNBits' else 'Not offloaded to NPU' for i in range(len(chart_data['x_labels']))]
    found_label = []
    for i, label in enumerate(labels):
        if label not in found_label:
            found_label.append(label)
        else:
            labels[i] = ''
    for attribute, measurement in chart_data.items():
        if attribute == 'x_labels':
            continue
        offset = width * multiplier
        above_threshold = np.maximum(np.array(measurement) - threshold, 0)
        below_threshold = np.minimum(np.array(measurement), threshold)
        if large_set:
            rects = ax.bar(x + offset, below_threshold, width, color='green', label=labels, edgecolor='black', hatch=hatches)
            rects = ax.bar(x + offset, above_threshold, width, color='red', label=labels, edgecolor='black', bottom=below_threshold, hatch=hatches)
        else:
                rects = ax.bar(x + offset, below_threshold, width, color='green')
                rects = ax.bar(x + offset, above_threshold, width, color='red', bottom=below_threshold)
        # ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Ratio of NPU:CPU Run Time')
    ax.set_xticks(x + (1/len(chart_data['x_labels']))*width, chart_data['x_labels'])
    # Stagger x axis ticks if there's a lot of columns
    if large_set:
        for tick in ax.xaxis.get_major_ticks()[1::2]:
            tick.set_pad(15)
        ax.legend(loc='best')
    # We change the fontsize of minor ticks label 
    if large_set:
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.tick_params(axis='both', which='minor', labelsize=5)
    plt.title(title)
    ax.axhline(y=1,linewidth=1, color='black', ls='dashed') 
    plt.tight_layout()
    if save_name is not None:
        plt.savefig(save_name + '.pdf', bbox_inches="tight")
    # plt.show()

def generate_pie_chart(data, title, save_name=None, order_values=False):
    fig, ax = plt.subplots(figsize=(12,6))
    model_run_time = data['model_run']
    chart_data = {'labels': data['labels'], 'model_run_ratio_pct': data['exec_time']}
    for i, run_time in enumerate(chart_data['model_run_ratio_pct']):
        chart_data['model_run_ratio_pct'][i] = 100 * (run_time / model_run_time)

    chart_data['labels'].append('UNKNOWN--Start/Stop Overhead?')
    chart_data['model_run_ratio_pct'].append(100-sum(chart_data['model_run_ratio_pct']))

    if order_values:
        new_order = np.array(chart_data['model_run_ratio_pct']).argsort()[::-1]  # Order in descending order
        for key, values in chart_data.items():
            values = [values[i] for i in new_order]
            chart_data[key] = values

    patches, texts = ax.pie(chart_data['model_run_ratio_pct'])
    labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(chart_data['labels'], chart_data['model_run_ratio_pct'])]
    plt.legend(patches, labels, loc='best', bbox_to_anchor=(-0.1, 1.),
           fontsize=8)
    plt.title(title)
    if save_name is not None:
        plt.savefig(save_name + '.pdf')
    # plt.show()


def compare_execution_times(file_prefix, num_runs):
    bar_chart_data = {}
    pie_chart_data = {}
    
    for run in range(1, 2):
        cpu_file_name = file_prefix + str(run) + '_cpu.json'
        npu_file_name = file_prefix + str(run) + '_npu.json'
        with open(cpu_file_name, 'r') as f_cpu, open(npu_file_name, 'r') as f_npu:
            # Use ijson to stream each item in the array of objects
            cpu_parser = ijson.items(f_cpu, 'item')  # Process CPU data items
            npu_parser = ijson.items(f_npu, 'item')  # Process NPU data items
            cpu_obj = next(cpu_parser, None)
            npu_obj = next(npu_parser, None)
            bar_chart_data = {'session_init': {'x_labels': [], 'cpu': [0, 0], 'npu': [0, 0]},
                            'model_run_first': {'x_labels': ['Task ' + str(i) for i in range(1, num_runs + 1)], 'cpu': [0 for i in range(1, num_runs + 1)], 'npu': [0 for i in range(1, num_runs + 1)]},
                            'model_run_subsequent': {'x_labels': ['Task ' + str(i) for i in range(1, num_runs + 1)], 'cpu': [0 for i in range(1, num_runs + 1)], 'npu': [0 for i in range(1, num_runs + 1)]},
                            'mat_mul_op_first': {'x_labels': ['Task ' + str(i) for i in range(1, num_runs + 1)], 'cpu': [0 for i in range(1, num_runs + 1)], 'npu': [0 for i in range(1, num_runs + 1)]},
                            'mat_mul_op_subsequent': {'x_labels': ['Task ' + str(i) for i in range(1, num_runs + 1)], 'cpu': [0 for i in range(1, num_runs + 1)], 'npu': [0 for i in range(1, num_runs + 1)]},
                            'each_op_task_10_first': {'x_labels': [], 'cpu': [0 for i in range(2,len(cpu_obj['Task 1'].keys()))], 'npu': [0 for i in range(2,len(npu_obj['Task 1'].keys()))]},
                            'each_op_task_10_subsequent': {'x_labels': [], 'cpu': [0 for i in range(2,len(cpu_obj['Task 1'].keys()))], 'npu': [0 for i in range(2,len(npu_obj['Task 1'].keys()))]}}
            pie_chart_data = {'cpu_run_ops_to_model_run_subsequent': {'model_run': 0, 'labels': [], 'exec_time': [0 for i in range(2,len(cpu_obj['Task 1'].keys()))]}, 
                            'npu_run_ops_to_model_run_subsequent': {'model_run': 0, 'labels': [], 'exec_time': [0 for i in range(2,len(npu_obj['Task 1'].keys()))]},
                            'cpu_run_ops_to_model_run_first': {'model_run': 0, 'labels': [], 'exec_time': [0 for i in range(2,len(cpu_obj['Task 1'].keys()))]}, 
                            'npu_run_ops_to_model_run_first': {'model_run': 0, 'labels': [], 'exec_time': [0 for i in range(2,len(npu_obj['Task 1'].keys()))]}}

    for run in range(1, num_runs+1):
        cpu_file_name = file_prefix + str(run) + '_cpu.json'
        npu_file_name = file_prefix + str(run) + '_npu.json'
        with open(cpu_file_name, 'r') as f_cpu, open(npu_file_name, 'r') as f_npu:
            # Use ijson to stream each item in the array of objects
            cpu_parser = ijson.items(f_cpu, 'item')  # Process CPU data items
            npu_parser = ijson.items(f_npu, 'item')  # Process NPU data items
            cpu_obj = next(cpu_parser, None)
            npu_obj = next(npu_parser, None)
            # Go through the CPU file and initialize data
            while cpu_obj is not None:
                cpu_runs = [key for key in cpu_obj.keys() if 'Task' in key]
                if len(cpu_runs) > 0: # Measurements for each run
                    for node, measurement in cpu_obj[cpu_runs[0]].items():
                        if node == 'model_run':
                            idx = [i for i, key in enumerate(bar_chart_data['model_run_first']['x_labels']) if key == cpu_runs[0]]
                            bar_chart_data['model_run_first']['cpu'][idx[0]] = bar_chart_data['model_run_first']['cpu'][idx[0]] + time_to_microseconds(measurement['First token time'])
                            idx = [i for i, key in enumerate(bar_chart_data['model_run_subsequent']['x_labels']) if key == cpu_runs[0]]
                            bar_chart_data['model_run_subsequent']['cpu'][idx[0]] = bar_chart_data['model_run_subsequent']['cpu'][idx[0]] + time_to_microseconds(measurement['Subsequent token average'])
                        if node == 'MatMulNBits':
                            idx = [i for i, key in enumerate(bar_chart_data['mat_mul_op_first']['x_labels']) if key == cpu_runs[0]]
                            bar_chart_data['mat_mul_op_first']['cpu'][idx[0]] = bar_chart_data['mat_mul_op_first']['cpu'][idx[0]] + time_to_microseconds(measurement['First token average'])
                            idx = [i for i, key in enumerate(bar_chart_data['mat_mul_op_subsequent']['x_labels']) if key == cpu_runs[0]]
                            bar_chart_data['mat_mul_op_subsequent']['cpu'][idx[0]] = bar_chart_data['mat_mul_op_subsequent']['cpu'][idx[0]] + time_to_microseconds(measurement['Subsequent token average'])
                    if '10' in cpu_runs[0]:
                        for node, measurement in cpu_obj[cpu_runs[0]].items():
                            if node == 'model_run':
                                pie_chart_data['cpu_run_ops_to_model_run_first']['model_run'] = pie_chart_data['cpu_run_ops_to_model_run_first']['model_run'] + time_to_microseconds(measurement['First token time'])
                                for run_time in measurement['Subsequent token times']:
                                    pie_chart_data['cpu_run_ops_to_model_run_subsequent']['model_run'] = pie_chart_data['cpu_run_ops_to_model_run_subsequent']['model_run'] + time_to_microseconds(run_time)
                            elif node == 'SequentialExecutor::Execute':
                                continue
                            else:
                                if node not in bar_chart_data['each_op_task_10_first']['x_labels']:
                                    bar_chart_data['each_op_task_10_first']['x_labels'].append(node)
                                idx = [i for i, key in enumerate(bar_chart_data['each_op_task_10_first']['x_labels']) if key == node]
                                bar_chart_data['each_op_task_10_first']['cpu'][idx[0]] = bar_chart_data['each_op_task_10_first']['cpu'][idx[0]] + time_to_microseconds(measurement['First token average'])
                                if node not in bar_chart_data['each_op_task_10_subsequent']['x_labels']:
                                    bar_chart_data['each_op_task_10_subsequent']['x_labels'].append(node)
                                idx = [i for i, key in enumerate(bar_chart_data['each_op_task_10_subsequent']['x_labels']) if key == node]
                                bar_chart_data['each_op_task_10_subsequent']['cpu'][idx[0]] =  bar_chart_data['each_op_task_10_subsequent']['cpu'][idx[0]] + time_to_microseconds(measurement['Subsequent token average'])

                                if node not in pie_chart_data['cpu_run_ops_to_model_run_first']['labels']:
                                    pie_chart_data['cpu_run_ops_to_model_run_first']['labels'].append(node)
                                idx = [i for i, key in enumerate(pie_chart_data['cpu_run_ops_to_model_run_first']['labels']) if key == node]
                                for run_time in measurement['First token times']:
                                    pie_chart_data['cpu_run_ops_to_model_run_first']['exec_time'][idx[0]] = pie_chart_data['cpu_run_ops_to_model_run_first']['exec_time'][idx[0]] + time_to_microseconds(run_time)
                                if node not in pie_chart_data['cpu_run_ops_to_model_run_subsequent']['labels']:
                                    pie_chart_data['cpu_run_ops_to_model_run_subsequent']['labels'].append(node)
                                idx = [i for i, key in enumerate(pie_chart_data['cpu_run_ops_to_model_run_subsequent']['labels']) if key == node]
                                for run_time in measurement['Subsequent token times']:
                                    pie_chart_data['cpu_run_ops_to_model_run_subsequent']['exec_time'][idx[0]] = pie_chart_data['cpu_run_ops_to_model_run_subsequent']['exec_time'][idx[0]] + time_to_microseconds(run_time)
                else: # Measurements for model loading and session initialization 
                    # For the bar chart, the label locations will be the object's keys
                    for key, vals in cpu_obj.items():
                        if key not in bar_chart_data['session_init']['x_labels']:
                            bar_chart_data['session_init']['x_labels'].append(key)
                        idx = [i for i, node in enumerate(bar_chart_data['session_init']['x_labels']) if key == node]
                        bar_chart_data['session_init']['cpu'][idx[0]] = bar_chart_data['session_init']['cpu'][idx[0]] + vals['Duration']
                cpu_obj = next(cpu_parser, None)

            # Go through the NPU file to get data for comparison. It's assumed that the 
            # keys in this JSON file are the same as the ones in the CPU file
            while npu_obj is not None:
                npu_runs = [key for key in npu_obj.keys() if 'Task' in key]
                if len(npu_runs) > 0: # Measurements for each run
                    for node, measurement in npu_obj[npu_runs[0]].items():
                        if node == 'model_run':
                            idx = [i for i, key in enumerate(bar_chart_data['model_run_first']['x_labels']) if key == npu_runs[0]]
                            bar_chart_data['model_run_first']['npu'][idx[0]] = bar_chart_data['model_run_first']['npu'][idx[0]] + time_to_microseconds(measurement['First token time'])
                            idx = [i for i, key in enumerate(bar_chart_data['model_run_subsequent']['x_labels']) if key == npu_runs[0]]
                            bar_chart_data['model_run_subsequent']['npu'][idx[0]] = bar_chart_data['model_run_subsequent']['npu'][idx[0]] + time_to_microseconds(measurement['Subsequent token average'])
                        if node == 'vitisaiep':
                            idx = [i for i, key in enumerate(bar_chart_data['mat_mul_op_first']['x_labels']) if key == npu_runs[0]]
                            bar_chart_data['mat_mul_op_first']['npu'][idx[0]] = bar_chart_data['mat_mul_op_first']['npu'][idx[0]] + time_to_microseconds(measurement['First token average'])
                            idx = [i for i, key in enumerate(bar_chart_data['mat_mul_op_subsequent']['x_labels']) if key == npu_runs[0]]
                            bar_chart_data['mat_mul_op_subsequent']['npu'][idx[0]] = bar_chart_data['mat_mul_op_subsequent']['npu'][idx[0]] + time_to_microseconds(measurement['Subsequent token average'])
                    if '10' in npu_runs[0]:
                        for node, measurement in npu_obj[npu_runs[0]].items():
                            if node == 'model_run':
                                pie_chart_data['npu_run_ops_to_model_run_first']['model_run'] = pie_chart_data['npu_run_ops_to_model_run_first']['model_run'] + time_to_microseconds(measurement['First token time'])
                                for run_time in measurement['Subsequent token times']:
                                    pie_chart_data['npu_run_ops_to_model_run_subsequent']['model_run'] = pie_chart_data['npu_run_ops_to_model_run_subsequent']['model_run'] + time_to_microseconds(run_time)
                            elif node == 'SequentialExecutor::Execute':
                                continue
                            else:
                                if node == 'vitisaiep':
                                    idx = [i for i, key in enumerate(bar_chart_data['each_op_task_10_first']['x_labels']) if key == 'MatMulNBits']
                                    bar_chart_data['each_op_task_10_first']['npu'][idx[0]] = bar_chart_data['each_op_task_10_first']['npu'][idx[0]] + time_to_microseconds(measurement['First token average'])
                                    idx = [i for i, key in enumerate(bar_chart_data['each_op_task_10_subsequent']['x_labels']) if key == 'MatMulNBits']
                                    bar_chart_data['each_op_task_10_subsequent']['npu'][idx[0]] = bar_chart_data['each_op_task_10_subsequent']['npu'][idx[0]] + time_to_microseconds(measurement['Subsequent token average'])

                                    if node + ' (MatMulNBits)' not in pie_chart_data['npu_run_ops_to_model_run_first']['labels']:
                                        pie_chart_data['npu_run_ops_to_model_run_first']['labels'].append(node + ' (MatMulNBits)')
                                    idx = [i for i, key in enumerate(pie_chart_data['npu_run_ops_to_model_run_first']['labels']) if node in key]
                                    for run_time in measurement['First token times']:
                                        pie_chart_data['npu_run_ops_to_model_run_first']['exec_time'][idx[0]] = pie_chart_data['npu_run_ops_to_model_run_first']['exec_time'][idx[0]] + time_to_microseconds(run_time)

                                    if node + ' (MatMulNBits)' not in pie_chart_data['npu_run_ops_to_model_run_subsequent']['labels']:
                                        pie_chart_data['npu_run_ops_to_model_run_subsequent']['labels'].append(node + ' (MatMulNBits)')
                                    idx = [i for i, key in enumerate(pie_chart_data['npu_run_ops_to_model_run_subsequent']['labels']) if node in key]
                                    for run_time in measurement['Subsequent token times']:
                                        pie_chart_data['npu_run_ops_to_model_run_subsequent']['exec_time'][idx[0]] = pie_chart_data['npu_run_ops_to_model_run_subsequent']['exec_time'][idx[0]] + time_to_microseconds(run_time)
                                else:
                                    idx = [i for i, key in enumerate(bar_chart_data['each_op_task_10_first']['x_labels']) if key == node]
                                    bar_chart_data['each_op_task_10_first']['npu'][idx[0]] = bar_chart_data['each_op_task_10_first']['npu'][idx[0]] + time_to_microseconds(measurement['First token average'])
                                    idx = [i for i, key in enumerate(bar_chart_data['each_op_task_10_subsequent']['x_labels']) if key == node]
                                    bar_chart_data['each_op_task_10_subsequent']['npu'][idx[0]] = bar_chart_data['each_op_task_10_subsequent']['npu'][idx[0]] + time_to_microseconds(measurement['Subsequent token average'])

                                    if node not in pie_chart_data['npu_run_ops_to_model_run_first']['labels']:
                                        pie_chart_data['npu_run_ops_to_model_run_first']['labels'].append(node)
                                    idx = [i for i, key in enumerate(pie_chart_data['npu_run_ops_to_model_run_first']['labels']) if key == node]
                                    for run_time in measurement['First token times']:
                                        pie_chart_data['npu_run_ops_to_model_run_first']['exec_time'][idx[0]] = pie_chart_data['npu_run_ops_to_model_run_first']['exec_time'][idx[0]] + time_to_microseconds(run_time)

                                    if node not in pie_chart_data['npu_run_ops_to_model_run_subsequent']['labels']:
                                        pie_chart_data['npu_run_ops_to_model_run_subsequent']['labels'].append(node)
                                    idx = [i for i, key in enumerate(pie_chart_data['npu_run_ops_to_model_run_subsequent']['labels']) if key == node]
                                    for run_time in measurement['Subsequent token times']:
                                        pie_chart_data['npu_run_ops_to_model_run_subsequent']['exec_time'][idx[0]] = pie_chart_data['npu_run_ops_to_model_run_subsequent']['exec_time'][idx[0]] + time_to_microseconds(run_time)
                else: # Measurements for model loading and session initialization 
                    # For the bar chart, the label locations will be the object's keys
                    for key, vals in npu_obj.items():
                        idx = [i for i, node in enumerate(bar_chart_data['session_init']['x_labels']) if key == node]
                        bar_chart_data['session_init']['npu'][idx[0]] = bar_chart_data['session_init']['npu'][idx[0]] + vals['Duration']
                npu_obj = next(npu_parser, None)

    for key, vals in bar_chart_data.items():
        if key == 'session_init':
            for node, times in vals.items():
                if node != 'x_labels':
                    for i, time in enumerate(times):
                        bar_chart_data[key][node][i] = time // num_runs
        else:
            for node, times in vals.items():
                if node != 'x_labels':
                    for i, time in enumerate(times):
                        bar_chart_data[key][node][i] = time // num_runs
    
    for key, vals in pie_chart_data.items():
        for node, times in vals.items():
            if node == 'model_run':
                pie_chart_data[key][node] = times // num_runs
            elif node != 'labels':
                for i, time in enumerate(times):
                    pie_chart_data[key][node][i] = time // num_runs
    # generate_bar_chart(bar_chart_data['session_init'], "Session Startup Comparison", 'sess_start_comparison')
    # TODO: Confirm that below is the correct description of these comparison
    generate_bar_chart(bar_chart_data['model_run_first'], "Model Run: 1st Prefill+Decoder+Post-Processing Pass", os.path.join(NPU_ANALYSIS_DIR, 'model_run_1st_pass')) 
    generate_bar_chart(bar_chart_data['model_run_subsequent'],"Model Run: Avg of Subsequent Prefill+Decoder+Post-Processing Pass", os.path.join(NPU_ANALYSIS_DIR, 'model_run_subseq_pass'))
    generate_bar_chart(bar_chart_data['mat_mul_op_first'], "Mat Mul: 1st Pass", os.path.join(NPU_ANALYSIS_DIR, 'mat_mul_1st_pass'))
    generate_bar_chart(bar_chart_data['mat_mul_op_subsequent'],"Mat Mul: Avg of Subsequent Pass", os.path.join(NPU_ANALYSIS_DIR, 'mat_mul_subseq_pass'))
    generate_bar_chart(bar_chart_data['each_op_task_10_first'], title="Promp Length=2048, All Ops: 1st Pass", save_name=os.path.join(NPU_ANALYSIS_DIR, 'task_10_all_ops_1st_pass'), large_set=True, order_values=True) 
    generate_bar_chart(bar_chart_data['each_op_task_10_subsequent'], title="Promp Length=2048, All Ops: Subsequent Pass", save_name=os.path.join(NPU_ANALYSIS_DIR, 'task_10_all_ops_subseq_pass'), large_set=True, order_values=True) 
    generate_pie_chart(pie_chart_data['cpu_run_ops_to_model_run_first'], title="Promp Length=2048, 1st Pass CPU: Op Distribution Over Model Run", save_name=os.path.join(NPU_ANALYSIS_DIR, 'rt_distr_over_model_run_cpu_first'), order_values=True)
    generate_pie_chart(pie_chart_data['npu_run_ops_to_model_run_first'], title="Promp Length=2048, 1st Pass NPU: Op Distribution Over Model Run", save_name=os.path.join(NPU_ANALYSIS_DIR, 'rt_distr_over_model_run_npu_first'), order_values=True)
    generate_pie_chart(pie_chart_data['cpu_run_ops_to_model_run_subsequent'], title="Promp Length=2048, Subsequent Pass CPU: Op Distribution Over Model Run", save_name=os.path.join(NPU_ANALYSIS_DIR, 'rt_distr_over_model_run_cpu_subseq'), order_values=True)
    generate_pie_chart(pie_chart_data['npu_run_ops_to_model_run_subsequent'], title="Promp Length=2048, Subsequent Pass NPU: Op Distribution Over Model Run", save_name=os.path.join(NPU_ANALYSIS_DIR, 'rt_distr_over_model_run_npu_subseq'), order_values=True)


if __name__ == "__main__":
    cpu_file = os.path.join(BENCHMARK_RUNS_DIR, "onnxruntime_profile_cpu_run_")
    npu_file = os.path.join(BENCHMARK_RUNS_DIR, "onnxruntime_profile_npu_run_")  

    Path(NPU_ANALYSIS_DIR).mkdir(parents=True, exist_ok=True)
    for run in range(1, NUM_BENCHMARK_RUNS + 1):
        file_extension = ".json"
        cpu_file_name = cpu_file + str(run) + file_extension
        npu_file_name = npu_file + str(run) + file_extension
        out_file_name = os.path.join(NPU_ANALYSIS_DIR, "mismatches", "mismatches_" + str(run) + file_extension)
        compare_json_incremental(cpu_file_name, npu_file_name, out_file_name)
    print("Finished finding mismatches")

    for run in range(1, NUM_BENCHMARK_RUNS + 1):
        file_extension = ".json"
        cpu_file_name = cpu_file + str(run) + file_extension
        npu_file_name = npu_file + str(run) + file_extension
        out_file_name = os.path.join(NPU_ANALYSIS_DIR, "execution_time_distribution", "run" + str(run))
        find_execution_distribution(cpu_file_name, npu_file_name, out_file_name)
    print("Finished extracting distributions")

    out_file_prefix = os.path.join(NPU_ANALYSIS_DIR, "execution_time_distribution", "run")
    compare_execution_times(out_file_prefix, NUM_BENCHMARK_RUNS)
    print("Finished comparing distributions")
