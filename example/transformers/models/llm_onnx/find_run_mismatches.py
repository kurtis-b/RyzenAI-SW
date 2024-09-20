import ijson
import json
import numpy as np
import matplotlib.pyplot as plt


NUM_FORWARD_PASSES = 11
NUM_PROMPTS = 10

def compare_json_incremental(cpu_file, npu_file):
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
    save_data_to_json(result, 'mismatches.json', indent=4)


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
        parsers['cpu'] = ijson.items(f_cpu, 'item')  # Process CPU data items
        parsers['npu'] = ijson.items(f_npu, 'item')  # Process NPU data items

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
                                    result[run][run_key]['vitisaiep'] = {f'{EXEC_TIME_PREFIX_1} token time': parser_obj[DURATION_PARSER_KEY],
                                                                f'{EXEC_TIME_PREFIX_2} token times': []}
                                else:
                                    result[run][run_key]['vitisaiep'][f'{EXEC_TIME_PREFIX_1} token time'] = parser_obj[DURATION_PARSER_KEY]
                            else:
                                result[run][run_key]['vitisaiep'][f'{EXEC_TIME_PREFIX_2} token times'].append(parser_obj[DURATION_PARSER_KEY])
                        else:
                            if counter == 1:
                                if parser_obj[ARGS_KEY][OP_NAME_KEY] not in result[run][run_key]:
                                    result[run][run_key][parser_obj[ARGS_KEY][OP_NAME_KEY]] = {f'{EXEC_TIME_PREFIX_1} token time': parser_obj[DURATION_PARSER_KEY],
                                                                                f'{EXEC_TIME_PREFIX_2} token times': []}
                                else:
                                    result[run][run_key][parser_obj[ARGS_KEY][OP_NAME_KEY]][f'{EXEC_TIME_PREFIX_1} token time'] = parser_obj[DURATION_PARSER_KEY]
                            else:
                                result[run][run_key][parser_obj[ARGS_KEY][OP_NAME_KEY]][f'{EXEC_TIME_PREFIX_2} token times'].append(parser_obj[DURATION_PARSER_KEY])
                parser_obj = next(parser, None)

            for run in range(NUM_PROMPTS):
                run_key = f"Task {run+1}"
                for exec_name, data in result[run][run_key].items():
                    if exec_name != 'model_loading_uri' and exec_name != 'session_initialization': 
                        key_prefixes = [EXEC_TIME_PREFIX_2]
                        for prefix in key_prefixes:
                            microseconds_data = []  # Need to convert the time values to microseconds since they're in a particular format
                            for times in data[f'{prefix} token times']:
                                microseconds_data.append(time_to_microseconds(times))
                            data[f'{prefix} token average'] = microseconds_to_time_format(round(np.average(microseconds_data)))
                            data[f'{prefix} token maximum'] = microseconds_to_time_format(round(np.max(microseconds_data)))
                            data[f'{prefix} token minimum'] = microseconds_to_time_format(round(np.min(microseconds_data)))
                    
            # Save data to an output JSON file
            save_data_to_json(result, f'{output_file_name}_{file_name}.json', indent=1)


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
    time1_microseconds = time_to_microseconds(time1)
    time2_microseconds = time_to_microseconds(time2)
    
    return time1_microseconds / time2_microseconds

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
    # plt.show()
    if save_name is not None:
        plt.savefig(save_name + '.pdf', bbox_inches="tight")

def compare_execution_times(cpu_file, npu_file):
    with open(cpu_file, 'r') as f_cpu, open(npu_file, 'r') as f_npu:
        # Use ijson to stream each item in the array of objects
        cpu_parser = ijson.items(f_cpu, 'item')  # Process CPU data items
        npu_parser = ijson.items(f_npu, 'item')  # Process NPU data items
        cpu_obj = next(cpu_parser, None)
        npu_obj = next(npu_parser, None)

        bar_chart_data = {'session_init': {'x_labels': [], 'cpu': [], 'npu': []},
                          'model_run_first': {'x_labels': [], 'cpu': [], 'npu': []},
                          'model_run_subsequent': {'x_labels': [], 'cpu': [], 'npu': []},
                          'mat_mul_op_first': {'x_labels': [], 'cpu': [], 'npu': []},
                          'mat_mul_op_subsequent': {'x_labels': [], 'cpu': [], 'npu': []},
                          'each_op_task_10_first': {'x_labels': [], 'cpu': [-1 for i in range(2,len(cpu_obj['Task 1'].keys()))], 'npu': [-1 for i in range(2,len(npu_obj['Task 1'].keys()))]},
                          'each_op_task_10_subsequent': {'x_labels': [], 'cpu': [-1 for i in range(2,len(cpu_obj['Task 1'].keys()))], 'npu': [-1 for i in range(2,len(npu_obj['Task 1'].keys()))]}}
        # Go through the CPU file and initialize data
        while cpu_obj is not None:
            cpu_runs = [key for key in cpu_obj.keys() if 'Task' in key]
            if len(cpu_runs) > 0: # Measurements for each run
                for node, measurement in cpu_obj[cpu_runs[0]].items():
                    if node == 'model_run':
                        bar_chart_data['model_run_first']['x_labels'].append(cpu_runs[0])
                        bar_chart_data['model_run_first']['cpu'].append(measurement['First token time'])
                        bar_chart_data['model_run_subsequent']['x_labels'].append(cpu_runs[0])
                        bar_chart_data['model_run_subsequent']['cpu'].append(measurement['Subsequent token average'])
                    if node == 'MatMulNBits':
                        bar_chart_data['mat_mul_op_first']['x_labels'].append(cpu_runs[0])
                        bar_chart_data['mat_mul_op_first']['cpu'].append(measurement['First token time'])
                        bar_chart_data['mat_mul_op_subsequent']['x_labels'].append(cpu_runs[0])
                        bar_chart_data['mat_mul_op_subsequent']['cpu'].append(measurement['Subsequent token average'])
                if '10' in cpu_runs[0]:
                    for node, measurement in cpu_obj[cpu_runs[0]].items():
                        if node == 'model_run' or node == 'SequentialExecutor::Execute':
                            continue
                        bar_chart_data['each_op_task_10_first']['x_labels'].append(node)
                        idx = [i for i, key in enumerate(bar_chart_data['each_op_task_10_first']['x_labels']) if key == node]
                        bar_chart_data['each_op_task_10_first']['cpu'][idx[0]] = measurement['First token time']
                        bar_chart_data['each_op_task_10_subsequent']['x_labels'].append(node)
                        idx = [i for i, key in enumerate(bar_chart_data['each_op_task_10_subsequent']['x_labels']) if key == node]
                        bar_chart_data['each_op_task_10_subsequent']['cpu'][idx[0]] = measurement['Subsequent token average']
            else: # Measurements for model loading and session initialization 
                # For the bar chart, the label locations will be the object's keys
                for key, vals in cpu_obj.items():
                    bar_chart_data['session_init']['x_labels'].append(key)
                    bar_chart_data['session_init']['cpu'].append(vals['Duration'])
            cpu_obj = next(cpu_parser, None)

        # Go through the NPU file to get data for comparison. It's assumed that the 
        # keys in this JSON file are the same as the ones in the CPU file
        while npu_obj is not None:
            npu_runs = [key for key in npu_obj.keys() if 'Task' in key]
            if len(npu_runs) > 0: # Measurements for each run
                for node, measurement in npu_obj[npu_runs[0]].items():
                    if node == 'model_run':
                        bar_chart_data['model_run_first']['npu'].append(measurement['First token time'])
                        bar_chart_data['model_run_subsequent']['npu'].append(measurement['Subsequent token average'])
                    if node == 'vitisaiep':
                        bar_chart_data['mat_mul_op_first']['npu'].append(measurement['First token time'])
                        bar_chart_data['mat_mul_op_subsequent']['npu'].append(measurement['Subsequent token average'])
                if '10' in npu_runs[0]:
                    for node, measurement in npu_obj[npu_runs[0]].items():
                        if node == 'model_run' or node == 'SequentialExecutor::Execute':
                            continue
                        if node == 'vitisaiep':
                            idx = [i for i, key in enumerate(bar_chart_data['each_op_task_10_first']['x_labels']) if key == 'MatMulNBits']
                            bar_chart_data['each_op_task_10_first']['npu'][idx[0]] = measurement['First token time']
                            idx = [i for i, key in enumerate(bar_chart_data['each_op_task_10_subsequent']['x_labels']) if key == 'MatMulNBits']
                            bar_chart_data['each_op_task_10_subsequent']['npu'][idx[0]] = measurement['Subsequent token average']
                        else:
                            idx = [i for i, key in enumerate(bar_chart_data['each_op_task_10_first']['x_labels']) if key == node]
                            bar_chart_data['each_op_task_10_first']['npu'][idx[0]] = measurement['First token time']
                            idx = [i for i, key in enumerate(bar_chart_data['each_op_task_10_subsequent']['x_labels']) if key == node]
                            bar_chart_data['each_op_task_10_subsequent']['npu'][idx[0]] = measurement['Subsequent token average']
            else: # Measurements for model loading and session initialization 
                # For the bar chart, the label locations will be the object's keys
                for key, vals in npu_obj.items():
                    bar_chart_data['session_init']['npu'].append(vals['Duration'])
            npu_obj = next(npu_parser, None)

        generate_bar_chart(bar_chart_data['session_init'], "Session Startup Comparison", 'sess_start_comparison')
        # TODO: Confirm that below is the correct description of these comparison
        generate_bar_chart(bar_chart_data['model_run_first'], "Model Run: 1st Prefill+Decoder+Post-Processing Pass", 'model_run_1st_pass') 
        generate_bar_chart(bar_chart_data['model_run_subsequent'], "Model Run: Avg of Subsequent Prefill+Decoder+Post-Processing Pass", 'model_run_subseq_pass') 
        generate_bar_chart(bar_chart_data['mat_mul_op_first'], "Mat Mul: 1st Pass", 'mat_mul_1st_pass') 
        generate_bar_chart(bar_chart_data['mat_mul_op_subsequent'], "Mat Mul: Avg of Subsequent Pass", 'mat_mul_subseq_pass') 

        # TODO: Compare between runs the other operators that aren't offloaded to the NPU
        # Split dict in half--no need to use this anymore since will just use all items in the same graph. but keeping here in case I need to use the method later
        # all_ops_first_half = {key:bar_chart_data['each_op_task_10_first'][key][len(measurement)//2:] for key, measurement in bar_chart_data['each_op_task_10_first'].items()}
        # al_ops_second_half = {key:bar_chart_data['each_op_task_10_first'][key][:len(measurement)//2] for key, measurement in bar_chart_data['each_op_task_10_first'].items()}

        # For the next graph, swap the first two items to make the graph look nicer. The second item has a long name, and doing the swap prevents the staggered
        # x tick labels from overlapping.
        # i, j = 0, 1
        # for key, measurements in bar_chart_data['each_op_task_10_first'].items():
        #     measurements[i], measurements[j] = measurements[j], measurements[i]
        #     bar_chart_data['each_op_task_10_first'][key] = measurements
        generate_bar_chart(bar_chart_data['each_op_task_10_first'], title="Task 10 All Ops: 1st Pass", save_name='task_10_all_ops_1st_pass', large_set=True, order_values=True) 
        generate_bar_chart(bar_chart_data['each_op_task_10_subsequent'], title="Task 10 All Ops: Subsequent Pass", save_name='task_10_all_ops_subseq_pass', large_set=True, order_values=True) 



if __name__ == "__main__":
    cpu_file = "onnxruntime_profile__cpu_run.json"
    npu_file = "onnxruntime_profile__npu_run.json"    
    output_file_name = 'execution_time_distribution'

    # compare_json_incremental(cpu_file, npu_file)
    # find_execution_distribution(cpu_file, npu_file, output_file_name)

    cpu_file = f"{output_file_name}_cpu.json"
    npu_file = f"{output_file_name}_npu.json"
    compare_execution_times(cpu_file, npu_file)
