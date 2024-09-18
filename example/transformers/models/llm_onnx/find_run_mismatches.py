import ijson
import json
import numpy as np

def compare_json_incremental(cpu_file, npu_file, output_file):
    result = []

    with open(cpu_file, 'r') as f_cpu, open(npu_file, 'r') as f_npu:
        # Use ijson to stream each item in the array of objects
        cpu_parser = ijson.items(f_cpu, 'item')  # Process CPU data items
        npu_parser = ijson.items(f_npu, 'item')  # Process NPU data items

        cpu_obj = next(cpu_parser, None)
        npu_obj = next(npu_parser, None)
        
        counter = 0
        counter_cpu_mat_muls = 0
        while cpu_obj is not None and npu_obj is not None:
            cpu_name = cpu_obj['name']
            npu_name = npu_obj['name']

            if "MatMul_Q4_fence_before" in cpu_name:
                counter_cpu_mat_muls = counter_cpu_mat_muls + 1
            if "VitisAIExecutionProvider" in npu_name and "fence_before" in npu_name:
                counter = counter + 1
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
                    npu_name = npu_obj['name']
                    mismatch_data['nodes'].append(npu_name)
                next_npu_name = npu_name

                # Collect mismatched CPU lines until names match again
                npu_obj = next(npu_parser, None)
                if npu_obj is not None:
                    next_npu_name = npu_obj['name']
                while cpu_obj is not None and cpu_name != next_npu_name:
                    mismatch_data['mismatches'].append(cpu_name)
                    cpu_obj = next(cpu_parser, None)
                    if cpu_obj is not None:
                        cpu_name = cpu_obj['name']

                result.append({header: mismatch_data})

                # Move to the next matching NPU line or break if CPU data ends
                if cpu_obj is None:
                    break
    
    print(f'Found {counter} mismatches')
    counter_cpu_mat_muls = counter_cpu_mat_muls
    result.insert(0, {'Mismatches found': counter,
                      'CPU Mat Mul executions': counter_cpu_mat_muls,
                      'Mat Mul executions not offloaded to NPU': counter_cpu_mat_muls - counter})
    # Save mismatched data to an output JSON file
    save_data_to_json(result, output_file)

def find_execution_distribution(cpu_file, npu_file, output_file):
    result_cpu = [{
        "model_loading_uri": {
            'Duration': -1,
            'Time start': -1
        },
        "session_initialization": {
            'Duration': -1,
            'Time start': -1
        },
        "SequentialExecutor::Execute": {
            'First token times': [],
            'Subsequent token times': [],
            'First token average': -1,
            'First token maximum': -1,
            'First token minimum': -1,
            'Subsequent token average': -1,
            'Subsequent token maximum': -1,
            'Subsequent token minimum': -1,
        },
        "model_run": {
            'First token times': [],
            'Subsequent token times': [],
            'First token average': -1,
            'First token maximum': -1,
            'First token minimum': -1,
            'Subsequent token average': -1,
            'Subsequent token maximum': -1,
            'Subsequent token minimum': -1,
        }
    }]
    result_npu = [{
        "model_loading_uri": {
            'Duration': -1,
            'Time start': -1
        },
        "session_initialization": {
            'Duration': -1,
            'Time start': -1
        },
        "SequentialExecutor::Execute": {
            'First token times': [],
            'Subsequent token times': [],
            'First token average': -1,
            'First token maximum': -1,
            'First token minimum': -1,
            'Subsequent token average': -1,
            'Subsequent token maximum': -1,
            'Subsequent token minimum': -1,
        },
        "model_run": {
            'First token times': [],
            'Subsequent token times': [],
            'First token average': -1,
            'First token maximum': -1,
            'First token minimum': -1,
            'Subsequent token average': -1,
            'Subsequent token maximum': -1,
            'Subsequent token minimum': -1,
        }
    }]

    with open(cpu_file, 'r') as f_cpu, open(npu_file, 'r') as f_npu:
        # Use ijson to stream each item in the array of objects
        cpu_parser = ijson.items(f_cpu, 'item')  # Process CPU data items
        npu_parser = ijson.items(f_npu, 'item')  # Process NPU data items

        cpu_obj = next(cpu_parser, None)
        npu_obj = next(npu_parser, None)

        num_forward_passes = 11
        counter = 1
        while cpu_obj is not None:
            cpu_name = cpu_obj['name']
            
            if cpu_name == "model_loading_uri" or cpu_name == "session_initialization":
                result_cpu[0][cpu_name]['Duration'] = cpu_obj['dur']
                result_cpu[0][cpu_name]['Time start'] = cpu_obj['ts']
            elif cpu_name == "SequentialExecutor::Execute":
                if counter == 1:
                    result_cpu[0][cpu_name]['First token times'].append(cpu_obj['dur'])
                else:
                    result_cpu[0][cpu_name]['Subsequent token times'].append(cpu_obj['dur'])
            elif cpu_name == "model_run":
                if counter == 1:
                    result_cpu[0][cpu_name]['First token times'].append(cpu_obj['dur'])
                else:
                    result_cpu[0][cpu_name]['Subsequent token times'].append(cpu_obj['dur'])
                counter = (counter + 1) % num_forward_passes
            else:
                if "kernel_time" in cpu_name:
                    if counter == 1:
                        if cpu_obj['args']['op_name'] not in result_cpu[0]:
                            result_cpu[0][cpu_obj['args']['op_name']] = {'First token times': [cpu_obj['dur']],
                                                                         'Subsequent token times': []}
                        else:
                            result_cpu[0][cpu_obj['args']['op_name']]['First token times'].append(cpu_obj['dur'])
                    else:
                        result_cpu[0][cpu_obj['args']['op_name']]['Subsequent token times'].append(cpu_obj['dur'])
            cpu_obj = next(cpu_parser, None)

        num_forward_passes = 11
        counter = 1
        while npu_obj is not None:
            npu_name = npu_obj['name']
            
            if npu_name == "model_loading_uri" or npu_name == "session_initialization":
                result_npu[0][npu_name]['Duration'] = npu_obj['dur']
                result_npu[0][npu_name]['Time start'] = npu_obj['ts']
            elif npu_name == "SequentialExecutor::Execute":
                if counter == 1:
                    result_npu[0][npu_name]['First token times'].append(npu_obj['dur'])
                else:
                    result_npu[0][npu_name]['Subsequent token times'].append(npu_obj['dur'])
            elif npu_name == "model_run":
                if counter == 1:
                    result_npu[0][npu_name]['First token times'].append(npu_obj['dur'])
                else:
                    result_npu[0][npu_name]['Subsequent token times'].append(npu_obj['dur'])
                counter = (counter + 1) % num_forward_passes
            else:
                if "kernel_time" in npu_name:
                    if "VitisAIExecutionProvider" in npu_name:
                        if counter == 1:
                            if 'vitisaiep' not in result_npu[0]:
                                result_npu[0]['vitisaiep'] = {'First token times': [npu_obj['dur']],
                                                              'Subsequent token times': []}
                            else:
                                result_npu[0]['vitisaiep']['First token times'].append(npu_obj['dur'])
                        else:
                            result_npu[0]['vitisaiep']['Subsequent token times'].append(npu_obj['dur'])
                    else:
                        if counter == 1:
                            if npu_obj['args']['op_name'] not in result_npu[0]:
                                result_npu[0][npu_obj['args']['op_name']] = {'First token times': [npu_obj['dur']],
                                                                             'Subsequent token times': []}
                            else:
                                result_npu[0][npu_obj['args']['op_name']]['First token times'].append(npu_obj['dur'])
                        else:
                            result_npu[0][npu_obj['args']['op_name']]['Subsequent token times'].append(npu_obj['dur'])
            npu_obj = next(npu_parser, None)

    for exec_name, data in result_cpu[0].items():
        if exec_name != 'model_loading_uri' and exec_name != 'session_initialization': 
            data['First token average'] = np.average(data['First token times']).tolist()
            data['First token maximum'] = np.max(data['First token times']).tolist()
            data['First token minimum'] = np.min(data['First token times']).tolist()
            data['Subsequent token average'] = np.average(data['Subsequent token times']).tolist()
            data['Subsequent token maximum'] = np.max(data['Subsequent token times']).tolist()
            data['Subsequent token minimum'] = np.min(data['Subsequent token times']).tolist()
    for exec_name, data in result_npu[0].items():
        if exec_name != 'model_loading_uri' and exec_name != 'session_initialization':
            data['First token average'] = np.average(data['First token times']).tolist()
            data['First token maximum'] = np.max(data['First token times']).tolist()
            data['First token minimum'] = np.min(data['First token times']).tolist()
            data['Subsequent token average'] = np.average(data['Subsequent token times']).tolist()
            data['Subsequent token maximum'] = np.max(data['Subsequent token times']).tolist()
            data['Subsequent token minimum'] = np.min(data['Subsequent token times']).tolist()
                
    # Save mismatched data to an output JSON file
    with open(output_file + '_cpu.json', 'w') as f_out:
        json.dump(result_cpu, f_out, indent=1)
    with open(output_file + '_npu.json', 'w') as f_out:
        json.dump(result_npu, f_out, indent=1)

def save_data_to_json(result, output_file):
    with open(output_file, 'w') as f_out:
        json.dump(result, f_out, indent=4)

if __name__ == "__main__":
    cpu_file = "onnxruntime_profile__cpu_run.json"
    npu_file = "onnxruntime_profile__npu_run.json"
    output_file = "execution_time_distribution"
    
    compare_json_incremental(cpu_file, npu_file, output_file)
    find_execution_distribution(cpu_file, npu_file, output_file)
    print(f"Data files saved")
