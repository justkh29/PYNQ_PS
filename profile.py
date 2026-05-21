import re
from collections import OrderedDict

def parse_inference_log(log_file):
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Patterns
    layer_start = re.compile(r'^={10}\s+(\w+):\s+(\S+)\s+={10}$')
    completed_pattern = re.compile(r'Layer\s+\S+\s+completed\s+in\s+([\d\.]+)\s+sec')
    cpu_done_pattern = re.compile(r'\[CPU\]\s+.+?\s+Done\s+in\s+([\d\.]+)\s+sec')
    chunk_pattern = re.compile(
        r'HW Time:\s*([\d\.]+)\s*sec\s*\|\s*Transfer Time:\s*([\d\.]+)\s*sec\s*\|\s*Unpack Time:\s*([\d\.]+)\s*sec'
    )

    layers = {}
    current_layer = None
    current_device = None
    hw_sum = 0.0
    transfer_sum = 0.0
    unpack_sum = 0.0
    completed_time = None

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        match = layer_start.match(line)
        if match:
            if current_layer is not None and completed_time is not None:
                if current_layer not in layers:
                    layers[current_layer] = {'device': current_device,
                                             'hw': [], 'transfer': [], 'unpack': [], 'completed': []}
                layers[current_layer]['hw'].append(hw_sum)
                layers[current_layer]['transfer'].append(transfer_sum)
                layers[current_layer]['unpack'].append(unpack_sum)
                layers[current_layer]['completed'].append(completed_time)

            name = match.group(2)
            current_layer = name
            current_device = None
            hw_sum = 0.0
            transfer_sum = 0.0
            unpack_sum = 0.0
            completed_time = None

            j = i + 1
            while j < len(lines):
                next_line = lines[j].strip()
                if layer_start.match(next_line):
                    break

                if '[CPU] Running' in next_line:
                    current_device = 'cpu'

                chunk_match = chunk_pattern.search(next_line)
                if chunk_match:
                    hw_sum += float(chunk_match.group(1))
                    transfer_sum += float(chunk_match.group(2))
                    unpack_sum += float(chunk_match.group(3))

                comp_match = completed_pattern.search(next_line)
                if comp_match:
                    completed_time = float(comp_match.group(1))

                cpu_match = cpu_done_pattern.search(next_line)
                if cpu_match and current_device == 'cpu' and completed_time is None:
                    completed_time = float(cpu_match.group(1))

                j += 1
            i = j
            continue
        i += 1

    if current_layer is not None and completed_time is not None:
        if current_layer not in layers:
            layers[current_layer] = {'device': current_device,
                                     'hw': [], 'transfer': [], 'unpack': [], 'completed': []}
        layers[current_layer]['hw'].append(hw_sum)
        layers[current_layer]['transfer'].append(transfer_sum)
        layers[current_layer]['unpack'].append(unpack_sum)
        layers[current_layer]['completed'].append(completed_time)

    for info in layers.values():
        if info['device'] is None:
            info['device'] = 'fpga'

    return layers

def main():
    log_file = 'inference_log.txt'
    layers = parse_inference_log(log_file)

    summary = []
    for name, info in layers.items():
        hw_avg = sum(info['hw']) / len(info['hw']) if info['hw'] else 0.0
        transfer_avg = sum(info['transfer']) / len(info['transfer']) if info['transfer'] else 0.0
        unpack_avg = sum(info['unpack']) / len(info['unpack']) if info['unpack'] else 0.0
        completed_avg = sum(info['completed']) / len(info['completed']) if info['completed'] else 0.0
        summary.append((name, info['device'], hw_avg, transfer_avg, unpack_avg, completed_avg))

    summary.sort(key=lambda x: x[5], reverse=True)

    # Print header
    print("Layer|device|avg_hw|avg_transfer|avg_unpack|avg_completed")
    for name, device, hw, transfer, unpack, completed in summary:
        print(f"{name}|{device}|{hw:.6f}|{transfer:.6f}|{unpack:.6f}|{completed:.6f}")

if __name__ == '__main__':
    main()