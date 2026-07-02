import re
from datetime import datetime
import ast

file_L1 = "/home/cristiano/Desktop/nrL1_stats.log"
file_MAC = "/home/cristiano/Desktop/nrMAC_stats.log"
file_L4 = "/home/cristiano/Desktop/logging/nrL4_stats.log"

numerical_pattern = r'[-+]?\b\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?\b'
patter_MAC = r'-?\b\d+(?:\.\d+)?\b'
rnti_pattern = r'UE (\w+)'
rtt_pattern = r"smoothed_rtt\(us\):(\d+) mean_deviation\(us\):(\d+)"
cw_pattern = r"congestion_window: (\d+)"
bw_pattern = r"New sustained bandwidth estimate \(KBytes/s\): (\d+)"

def get_RNTI(line):
    parts = line.split(':')[0].split(',')
    for part in parts:
        if 'RNTI' in part:
            # Extract the RNTI value
            rnti = part.split()[-1].strip()

    return rnti

def parse_L1(line):
    if "timestamp" in line:
        pattern = r'\b\d+\b'
        return [int(re.search(pattern, line).group())], False
    if "Blacklisted" in line or "max_IO" in line or "PRACH" in line:
        return re.findall(numerical_pattern, line), False
    if "DLSCH" in line:
        rnti = get_RNTI(line)
        return [rnti] + re.findall(numerical_pattern, line.split(":")[1]), False
    if "ULSCH" in line:
        return [value.replace(',', '.')for value in re.findall(numerical_pattern, line.split(':')[1])], False
    if "round_trials" in line:
        return re.findall(numerical_pattern, line), True
    
    return [], False

def parse_MAC(line):
    if "timestamp" in line:
        pattern = r'\b\d+\b'
        return [int(re.search(pattern, line).group())], False
    if "PH" in line:
        rnti = re.search(r'UE RNTI (\w+)', line).group(1)
        return [rnti] + re.findall(patter_MAC, line[12:]), False
    if "TPMI" in line:
        return re.findall(patter_MAC, line[9:]), False
    if "dlsch" in line or "ulsch_rounds" in line:
        return re.findall(patter_MAC, line[9:]), False
    if "ulsch_total_bytes_scheduled" in line:
        return re.findall(patter_MAC, line[9:]), True
    
    return [], False

def parse_quiche(line):
    if line[0] == "[" and len(line) > 20:
        timestamp_str = line.split(":")[0][1:]
        current_year = datetime.now().year
        timestamp_str_with_year = f"{current_year}/{timestamp_str}"
        timestamp = datetime.strptime(timestamp_str_with_year, "%Y/%m%d/%H%M%S.%f")
        unix_timestamp = int(timestamp.timestamp()) - 7 * 3600
        if "smoothed_rtt" in line:
            return [unix_timestamp] + list(re.findall(rtt_pattern, line)[0]), "rtt"
        if "Final target congestion_window" in line:
            return [unix_timestamp] + [re.search(cw_pattern, line).group(1)], "cw"
        if "New sustained bandwidth estimate" in line:
            return [unix_timestamp] + [re.search(bw_pattern, line).group(1)], "bw"

    return [], "skip"

def parse_aioquic(line):
    timestamp_str, dict_str = line.split(',', 1)
    unix_timestamp = int(timestamp_str)
    dict_str = dict_str.strip()
    data = ast.literal_eval(dict_str)

    return [unix_timestamp, data.get("cwnd"), data.get("smoothed_rtt"), data.get("rtt_variance")], "aioquic"

def parse_bw(line):
    timestamp, bw = line.split("/")
    return [int(timestamp), float(bw)], "bw"
        
if __name__ == "__main__":
    file_L1 = open(file_L1, "r")
    file_MAC = open(file_MAC, "r")
    file_L4 = open(file_L4, "r")

    #for line in file_L1:
    #    res, flag = parse_L1(line)
    #    print(res, flag)

    #for line in file_MAC:
    #    res, flag = parse_MAC(line)
    #    print(res, flag)

    for line in file_L4:
        res, flag = parse_quiche(line)
        print(res, flag)

