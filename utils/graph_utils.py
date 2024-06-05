import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pandas as pd

import torch
from torch_geometric.data import Data
from collections import Counter


def getTraceGraphPyG(trace, faults, processes, svc_name_mapping, metrics, span_level=True):
    graph_label = 0
    nodes = {}  # 保存了节点ID(spanID)和节点内容(node_number,span)的字典
    from_node = []
    to_node = []
    durations = []
    root_cause_label = [0. for _ in range(len(svc_name_mapping))]
    cpu_usage_system = []
    cpu_usage_total = []
    cpu_usage_user = []
    memory_usage = []
    memory_working_set = []
    rx_bytes = []
    tx_bytes = []

    node_features = []

    for span in trace['spans']:
        serviceName = trace['processes'][span['processID']]['serviceName']
        metric = metrics[f'{serviceName}.csv']  # dataframe
        t = int(span['startTime'] / 1e6 - 28800)
        try:
            row = metric.loc[metric['timestamp'] == t].iloc[0]
        except IndexError as e:
            # print(f'{t} is not found,add zero value')
            rx_bytes.append(float(0))
            tx_bytes.append(float(0))
            cpu_usage_system.append(float(0))
            cpu_usage_total.append(float(0))
            cpu_usage_user.append(float(0))
            memory_usage.append(float(0))
            memory_working_set.append(float(0))
        else:
            rx_bytes.append(float(row['rx_bytes']))
            tx_bytes.append(float(row['tx_bytes']))
            cpu_usage_system.append(float(row['cpu_usage_system']))
            cpu_usage_total.append(float(row['cpu_usage_total']))
            cpu_usage_user.append(float(row['cpu_usage_user']))
            memory_usage.append(float(row['memory_usage']))
            memory_working_set.append(float(row['memory_working_set']))
        if span['spanID'] not in nodes:
            if span_level:
                nodes[span['spanID']] = [len(nodes), span]
            else:
                # TODO:消融实验-service level的建模
                pass
        if span['duration']:
            durations.append(float(span['duration']))
        else:
            durations.append(max(durations))

    # durations = normalize(durations)
    # rx_bytes = normalize(rx_bytes)
    # tx_bytes = normalize(tx_bytes)

    # 连接边
    for _, node in nodes.items():
        _, span = node
        if len(span['references']) != 0:
            if span['references'][0]:
                if span['references'][0]['spanID'] in nodes.keys():
                    if span['references'][0]['refType'] == 'CHILD_OF':
                        from_node.append(
                            nodes[span['references'][0]['spanID']][0])
                        to_node.append(nodes[span['spanID']][0])

    # 打标签
    for _, node in nodes.items():
        node_number, span = node
        for fault in faults['faults']:
            # print(f"{fault['start']} < {span['startTime']/1e6} < {fault['start'] + fault['duration']} and {trace['processes'][span['processID']]['serviceName']} == {fault['name']}")
            if (fault['start'] < (span['startTime']) / 1e6 - 28800 < fault['start'] + fault['duration']) and (
                    trace['processes'][span['processID']]['serviceName'] == fault['name']):
                graph_label = 1
                root_cause_label[svc_name_mapping[fault['name']]] = 1
    from_node += [i for i in range(len(durations))]
    to_node += [i for i in range(len(durations))]
    edge_index = torch.tensor([from_node,
                               to_node], dtype=torch.long)
    # out_degrees = G.out_degrees().numpy().tolist()
    out_degrees = [0 for _ in range(len(durations))]
    c = Counter(out_degrees)
    for k in c:
        out_degrees[k] = c[k]
    out_degrees = [i / max(out_degrees) for i in out_degrees]
    for i in range(len(durations)):
        node_features.append(
            [out_degrees[i], durations[i], rx_bytes[i], tx_bytes[i], cpu_usage_total[i], memory_usage[i]])
    x = torch.tensor(node_features)
    G = Data(x=x, edge_index=edge_index, label=torch.tensor(graph_label), r_label=torch.tensor(root_cause_label))
    return G


def prepare_metrics(metrics_path):
    metrics = {}
    svc_name_mapping = {}
    csv_file_names = [f for f in os.listdir(metrics_path) if f.endswith('.csv')]

    for i in csv_file_names:
        svc_name_mapping[i.split('.')[0]] = len(svc_name_mapping)

    # 遍历csv文件名列表，读取每个csv文件的内容，并将其存储到字典metrics中
    for csv_file_name in csv_file_names:
        csv_file_path = os.path.join(metrics_path, csv_file_name)
        df = pd.read_csv(csv_file_path)
        metrics[csv_file_name] = df

    return metrics, svc_name_mapping


def prepare_spans_and_faults(s, f):
    with open(s, 'r') as _f:
        # 读取 JSON 数据
        dataset = json.load(_f)

    with open(f, 'r') as _f:
        # 读取 JSON 数据
        fault = json.load(_f)

    return dataset, fault


def getTraceGraphsPyG(dataset, faults, svc_name_mapping, metrics):
    PyGGraphs = []
    for trace in tqdm(dataset, desc="trace processing"):
        processes = trace['processes']
        G = getTraceGraphPyG(trace, faults, processes, svc_name_mapping, metrics)
        PyGGraphs.append(G)

    total = 0
    normal = 0
    abnormal_graph = 0
    for i, g in tqdm(enumerate(PyGGraphs), desc="labeling graphs"):
        total += 1
        if g['label'] != torch.tensor([0]):
            # print(root_cause_labels[i])
            abnormal_graph += 1
    print(f"traceGraph  description: total:{total},normal:{total - abnormal_graph},abnormal{abnormal_graph}")

    return PyGGraphs


def prepare(root_path):
    dataset_path = os.path.join(root_path, 'spans.json')
    faults_path = os.path.join(root_path, 'fault.json')
    metrics_path = os.path.join(root_path, 'metrics')
    metrics, svc_name_mapping = prepare_metrics(str(metrics_path))
    s, f = prepare_spans_and_faults(str(dataset_path), str(faults_path))

    return getTraceGraphsPyG(s, f, svc_name_mapping, metrics)
