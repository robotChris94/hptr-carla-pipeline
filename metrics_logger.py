# metrics_logger.py
# 將每回合 log 統整成 CSV 表格與平均結果

'''
功能：
將你實驗 log（如 logs/hptr_town05.json）解析成 summary.csv
自動計算平均 minADE、Success Rate、Collision Rate
CLI 使用方式：python3 metrics_logger.py --input logs/hptr_town05.json --output summary.csv

'''

import json
import csv
import argparse
from statistics import mean

def parse_log(input_path, output_csv):
    with open(input_path, 'r') as f:
        lines = f.readlines()
        data = [json.loads(line.strip()) for line in lines if line.strip()]

    if not data:
        print("No data found.")
        return

    keys = ["timestamp", "route", "minADE", "completion", "collision", "seed"]

    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for d in data:
            row = {k: d.get(k, None) for k in keys}
            writer.writerow(row)

    # 顯示簡單統計
    minades = [d["minADE"] for d in data if d.get("minADE") is not None]
    completions = [d["completion"] for d in data if d.get("completion") is not None]
    collisions = [d["collision"] for d in data if d.get("collision") is not None]

    print("\n--- Summary ---")
    print(f"Total rounds: {len(data)}")
    print(f"Avg ADE: {mean(minades):.3f}")
    print(f"Success Rate: {sum(completions)/len(completions):.2%}")
    print(f"Collision Rate: {sum(collisions)/len(collisions):.2%}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='experiment log json file')
    parser.add_argument('--output', type=str, default='summary.csv', help='summary output csv')
    args = parser.parse_args()

    parse_log(args.input, args.output)
