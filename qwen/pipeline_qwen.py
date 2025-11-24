# pipeline.py
# 控制整体流程，跑 2Wiki / HotpotQA / MuSiQue，并统计六个指标

import os
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set
from tqdm import tqdm

from gamma_qwen import LLMClient
from theta_qwen import ThetaAgent

from dotenv import load_dotenv
load_dotenv()

DATASET_FILES = {
    "2wiki": "2wiki_500.json",
    "hotpotqa": "hotpotqa_500.json",
    "musique": "musique_500.json",
}


def ensure_qwen_default() -> None:
    """
    如果用户未在 .env 或环境变量中指定模型，默认用 Qwen-2.5。
    """

    os.environ["MODEL_NAME"] = "qwen2.5:7b"
        # 若 API_URL 未指定，则沿用 gamma.py 中的默认值


def load_dataset(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Dataset JSON must be a list of examples")
    return data


def load_processed_indices(log_path: str) -> Tuple[List[Dict[str, Any]], Set[int], Dict[str, float]]:
    """
    从日志文件中加载已处理的样本索引、已有结果列表和累积指标
    
    返回:
        existing_results: 已有的结果列表（JSON 数组或从旧的 JSONL 合并）
        processed_indices: 已处理的样本索引集合
        accumulated_metrics: 累积的指标值
    """
    existing_results: List[Dict[str, Any]] = []
    processed_indices: Set[int] = set()
    accumulated_metrics = {
        "sum_answer_em": 0.0,
        "sum_answer_f1": 0.0,
        "sum_support_em": 0.0,
        "sum_support_f1": 0.0,
        "sum_support_prec": 0.0,
        "sum_support_rec": 0.0,
    }
    
    if not os.path.exists(log_path):
        return existing_results, processed_indices, accumulated_metrics
    
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if not content:
            return existing_results, processed_indices, accumulated_metrics

        # 支持 JSON 数组（推荐）与 JSONL（兼容旧格式）两种形式
        if content.lstrip().startswith("["):
            try:
                loaded = json.loads(content)
                if not isinstance(loaded, list):
                    raise ValueError("Log JSON must be a list")
                existing_results = loaded
            except Exception as e:
                raise ValueError(f"Invalid JSON log format: {e}")
        else:
            for line in content.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    existing_results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        for result in existing_results:
            if not isinstance(result, dict):
                continue
            idx = result.get("example_index")
            if idx is None:
                continue
            processed_indices.add(int(idx))
            # 累积指标
            accumulated_metrics["sum_answer_em"] += float(result.get("answer_em", 0.0))
            accumulated_metrics["sum_answer_f1"] += float(result.get("answer_f1", 0.0))
            accumulated_metrics["sum_support_em"] += float(result.get("support_em", 0.0))
            accumulated_metrics["sum_support_f1"] += float(result.get("support_f1", 0.0))
            accumulated_metrics["sum_support_prec"] += float(result.get("support_precision", 0.0))
            accumulated_metrics["sum_support_rec"] += float(result.get("support_recall", 0.0))
    except Exception as e:
        print(f"Warning: Failed to load processed indices from {log_path}: {e}")
    
    return existing_results, processed_indices, accumulated_metrics


def run_dataset(
    dataset_name: str,
    data: List[Dict[str, Any]],
    log_dir: str,
    limit: int = -1,
) -> None:
    ensure_qwen_default()
    os.makedirs(log_dir, exist_ok=True)
    out_path = os.path.join(log_dir, f"theta_gamma_{dataset_name}.jsonl")

    if limit is not None and limit > 0:
        eval_data = data[:limit]
    else:
        eval_data = data

    total = len(eval_data)
    if total == 0:
        print(f"[{dataset_name}] No examples to run.")
        return

    # 加载已处理的样本、已有结果和累积指标
    _, processed_indices, accumulated_metrics = load_processed_indices(out_path)
    processed_count = len(processed_indices)
    
    if processed_count > 0:
        print(f"[{dataset_name}] 发现已处理 {processed_count}/{total} 个样本，将从断点继续...")
    
    # 六个指标的累积和（从已处理的样本中恢复）
    sum_answer_em = accumulated_metrics["sum_answer_em"]
    sum_answer_f1 = accumulated_metrics["sum_answer_f1"]
    sum_support_em = accumulated_metrics["sum_support_em"]
    sum_support_f1 = accumulated_metrics["sum_support_f1"]
    sum_support_prec = accumulated_metrics["sum_support_prec"]
    sum_support_rec = accumulated_metrics["sum_support_rec"]

    # 过滤出需要处理的样本
    remaining_data = [(idx, ex) for idx, ex in enumerate(eval_data) if idx not in processed_indices]
    remaining_count = len(remaining_data)
    
    if remaining_count == 0:
        print(f"[{dataset_name}] 所有样本已处理完成，无需继续运行。")
        return

    # 使用 tqdm 创建进度条，只遍历需要处理的样本
    pbar = tqdm(
        remaining_data,
        total=total,
        initial=processed_count,
        desc=f"[{dataset_name}]",
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )
    
    current_processed = processed_count  # 当前已处理的样本总数

    # 追加写入 JSONL，避免整文件重写
    with open(out_path, "a", encoding="utf-8") as fout:
        for idx, ex in pbar:
            call_log: List[Dict[str, Any]] = []
            llm_client = LLMClient(call_log=call_log)
            theta = ThetaAgent(dataset_name=dataset_name, llm_client=llm_client)

            result = theta.solve_one(ex, example_index=idx)

            sum_answer_em += float(result.get("answer_em", 0.0))
            sum_answer_f1 += float(result.get("answer_f1", 0.0))
            sum_support_em += float(result.get("support_em", 0.0))
            sum_support_f1 += float(result.get("support_f1", 0.0))
            sum_support_prec += float(result.get("support_precision", 0.0))
            sum_support_rec += float(result.get("support_recall", 0.0))

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()  # 立即刷新，确保断点安全

            # 更新已处理的样本数
            current_processed += 1
            
            # 每个样本都更新进度条，显示当前平均指标
            if current_processed > 0:
                pbar.set_postfix({
                    "ans_EM": f"{sum_answer_em/current_processed:.3f}",
                    "ans_F1": f"{sum_answer_f1/current_processed:.3f}",
                    "sup_EM": f"{sum_support_em/current_processed:.3f}",
                    "sup_F1": f"{sum_support_f1/current_processed:.3f}"
                })

    n = float(total)
    print(f"[{dataset_name}] DONE on {total} examples.")
    print(
        f"  answer_em = {sum_answer_em/n:.4f}, "
        f"answer_f1 = {sum_answer_f1/n:.4f}"
    )
    print(
        f"  support_em = {sum_support_em/n:.4f}, "
        f"support_f1 = {sum_support_f1/n:.4f}, "
        f"support_precision = {sum_support_prec/n:.4f}, "
        f"support_recall = {sum_support_rec/n:.4f}"
    )
    print(f"  Results saved to: {out_path}")


def main():
    default_data_dir = Path(__file__).resolve().parent.parent  # 仓库根目录
    parser = argparse.ArgumentParser(
        description="Theta-Gamma dual-agent pipeline for 2Wiki / HotpotQA / MuSiQue"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(default_data_dir),
        help="Directory containing the dataset json files",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="2wiki,hotpotqa,musique",
        help="Comma-separated list of datasets: 2wiki,hotpotqa,musique",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Maximum number of examples per dataset (-1 = all)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory to store JSONL logs",
    )

    args = parser.parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    for dataset_name in datasets:
        if dataset_name not in DATASET_FILES:
            print(f"Unknown dataset: {dataset_name}, skip.")
            continue
        data_dir = Path(args.data_dir).expanduser()
        data_path = data_dir / DATASET_FILES[dataset_name]
        if not os.path.exists(data_path):
            print(f"File not found for {dataset_name}: {data_path}, skip.")
            continue

        print(f"Loading {dataset_name} from {data_path} ...")
        data = load_dataset(data_path)

        limit = args.limit if args.limit and args.limit > 0 else -1
        run_dataset(dataset_name, data, log_dir=args.log_dir, limit=limit)


if __name__ == "__main__":
    main()
