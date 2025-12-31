import os
import argparse
from src.compare import ComparativeAnalyzer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--reference_data', type=str, required=True,
        help="Path to the reference model's recommendation list (e.g., SerenUplift)"
    )
    parser.add_argument(
        '--baseline_data', nargs='+', required=True,
        help="List of paths to baseline models' recommendation lists"
    )
    parser.add_argument(
        '--paired_t', action='store_true',
        help='Perform paired t-test between the reference and target recommendation lists'
    )
    parser.add_argument(
        '--cohen', action='store_true',
        help='Calculate Cohen\'s d between the reference and target recommendation lists'
    )
    parser.add_argument(
        '--wilcoxon', action='store_true',
        help='Perform Wilcoxon signed-rank test (robust to non-normality)'
    )

    return parser.parse_args()


def main():
    args = parse_args()
    analyzer = ComparativeAnalyzer(args.reference_data)

    results = []

    ref_name = os.path.basename(args.reference_data).replace('.json', '')
    ref_result = analyzer.analyze(ref_name, args.reference_data)
    results.append(ref_result)

    for path in args.baseline_data:
        model_name = os.path.basename(path).replace('.json', '')
        result = analyzer.analyze(model_name, path)
        results.append(result)

    header = f"| {'Model':^30} | {'Popularity':^12} | {'Unique':^10} | {'Coverage':^10} | {'Serendipity':^12} |"
    divider = f"|{'-' * 32}|{'-' * 14}|{'-' * 12}|{'-' * 12}|{'-' * 14}|"

    print("\n" + divider)
    print(header)
    print(divider)

    for res in results:
        print(
            f"| {res['Model']:<30} | {res['Popularity']:>12.4f} | "
            f"{res['Unique_Items']:>10,} | {res['Coverage']:>10.2%} | {res['Serendipity']:>12.4f} |"
        )

    print(divider + "\n")

    if args.paired_t:
        print('[Paired T-Test Results]')
        for path in args.baseline_data:
            model_name = os.path.basename(path).replace('.json', '')
            t_res = analyzer.paired_t_test(path)
            print(f"> {model_name}: t-stat={t_res['t_stat']:.4f}, p-value={t_res['p_value']:.4e}")

    if args.cohen:
        print('[Cohen\'s d Results]')
        for path in args.baseline_data:
            model_name = os.path.basename(path).replace('.json', '')
            c_res = analyzer.cohen_d(path)
            print(f"> {model_name}: Cohen's d={c_res['cohen_d']:.4f}")

    if args.wilcoxon:
        print('[Wilcoxon Signed-Rank Test Results]')
        for path in args.baseline_data:
            model_name = os.path.basename(path).replace('.json', '')
            w_res = analyzer.wilcoxon_test(path)
            print(f"> {model_name}: stat={w_res['stat']:.4f}, p-value={w_res['p_value']:.4e}")


if __name__ == '__main__':
    main()
