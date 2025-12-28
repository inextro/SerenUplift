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

    return parser.parse_args()


def main():
    args = parse_args()
    analyzer = ComparativeAnalyzer(args.reference_data)

    results = []

    ref_name = os.path.basename(args.reference_data).replace('.json', '')
    ref_result = analyzer.analyze(args.reference_data, ref_name)
    results.append(ref_result)

    for path in args.baseline_data:
        model_name = os.path.basename(path).replace('.json', '')
        result = analyzer.analyze(path, model_name)
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


if __name__ == '__main__':
    main()
