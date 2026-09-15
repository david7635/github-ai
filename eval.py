import sys
import yaml
import argparse
from tabulate import tabulate

from retrieval import retrieve_context

def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval precision@k")
    parser.add_argument("repo_id", help="The repo ID (e.g. user_repo) to evaluate against")
    parser.add_argument("eval_file", help="Path to the YAML/JSON eval file")
    parser.add_argument("--k", type=int, default=6, help="Top K to consider for precision")
    args = parser.parse_args()

    with open(args.eval_file, 'r') as f:
        data = yaml.safe_load(f)
        
    results = []
    success_count = 0
    total = len(data)

    print(f"Running evaluation on {args.repo_id} with {total} questions...")

    for item in data:
        q = item.get("question")
        expected = item.get("expected_file_or_symbol")
        
        # We only need sources, so we can ignore the context string
        try:
            _, sources = retrieve_context(q, args.repo_id, top_k_initial=15, top_k_final=args.k)
        except Exception as e:
            print(f"Error retrieving context for query '{q}': {e}")
            sources = []
            
        found = False
        for s in sources:
            if expected in s["file"] or expected in s["symbol"]:
                found = True
                break
                
        results.append([q, expected, "Pass" if found else "Fail"])
        if found:
            success_count += 1
            
    print("\nResults:")
    print(tabulate(results, headers=["Question", "Expected Target", "Result"], tablefmt="grid"))
    print(f"\nPrecision@{args.k}: {success_count}/{total} ({(success_count/total)*100:.2f}%)")

if __name__ == "__main__":
    main()
