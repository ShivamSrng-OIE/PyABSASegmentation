import os
import json
import argparse
from datetime import datetime
from warnings import filterwarnings

# Quiet noisy logs if TF is installed
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
filterwarnings("ignore")

from pyabsa import ATEPCCheckpointManager


DEFAULT_SAMPLES = [
    "The camera is excellent, but the battery life drains too quickly.",
    "The screen is too dim, the speakers are weak, but the keyboard feels amazing.",
    "The build quality is solid; however, the software is buggy and the updates are slow.",
    "The display is sharp and colorful, yet the device overheats after prolonged use.",
    "The charger is included, the battery lasts all day, but the phone overheats sometimes.",
    "Although the camera is decent, the low-light performance is disappointing.",
    "The design is sleek. The weight is manageable, but the price is far too high!",
    "Battery life is okay, the camera is outstanding, however the charging speed is terrible.",
]


def read_lines(path: str):
    """Read non-empty lines from a text file."""
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]
    return [ln for ln in lines if ln]


def main():
    parser = argparse.ArgumentParser(
        description="Run PyABSA ATEPC with aspect-centric positive/negative snippets and save JSON results."
    )
    parser.add_argument(
        "--checkpoint",
        default="multilingual",
        help="ATEPC checkpoint to load (default: multilingual)",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Optional path to a .txt file (one sentence per line). If omitted, built-in complex samples are used.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON file path. Default: ./aspect_snippets.<timestamp>.json",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=4,
        help="Token window (each side) used as fallback when no boundary is found. Default: 4",
    )
    parser.add_argument(
        "--no-connector",
        action="store_true",
        help="If set, do NOT include the boundary connector word (e.g., 'but') in snippets.",
    )
    args = parser.parse_args()

    # Prepare sentences
    if args.input:
        sentences = read_lines(args.input)
        if not sentences:
            raise SystemExit(f"No sentences found in {args.input}")
    else:
        sentences = DEFAULT_SAMPLES

    # Load your patched extractor
    print("\nLoading ATEPC extractor...")
    extractor = ATEPCCheckpointManager.get_aspect_extractor(
        checkpoint=args.checkpoint,
        auto_device=True,
    )

    # Configure your snippet behavior (read by your patched code)
    extractor.config.context_radius = args.window
    extractor.config.include_connector_in_snippet = (not args.no_connector)

    results = []
    print(f"\nRunning inference on {len(sentences)} sentence(s)...")
    for i, s in enumerate(sentences):
        try:
            out = extractor.predict(
                s, pred_sentiment=True, print_result=False, save_result=False
            )
            results.append(
                {
                    "id": i,
                    "sentence": out.get("sentence", s),
                    "aspects": out.get("aspect", []),
                    "sentiments": out.get("sentiment", []),
                    "positions": out.get("position", []),
                    "positive": out.get("positive", []),
                    "negative": out.get("negative", []),
                    "confidence": out.get("confidence", []),
                    "probs": out.get("probs", []),
                }
            )
        except Exception as e:
            results.append({"id": i, "sentence": s, "error": str(e)})
            print(f"[Warn] Failed on sample #{i}: {e}")
    
    out_path = args.output or os.path.abspath(f"./aspect_snippets.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Saved results to:\n{out_path}\n")


if __name__ == "__main__":
    main()