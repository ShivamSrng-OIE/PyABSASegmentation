import os
import json
import argparse
from datetime import datetime
from warnings import filterwarnings

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
filterwarnings("ignore")

from pyabsa import ATEPCCheckpointManager


DEFAULT_SAMPLES = [
    # A) Coordination & mixed polarity
    "The camera is excellent, but the battery life drains too quickly.",
    "The speakers are loud and clear, and the mic is fine, but the fans are obnoxiously noisy.",
    "The keyboard feels great and the trackpad is responsive, yet the palm rest gets sticky.",
    "The app loads fast but the login screen freezes and the captcha never shows.",
    "Build quality is premium, the hinge is sturdy, and the screen is gorgeous, but the coil whine is unbearable.",

    # B) Clause order & leading connectors
    "However, the updates are slow and the notifications arrive late.",
    "Although the camera is decent, low-light performance is disappointing and stabilization is worse.",
    "Yet the battery remains weak even after the latest patch.",
    "But the charger stops working unless the cable is held at an angle.",
    "And the speakers distort above 70% volume.",

    # C) Enumerations & comma splices
    "The UI is intuitive, navigation is smooth, onboarding is clunky, the help button does nothing.",
    "The lens, the sensor, the ISP—all top-notch; the autofocus, not so much.",
    "Maps, music, calls: fine; hotspot: terrible; Bluetooth pairing: flaky; car mode: brilliant.",
    "Screen: vibrant, touch: accurate, oleophobic coating: absent.",
    "The price is fair, the warranty is generous, the customer chat is robotic.",

    # D) Parentheticals & asides
    "The keyboard (to my surprise) is silent, but the spacebar rattles.",
    "The display—while bright—bleeds along the bottom edge.",
    "The hinge (rev 2) is solid; the first batch squeaked constantly.",
    "The camera, frankly, is overrated; the selfie mode over-smooths faces.",
    "Performance (single-core) is superb; (multi-core) throttles quickly.",

    # E) Negation scope & double cues
    "The camera isn't bad, but it's not good either.",
    "I don't dislike the speakers; I just can't understand dialogue.",
    "The update didn't fix the lag; it didn't make it worse, either.",
    "Not only is the battery small, it's also non-replaceable.",
    "It's no longer slow, just inconsistent under load.",

    # F) Comparatives & superlatives
    "The display is brighter than last year, but colors look worse than budget models.",
    "The keyboard is the best part; the trackpad is the worst.",
    "Faster than my old phone, slower than every competitor in this price.",
    "The charger is heavier but less efficient than the 45W brick.",
    "The software is more polished now, yet the camera is less reliable.",

    # G) Fragments, ellipses, SMS-style
    "The camera? Stunning. Battery? Nope.",
    "Great mic; terrible echo cancelation. Meetings ruined.",
    "Fast boot. Then… random crashes.",
    "Love the design—hate the fingerprints.",
    "Good speakers, meh bass, zero sub-bass.",

    # H) Cross-sentence coreference & pronouns
    "I bought the Pro model. It's fast, but it overheats after ten minutes.",
    "The earbuds connect instantly. They drop on calls though, which is annoying.",
    "I tried the stylus; its latency is fine, its palm rejection isn't.",
    "The webcam is wide. It makes faces look distorted and the colors look sickly.",
    "The dock seemed sturdy. It bent anyway.",

    # I) Sarcasm-ish / hedged tone
    "Fantastic battery—if you never turn it on.",
    "The camera is “pro-grade”, sure, in daylight only.",
    "The AI assistant is helpful… when it actually listens.",
    "Lovely screen protector that scratches if you breathe on it.",
    "The “silent” fan screams in performance mode.",

    # J) Domain mix & rare connectors
    "The VPN is stable; nevertheless, the kill switch fails silently.",
    "Storage is ample; notwithstanding the claim, read speeds are mediocre.",
    "The stand is adjustable; conversely, the clamp slips under pressure.",
    "The firmware ships weekly; ergo, bugs ship weekly.",
    "The BIOS is modern; yet, PXE boot is broken.",

    # K) Non-ASCII, emojis, punctuation quirks
    "Speakers are great—bass is thin—treble is harsh.",
    "Battery life is okay… camera? 🤷‍♂️",
    "Touch is responsive — except near the edges.",
    "Screen is 120 Hz; scrolling still feels jittery.",
    "The mic is clear; sibilance (“s”) sounds bitey.",

    # L) Multilingual bits (Spanish)
    "La batería dura poco, pero la cámara es excelente.",
    "El teclado es cómodo; sin embargo, el trackpad falla a veces.",
    "La pantalla es brillante y nítida, aunque el software se cuelga.",
    "La carga rápida funciona, pero el teléfono se calienta demasiado.",
    "El altavoz está bien; los graves, no.",
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

    if args.input:
        sentences = read_lines(args.input)
        if not sentences:
            raise SystemExit(f"No sentences found in {args.input}")
    else:
        sentences = DEFAULT_SAMPLES

    
    print("\nLoading ATEPC extractor...")
    extractor = ATEPCCheckpointManager.get_aspect_extractor(
        checkpoint=args.checkpoint,
        auto_device=True,
    )

    
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