import argparse
from pathlib import Path
from importlib import util


def load_train_module(baseline_model: str) -> object:
    train_path = Path(__file__).absolute().parent / baseline_model / "train.py"
    if not train_path.exists():
        raise FileNotFoundError(f"Expected train.py at {train_path}")
    spec = util.spec_from_file_location(
        f"{baseline_model}_train", 
        str(train_path)
    )
    module = util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def combo_name(resampled_bool: bool, class_weights_bool: bool) -> str:
    return f"{'resampled' if resampled_bool else 'original'}_{'cw' if class_weights_bool else 'nocw'}"


def main():
    """
    Run all 4 combinations of training a LSTM baseline model.
    Specify which baseline model to use (multi_class_lstm or ensemble_lstm).
    """
    ap = argparse.ArgumentParser(description="Run all 4 LSTM combos")
    ap.add_argument("--baseline_model", default="ensemble_lstm", help="Which baseline model to run (multi_class or ensemble)")
    ap.add_argument("--dry-run", action="store_true", help="Print combos but don't execute training")
    args = ap.parse_args()

    # The four combos (resampled?, class_weights?)
    combos = [
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ]

    train_mod = load_train_module(args.baseline_model)
    output_dir = f"src/DARPA/baselines/{args.baseline_model}" 

    print(f"\n=== Baseline Training Runner ===")
    print(f"Using baseline model: {args.baseline_model}")
    print("Will run the following combos:")
    for resampled_bool, class_weights_bool in combos:
        print("  ", combo_name(resampled_bool, class_weights_bool))
    print(f"Output directory: {output_dir}")

    if args.dry_run:
        return

    for resampled_bool, class_weights_bool in combos:
        name = combo_name(resampled_bool, class_weights_bool)
        print(f"\nRunning: {name} ===")

        train_mod.train_lstm(
            output_dir=output_dir,
            resampled_bool=resampled_bool,
            class_weights_bool=class_weights_bool
        )


if __name__ == "__main__":
    main()
