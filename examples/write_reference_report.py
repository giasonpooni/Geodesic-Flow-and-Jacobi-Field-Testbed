"""Write the first constant-curvature validation and application report."""

from pathlib import Path

from geodesic_testbed.reports import write_reference_report


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    json_path, markdown_path = write_reference_report(root / "results")
    print(json_path)
    print(markdown_path)


if __name__ == "__main__":
    main()
