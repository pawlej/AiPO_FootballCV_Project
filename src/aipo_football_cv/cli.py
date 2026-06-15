from __future__ import annotations

import argparse
from pathlib import Path

from .infer_model import infer_frames
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AiPO project: local football CV pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run full video analysis pipeline")
    run.add_argument("--video", required=True, help="Path to input video, e.g. content/match.mp4")
    run.add_argument("--model", required=True, help="Path to YOLO model, e.g. models/best.pt")
    run.add_argument("--output-dir", default="outputs")
    run.add_argument("--confidence", type=float, default=0.3)
    run.add_argument("--batch-size", type=int, default=20)
    run.add_argument("--max-frames", type=int, default=None)
    run.add_argument("--stride", type=int, default=1)
    run.add_argument("--use-stub", action="store_true")
    run.add_argument("--stub-path", default="stubs/tracks.pkl")
    run.add_argument("--enable-view-transform", action="store_true")
    run.add_argument("--roboflow-api-key", default=None)
    run.add_argument("--meter-scale", type=float, default=0.01)

    infer = subparsers.add_parser("infer", help="Run model-only inference and save detections CSV")
    infer.add_argument("--video", required=True)
    infer.add_argument("--model", required=True)
    infer.add_argument("--output-csv", default="outputs/model_detections.csv")
    infer.add_argument("--confidence", type=float, default=0.3)
    infer.add_argument("--max-frames", type=int, default=200)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        result = run_pipeline(
            video_path=args.video,
            model_path=args.model,
            output_dir=args.output_dir,
            confidence=args.confidence,
            batch_size=args.batch_size,
            max_frames=args.max_frames,
            stride=args.stride,
            use_stub=args.use_stub,
            stub_path=args.stub_path,
            enable_view_transform=args.enable_view_transform,
            roboflow_api_key=args.roboflow_api_key,
            meter_scale=args.meter_scale,
        )
        print("Saved annotated video:", result.output_video_path)
        print("Saved team stats:", result.team_stats_path)
        print("Saved player stats:", result.player_stats_path)
    elif args.command == "infer":
        df = infer_frames(
            model_path=args.model,
            video_path=args.video,
            output_csv=args.output_csv,
            confidence=args.confidence,
            max_frames=args.max_frames,
        )
        print(f"Saved {len(df)} detections to {Path(args.output_csv)}")


if __name__ == "__main__":
    main()
