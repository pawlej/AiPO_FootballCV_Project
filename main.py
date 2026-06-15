"""

Przykłady:
    python main.py run --video content/example_match.mp4 --model models/best.pt
    python main.py infer --video content/example_match.mp4 --model models/best.pt
"""

from aipo_football_cv.cli import main


if __name__ == "__main__":
    main()
