"""Regenerate the README schematics: python docs/visuals/render.py.

Requires Pillow (already a project dependency). These are software diagrams,
not experimental results. Fonts are selected locally; no network is used.
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
NAVY, TEAL, INK = "#142C45", "#087F83", "#21384F"
MUTED, LINE, PALE, WHITE = "#53697B", "#CDDCE4", "#EFF7F8", "#FFFFFF"


def font(size, bold=False):
    names = [
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "Arial Bold.ttf" if bold else "Arial.ttf",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


class Diagram:
    def __init__(self, title, subtitle, height=1030):
        self.image = Image.new("RGB", (1600, height), WHITE)
        self.draw = ImageDraw.Draw(self.image)
        self.height = height
        self.draw.rounded_rectangle((1, 1, 1598, height - 2), 24, outline=LINE, width=2)
        self.text((60, 40), "THYROID NODULE CNN · SOFTWARE DESIGN", 24, TEAL, True)
        self.text((60, 86), title, 47, NAVY, True)
        self.text((60, 154), subtitle, 29, MUTED)

    def text(self, pos, value, size=29, color=INK, bold=False):
        self.draw.text(pos, value, fill=color, font=font(size, bold))

    def card(self, x, y, w, h, title, lines, accent=False, label=None):
        self.draw.rounded_rectangle((x, y, x + w, y + h), 17, fill=PALE if accent else WHITE, outline=LINE, width=2)
        self.draw.rounded_rectangle((x, y, x + 7, y + h), 3, fill=TEAL)
        if label:
            self.text((x + 26, y + 20), label, 22, TEAL, True)
        offset = 56 if label else 23
        self.text((x + 26, y + offset), title, 32, NAVY, True)
        for i, line in enumerate(lines):
            self.text((x + 26, y + offset + 53 + i * 39), line, 28, MUTED)

    def arrow(self, pts, color=TEAL):
        self.draw.line(pts, fill=color, width=4, joint="curve")
        (px, py), (x, y) = pts[-2:]
        if x > px:
            triangle = [(x, y), (x - 13, y - 8), (x - 13, y + 8)]
        elif x < px:
            triangle = [(x, y), (x + 13, y - 8), (x + 13, y + 8)]
        elif y > py:
            triangle = [(x, y), (x - 8, y - 13), (x + 8, y - 13)]
        else:
            triangle = [(x, y), (x - 8, y + 13), (x + 8, y + 13)]
        self.draw.polygon(triangle, fill=color)

    def footer(self, text):
        self.draw.line((60, self.height - 86, 1540, self.height - 86), fill=LINE, width=2)
        self.text((60, self.height - 63), text, 25, MUTED)

    def save(self, name):
        self.image.save(ROOT / name, optimize=True)


def pipeline():
    d = Diagram("From grouped images to traceable evaluation", "A fixed split separates fitting, model selection and final test predictions.")
    d.card(60, 232, 440, 249, "Authorized images", ["Two labelled class directories", "Patient / case group CSV", "SHA-256 for exact copies"], label="01  DEFINE THE DATA")
    d.card(580, 232, 440, 249, "Immutable manifest", ["Whole groups stay together", "Duplicates join components", "Seed + hashes + class order"], label="02  FREEZE THE SPLIT")
    d.card(1100, 232, 440, 249, "Three partitions", ["Train: fit model parameters", "Validation: early stopping", "Test: reserved evaluation"], label="03  KEEP ROLES SEPARATE")
    d.arrow([(500, 340), (580, 340)])
    d.arrow([(1020, 340), (1100, 340)])
    d.card(60, 583, 440, 239, "CNN training", ["AlexNet-style or Inception", "RGB input; model normalizes", "Train + validation only"], accent=True, label="04  BUILD THE MODEL")
    d.card(580, 583, 440, 239, "Saved experiment", [".keras model", ".metadata.json + history", "Model and split identities"], accent=True, label="05  PRESERVE THE CONTRACT")
    d.card(1100, 583, 440, 239, "Held-out evaluation", ["Aligned sample probabilities", "Individual + mean ensemble", "Confusion matrix, AUC + more"], accent=True, label="06  REPORT WHAT HAPPENED")
    d.arrow([(500, 695), (580, 695)])
    d.arrow([(1020, 695), (1100, 695)])
    d.arrow([(1320, 481), (1320, 532), (280, 532), (280, 583)])
    d.text((515, 487), "train + validation", 25, TEAL, True)
    d.arrow([(1470, 481), (1470, 583)])
    d.text((1388, 505), "test", 25, TEAL, True)
    d.text((60, 861), "Evaluation rechecks bytes, labels, manifest identity and known training overlap.", 30, NAVY, True)
    d.footer("Schematic of the maintained implementation · No dataset, clinical model or new accuracy result is shown.")
    d.save("experiment-flow.png")


def artifacts():
    d = Diagram("Trace a prediction through its experiment records", "Link each model to the dataset, preprocessing and predictions that define the experiment.", 1090)
    d.card(60, 236, 455, 301, "Dataset manifest", ["Ordered classes and rows", "Sample ID, group and label", "Image SHA-256 hashes", "Split seed and assignments"], label="manifest.json")
    d.card(572, 236, 455, 301, "Model contract", ["Model hash and input shape", "RGB preprocessing contract", "Architecture and train seed", "Train / validation hashes"], accent=True, label=".keras + .metadata.json")
    d.card(1084, 236, 455, 301, "Evaluation report", ["Sample-aligned probabilities", "Model identities and threshold", "Individual + ensemble metrics", "Undefined quantities → null"], label="evaluation.json")
    d.arrow([(515, 386), (572, 386)])
    d.arrow([(1027, 386), (1084, 386)])
    d.draw.rounded_rectangle((60, 602, 1540, 940), 18, fill=PALE)
    d.text((91, 630), "Read a binary probability correctly", 36, NAVY, True)
    d.text((91, 690), "Sorted directory names define class 0 and class 1; metadata preserves that order.", 30, INK)
    d.card(92, 761, 430, 126, "class_names[0]", ["Probability = 1 − p"])
    d.card(584, 761, 430, 126, "Sigmoid output p", ["Probability of class index 1"], accent=True)
    d.card(1076, 761, 430, 126, "class_names[1]", ["Probability = p"])
    d.footer("Schematic contracts, not sample medical predictions · Threshold selection belongs before final testing.")
    d.save("experiment-records.png")


if __name__ == "__main__":
    pipeline()
    artifacts()
