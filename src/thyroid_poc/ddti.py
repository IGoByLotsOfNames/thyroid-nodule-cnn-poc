"""Import a byte-preserving, case-grouped cohort from original DDTI XML/JPGs.

Targets reproduce the recorded TIRADS groups; they are not pathology diagnoses.
All cases participate in duplicate grouping before missing labels are excluded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from PIL import Image

CLASS_NAMES = ("0_tirads_2_3", "1_tirads_4a_4b_4c_5")
TIRADS_LABELS = {"2": 0, "3": 0, "4a": 1, "4b": 1, "4c": 1, "5": 1}
CASE_FILE = re.compile(r"([1-9][0-9]*)\.xml")
IMAGE_FILE = re.compile(r"([1-9][0-9]*)_([1-9][0-9]*)\.jpg")
IDENTIFIER = re.compile(r"[1-9][0-9]*")
TARGET_SCHEMA = {
    "id": "ddti_tirads_binary_v1",
    "source_field": "case/tirads",
    "class_names": list(CLASS_NAMES),
    "source_value_to_label": TIRADS_LABELS,
    "missing_policy": "exclude_and_record",
    "unsupported_policy": "reject",
    "positive_class": 1,
    "interpretation": "Lower (2/3) versus higher (4a/4b/4c/5) source TIRADS annotations; not pathology-confirmed benign/malignant diagnoses or current ACR TI-RADS.",
}


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def cohort_fingerprint(manifest: dict) -> str:
    """Hash all cohort provenance except this derived fingerprint field itself."""
    return _hash(
        _canonical({key: value for key, value in manifest.items() if key != "cohort_fingerprint"})
    )


def _no_links(path: Path) -> None:
    for part in (path, *path.parents):
        if part.is_symlink() or (hasattr(part, "is_junction") and part.is_junction()):
            raise ValueError("Symlinks and junctions are not accepted")


def _inventory(source: Path) -> list[Path]:
    _no_links(source)
    if not source.is_dir():
        raise ValueError("Source must be an existing flat DDTI directory")
    result = sorted(source.iterdir(), key=lambda path: path.name)
    if not result:
        raise ValueError("Source is empty")
    for path in result:
        _no_links(path)
        if not path.is_file() or not (
            CASE_FILE.fullmatch(path.name) or IMAGE_FILE.fullmatch(path.name)
        ):
            raise ValueError(f"Unexpected or nested DDTI source entry: {path.name!r}")
    return result


def _one_text(root: ET.Element, tag: str) -> str:
    elements = root.findall(tag)
    if len(elements) != 1 or len(elements[0]):
        raise ValueError(f"Each XML record must contain one simple {tag} element")
    return (elements[0].text or "").strip()


def _scan(source: Path) -> tuple[list[dict], list[dict], list[dict]]:
    """Read every source file once, including all missing-label cases."""
    files, cases, images = [], {}, {}
    for path in _inventory(source):
        data = path.read_bytes()
        file_row = {"path": path.name, "bytes": len(data), "sha256": _hash(data)}
        files.append(file_row)
        case_match = CASE_FILE.fullmatch(path.name)
        if case_match:
            try:
                xml_text = data.decode("utf-8-sig")
            except UnicodeDecodeError as error:
                raise ValueError("Case XML must use UTF-8 encoding") from error
            if "\x00" in xml_text:
                raise ValueError("Case XML must use UTF-8 encoding without NUL characters")
            if "<!DOCTYPE" in xml_text.upper() or "<!ENTITY" in xml_text.upper():
                raise ValueError("XML declarations defining entities are not accepted")
            try:
                root = ET.fromstring(xml_text)
            except ET.ParseError as error:
                raise ValueError("Malformed case XML") from error
            if root.tag != "case":
                raise ValueError("Expected a case XML root")
            case_id = _one_text(root, "number")
            if not IDENTIFIER.fullmatch(case_id) or case_id != case_match[1]:
                raise ValueError("Full XML case number must match the XML filename")
            tirads = _one_text(root, "tirads")
            if tirads and tirads not in TIRADS_LABELS:
                raise ValueError(f"Unsupported TIRADS value: {tirads!r}")
            views = [_one_text(mark, "image") for mark in root.findall("mark")]
            if (
                not views
                or any(not IDENTIFIER.fullmatch(view) for view in views)
                or len(set(views)) != len(views)
            ):
                raise ValueError("Image marks must have unique, positive full decimal view IDs")
            cases[case_id] = {
                "case": case_id,
                "source_xml": path.name,
                "xml_sha256": file_row["sha256"],
                "tirads": tirads,
                "views": sorted(views, key=int),
            }
        else:
            image_match = IMAGE_FILE.fullmatch(path.name)
            try:
                with Image.open(io.BytesIO(data)) as image:
                    if image.format != "JPEG":
                        raise ValueError("DDTI .jpg files must contain JPEG data")
                    image.load()
                    rgb = image.convert("RGB")
                    rgb_hash = _hash(
                        f"{image.width}x{image.height}:RGB:".encode("ascii") + rgb.tobytes()
                    )
                    width, height = image.size
            except Exception as error:
                raise ValueError(f"Image cannot be decoded as JPEG: {path.name}") from error
            images[path.name] = {
                "source_image": path.name,
                "case": image_match[1],
                "view": image_match[2],
                "sha256": file_row["sha256"],
                "size": len(data),
                "rgb_sha256": rgb_hash,
                "width": width,
                "height": height,
            }
    expected = {f"{case['case']}_{view}.jpg" for case in cases.values() for view in case["views"]}
    if not cases or set(images) != expected:
        raise ValueError("XML image marks must agree exactly with all available source JPGs")
    case_rows = sorted(cases.values(), key=lambda row: int(row["case"]))
    image_rows = []
    for row in sorted(images.values(), key=lambda row: (int(row["case"]), int(row["view"]))):
        case = cases[row["case"]]
        image_rows.append(
            {
                **row,
                "source_xml": case["source_xml"],
                "xml_sha256": case["xml_sha256"],
                "tirads": case["tirads"],
            }
        )
    return files, case_rows, image_rows


def _source_manifest(path: Path | None, files: list[dict]) -> dict | None:
    if path is None:
        return None
    path = Path(path).absolute()
    _no_links(path)
    raw = path.read_bytes()
    record = json.loads(raw)
    expected = record.get("files")
    if not isinstance(expected, list) or len(expected) != len(files):
        raise ValueError("Expected source manifest does not cover the exact source inventory")
    for item in expected:
        if not isinstance(item, dict) or set(item) != {"path", "bytes", "sha256"}:
            raise ValueError("Invalid source manifest record")
    if sorted(expected, key=lambda row: row["path"]) != files:
        raise ValueError("Source files differ from the expected source manifest")
    return {"sha256": _hash(raw), "archive_sha256_recorded": record.get("archive_sha256")}


def _group(cases: list[dict], images: list[dict]) -> list[dict]:
    parent = {case["case"]: case["case"] for case in cases}

    def find(case: str) -> str:
        while parent[case] != case:
            parent[case] = parent[parent[case]]
            case = parent[case]
        return case

    def union(a: str, b: str) -> None:
        a, b = find(a), find(b)
        low, high = sorted((a, b), key=int)
        parent[high] = low

    seen_bytes, seen_pixels = {}, {}
    for row in images:  # Includes missing TIRADS: they may bridge other cases.
        for key, registry in ((row["sha256"], seen_bytes), (row["rgb_sha256"], seen_pixels)):
            if key in registry:
                union(row["case"], registry[key])
            registry[key] = row["case"]
    components = {}
    for case in cases:
        group = f"ddti-case-{find(case['case'])}"
        case["group"] = group
        components.setdefault(group, []).append(case["case"])
    case_groups = {case["case"]: case["group"] for case in cases}
    byte_labels, pixel_labels = {}, {}
    for row in images:
        row["group"] = case_groups[row["case"]]
        if row["tirads"]:
            label = TIRADS_LABELS[row["tirads"]]
            for key, registry in ((row["sha256"], byte_labels), (row["rgb_sha256"], pixel_labels)):
                previous = registry.setdefault(key, label)
                if previous != label:
                    raise ValueError(
                        "Identical image bytes or RGB pixels have conflicting included binary labels"
                    )
    return [
        {"group": group, "cases": members}
        for group, members in sorted(components.items(), key=lambda pair: int(pair[1][0]))
    ]


def _revalidate(source: Path, files: list[dict]) -> None:
    paths = _inventory(source)
    if [path.name for path in paths] != [row["path"] for row in files]:
        raise ValueError("Source inventory changed during import")
    for path, row in zip(paths, files):
        raw = path.read_bytes()
        if len(raw) != row["bytes"] or _hash(raw) != row["sha256"]:
            raise ValueError("Source bytes changed during import")


def import_ddti(
    source: Path, destination: Path, *, expected_source_manifest: Path | None = None
) -> dict:
    """Create a fresh cohort, groups.csv and cohort.json without any data split.

    ``destination.parent`` must already exist. Neither source nor destination may
    contain the other. Metadata contains relative source names, not absolute paths
    or timestamps, so identical inputs produce identical outputs.
    """
    source, destination = Path(source).absolute(), Path(destination).absolute()
    _no_links(source)
    _no_links(destination)
    if destination.exists():
        raise FileExistsError("Cohort destination already exists; choose a fresh path")
    source, destination = source.resolve(), destination.resolve()
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("Source and destination must not contain each other")
    if not destination.parent.is_dir():
        raise ValueError("Destination parent must already exist")
    files, cases, images = _scan(source)
    source_binding = _source_manifest(expected_source_manifest, files)
    components = _group(cases, images)
    rows, excluded = [], []
    for image in images:
        if not image["tirads"]:
            excluded.append({**image, "reason": "missing_tirads"})
        else:
            label = TIRADS_LABELS[image["tirads"]]
            rows.append(
                {**image, "label": label, "path": f"{CLASS_NAMES[label]}/{image['source_image']}"}
            )
    if {row["label"] for row in rows} != {0, 1}:
        raise ValueError("Both explicit TIRADS target groups must have included images")
    manifest = {
        "schema_version": 1,
        "kind": "ddti_imported_cohort",
        "class_names": list(CLASS_NAMES),
        "target_schema": json.loads(_canonical(TARGET_SCHEMA)),
        "source_fingerprint": _hash(_canonical(files)),
        "source_binding": source_binding,
        "source_files": files,
        "cases": cases,
        "components": components,
        "grouping": "full_case_connected_by_exact_bytes_and_decoded_rgb_before_exclusions",
        "generalization_unit": "case_disjoint_only_patient_identity_across_cases_unverified",
        "rows": rows,
        "excluded": excluded,
        "counts": {
            "source_cases": len(cases),
            "source_images": len(images),
            "included_cases": len({row["case"] for row in rows}),
            "included_images": len(rows),
            "excluded_cases": len({row["case"] for row in excluded}),
            "excluded_images": len(excluded),
            "all_case_components": len(components),
            "included_components": len({row["group"] for row in rows}),
            "images_by_class": dict(sorted(Counter(str(row["label"]) for row in rows).items())),
        },
    }
    manifest["cohort_fingerprint"] = cohort_fingerprint(manifest)
    stage = Path(tempfile.mkdtemp(prefix=".ddti-import-", dir=destination.parent)).resolve()
    try:
        for name in CLASS_NAMES:
            (stage / "data" / name).mkdir(parents=True)
        for row in rows:
            raw = (source / row["source_image"]).read_bytes()
            if len(raw) != row["size"] or _hash(raw) != row["sha256"]:
                raise ValueError("Source image changed before copying")
            target = stage / "data" / row["path"]
            target.write_bytes(raw)
            if _hash(target.read_bytes()) != row["sha256"]:
                raise ValueError("Copied image checksum mismatch")
        with (stage / "groups.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(("path", "group"))
            writer.writerows((row["path"], row["group"]) for row in rows)
        (stage / "cohort.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        _revalidate(source, files)  # Includes excluded JPGs and every XML record.
        if _source_manifest(expected_source_manifest, files) != source_binding:
            raise ValueError("Expected source manifest changed during import")
        _no_links(destination)
        _no_links(stage)
        if destination.exists():
            raise FileExistsError("Destination appeared during import")
        stage.rename(destination)
    finally:
        if stage.exists():
            # Recursive cleanup is confined to the exact staging directory created
            # above, under the verified destination parent; never clean source/output.
            _no_links(stage)
            if stage.parent != destination.parent or not stage.name.startswith(".ddti-import-"):
                raise ValueError("Refusing cleanup outside the allocated staging directory")
            shutil.rmtree(stage)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--source-manifest", type=Path, dest="expected_source_manifest")
    args = parser.parse_args()
    manifest = import_ddti(
        args.source, args.destination, expected_source_manifest=args.expected_source_manifest
    )
    print(
        json.dumps(
            {"cohort_fingerprint": manifest["cohort_fingerprint"], "counts": manifest["counts"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
