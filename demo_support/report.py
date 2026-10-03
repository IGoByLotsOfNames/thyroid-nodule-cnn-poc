"""Self-contained, escaped HTML presentation for the local evaluation demo."""

from __future__ import annotations

import math
import re
from html import escape
from numbers import Real


def _text(value):
    return escape("—" if value is None else str(value), quote=True)


def _number(value, digits=2, scale=1, signed=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        return "—"
    return format(value * scale, f"{'+' if signed else ''}.{digits}f")


def _percent(value):
    return _number(value, scale=100) + ("%" if _number(value) != "—" else "")


def _points(value):
    return _number(value, scale=100, signed=True) + (" pp" if _number(value) != "—" else "")


def _class(classes, label):
    return f"{label} · {_text(classes[label])}" if type(label) is int and label in (0, 1) else "—"


def _matrix(metrics, classes, title):
    matrix = metrics.get("confusion_matrix", [[None, None], [None, None]])
    rows = "".join(
        f'<tr><th scope="row">{_class(classes, i)}</th>'
        + "".join(f"<td>{_number(matrix[i][j], 0)}</td>" for j in (0, 1))
        + "</tr>"
        for i in (0, 1)
    )
    return f"""<div class="panel matrix"><h3>{_text(title)}</h3>
<p class="muted">Threshold {_number(metrics.get("threshold"), 3)} · balanced accuracy {_percent(metrics.get("balanced_accuracy"))}</p>
<table><caption>Rows: true class. Columns: predicted class.</caption><thead><tr>
<th scope="col">True ↓ / predicted →</th><th scope="col">{_class(classes, 0)}</th>
<th scope="col">{_class(classes, 1)}</th></tr></thead><tbody>{rows}</tbody></table></div>"""


def _interval(interval, points=False):
    if not isinstance(interval, dict):
        return "Unavailable"
    format_value = _points if points else _percent
    return f"{format_value(interval.get('lower'))} to {format_value(interval.get('upper'))}"


def _samples(synthetic, images):
    classes, threshold = synthetic["class_names"], synthetic["selection"]["threshold"]
    rows = []
    for sample in synthetic.get("samples", []):
        source, probability = sample.get("source"), sample.get("probability")
        uri = images.get(source, "")
        valid_uri = isinstance(uri, str) and re.fullmatch(
            r"data:image/(?:png|jpeg|webp);base64,[A-Za-z0-9+/]+={0,2}", uri
        )
        preview = (
            (
                f'<img src="{_text(uri)}" alt="Synthetic fixture {_text(source)}" '
                'width="64" height="64">'
            )
            if valid_uri
            else '<span class="muted">No image</span>'
        )
        finite = (
            isinstance(probability, Real)
            and not isinstance(probability, bool)
            and math.isfinite(probability)
        )
        fixed = _class(classes, int(probability >= 0.5)) if finite else "—"
        selected = _class(classes, int(probability >= threshold)) if finite else "—"
        rows.append(f"""<tr><td>{preview}</td><th scope="row"><code>{_text(source)}</code>
<small>Group: {_text(sample.get("group"))}</small></th><td>{_class(classes, sample.get("label"))}</td>
<td>{_number(probability, 3)}</td><td>{fixed}</td><td>{selected}</td></tr>""")
    return (
        """<div class="table-wrap"><table><caption>Test fixtures · assigned probabilities of class 1</caption>
<thead><tr><th scope="col">Image</th><th scope="col">Sample / group</th><th scope="col">True class</th>
<th scope="col">Assigned p(class 1)</th><th scope="col">At 0.5</th><th scope="col">At selected threshold</th>
</tr></thead><tbody>"""
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _recorded(recorded):
    if not recorded:
        return '<section id="recorded"><h2>02 · Recorded TIRADS measurement</h2><p>No recorded measurement is bundled.</p></section>'
    stats, support = recorded.get("across_seeds", {}), recorded.get("test_support", {})
    bootstrap = recorded.get("group_bootstrap", {})
    interval = (bootstrap.get("intervals") or {}).get("mean_paired_delta_vs_fixed")
    crosses_zero = (
        isinstance(interval, dict)
        and isinstance(interval.get("lower"), Real)
        and isinstance(interval.get("upper"), Real)
        and interval["lower"] <= 0 <= interval["upper"]
    )
    interval_note = (
        "The paired interval includes zero: this comparison does not establish a clear improvement."
        if crosses_zero
        else "Interpret the paired interval under the stated fixed-model and grouping assumptions."
    )
    rows = "".join(
        f'<tr><th scope="row">{_text(row.get("seed"))}</th><td>{_number(row.get("selected_threshold"), 3)}</td>'
        f"<td>{_percent(row.get('balanced_accuracy'))}</td><td>{_percent(row.get('fixed_threshold_balanced_accuracy'))}</td>"
        f"<td>{_points(row.get('paired_delta_vs_fixed'))}</td></tr>"
        for row in recorded.get("per_seed", [])
    )
    selected = stats.get("balanced_accuracy", {})
    delta = stats.get("paired_delta_vs_fixed", {})
    negatives = recorded.get("test_resolution", {}).get("images_by_class", [None, None])[0]
    return f"""<section id="recorded" aria-labelledby="recorded-title"><div class="section-top"><span class="tag">Recorded evidence</span>
<h2 id="recorded-title">02 · A small, fixed TIRADS experiment</h2></div>
<p>{_text(recorded.get("interpretation", "Read these modest observed scores with their uncertainty and limited test support."))}</p>
<div class="stats"><div><span>Selected-threshold mean BA</span><strong>{_percent(selected.get("mean"))}</strong></div>
<div><span>Fixed-0.5 mean BA</span><strong>{_percent(stats.get("fixed_threshold_balanced_accuracy", {}).get("mean"))}</strong></div>
<div><span>Training-prior baseline BA</span><strong>{_percent(recorded.get("baseline", {}).get("balanced_accuracy"))}</strong></div></div>
<div class="callout"><strong>Paired shift: {_points(delta.get("mean"))}</strong>
<p>Approximate 95% group-bootstrap interval: {_interval(interval, points=True)}. {_text(interval_note)}</p></div>
<div class="table-wrap"><table><caption>Every recorded training seed · image-level balanced accuracy (BA)</caption>
<thead><tr><th scope="col">Seed</th><th scope="col">Selected threshold</th><th scope="col">Selected BA</th>
<th scope="col">Fixed-0.5 BA</th><th scope="col">Paired change</th></tr></thead><tbody>{rows}</tbody></table></div>
<p class="muted">Across-seed sample SD: {_number(selected.get("sample_sd"), scale=100)} percentage points; variance: {_number(selected.get("sample_variance"), 6)} (BA proportion squared), ddof=1.
Paired-change sample SD: {_number(delta.get("sample_sd"), scale=100)} percentage points. These are means of separate fits, not ensemble scores.</p>
<div class="two"><div class="panel"><h3>Support, not patient counts</h3><p>{_text(support.get("image_count"))} test images · {_text(support.get("declared_group_count"))} declared groups · {_text(negatives)} class-0 images.</p>
<p>Class 0: {_text(recorded.get("class_names", ["—", "—"])[0])}. Class 1: {_text(recorded.get("class_names", ["—", "—"])[1])}.</p>
<p>Source TIRADS annotations are risk categories, not pathology diagnoses. Declared groups are not verified independent patients.</p></div>
<div class="panel"><h3>What the interval covers</h3><p>Shared whole-group draws preserve paired comparisons across frozen models and thresholds. The approximate interval conditions on draws containing both classes.</p>
<p>Bootstrap status: {_text(bootstrap.get("status"))}. {_text(bootstrap.get("valid_draws"))} valid draws; {_text(bootstrap.get("invalid_single_class_draws"))} rejected single-class draws. Small class counts limit precision. This does not cover retraining or external-site uncertainty.</p></div></div>
</section>"""


def render_report(payload, images: dict[str, str]) -> str:
    """Return an offline HTML document; all payload text and image attributes are escaped."""
    if (
        type(payload.get("schema_version")) is not int
        or payload["schema_version"] != 1
        or payload.get("kind") != "thyroid_demo"
    ):
        raise ValueError("Unsupported demo report payload")
    synthetic, recorded = payload["synthetic"], payload.get("recorded_measurement")
    classes, selection = synthetic["class_names"], synthetic["selection"]
    mean = (recorded or {}).get("across_seeds", {}).get("balanced_accuracy", {}).get("mean")
    provenance = (recorded or {}).get("provenance", {})
    recorded_heading = (
        "Limited evidence of predictive utility" if recorded else "No recorded measurement bundled"
    )
    recorded_intro = (
        f"Observed mean balanced accuracy: <strong>{_percent(mean)}</strong>. "
        "All seeds and the paired uncertainty are shown below."
        if recorded
        else "This report includes the synthetic walkthrough only."
    )
    sample_hashes = "".join(
        f"<dt>{_text(row.get('source'))}</dt><dd><code>{_text(row.get('sha256'))}</code></dd>"
        for row in synthetic.get("samples", [])
    )
    smoke, smoke_html = payload.get("cnn_smoke"), ""
    if smoke:
        smoke_html = f"""<section><span class="tag">Optional execution check</span><h2>CNN smoke test on synthetic images</h2>
<p>{_text(smoke.get("notice"))}</p><p>Seed {_text(smoke.get("seed"))} · {_text(smoke.get("epochs"))} epochs. This checks software execution, not ultrasound performance.</p>
{_matrix(smoke.get("metrics", {}), smoke.get("class_names", ["0", "1"]), "Synthetic CNN predictions")}</section>"""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>Thyroid CNN · inspectable evaluation</title><style>{_CSS}</style></head><body><main>
<header><p class="eyebrow">Research software · evidence and limits</p><h1>Thyroid CNN <span>· inspectable evaluation</span></h1>
<p class="intro">Follow a threshold decision, then inspect a separate recorded experiment. Every score below has an explicit source.</p></header>
<div class="two overview"><div class="panel"><span class="tag">01 · Synthetic walkthrough</span><h2>Constructed examples, visible decisions</h2>
<p>These probabilities are handcrafted, not CNN predictions. The fixture images and scores are not medical evidence.</p>
<p class="metric-line">Selected threshold <strong>{_number(selection.get("threshold"), 3)}</strong> · constructed test BA <strong>{_percent(synthetic.get("selected_metrics", {}).get("balanced_accuracy"))}</strong></p></div>
<div class="panel"><span class="tag">02 · Recorded TIRADS measurement</span><h2>{_text(recorded_heading)}</h2>
<p>{recorded_intro}</p>
<p class="muted">This separate result uses source risk annotations. It is not a diagnostic validation.</p></div></div>
<section id="synthetic" aria-labelledby="synthetic-title"><div class="section-top"><span class="tag">Transparent fixture</span><h2 id="synthetic-title">01 · Trace the synthetic decision</h2></div>
<p>{_text(synthetic.get("notice"))}</p>
<ol class="pipeline"><li><b>1 · Establish the fixture</b><span>{_text(synthetic.get("counts", {}).get("train"))} train / {_text(synthetic.get("counts", {}).get("validation"))} validation / {_text(synthetic.get("counts", {}).get("test"))} test samples</span></li>
<li><b>2 · Use validation scores</b><span>Choose by balanced accuracy</span></li><li><b>3 · Freeze the threshold</b><span>{_number(selection.get("threshold"), 3)}; validation BA {_percent(selection.get("balanced_accuracy"))}</span></li>
<li><b>4 · Inspect held-out decisions</b><span>Same probabilities, two thresholds</span></li></ol>
<p>Class 1 is predicted when its assigned probability is at least the threshold. Balanced accuracy is the mean recall of the two classes.</p>
{_samples(synthetic, images)}<div class="two matrices">{_matrix(synthetic.get("fixed_metrics", {}), classes, "Fixed threshold")}
{_matrix(synthetic.get("selected_metrics", {}), classes, "Validation-selected threshold")}</div></section>
{_recorded(recorded)}{smoke_html}
<footer><h2>Provenance and execution</h2><p>The default walkthrough performs no runtime CNN training or inference: it uses constructed probabilities and a saved aggregate measurement. An optional CNN smoke check is labelled separately when present.</p>
<dl><dt>Synthetic fixture SHA-256</dt><dd><code>{_text(synthetic.get("fixture_sha256"))}</code></dd>
<dt>Recorded plan SHA-256</dt><dd><code>{_text(provenance.get("plan_sha256"))}</code></dd>
<dt>Recorded results SHA-256</dt><dd><code>{_text(provenance.get("results_sha256"))}</code></dd>
<dt>Recorded verification</dt><dd>{_text(provenance.get("verification"))}</dd></dl>
<details><summary>Synthetic sample hashes</summary><dl>{sample_hashes}</dl></details>
<p class="muted">Self-contained static report · no server, scripts or external assets · no comparison with historical study accuracy figures.</p></footer>
</main></body></html>"""


_CSS = """
:root{color-scheme:light;--ink:#183247;--teal:#00695f;--paper:#f7f6ef;--line:#c7d6d1;--muted:#465c67}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.6 "Segoe UI",Arial,sans-serif}
main{max-width:1160px;margin:auto;padding:48px 36px}header{border-top:6px solid var(--teal);padding-top:24px;margin-bottom:26px}
h1{font-size:clamp(2rem,4vw,3.25rem);line-height:1.14;letter-spacing:-.035em;margin:12px 0 20px}h1 span{font-weight:400}
h2{font-size:1.5rem;line-height:1.3;margin:12px 0 18px}h3{font-size:1.05rem;line-height:1.4;margin:0 0 12px}
p{margin:0 0 16px}.intro{font-size:1.12rem;max-width:800px}.eyebrow,.tag{text-transform:uppercase;letter-spacing:.12em;font-weight:700;font-size:.73rem;color:var(--teal)}
.two{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:22px}.panel{background:#fff;border:1px solid var(--line);border-radius:12px;padding:24px;min-width:0}.overview .panel{border-top:4px solid var(--teal)}
section{padding-top:38px;margin-top:34px;border-top:1px solid var(--line)}.section-top{margin-bottom:18px}.metric-line{padding-top:14px;border-top:1px solid var(--line);margin-bottom:0}.muted,small{color:var(--muted);font-size:.9rem}
.pipeline{list-style:none;padding:0;margin:24px 0;display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.pipeline li{padding:16px;background:#e6efea;border-radius:8px}.pipeline b,.pipeline span{display:block}.pipeline b{font-size:.86rem;margin-bottom:8px}.pipeline span{font-size:.86rem}
.table-wrap{overflow-x:auto;margin:22px 0}table{width:100%;border-collapse:collapse;font-size:.9rem;text-align:left}caption{text-align:left;font-weight:700;margin:0 0 12px}th,td{padding:13px 12px;border-bottom:1px solid var(--line);vertical-align:middle}thead th{background:#e6efea;font-weight:700}tbody th{font-weight:600}small{display:block;font-weight:400}img{display:block;object-fit:contain;background:var(--paper);border:1px solid var(--line);border-radius:6px}
.matrices{margin-top:28px}.matrix table{table-layout:fixed}.matrix td{text-align:center;font-weight:700;font-size:1.2rem}.matrix th{font-size:.8rem;overflow-wrap:anywhere}.matrix caption{font-size:.85rem;color:var(--muted)}
.stats{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:20px;margin:24px 0}.stats>div{border-left:3px solid var(--teal);padding:8px 18px}.stats span{display:block;font-size:.85rem;color:var(--muted)}.stats strong{display:block;font-size:1.65rem;line-height:1.4}
.callout{background:#e6efea;border-left:4px solid var(--teal);padding:20px 24px;border-radius:0 8px 8px 0;margin:24px 0}.callout p{margin:8px 0 0}code{font-family:Consolas,monospace;font-size:.83em;overflow-wrap:anywhere}
footer{border-top:2px solid var(--teal);margin-top:46px;padding-top:24px;font-size:.9rem}dl{display:grid;grid-template-columns:minmax(150px,1fr) minmax(0,3fr);gap:10px 20px}dt{font-weight:600}dd{margin:0;overflow-wrap:anywhere}summary{cursor:pointer;font-weight:600;padding:12px 0}summary:focus-visible{outline:3px solid var(--teal);outline-offset:4px}
@media(max-width:760px){main{padding:28px 18px}.two,.stats{grid-template-columns:1fr}.pipeline{grid-template-columns:repeat(2,minmax(0,1fr))}.panel{padding:20px}dl{grid-template-columns:1fr;gap:4px}dd{margin-bottom:14px}th,td{padding:10px 8px}}
@media print{body{background:#fff;font-size:10pt;print-color-adjust:exact;-webkit-print-color-adjust:exact}main{max-width:none;padding:0}h1{font-size:25pt}.two{gap:12px}.panel{padding:14px}section{padding-top:18px;margin-top:22px}tr,.panel,.callout{break-inside:avoid}h2,h3,caption{break-after:avoid}.table-wrap{overflow:visible}footer{font-size:8pt}details{display:block}summary{display:none}details dl{display:grid}}
"""
