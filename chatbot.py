import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

_LAST_ERROR: Optional[str] = None
CLASS_LABELS = {
    0: "Vegetation",
    1: "Built-up",
    2: "Water",
    3: "Other",
}


def _set_last_error(message: Optional[str]) -> None:
    global _LAST_ERROR
    _LAST_ERROR = message


def get_last_error() -> Optional[str]:
    return _LAST_ERROR


def _load_dotenv_file() -> None:
    dotenv_path = Path(__file__).resolve().parent / ".env"
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _model_name() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def _timeout_seconds() -> float:
    try:
        return float(os.getenv("OPENAI_TIMEOUT_SECONDS", "10"))
    except Exception:
        return 10.0


def _http_json(url: str, payload: Dict[str, Any], timeout: float) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None, "OPENAI_API_KEY is missing"

    req = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
        parsed = json.loads(body)
        if isinstance(parsed, dict):
            return parsed, None
        return None, "Response was not a JSON object"
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8")
        except Exception:
            body = str(e)
        return None, f"HTTP {e.code}: {body[:300]}"
    except Exception as e:
        return None, str(e)


def get_chat_runtime_status() -> Dict[str, Any]:
    _load_dotenv_file()
    model = _model_name()
    has_key = bool(os.getenv("OPENAI_API_KEY"))
    if not has_key:
        return {
            "ready": False,
            "provider": "fallback",
            "model": model,
            "message": "OPENAI_API_KEY is missing, fallback mode is active.",
            "last_error": get_last_error(),
        }

    return {
        "ready": True,
        "provider": "openai_http",
        "model": model,
        "message": "OpenAI API is configured.",
        "last_error": get_last_error(),
    }


def format_class_summary(class_counts: Dict[Any, Any]) -> str:
    total = sum(class_counts.values()) if class_counts else 0
    if total <= 0:
        return "No class counts are available."

    parts = []
    for class_id in [0, 1, 2, 3]:
        count = int(class_counts.get(class_id, 0))
        pct = (count / total) * 100.0
        parts.append(f"{CLASS_LABELS[class_id]}: {count} cells ({pct:.1f}%)")
    return ", ".join(parts)


def _normalize_counts(raw_counts: Dict[Any, Any]) -> Dict[int, int]:
    return {i: int(raw_counts.get(i, 0)) for i in [0, 1, 2, 3]}


def _format_counts_block(raw_counts: Dict[Any, Any]) -> str:
    counts = _normalize_counts(raw_counts)
    total = sum(counts.values())
    lines = []
    for i in [0, 1, 2, 3]:
        pct = (counts[i] / total * 100.0) if total > 0 else 0.0
        lines.append(f"- {CLASS_LABELS[i]}: {counts[i]} cells ({pct:.1f}%)")
    return "\n".join(lines)


def _format_confusion_explanation(matrix: Any, model_name: str) -> str:
    if not isinstance(matrix, list) or len(matrix) != 4:
        return f"{model_name} confusion matrix is available but not in expected 4x4 shape."
    lines = [f"{model_name} confusion matrix summary (rows=actual, columns=predicted):"]
    for row_idx, row in enumerate(matrix):
        if not isinstance(row, list) or len(row) != 4:
            continue
        row_total = sum(int(x) for x in row)
        correct = int(row[row_idx]) if row_idx < len(row) else 0
        row_acc = (correct / row_total * 100.0) if row_total > 0 else 0.0
        pred_breakdown = ", ".join(
            [f"{CLASS_LABELS[col_idx]}={int(row[col_idx])}" for col_idx in [0, 1, 2, 3]]
        )
        lines.append(
            f"- Actual {CLASS_LABELS[row_idx]}: {pred_breakdown} (class recall {row_acc:.1f}%)"
        )
    return "\n".join(lines)


def _compose_answer(summary: str, key_points: Any, interpretation: str) -> str:
    lines = [summary, "", "Key numbers:"]
    for point in key_points:
        lines.append(f"- {point}")
    lines.extend(["", f"Interpretation: {interpretation}"])
    return "\n".join(lines)


def _humanize_class_ids(text: str) -> str:
    replacements = {
        r"\b[Cc]lass 0\b": "Vegetation",
        r"\b[Cc]lass 1\b": "Built-up",
        r"\b[Cc]lass 2\b": "Water",
        r"\b[Cc]lass 3\b": "Other",
        r"\blabel 0\b": "Vegetation",
        r"\blabel 1\b": "Built-up",
        r"\blabel 2\b": "Water",
        r"\blabel 3\b": "Other",
    }
    out = text
    for pattern, repl in replacements.items():
        out = re.sub(pattern, repl, out)
    return out


def _format_net_change_human(net_change: Dict[Any, Any], year1: Any, year2: Any) -> str:
    normalized = {i: int(net_change.get(i, 0)) for i in [0, 1, 2, 3]}
    veg = normalized[0]
    built = normalized[1]
    water = normalized[2]
    other = normalized[3]
    top_class = max(normalized, key=lambda k: abs(normalized[k]))
    top_dir = "increase" if normalized[top_class] >= 0 else "decrease"
    return (
        f"From {year1} to {year2}, the clearest trend is that {CLASS_LABELS[1]} areas "
        f"{'expanded' if built >= 0 else 'declined'} by {abs(built)} cells ({built:+d}). "
        f"Over the same period, {CLASS_LABELS[0]} changed by {veg:+d} cells, "
        f"{CLASS_LABELS[2]} by {water:+d} cells, and {CLASS_LABELS[3]} by {other:+d} cells.\n\n"
        f"The largest absolute shift is in {CLASS_LABELS[top_class]} ({normalized[top_class]:+d}, {top_dir}). "
        f"In simple terms, positive numbers mean that class covers more area in {year2}, and negative numbers mean it covers less."
    )


def _format_transition_summary(matrix: Any, year1: Any = None, year2: Any = None) -> str:
    if not isinstance(matrix, list) or len(matrix) != 4:
        return "Transition matrix is available but not in expected 4x4 shape."
    title = "transition matrix"
    if year1 is not None and year2 is not None:
        title = f"transition matrix for {year1} to {year2}"
    lines = [f"Looking at the {title}, most cells stay in the same class, which suggests overall stability."]
    biggest_move = None
    stability_parts = []
    for i in [0, 1, 2, 3]:
        row = matrix[i] if i < len(matrix) and isinstance(matrix[i], list) else [0, 0, 0, 0]
        row = [int(x) for x in row[:4]]
        row_total = sum(row)
        stay = row[i] if i < len(row) else 0
        stay_rate = (stay / row_total * 100.0) if row_total > 0 else 0.0
        stability_parts.append(f"{CLASS_LABELS[i]} stays {stay_rate:.1f}%")
        for j in [0, 1, 2, 3]:
            if i == j:
                continue
            val = row[j]
            if biggest_move is None or val > biggest_move[0]:
                biggest_move = (val, i, j)
    lines.append("Class stability: " + ", ".join(stability_parts) + ".")
    if biggest_move is not None:
        val, src, dst = biggest_move
        lines.append(
            f"The most important transition is {CLASS_LABELS[src]} -> {CLASS_LABELS[dst]} ({val} cells)."
        )
    lines.append(
        "In plain language, this highlights where land use is persistent and where the biggest shifts are happening."
    )
    return "\n".join(lines)


def _confusion_user_takeaway(overall_acc: float, biggest: Any) -> str:
    confidence = "high" if overall_acc >= 85 else ("moderate" if overall_acc >= 70 else "limited")
    if biggest is None:
        return (
            f"This model has {confidence} reliability overall ({overall_acc:.2f}%). "
            "Treat individual cells as estimates, but use class-level trends with more confidence."
        )
    count, actual_cls, pred_cls = biggest
    return (
        f"This model has {confidence} reliability overall ({overall_acc:.2f}%). "
        f"The main confusion is {CLASS_LABELS[actual_cls]} being read as {CLASS_LABELS[pred_cls]} "
        f"({int(count)} cells), so decisions that depend on separating those two classes should be checked carefully."
    )


def _error_map_user_takeaway(err_rate: float) -> str:
    if err_rate < 10:
        level = "low"
    elif err_rate < 20:
        level = "moderate"
    else:
        level = "high"
    return (
        f"The overall error level is {level} ({err_rate:.2f}%). "
        "Use green areas as more dependable signals, and treat red clusters as review zones before making final conclusions."
    )


def _build_nontechnical_page_summary(context: Dict[str, Any], chat_metrics: Dict[str, Any]) -> Optional[str]:
    view_mode = context.get("view_mode", "Single Year")
    year = context.get("year")
    class_counts = _normalize_counts(context.get("class_counts", {}))
    total = sum(class_counts.values())
    if total == 0:
        return None

    dominant = max(class_counts, key=lambda k: class_counts[k])
    dominant_pct = (class_counts[dominant] / total) * 100.0 if total > 0 else 0.0

    if view_mode == "Multiple Years":
        comp = chat_metrics.get("comparison", {})
        y1 = comp.get("first_year")
        y2 = comp.get("second_year")
        net = comp.get("net_change", {})
        built = int(net.get(1, 0)) if isinstance(net, dict) else 0
        return _compose_answer(
            f"This page compares land-cover change from {y1} to {y2}.",
            [
                f"Main shift: Built-up changed by {built:+d} cells",
                f"Largest current class in selected output: {CLASS_LABELS[dominant]} ({class_counts[dominant]} cells, {dominant_pct:.1f}%)",
                "Charts show composition, net change, percentage change, and class transitions",
            ],
            "Use this page to understand which land types are expanding or shrinking and where change is concentrated.",
        )

    if str(year) == "2024":
        disagreement = chat_metrics.get("model_disagreement", {})
        d_count = int(disagreement.get("count", 0)) if isinstance(disagreement, dict) else 0
        d_rate = float(disagreement.get("rate", 0.0)) * 100.0 if isinstance(disagreement, dict) else 0.0
        return _compose_answer(
            "This page shows 2024 predicted land cover (no ground-truth labels available).",
            [
                f"Dominant predicted class: {CLASS_LABELS[dominant]} ({class_counts[dominant]} cells, {dominant_pct:.1f}%)",
                f"Model disagreement: {d_count} cells ({d_rate:.2f}%)",
                f"Total predicted cells: {total}",
            ],
            "Treat this as a scenario map: agreement zones are more reliable, disagreement zones need caution or validation.",
        )

    acc = context.get("accuracy")
    accs = chat_metrics.get("accuracies", {})
    if isinstance(accs, dict) and "mlp" in accs and "ridge" in accs:
        acc_text = f"MLP {float(accs['mlp']) * 100:.2f}% vs Ridge {float(accs['ridge']) * 100:.2f}%"
    else:
        acc_text = f"{float(acc) * 100:.2f}%" if acc is not None else "not shown for this specific view"
    return _compose_answer(
        f"This page shows land-cover results for {year}.",
        [
            f"Dominant class: {CLASS_LABELS[dominant]} ({class_counts[dominant]} cells, {dominant_pct:.1f}%)",
            f"Total cells analyzed: {total}",
            f"Model accuracy in this view: {acc_text}",
        ],
        "Use this to see what land type dominates and how confident the model is before making decisions.",
    )


def _build_key_takeaway(context: Dict[str, Any], chat_metrics: Dict[str, Any]) -> Optional[str]:
    class_counts = _normalize_counts(context.get("class_counts", {}))
    total = sum(class_counts.values())
    if total == 0:
        return None
    dominant = max(class_counts, key=lambda k: class_counts[k])
    dominant_pct = (class_counts[dominant] / total) * 100.0
    if context.get("view_mode") == "Multiple Years":
        comp = chat_metrics.get("comparison", {})
        net = comp.get("net_change", {})
        top = max([0, 1, 2, 3], key=lambda k: abs(int(net.get(k, 0)))) if isinstance(net, dict) else dominant
        top_val = int(net.get(top, 0)) if isinstance(net, dict) else 0
        return (
            f"The key takeaway is that {CLASS_LABELS[top]} changed the most ({top_val:+d} cells) "
            f"between {comp.get('first_year')} and {comp.get('second_year')}."
        )
    accs = chat_metrics.get("accuracies", {})
    if isinstance(accs, dict) and "mlp" in accs and "ridge" in accs:
        better = "MLP" if float(accs["mlp"]) >= float(accs["ridge"]) else "Ridge"
        return (
            f"The key takeaway is that {CLASS_LABELS[dominant]} is the dominant class ({class_counts[dominant]} cells, {dominant_pct:.1f}%), "
            f"and {better} is the more accurate model in this view."
        )
    return (
        f"The key takeaway is that {CLASS_LABELS[dominant]} is the dominant class "
        f"({class_counts[dominant]} cells, {dominant_pct:.1f}% of the area)."
    )


def _fallback_response(user_query: str, context: Dict[str, Any]) -> str:
    year = context.get("year")
    model = context.get("model")
    view_mode = context.get("view_mode", "Single Year")
    accuracy = context.get("accuracy")
    class_counts = context.get("class_counts", {})
    chat_metrics = context.get("chat_metrics", {})
    summary = format_class_summary(class_counts)
    user_input = (user_query or "").lower()

    if "confusion" in user_input:
        cms = chat_metrics.get("confusion_matrices", {})
        if cms:
            selected = str(chat_metrics.get("selected_model") or "").lower()
            if selected in {"mlp", "ridge"} and selected in cms:
                return f"{selected.upper()} confusion matrix (rows=actual, cols=predicted): {cms[selected]}"
            if "mlp" in cms and "ridge" in cms:
                return f"MLP confusion matrix: {cms['mlp']} | Ridge confusion matrix: {cms['ridge']}"

    if "pie" in user_input:
        return f"The pie chart on this page summarizes class composition for year {year}: {summary}"

    if "distribution" in user_input or "bar" in user_input:
        if chat_metrics.get("selected_model_counts"):
            return f"Class distribution (selected model): {chat_metrics['selected_model_counts']}"
        if chat_metrics.get("actual_counts") and chat_metrics.get("mlp_counts"):
            return (
                f"Actual: {chat_metrics['actual_counts']}, "
                f"MLP: {chat_metrics['mlp_counts']}, Ridge: {chat_metrics.get('ridge_counts')}"
            )
        return f"The distribution charts compare grid-cell counts by class. Current page summary: {summary}"

    if "trend" in user_input or "transition" in user_input or "growth" in user_input:
        comp = chat_metrics.get("comparison", {})
        if comp.get("net_change"):
            return _format_net_change_human(
                comp.get("net_change", {}),
                comp.get("first_year"),
                comp.get("second_year")
            )

    if "map" in user_input:
        return (
            "The map colors each grid cell as vegetation, built-up, water, or other using the active "
            f"page context (year={year}, model={model})."
        )

    if "ndvi" in user_input or "ndbi" in user_input or "ndwi" in user_input:
        return "NDVI tracks vegetation signal, NDBI highlights built-up signal, and NDWI highlights water signal."

    if "accuracy" in user_input or "reliable" in user_input or "trust" in user_input:
        if str(year) == "2024":
            return (
                "No ground-truth labels are available for 2024 in this dashboard, "
                "so reliability can only be inferred from earlier-year performance."
            )
        if accuracy is not None:
            return f"Estimated accuracy for this page is {accuracy * 100:.2f}%."
        return "No explicit accuracy metric is available for the current page state."

    if "change" in user_input or "difference" in user_input:
        y1 = context.get("first_year")
        y2 = context.get("second_year")
        if view_mode == "Multiple Years" and y1 and y2:
            return f"You are comparing {y1} vs {y2}. Current class mix context: {summary}"
        return f"Current class mix context: {summary}"

    return (
        f"I can explain this page using the current context ({view_mode}, year={year}, model={model}). "
        f"Class summary: {summary}"
    )


def _extract_response_text(payload: Dict[str, Any]) -> Optional[str]:
    text = payload.get("output_text")
    if isinstance(text, str) and text.strip():
        return text.strip()

    output = payload.get("output", [])
    for item in output:
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"} and content.get("text"):
                return str(content["text"]).strip()

    return None


def _deterministic_response(user_query: str, context: Dict[str, Any]) -> Optional[str]:
    user_input = (user_query or "").lower()
    chat_metrics = context.get("chat_metrics", {}) or {}
    selected_model = str(chat_metrics.get("selected_model") or context.get("model") or "").lower()
    comparison = chat_metrics.get("comparison", {}) or {}
    class_counts = context.get("class_counts", {}) or {}
    year = context.get("year")
    # Balanced mode: deterministic only for explicit metric/chart-value asks.
    metric_markers = [
        "confusion",
        "accuracy",
        "class distribution",
        "distribution chart",
        "bar chart",
        "prediction difference map",
        "uncertainty map",
        "agreement map",
        "error map",
        "net change",
        "percentage change",
        "transition matrix",
        "trend between",
        "how many",
        "count",
        "rate",
        "percent",
    ]
    if not any(marker in user_input for marker in metric_markers):
        return None

    def _most_common_class(counts: Dict[Any, Any]) -> str:
        c = _normalize_counts(counts)
        top = max(c, key=lambda k: c[k])
        return f"{CLASS_LABELS[top]} ({c[top]} cells)"

    if "confusion" in user_input:
        cms = chat_metrics.get("confusion_matrices", {})
        if selected_model in {"mlp", "ridge"} and selected_model in cms:
            m = cms[selected_model]
            if isinstance(m, list) and len(m) == 4:
                total = sum(sum(int(x) for x in row) for row in m if isinstance(row, list))
                correct = sum(int(m[i][i]) for i in [0, 1, 2, 3] if isinstance(m[i], list) and len(m[i]) > i)
                overall = (correct / total * 100.0) if total > 0 else 0.0
                biggest = None
                for r in [0, 1, 2, 3]:
                    row = m[r]
                    if not isinstance(row, list) or len(row) != 4:
                        continue
                    for c in [0, 1, 2, 3]:
                        if r == c:
                            continue
                        val = int(row[c])
                        if biggest is None or val > biggest[0]:
                            biggest = (val, r, c)
                key_points = [f"Overall accuracy: {overall:.2f}%"]
                if biggest is not None:
                    key_points.append(
                        f"Most common mistake: actual {CLASS_LABELS[biggest[1]]} predicted as {CLASS_LABELS[biggest[2]]} ({biggest[0]} cells)"
                    )
                key_points.append("Rows are the real class, columns are the model prediction")
                return _compose_answer(
                    f"{selected_model.upper()} confusion matrix in simple words:",
                    key_points,
                    _confusion_user_takeaway(overall, biggest),
                )
            return _format_confusion_explanation(cms[selected_model], selected_model.upper())
        if "mlp" in cms and "ridge" in cms:
            mlp_text = _format_confusion_explanation(cms["mlp"], "MLP")
            ridge_text = _format_confusion_explanation(cms["ridge"], "Ridge")
            return f"{mlp_text}\n\n{ridge_text}"

    if ("what this map" in user_input or "what does this map" in user_input or "explain map" in user_input) and class_counts:
        c = _normalize_counts(class_counts)
        total = sum(c.values())
        dominant = max(c, key=lambda k: c[k])
        return _compose_answer(
            f"This map shows land-cover types across Nuremberg for {year}.",
            [
                f"Total grid cells: {total}",
                f"Dominant class: {CLASS_LABELS[dominant]} ({c[dominant]} cells, {(c[dominant] / total * 100.0 if total else 0.0):.1f}%)",
                f"Vegetation: {c[0]}, Built-up: {c[1]}, Water: {c[2]}, Other: {c[3]}",
            ],
            "Each cell is colored by land-cover type, so you can see where each type is concentrated.",
        )

    if ("dominant" in user_input or "most common" in user_input or "highest" in user_input) and class_counts:
        c = _normalize_counts(class_counts)
        ranked = sorted(c.items(), key=lambda kv: kv[1], reverse=True)
        top_i, top_v = ranked[0]
        second_i, second_v = ranked[1]
        return _compose_answer(
            f"The dominant class in {year} is {CLASS_LABELS[top_i]}.",
            [
                f"{CLASS_LABELS[top_i]}: {top_v} cells",
                f"Next highest is {CLASS_LABELS[second_i]}: {second_v} cells",
                f"Gap: {top_v - second_v} cells",
            ],
            "A larger gap means the dominant class clearly outweighs the second class.",
        )

    if "accuracy" in user_input:
        accs = chat_metrics.get("accuracies", {})
        if selected_model in {"mlp", "ridge"} and selected_model in accs:
            return (
                f"{selected_model.upper()} accuracy on this page is {accs[selected_model] * 100:.2f}%.\n"
                "This value is computed directly from the current page's actual vs predicted labels."
            )
        if "mlp" in accs and "ridge" in accs:
            better = "MLP" if accs["mlp"] >= accs["ridge"] else "Ridge"
            return (
                f"Model accuracy comparison on this page:\n"
                f"- MLP: {accs['mlp'] * 100:.2f}%\n"
                f"- Ridge: {accs['ridge'] * 100:.2f}%\n"
                f"Higher accuracy: {better}."
            )

    if "distribution" in user_input or "class comparison" in user_input or "bar" in user_input:
        if chat_metrics.get("selected_model_counts"):
            model_name = str(chat_metrics.get("selected_model") or context.get("model") or "Selected model")
            counts = _normalize_counts(chat_metrics["selected_model_counts"])
            total = sum(counts.values())
            return _compose_answer(
                f"{model_name} class distribution on this page:",
                [
                    f"Total cells: {total}",
                    f"Most common class: {_most_common_class(counts)}",
                    f"Vegetation: {counts[0]}",
                    f"Built-up: {counts[1]}",
                    f"Water: {counts[2]}",
                    f"Other: {counts[3]}",
                ],
                "This chart shows how the model allocates land-cover classes in the currently selected view.",
            )
        if chat_metrics.get("actual_counts") and chat_metrics.get("mlp_counts"):
            actual = _normalize_counts(chat_metrics["actual_counts"])
            mlp = _normalize_counts(chat_metrics["mlp_counts"])
            ridge = _normalize_counts(chat_metrics.get("ridge_counts", {}))
            return (
                "Class distribution comparison for this page:\n"
                f"Actual:\n{_format_counts_block(actual)}\n"
                f"MLP prediction:\n{_format_counts_block(mlp)}\n"
                f"Ridge prediction:\n{_format_counts_block(ridge)}"
            )
        if chat_metrics.get("mlp_counts") and chat_metrics.get("ridge_counts"):
            mlp = _normalize_counts(chat_metrics["mlp_counts"])
            ridge = _normalize_counts(chat_metrics["ridge_counts"])
            return _compose_answer(
                "Class distribution comparison for 2024 predictions:",
                [
                    f"MLP dominant class: {_most_common_class(mlp)}",
                    f"Ridge dominant class: {_most_common_class(ridge)}",
                    f"MLP counts -> Vegetation {mlp[0]}, Built-up {mlp[1]}, Water {mlp[2]}, Other {mlp[3]}",
                    f"Ridge counts -> Vegetation {ridge[0]}, Built-up {ridge[1]}, Water {ridge[2]}, Other {ridge[3]}",
                ],
                "Both models suggest the same broad composition if dominant classes and totals are close; differences indicate prediction uncertainty.",
            )
        comp = chat_metrics.get("comparison", {})
        if comp.get("counts_first") and comp.get("counts_second"):
            return (
                f"Class distribution comparison ({comp.get('first_year')} -> {comp.get('second_year')}):\n"
                f"{comp.get('first_year')}:\n{_format_counts_block(comp['counts_first'])}\n"
                f"{comp.get('second_year')}:\n{_format_counts_block(comp['counts_second'])}\n"
                f"{_format_net_change_human(comp.get('net_change', {}), comp.get('first_year'), comp.get('second_year'))}"
            )

    if (
        "change" in user_input
        or "difference" in user_input
        or "trend" in user_input
        or "growth" in user_input
        or "increase" in user_input
        or "decrease" in user_input
    ):
        comp = comparison
        if comp.get("net_change"):
            base_text = _format_net_change_human(
                comp.get("net_change", {}),
                comp.get("first_year"),
                comp.get("second_year"),
            )
            builtup = comp.get("builtup_change")
            if isinstance(builtup, dict):
                base_text += (
                    f"\nBuilt-up specific change: {int(builtup.get('delta', 0)):+d} cells "
                    f"({float(builtup.get('percent', 0.0)):+.2f}%)."
                )
            return base_text
        if chat_metrics.get("mlp_counts") and chat_metrics.get("ridge_counts"):
            mlp = _normalize_counts(chat_metrics.get("mlp_counts", {}))
            ridge = _normalize_counts(chat_metrics.get("ridge_counts", {}))
            diffs = {k: int(ridge[k] - mlp[k]) for k in [0, 1, 2, 3]}
            top = max(diffs, key=lambda k: abs(diffs[k]))
            direction = "higher in Ridge" if diffs[top] > 0 else "higher in MLP"
            return _compose_answer(
                "Main model-to-model change in this 2021 comparison:",
                [
                    f"Largest class gap: {CLASS_LABELS[top]} ({diffs[top]:+d} cells, {direction})",
                    f"MLP {CLASS_LABELS[top]} count: {mlp[top]}",
                    f"Ridge {CLASS_LABELS[top]} count: {ridge[top]}",
                ],
                "This is where the two models disagree most in totals, so this class should be reviewed first.",
            )
        disagreement = chat_metrics.get("model_disagreement")
        if disagreement:
            return _compose_answer(
                "Prediction difference map explanation:",
                [
                    f"Cells where MLP and Ridge differ: {disagreement.get('count')}",
                    f"Disagreement rate: {disagreement.get('rate', 0.0) * 100:.2f}%",
                    "Map colors: red = models differ, gray = models agree",
                ],
                "Higher disagreement means lower confidence in those locations; agreement zones are more stable between models.",
            )

    if "transition" in user_input and comparison.get("transition_matrix"):
        return _format_transition_summary(
            comparison.get("transition_matrix"),
            comparison.get("first_year"),
            comparison.get("second_year"),
        )

    if "percentage" in user_input and comparison.get("percentage_change"):
        pct = comparison.get("percentage_change", {})
        return (
            f"Percentage change from {comparison.get('first_year')} to {comparison.get('second_year')}:\n"
            f"- Vegetation: {float(pct.get(0, 0.0)):+.1f}%\n"
            f"- Built-up: {float(pct.get(1, 0.0)):+.1f}%\n"
            f"- Water: {float(pct.get(2, 0.0)):+.1f}%\n"
            f"- Other: {float(pct.get(3, 0.0)):+.1f}%"
        )

    if "error map" in user_input:
        accs = chat_metrics.get("accuracies", {})
        sel = selected_model if selected_model in {"mlp", "ridge"} else "mlp"
        if sel in accs:
            err_rate = (1.0 - float(accs[sel])) * 100.0
            return _compose_answer(
                "Error map explanation:",
                [
                    f"Selected model: {sel.upper()}",
                    f"Estimated error rate: {err_rate:.2f}%",
                    "Map colors: green = correct prediction, red = incorrect prediction",
                ],
                _error_map_user_takeaway(err_rate),
            )

    if ("mistake" in user_input or "error" in user_input) and chat_metrics.get("confusion_matrices"):
        cms = chat_metrics.get("confusion_matrices", {})
        matrix = None
        model_name = "MLP"
        if selected_model in {"mlp", "ridge"} and selected_model in cms:
            matrix = cms[selected_model]
            model_name = selected_model.upper()
        elif "mlp" in cms:
            matrix = cms["mlp"]
            model_name = "MLP"

        if isinstance(matrix, list) and len(matrix) == 4:
            biggest = None
            for r in [0, 1, 2, 3]:
                row = matrix[r]
                if not isinstance(row, list) or len(row) != 4:
                    continue
                for c in [0, 1, 2, 3]:
                    if r == c:
                        continue
                    val = int(row[c])
                    if biggest is None or val > biggest[0]:
                        biggest = (val, r, c)
            if biggest is not None:
                count, actual_cls, pred_cls = biggest
                return _compose_answer(
                    f"Where {model_name} makes the biggest error:",
                    [
                        f"Largest confusion: Actual {CLASS_LABELS[actual_cls]} -> Predicted {CLASS_LABELS[pred_cls]} ({count} cells)"
                    ],
                    "This is the most frequent misclassification pattern in the current view.",
                )
    if "prediction difference map" in user_input or "uncertainty map" in user_input or "agreement map" in user_input:
        disagreement = chat_metrics.get("model_disagreement")
        if disagreement:
            agree_rate = 100.0 - (disagreement.get("rate", 0.0) * 100.0)
            return _compose_answer(
                "Map meaning for model agreement/difference:",
                [
                    f"Model disagreement: {disagreement.get('count')} cells ({disagreement.get('rate', 0.0) * 100:.2f}%)",
                    f"Model agreement: {agree_rate:.2f}%",
                    "Color meaning: green/gray = agreement, yellow/red = disagreement depending on the map",
                ],
                "Disagreement highlights uncertain areas where the two models give different classes.",
            )

    return None


def _call_openai_http(user_query: str, context: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    _load_dotenv_file()

    compact_context = {
        "view_mode": context.get("view_mode"),
        "year": context.get("year"),
        "model": context.get("model"),
        "compare_mode": context.get("compare_mode"),
        "first_year": context.get("first_year"),
        "second_year": context.get("second_year"),
        "accuracy": context.get("accuracy"),
        "class_counts": context.get("class_counts"),
        "mlp_counts": context.get("mlp_counts"),
        "ridge_counts": context.get("ridge_counts"),
        "total_cells": context.get("total_cells"),
        "selected_label_column": context.get("selected_label_column"),
        "chat_metrics": context.get("chat_metrics"),
    }

    system_prompt = (
        "You are the in-dashboard assistant for a Nuremberg land-cover analysis app. "
        "Answer using only the provided page context and user question. "
        "Use numeric values from chat_metrics when available and do not invent values. "
        "Explain numbers in plain, user-friendly language (no raw Python dict style). "
        "Avoid unexplained technical jargon. When a technical term appears, briefly explain it in simple words. "
        "Always include what the result means for a non-technical user and what action/caution follows from it. "
        "If the user asks as a city planner, provide practical steps and expected impact in simple terms. "
        "If the user asks for detailed explanation, provide a longer, structured explanation; if brief, keep it short. "
        "If user asks confusion matrix, accuracy, class comparison, transition, or trend and data exists in chat_metrics, return exact values with interpretation. "
        "If data is genuinely missing, explicitly say what is missing and suggest the relevant view in one line. "
        "Be precise, concise, and practical."
    )
    user_prompt = (
        "Page context JSON:\n"
        f"{json.dumps(compact_context, ensure_ascii=True)}\n\n"
        f"User question:\n{user_query}\n\n"
        "Respond in plain text."
    )

    payload = {
        "model": _model_name(),
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": user_prompt}],
            },
        ],
        "temperature": 0.2,
    }

    response, err = _http_json(
        url="https://api.openai.com/v1/responses",
        payload=payload,
        timeout=_timeout_seconds(),
    )
    if err:
        return None, err
    if not response:
        return None, "Empty response from OpenAI"

    text = _extract_response_text(response)
    if text:
        return text, None
    return None, "OpenAI did not return output text"


def generate_response(user_query: str, context: Dict[str, Any]) -> str:
    deterministic = _deterministic_response(user_query, context or {})
    if deterministic:
        return _humanize_class_ids(deterministic)

    text, error = _call_openai_http(user_query, context or {})
    if text:
        _set_last_error(None)
        return _humanize_class_ids(text)

    _set_last_error(error or "Unknown OpenAI error")
    fallback = _fallback_response(user_query, context or {})
    if error:
        reason = "OpenAI API request failed"
        if "insufficient_quota" in error:
            reason = "OpenAI API quota is exceeded (insufficient_quota)"
        elif "HTTP 401" in error:
            reason = "OpenAI API authentication failed (invalid API key)"
        elif "HTTP 429" in error:
            reason = "OpenAI API rate limit or quota issue (HTTP 429)"
        return _humanize_class_ids(f"{fallback}\n\n[Fallback mode active: {reason}]")
    return _humanize_class_ids(fallback)


def build_context(
    selected_year,
    selected_model=None,
    compare_mode=False,
    accuracy=None,
    view_mode="Single Year",
    first_year=None,
    second_year=None,
    selected_label_column=None,
):
    return {
        "year": selected_year,
        "model": selected_model,
        "compare_mode": compare_mode,
        "accuracy": accuracy,
        "view_mode": view_mode,
        "first_year": first_year,
        "second_year": second_year,
        "selected_label_column": selected_label_column,
    }
