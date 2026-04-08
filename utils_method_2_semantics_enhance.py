import json
import os

from utils_windows_long_path import maybe_windows_long_path, safe_open


def load_relation_profiles(profile_path):
    """
    Load offline-built relation profiles from JSON.
    Returns empty dict on missing/invalid input to keep baseline unchanged.
    """
    if not profile_path:
        return {}

    profile_path = maybe_windows_long_path(profile_path)
    if not os.path.exists(profile_path):
        return {}

    try:
        with safe_open(profile_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        return {}

    return {}


def _format_percent(ratio, *, decimals=1):
    try:
        pct = float(ratio) * 100.0
    except (TypeError, ValueError):
        return None
    if decimals <= 0:
        return f"{int(round(pct))}%"
    text = f"{pct:.{decimals}f}".rstrip("0").rstrip(".")
    return f"{text}%"


def _safe_int(x):
    try:
        return int(x)
    except Exception:
        return None


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def build_profile_block(
    head_name,
    profiles,
    *,
    use_country=True,
    use_time=True,
    use_events=True,
    topk_pairs=10,
    events_k=2,
):
    """
    Output format (paper/defense friendly):

    [Relation Profile]
    Head: {head}

    GeoBias(topK):
    - India -> India (6.4%), Philippines -> Philippines (6.3%), ...

    TemporalActivity:
    - level: LOW|MEDIUM|HIGH
    - count_train: 10
    - span_steps: 270
    - density: 0.037 (below median 0.081)

    ReprEvents(2):
    - 2014-04 South Africa->South Africa | South_Africa->Police_(South_Africa)
    - 2014-09 United States->Kenya | Police_(United_States)->Citizen_(Kenya)

    Guidance:
    - Prefer body relations consistent with GeoBias.
    - For LOW activity, prefer 1-2 hop rules; for HIGH activity, allow 2-3 hop rules.
    [/Relation Profile]
    """
    if not profiles or not head_name:
        return ""

    profile = profiles.get(head_name)
    if not isinstance(profile, dict):
        return ""

    lines = ["[Relation Profile]", f"Head: {head_name}"]
    has_any = False

    # -------- GeoBias --------
    if use_country:
        geo = profile.get("geo_bias") or {}
        pairs = geo.get("country_pairs_topk") or []
        k = max(0, int(topk_pairs))
        shown = []
        if isinstance(pairs, list) and pairs and k > 0:
            for entry in pairs[:k]:
                if not isinstance(entry, dict):
                    continue
                src = entry.get("src")
                tgt = entry.get("tgt")
                ratio_text = _format_percent(entry.get("ratio"), decimals=1)
                if src and tgt and ratio_text:
                    shown.append(f"{src} -> {tgt} ({ratio_text})")

        if shown:
            lines.append(f"GeoBias(top{k}):")
            lines.append("- " + ", ".join(shown))
            has_any = True

    # -------- TemporalActivity --------
    if use_time:
        ta = profile.get("temporal_activity") or {}
        if isinstance(ta, dict) and ta:
            level = (ta.get("level") or "").strip()
            count_train = _safe_int(ta.get("count_train"))
            span_steps = _safe_int(ta.get("span_steps"))
            density = _safe_float(ta.get("density"))
            median = _safe_float(ta.get("density_median"))

            # Consider this section present if any key info exists
            if level or count_train is not None or span_steps is not None or density is not None:
                lines.append("TemporalActivity:")
                if level:
                    lines.append(f"- level: {level}")
                if count_train is not None:
                    lines.append(f"- count_train: {count_train}")
                if span_steps is not None:
                    lines.append(f"- span_steps: {span_steps}")

                # density line with optional median comparison
                if density is not None:
                    dens_txt = f"{density:.3f}"
                    if median is not None:
                        med_txt = f"{median:.3f}"
                        cmp = "below" if density < median else ("above" if density > median else "equal to")
                        lines.append(f"- density: {dens_txt} ({cmp} median {med_txt})")
                    else:
                        lines.append(f"- density: {dens_txt}")

                has_any = True

    # -------- ReprEvents --------
    if use_events:
        events = profile.get("repr_events") or []
        max_k = max(0, int(events_k))
        shown_events = []
        if isinstance(events, list) and events and max_k > 0:
            for ev in events[:max_k]:
                if not isinstance(ev, dict):
                    continue
                date_ym = (ev.get("date_ym") or "").strip()
                src_country = (ev.get("src_country") or "").strip()
                tgt_country = (ev.get("tgt_country") or "").strip()
                src_name = (ev.get("src_name") or "").strip()
                tgt_name = (ev.get("tgt_name") or "").strip()

                if not date_ym or not src_name or not tgt_name:
                    continue

                if src_country and tgt_country:
                    shown_events.append(f"{head_name}({src_name},{tgt_name},{date_ym})")
                else:
                    shown_events.append(f"{head_name}({src_name},{tgt_name},{date_ym})")

        if shown_events:
            lines.append(f"ReprEvents({len(shown_events)}):")
            lines.extend(shown_events)
            has_any = True

    # -------- Guidance (only if we have any section) --------
    if has_any:
        lines.append("Guidance:")
        lines.append("- Prefer body relations consistent with GeoBias.")
        if use_time == True:
            lines.append("- For LOW activity, prefer 1-2 hop rules; for HIGH activity, allow 2-3 hop rules.")
        lines.append("[/Relation Profile]")
        return "\n".join(lines)

    return ""
