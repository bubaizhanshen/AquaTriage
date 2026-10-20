"""AquaTriage review workspace, shared by the local and hosted entrypoints."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import io
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("AQUATRIAGE_WORK_DIR", ROOT / "tmp" / "platform"))
WORK.mkdir(parents=True, exist_ok=True, mode=0o700)
for key in ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "MPLCONFIGDIR"):
    os.environ[key] = str(WORK)
sys.path.insert(0, str(ROOT / "src"))
CLOUD = os.environ.get("AQUATRIAGE_CLOUD") == "1"
MAX_UPLOAD_MB = 2 if CLOUD else 20
MAX_RECORDS = 1000 if CLOUD else 10000

import pandas as pd
import altair as alt
from rdkit import Chem
from rdkit.Chem import Draw
import streamlit as st

from ecoood.review_workflow import (
    ENDPOINTS, OPTIONAL, REQUIRED, RULES, export_csv, export_run, fit_reference,
    load_reference, prepare_inputs, read_input_csv, run_review,
    template_csv,
)
from ecoood.review_presentation import (
    CHECK_LABELS, ENDPOINT_LABELS, RULE_HELP, RULE_LABELS, SCORE_LABELS, TASK_LABELS,
    endpoint_plot_table, review_queue_view, run_summary, summary_text,
)


st.set_page_config(page_title="AquaTriage", page_icon=":material/science:", layout="wide")
st.markdown("""
<style>
 .block-container {padding-top:2rem; padding-bottom:2rem; max-width:1500px}
 h1 {font-size:2rem !important; letter-spacing:0 !important}
 h2 {font-size:1.35rem !important; letter-spacing:0 !important}
 h3 {font-size:1.1rem !important; letter-spacing:0 !important}
 [data-testid="stMetricValue"] {font-size:1.8rem}
 [data-testid="stSidebar"] {background:#f3f5f6}
 button {border-radius:5px !important}
 @media (max-width:640px) {
   [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) {flex-wrap:wrap}
   [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) > [data-testid="stColumn"] {
     flex:1 1 calc(50% - 1rem) !important; width:calc(50% - 1rem) !important;
     min-width:0 !important;
   }
 }
 </style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading reference model...")
def installed_reference(path: str, modified: int):
    checksum = None
    if CLOUD:
        checksum = json.loads(Path(path).with_suffix(".json").read_text())["artifact_sha256"]
    return load_reference(Path(path), expected_sha256=checksum)


@st.cache_resource(show_spinner=False)
def review_lock():
    return threading.BoundedSemaphore(1)


def load_queue(upload, source):
    if source == "Example queue":
        path = ROOT / "data" / "examples" / "review_queue.csv"
        if not path.exists():
            return None
        return read_input_csv(path.read_bytes())
    return read_input_csv(
        upload.getvalue(), max_bytes=MAX_UPLOAD_MB * 1024 * 1024, max_records=MAX_RECORDS,
    ) if upload else None


st.title("AquaTriage")
st.caption("Aquatic toxicity review")


def entered_cases(reference):
    records = st.session_state.setdefault("entered_cases", [])
    endpoint = st.selectbox("Endpoint", list(ENDPOINTS), format_func=ENDPOINT_LABELS.get,
                            key="entry_endpoint")
    species_options = sorted(reference.reference_cases.loc[
        reference.reference_cases.endpoint.eq(endpoint), "species",
    ].unique()) if reference else []
    with st.form("case_entry", border=False):
        identity_col, name_col = st.columns(2)
        chemical_id = identity_col.text_input("Chemical identifier", max_chars=128)
        chemical_name = name_col.text_input("Chemical name (optional)", max_chars=300)
        smiles = st.text_input("Confirmed SMILES", max_chars=5000)
        species = st.selectbox(
            "Species", species_options, index=None, accept_new_options=True,
            placeholder="Scientific species name", key=f"entry_species_{endpoint}",
        )
        duration = st.number_input(
            "Actual exposure duration (h)", min_value=0.0,
            value={"fish_96h_lc50": 96.0, "daphnia_48h_ec50": 48.0, "algae_72_96h_ec50": 72.0}[endpoint],
            step=1.0, key=f"entry_duration_{endpoint}",
            help=f"Accepted window for this endpoint: {ENDPOINTS[endpoint][0]:g}-{ENDPOINTS[endpoint][1]:g} h.",
        )
        with st.expander("Test conditions (optional)"):
            medium = st.selectbox(
                "Medium", ["", "FW", "AQU", "CUL"],
                format_func=lambda code: {"": "Not provided", "FW": "Freshwater (FW)",
                                          "AQU": "Aqueous (AQU)", "CUL": "Culture (CUL)"}[code],
            )
            temp_col, ph_col = st.columns(2)
            temperature = temp_col.number_input("Temperature (degrees C)", value=None, step=1.0)
            ph = ph_col.number_input("pH", value=None, min_value=0.0, max_value=14.0, step=0.1)
        verified = st.checkbox("I have confirmed the test-substance identity and its molecular structure.")
        add = st.form_submit_button("Add case", icon=":material/add:", disabled=len(records) >= MAX_RECORDS)
    if add:
        row = {
            "chemical_id": chemical_id, "chemical_name": chemical_name, "smiles": smiles,
            "species": species or "", "endpoint": endpoint, "duration_h": duration,
            "identity_verified": verified, "medium": medium,
            "temperature_c": temperature, "ph": ph,
        }
        candidate = pd.DataFrame([*records, row])
        try:
            _, rejected = prepare_inputs(candidate)
            if not rejected.empty:
                st.error("; ".join(rejected.rejection_reason.unique()))
            elif len(candidate.to_csv(index=False).encode()) > MAX_UPLOAD_MB * 1024 * 1024:
                st.error(f"Input exceeds the {MAX_UPLOAD_MB} MB limit.")
            else:
                records.append(row)
        except ValueError as exc:
            st.error(str(exc))
    if records:
        case_col, remove_col, clear_col = st.columns([3, 1, 1])
        selected = case_col.selectbox(
            "Queued case", list(range(len(records))),
            format_func=lambda i: f"{i + 1}. {records[i]['chemical_id']} / {ENDPOINT_LABELS[records[i]['endpoint']]}",
            key=f"entry_remove_{len(records)}",
        )
        if remove_col.button("Remove", icon=":material/delete:", help="Remove the selected input case"):
            records.pop(selected)
            st.rerun()
        if clear_col.button("Clear", icon=":material/clear_all:", help="Clear manually entered cases"):
            st.session_state.entered_cases = []
            st.rerun()
    return pd.DataFrame(records) if records else None


def input_validation(frame):
    digest = hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()
    cached = st.session_state.get("input_validation")
    if cached is None or cached["digest"] != digest:
        cached = {"digest": digest, "accepted": None, "rejected": None, "error": None}
        try:
            cached["accepted"], cached["rejected"] = prepare_inputs(frame)
        except ValueError as exc:
            cached["error"] = str(exc)
        st.session_state.input_validation = cached
    return cached


def check_table(frame):
    result = frame.copy()
    if "endpoint" in result:
        result["endpoint"] = result.endpoint.replace(ENDPOINT_LABELS)
    return result.rename(columns=CHECK_LABELS)


def endpoint_chart(run, chemical_id):
    data = endpoint_plot_table(run, chemical_id)
    y = alt.Y("label:N", title=None, sort=list(ENDPOINT_LABELS.values()),
              axis=alt.Axis(labelLimit=180))
    points = alt.Chart(data[data.prediction_mg_l.notna()]).mark_point(
        filled=True, size=100, color="#127f83",
    ).encode(
        x=alt.X("prediction_mg_l:Q", title="Predicted concentration (mg/L)",
                scale=alt.Scale(type="log", zero=False),
                axis=alt.Axis(tickCount=5, format="~g", domain=True)),
        y=y,
        tooltip=[alt.Tooltip("label:N", title="Endpoint"),
                 alt.Tooltip("prediction_mg_l:Q", title="Median prediction (mg/L)", format=".4g"),
                 alt.Tooltip("case_count:Q", title="Eligible cases", format=".0f")],
    )
    missing = alt.Chart(data[data.prediction_mg_l.isna()]).mark_text(
        align="left", color="#666666", fontSize=12,
    ).encode(x=alt.value(8), y=y, text="status:N")
    chart = points + missing
    if run.manifest["task"] == "hazard_omission":
        criterion = pd.DataFrame({"criterion": [run.manifest["cutoff_mg_l"]]})
        chart += alt.Chart(criterion).mark_rule(color="#333333", strokeDash=[5, 4]).encode(
            x="criterion:Q",
            tooltip=[alt.Tooltip("criterion:Q", title="Concentration criterion (mg/L)")],
        )
    return chart.properties(height=190).configure_axis(
        labelFontSize=12, titleFontSize=12, grid=False, domainColor="#333333",
        labelColor="#333333", titleColor="#333333", titleFontWeight="normal",
    ).configure_view(stroke=None)


@st.cache_data(show_spinner=False, max_entries=128, ttl=3600)
def molecule_image(smiles):
    options = Draw.rdMolDraw2D.MolDrawOptions()
    options.useBWAtomPalette()
    options.bondLineWidth = 2.0
    image = Draw.MolToImage(Chem.MolFromSmiles(smiles), size=(440, 330), options=options)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()

with st.sidebar:
    st.subheader("Reference model")
    installed = Path(os.environ.get("AQUATRIAGE_REFERENCE", ROOT / "models" / "review-v1" / "reference.joblib"))
    if CLOUD:
        source = "Installed reference"
        st.caption("Published reference v1")
    else:
        source = st.radio("Training reference", ["Installed reference", "Fit from my data"], label_visibility="collapsed")
    reference = None
    if source == "Installed reference":
        if installed.is_file():
            try:
                reference = installed_reference(str(installed), installed.stat().st_mtime_ns)
            except Exception as exc:
                st.error(f"Reference could not be loaded: {exc}")
        else:
            st.info("No fitted reference installed.")
    else:
        training = st.file_uploader("Measured reference data", type="csv", key="training_file",
                                    help="One chemical-species-endpoint case per row; measured target_log_molar is required.")
        st.download_button("Training template", template_csv(training=True, full=True),
                           "training_template.csv", "text/csv", icon=":material/download:", on_click="ignore")
        if training:
            training_hash = hashlib.sha256(training.getvalue()).hexdigest()
            if st.button("Fit reference", icon=":material/model_training:", use_container_width=True):
                try:
                    with st.spinner("Fitting five bootstrap models..."):
                        st.session_state.own_reference = fit_reference(read_input_csv(training.getvalue()))
                        st.session_state.own_reference_hash = training_hash
                except (ValueError, RuntimeError) as exc:
                    st.error(str(exc))
            if st.session_state.get("own_reference_hash") == training_hash:
                reference = st.session_state.get("own_reference")
    if reference:
        st.metric("Training chemicals", f"{reference.manifest['chemicals']:,}")
        st.caption(f"{reference.manifest['records']:,} records · 5 bootstrap models")
        st.caption("Reference " + reference.manifest["reference_sha256"][:12])
    st.divider()
    st.page_link("https://github.com/bubaizhanshen/AquaTriage", label="Source code", icon=":material/code:")
    st.page_link("https://github.com/bubaizhanshen/AquaTriage/blob/main/docs/review_workspace.md",
                 label="Input and result guide", icon=":material/help:")
    with st.expander("Model and data notes"):
        st.markdown(
            "Predictions concern acute freshwater LC50/EC50. Optional test conditions and bioactivity "
            "are imputed from the fitted reference; missingness is exported. A structure check does "
            "not verify a substance name. Uploaded identity-structure pairs must already be confirmed. "
            "Not selected does not mean safe or accurate. Uploads are processed in this server session; "
            "the application does not save uploaded tables to disk. Do not upload confidential data "
            "to a public instance."
        )
    if CLOUD:
        st.caption(f"Batch limit: {MAX_RECORDS:,} cases / {MAX_UPLOAD_MB} MB")

setup, queue_tab, record_tab, checks_tab, detail_tab = st.tabs(
    ["Review setup", "Review queue", "Chemical record", "Input checks", "Run details"]
)
with setup:
    left, right = st.columns([1.45, 1], gap="large")
    with left:
        st.subheader("Input cases")
        input_mode = st.segmented_control("Input source", ["Example queue", "Enter cases", "Upload CSV"],
                                          default="Example queue", label_visibility="collapsed")
        upload = None
        if input_mode == "Upload CSV":
            upload = st.file_uploader("Case table", type="csv", key="queue_file",
                                     max_upload_size=MAX_UPLOAD_MB,
                                     help="SMILES, chemical ID, species, endpoint, duration, and confirmed identity.")
        template_col, full_col = st.columns(2)
        template_col.download_button("Input template", template_csv(), "queue_template.csv", "text/csv",
                                     icon=":material/download:", use_container_width=True, on_click="ignore")
        full_col.download_button("All input columns", template_csv(full=True), "queue_full_template.csv", "text/csv",
                                 icon=":material/download:", use_container_width=True, on_click="ignore")
        frame, validation = None, None
        eligible_count, ignored = 0, []
        try:
            frame = entered_cases(reference) if input_mode == "Enter cases" else load_queue(upload, input_mode)
        except ValueError as exc:
            st.error(str(exc))
        if frame is not None:
            validation = input_validation(frame)
            ignored = sorted(set(frame.columns) - set(REQUIRED) - set(OPTIONAL))
            if validation["error"]:
                st.error(validation["error"])
            else:
                eligible_count = validation["accepted"].chemical_id.nunique()
                rejected_count = len(validation["rejected"])
                st.caption(f"Eligible chemicals: {eligible_count:,} · Rejected cases: {rejected_count:,}")
                if rejected_count:
                    st.warning(f"{rejected_count:,} {'case' if rejected_count == 1 else 'cases'} "
                               "did not pass input checks.")
                    with st.expander("Input issues", expanded=eligible_count == 0):
                        st.dataframe(check_table(validation["rejected"]), hide_index=True, width="stretch")
            if ignored:
                st.warning("Not used for prediction or ranking: " + ", ".join(ignored))
            st.caption(f"{len(frame):,} cases · {frame.chemical_id.nunique() if 'chemical_id' in frame else '?'} submitted chemical IDs")
            preview_cols = [c for c in ("chemical_name", "chemical_id", "species", "endpoint", "duration_h") if c in frame]
            preview = frame[preview_cols].copy()
            if "endpoint" in preview:
                preview["endpoint"] = preview.endpoint.replace(ENDPOINT_LABELS)
            st.dataframe(preview, hide_index=True, width="stretch", height=330,
                         column_config={
                             "chemical_name": "Chemical", "chemical_id": "Identifier",
                             "species": "Species", "endpoint": "Endpoint", "duration_h": "Duration (h)",
                         })
    with right:
        st.subheader("Review objective")
        objective = st.selectbox("Objective", ["Check concentration accuracy", "Recover hazard omissions"],
                                  help="Accuracy: prioritize potentially inaccurate concentrations. Omissions: prioritize "
                                       "predicted-low-concern chemicals whose toxicity may be underestimated.")
        task = "general_error" if objective == "Check concentration accuracy" else "hazard_omission"
        rule = st.selectbox("Ranking", RULES[task], format_func=RULE_LABELS.get,
                            help="\n\n".join(RULE_HELP[value] for value in RULES[task]))
        quota_mode = st.segmented_control("Review workload", ["Percentage", "Chemical count"], default="Percentage")
        fraction, count = None, None
        if quota_mode == "Percentage":
            fraction = st.slider("Eligible chemicals to review (%)", 0, 100, 25, 1) / 100
        else:
            count = st.number_input("Chemicals to review", min_value=0, max_value=int(eligible_count),
                                    value=min(5, int(eligible_count)), step=1)
        planned_count = count if count is not None else round(fraction * eligible_count)
        st.caption(f"Review allocation: {planned_count:,} of {eligible_count:,} eligible chemicals")
        cutoff = st.number_input("Acute-effect concentration criterion (mg/L)", min_value=0.000001,
                                  value=1.0, format="%.6g", disabled=task == "general_error",
                                  help="1 mg/L is the study reference, not a complete regulatory hazard classification.")
        active_config = {
            "queue": hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest() if frame is not None else None,
            "reference": reference.manifest["reference_sha256"] if reference else None,
            "task": task, "rule": rule, "fraction": fraction, "count": count, "cutoff": cutoff,
        }
        if st.button("Run review", type="primary", icon=":material/play_arrow:",
                     use_container_width=True, disabled=frame is None or reference is None or eligible_count == 0):
            guard = review_lock() if CLOUD else None
            if guard is not None and not guard.acquire(blocking=False):
                st.warning("Another review is running. Please retry shortly.")
            else:
                try:
                    with st.spinner("Predicting cases and ranking chemicals..."):
                        result = run_review(reference, frame, task=task, rule=rule,
                                            review_fraction=fraction, review_count=count, cutoff_mg_l=cutoff)
                    st.session_state.run = result
                    st.session_state.run_config = active_config
                    if result.manifest["eligible_chemicals"] == 0:
                        st.warning("All submitted cases were rejected.")
                    else:
                        n = result.manifest["review_count"]
                        st.success(f"{n} {'chemical' if n == 1 else 'chemicals'} selected")
                except (ValueError, RuntimeError) as exc:
                    st.error(str(exc))
                finally:
                    if guard is not None:
                        guard.release()

run = st.session_state.get("run")
stale = run is not None and st.session_state.get("run_config") != active_config
if stale:
    st.warning("Settings or input changed. Displayed results belong to the previous run.")
with queue_tab:
    if run is None:
        st.info("No completed review.")
    else:
        view = review_queue_view(run)
        completed_rule = run.manifest["rule"]
        omission = run.manifest["task"] == "hazard_omission"
        st.caption(f"{TASK_LABELS[run.manifest['task']]} · {RULE_LABELS[completed_rule]}")
        if omission:
            st.caption(f"Concentration criterion: {run.manifest['cutoff_mg_l']:g} mg/L")
        metrics = st.columns(4)
        metrics[0].metric("Eligible chemicals", run.manifest["eligible_chemicals"])
        metrics[1].metric("Selected", run.manifest["review_count"])
        metrics[2].metric("Rejected cases", run.manifest["rejected_records"])
        metrics[3].metric("Review fraction", f"{(run.manifest['realized_review_fraction'] or 0):.1%}")
        complete_col, selected_col = st.columns(2)
        complete_col.download_button("Download complete results", export_run(run), "aquatriage_review.zip",
                                    "application/zip", icon=":material/download:", disabled=stale, on_click="ignore")
        export_columns = ["rank", "chemical_id", "chemical_name", "prediction_mg_l", "represented_endpoints",
                          "review_score", "review_measure", "ranking_rule", "review_reason"]
        if omission:
            export_columns += ["concentration_criterion_mg_l", "criterion_position", "criterion_ratio"]
        selected_view = view.loc[view.selected_for_review] if not view.empty else view
        selected_col.download_button(
            "Download selected chemicals",
            export_csv(selected_view[[c for c in export_columns if c in selected_view]]),
            "aquatriage_selected.csv", "text/csv", icon=":material/download:",
            disabled=stale, on_click="ignore",
        )
        if not run.chemicals.empty:
            search = st.text_input("Find a chemical", placeholder="Name or identifier")
            selected_only = st.toggle("Selected only")
            shown = view.copy()
            if selected_only:
                shown = shown[shown.selected_for_review]
            if search:
                shown = shown[shown.chemical_name.str.contains(search, case=False, regex=False) |
                              shown.chemical_id.str.contains(search, case=False, regex=False)]
            cols = ["rank", "selected_for_review", "chemical_name", "review_score", "prediction_mg_l"]
            if omission:
                cols += ["criterion_position", "criterion_ratio"]
            cols += ["represented_endpoints", "chemical_id", "review_reason"]
            st.dataframe(shown[cols], hide_index=True, width="stretch", height=480,
                         column_config={
                             "prediction_mg_l": st.column_config.NumberColumn(
                                 "Prediction (mg/L)", format="%.4g",
                                 help="Lowest of the within-endpoint median predictions; unrepresented endpoints are not predicted."),
                             "selected_for_review": st.column_config.CheckboxColumn("Review"),
                             "rank": "Rank", "chemical_name": "Chemical", "chemical_id": "Identifier",
                             "represented_endpoints": "Represented endpoints",
                             "review_score": st.column_config.NumberColumn(
                                 SCORE_LABELS[completed_rule], format="%.4g", help=RULE_HELP[completed_rule]),
                             "review_reason": "Review allocation",
                             "criterion_position": "Prediction vs criterion",
                             "criterion_ratio": st.column_config.NumberColumn(
                                 "Prediction / criterion", format="%.4g",
                                 help="A ratio above 1 means the predicted concentration is above the criterion. "
                                      "The proximity rule uses its base-10 logarithm."),
                         })
            with st.expander("All ranking measures"):
                st.dataframe(shown[["chemical_name", "block_knn", "similarity_ad", "ensemble_sd"]],
                             hide_index=True, width="stretch", column_config={
                                 "chemical_name": "Chemical",
                                 **{key: st.column_config.NumberColumn(SCORE_LABELS[key], format="%.4g", help=RULE_HELP[key])
                                    for key in ("block_knn", "similarity_ad", "ensemble_sd")},
                             })
with record_tab:
    if run is not None and not run.chemicals.empty:
        ids = run.chemicals.chemical_id.tolist()
        names = run.chemicals.set_index("chemical_id").chemical_name.to_dict()
        identity = st.selectbox("Chemical", ids, format_func=lambda key: f"{names[key]} · {key}")
        chem = run.chemicals.set_index("chemical_id").loc[identity]
        mol_col, data_col = st.columns([1, 2], gap="large")
        with mol_col:
            st.image(molecule_image(chem.smiles), width="stretch")
            with st.expander("SMILES"):
                st.code(chem.smiles, language=None, wrap_lines=True)
        with data_col:
            st.subheader(names[identity])
            st.write("Selected for review" if chem.selected_for_review else "Not selected at this workload")
            st.caption(f"Rank {int(chem['rank'])} of {run.manifest['eligible_chemicals']} · "
                       f"{run.manifest['review_count']} review "
                       f"{'place' if run.manifest['review_count'] == 1 else 'places'}")
            st.caption(chem.selection_basis)
            actual_rule = run.manifest["rule"]
            actual_score = chem.margin_log10 if actual_rule == "threshold_proximity" else chem[actual_rule]
            st.metric(SCORE_LABELS[actual_rule], f"{actual_score:.4g}", help=RULE_HELP[actual_rule])
            if chem.unseen_species:
                st.warning("At least one submitted species was absent from training.")
            if chem.rejected_case_count:
                st.warning(f"{chem.rejected_case_count} other cases for this chemical were rejected.")
            if chem.ids_sharing_representation > 1:
                st.warning("Other submitted IDs share this molecular representation. Confirm distinct substance identities.")
            st.altair_chart(endpoint_chart(run, identity), width="stretch", theme=None)
            if run.manifest["task"] == "hazard_omission":
                st.caption(f"Concentration criterion: {run.manifest['cutoff_mg_l']:g} mg/L")
            endpoint_view = run.endpoints[run.endpoints.chemical_id.eq(identity)][
                ["endpoint", "prediction_mg_l", "case_count", "block_knn", "similarity_ad", "ensemble_sd"]
            ].copy()
            endpoint_view["endpoint"] = endpoint_view.endpoint.replace(ENDPOINT_LABELS)
            st.dataframe(endpoint_view, hide_index=True, width="stretch", column_config={
                "endpoint": "Endpoint", "case_count": "Cases",
                "prediction_mg_l": st.column_config.NumberColumn("Median prediction (mg/L)", format="%.4g"),
                **{key: st.column_config.NumberColumn(SCORE_LABELS[key], format="%.4g", help=RULE_HELP[key])
                   for key in ("block_knn", "similarity_ad", "ensemble_sd")},
            })
        st.subheader("Case predictions and training neighbors")
        subset = run.cases[run.cases.chemical_id.eq(identity)]
        case_numbers = {case: i + 1 for i, case in enumerate(subset.case_id)}
        case_view = subset[["case_id", "species", "endpoint", "duration_h", "prediction_mg_l",
                            "block_knn", "similarity_ad", "ensemble_sd"]].copy()
        case_view["case_id"] = case_view.case_id.map(case_numbers)
        case_view["endpoint"] = case_view.endpoint.replace(ENDPOINT_LABELS)
        st.dataframe(case_view, hide_index=True, width="stretch", column_config={
            "case_id": "Case", "species": "Species", "endpoint": "Endpoint", "duration_h": "Duration (h)",
            "prediction_mg_l": st.column_config.NumberColumn("Prediction (mg/L)", format="%.4g"),
            **{key: st.column_config.NumberColumn(SCORE_LABELS[key], format="%.4g", help=RULE_HELP[key])
               for key in ("block_knn", "similarity_ad", "ensemble_sd")},
        })
        neighbor_view = run.neighbors[run.neighbors.case_id.isin(subset.case_id)][
            ["case_id", "neighbor_rank", "reference_chemical_name", "reference_species", "reference_endpoint", "distance"]
        ].copy()
        neighbor_view["case_id"] = neighbor_view.case_id.map(case_numbers)
        neighbor_view["reference_endpoint"] = neighbor_view.reference_endpoint.replace(ENDPOINT_LABELS)
        st.dataframe(neighbor_view, hide_index=True, width="stretch", column_config={
            "case_id": "Case", "neighbor_rank": "Neighbor", "reference_chemical_name": "Training chemical",
            "reference_species": "Training species", "reference_endpoint": "Training endpoint",
            "distance": st.column_config.NumberColumn("Distance", format="%.3f"),
        })
    else:
        st.info("No predicted chemical records.")
with checks_tab:
    st.subheader("Current input checks")
    if validation is None:
        st.info("No input cases.")
    elif validation["error"]:
        st.error(validation["error"])
    else:
        st.caption(f"{len(validation['accepted']):,} eligible cases · {eligible_count:,} eligible chemicals")
        if validation["rejected"].empty:
            st.success("No cases rejected.")
        else:
            st.dataframe(check_table(validation["rejected"]), hide_index=True, width="stretch")
            st.download_button("Download input issues", export_csv(validation["rejected"]),
                               "aquatriage_input_issues.csv", "text/csv", icon=":material/download:", on_click="ignore")
    if ignored:
        st.warning("Not used for prediction or ranking: " + ", ".join(ignored))
    if run is not None:
        if not run.rejected.empty:
            st.subheader("Rejected cases in completed run")
            st.dataframe(check_table(run.rejected), hide_index=True, width="stretch")
        if not run.cases.empty:
            st.subheader("Completed run: missing inputs and reference overlap")
            st.dataframe(check_table(run.cases[[
                "case_id", "chemical_id", "endpoint", "missing_input_count",
                "missing_bioactivity_count", "taxonomy_filled_from_reference",
                "unseen_species", "chemical_in_training",
                "ids_sharing_representation",
            ]]), hide_index=True, width="stretch")
with detail_tab:
    if run is not None:
        st.dataframe(run_summary(run), hide_index=True, width="stretch")
        st.download_button("Download run summary", summary_text(run), "aquatriage_summary.txt",
                           "text/plain", icon=":material/download:", disabled=stale, on_click="ignore")
        with st.expander("Machine-readable run record"):
            st.json(run.manifest, expanded=False)
    elif reference is not None:
        st.json(reference.manifest, expanded=False)
