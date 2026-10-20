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
from rdkit import Chem
from rdkit.Chem import Draw
import streamlit as st

from ecoood.review_workflow import (
    RULES, RULE_NAMES, export_run, fit_reference,
    load_reference, read_input_csv, run_review,
    template_csv,
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
ENDPOINT_LABELS = {
    "fish_96h_lc50": "Fish LC50 (96 h)",
    "daphnia_48h_ec50": "Daphnid EC50 (48 h)",
    "algae_72_96h_ec50": "Algal EC50 (72-96 h)",
}


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
        input_mode = st.segmented_control("Input source", ["Example queue", "Upload CSV"],
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
        frame = None
        try:
            frame = load_queue(upload, input_mode)
        except ValueError as exc:
            st.error(str(exc))
        if frame is not None:
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
                                  help="Accuracy: all eligible chemicals. Omissions: predicted-low-concern chemicals first.")
        task = "general_error" if objective == "Check concentration accuracy" else "hazard_omission"
        rule = st.selectbox("Ranking", RULES[task], format_func=RULE_NAMES.get,
                            help="The first option is the task preset. Alternatives use the same queue and quota.")
        quota_mode = st.segmented_control("Review workload", ["Percentage", "Chemical count"], default="Percentage")
        fraction, count = None, None
        if quota_mode == "Percentage":
            fraction = st.slider("Eligible chemicals to review (%)", 0, 100, 25, 1) / 100
        else:
            count = st.number_input("Chemicals to review", min_value=0, value=5, step=1)
        cutoff = st.number_input("Acute-effect concentration criterion (mg/L)", min_value=0.000001,
                                  value=1.0, format="%.6g", disabled=task == "general_error",
                                  help="1 mg/L is the study reference, not a complete regulatory hazard classification.")
        active_config = {
            "queue": hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest() if frame is not None else None,
            "reference": reference.manifest["reference_sha256"] if reference else None,
            "task": task, "rule": rule, "fraction": fraction, "count": count, "cutoff": cutoff,
        }
        if st.button("Run review", type="primary", icon=":material/play_arrow:",
                     use_container_width=True, disabled=frame is None or reference is None):
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
        metrics = st.columns(4)
        metrics[0].metric("Eligible chemicals", run.manifest["eligible_chemicals"])
        metrics[1].metric("Selected", run.manifest["review_count"])
        metrics[2].metric("Rejected cases", run.manifest["rejected_records"])
        metrics[3].metric("Review fraction", f"{(run.manifest['realized_review_fraction'] or 0):.1%}")
        st.download_button("Download complete results", export_run(run), "aquatriage_review.zip",
                           "application/zip", icon=":material/download:", disabled=stale, on_click="ignore")
        if not run.chemicals.empty:
            search = st.text_input("Find a chemical", placeholder="Name or identifier")
            selected_only = st.toggle("Selected only")
            shown = run.chemicals.copy()
            if selected_only:
                shown = shown[shown.selected_for_review]
            if search:
                shown = shown[shown.chemical_name.str.contains(search, case=False, regex=False) |
                              shown.chemical_id.str.contains(search, case=False, regex=False)]
            cols = ["rank", "selected_for_review", "chemical_name", "prediction_mg_l",
                    "endpoint_count", "block_knn", "similarity_ad", "ensemble_sd", "selection_basis"]
            st.dataframe(shown[cols], hide_index=True, width="stretch", height=480,
                         column_config={
                             "prediction_mg_l": st.column_config.NumberColumn("Lowest endpoint median (mg/L)", format="%.4g"),
                             "selected_for_review": st.column_config.CheckboxColumn("Review"),
                             "rank": "Rank", "chemical_name": "Chemical", "endpoint_count": "Endpoints",
                             "selection_basis": "Ranking basis",
                             "block_knn": st.column_config.NumberColumn("Block kNN", format="%.3f"),
                             "similarity_ad": st.column_config.NumberColumn("Similarity AD", format="%.3f"),
                             "ensemble_sd": st.column_config.NumberColumn("Ensemble SD", format="%.3f"),
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
            st.caption(chem.selection_basis)
            if chem.unseen_species:
                st.warning("At least one submitted species was absent from training.")
            if chem.rejected_case_count:
                st.warning(f"{chem.rejected_case_count} other cases for this chemical were rejected.")
            if chem.ids_sharing_representation > 1:
                st.warning("Other submitted IDs share this molecular representation. Confirm distinct substance identities.")
            endpoint_view = run.endpoints[run.endpoints.chemical_id.eq(identity)][
                ["endpoint", "prediction_mg_l", "case_count", "block_knn", "similarity_ad", "ensemble_sd"]
            ].copy()
            endpoint_view["endpoint"] = endpoint_view.endpoint.replace(ENDPOINT_LABELS)
            st.dataframe(endpoint_view, hide_index=True, width="stretch", column_config={
                "endpoint": "Endpoint", "case_count": "Cases",
                "prediction_mg_l": st.column_config.NumberColumn("Median prediction (mg/L)", format="%.4g"),
                "block_knn": st.column_config.NumberColumn("Block kNN", format="%.3f"),
                "similarity_ad": st.column_config.NumberColumn("Similarity AD", format="%.3f"),
                "ensemble_sd": st.column_config.NumberColumn("Ensemble SD", format="%.3f"),
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
            "block_knn": st.column_config.NumberColumn("Block kNN", format="%.3f"),
            "similarity_ad": st.column_config.NumberColumn("Similarity AD", format="%.3f"),
            "ensemble_sd": st.column_config.NumberColumn("Ensemble SD", format="%.3f"),
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
    if run is not None:
        st.subheader("Rejected cases")
        if run.rejected.empty:
            st.success("No cases rejected.")
        else:
            st.dataframe(run.rejected, hide_index=True, width="stretch")
        if not run.cases.empty:
            st.subheader("Missing inputs and reference overlap")
            st.dataframe(run.cases[[
                "case_id", "chemical_id", "endpoint", "missing_input_count",
                "missing_bioactivity_count", "taxonomy_filled_from_reference",
                "unseen_species", "chemical_in_training",
                "ids_sharing_representation",
            ]], hide_index=True, width="stretch")
    else:
        st.info("No completed input checks.")
with detail_tab:
    if run is not None:
        st.json(run.manifest, expanded=False)
    elif reference is not None:
        st.json(reference.manifest, expanded=False)
