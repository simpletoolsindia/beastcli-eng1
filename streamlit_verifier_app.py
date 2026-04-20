from __future__ import annotations

import subprocess
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from dataset_verifier import (
    DEFAULT_OLLAMA_URL,
    discover_jsonl_files,
    list_ollama_models,
    verify_dataset,
)
from ollama_generation_loop import run_ollama_generation_loop


st.set_page_config(
    page_title="Dataset Verifier",
    page_icon="✅",
    layout="wide",
)

st.title("Dataset Verifier")
st.caption("Review JSONL dataset rows with a local Ollama model and export a CSV of incorrect samples.")

if "verification_rows" not in st.session_state:
    st.session_state.verification_rows = []
if "csv_path" not in st.session_state:
    st.session_state.csv_path = ""
if "selected_dataset" not in st.session_state:
    st.session_state.selected_dataset = ""
if "last_generated_dataset" not in st.session_state:
    st.session_state.last_generated_dataset = ""
if "ollama_loop_report" not in st.session_state:
    st.session_state.ollama_loop_report = {}
if "ollama_loop_rejections" not in st.session_state:
    st.session_state.ollama_loop_rejections = []


def _save_upload(uploaded_file) -> Path:
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    destination = upload_dir / uploaded_file.name
    destination.write_bytes(uploaded_file.getbuffer())
    return destination


def _generate_dataset(count: int, seed: int, output_name: str) -> tuple[Path, str]:
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = output_name.strip() or "generated_dataset"
    if not safe_name.endswith(".jsonl"):
        safe_name = f"{safe_name}.jsonl"
    output_path = output_dir / safe_name

    command = [
        "python3",
        "regenerate_dataset.py",
        "--count",
        str(count),
        "--output",
        str(output_path),
        "--seed",
        str(seed),
    ]
    result = subprocess.run(
        command,
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        check=False,
    )
    output_text = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
    if result.returncode != 0:
        raise RuntimeError(output_text.strip() or "Dataset generation failed.")
    return output_path, output_text.strip()


with st.sidebar:
    st.header("Source")
    existing_files = discover_jsonl_files()
    uploaded_file = st.file_uploader("Upload JSONL", type=["jsonl"])
    selected_existing = st.selectbox(
        "Or choose existing JSONL",
        [""] + [str(path) for path in existing_files],
        index=([""] + [str(path) for path in existing_files]).index(st.session_state.selected_dataset)
        if st.session_state.selected_dataset in [""] + [str(path) for path in existing_files]
        else 0,
        key="dataset_selectbox",
    )

    st.header("Generate dataset")
    generate_count = st.number_input("Records", min_value=1, value=100, step=1)
    generate_seed = st.number_input("Seed", min_value=1, value=2030, step=1)
    generate_name = st.text_input("Output name", value="generated_from_ui.jsonl")
    if st.button("Run dataset generator"):
        with st.spinner("Generating dataset..."):
            try:
                generated_path, generation_log = _generate_dataset(
                    count=int(generate_count),
                    seed=int(generate_seed),
                    output_name=generate_name,
                )
            except Exception as exc:
                st.error(str(exc))
            else:
                st.session_state.selected_dataset = str(generated_path)
                st.session_state.last_generated_dataset = generation_log
                st.success(f"Generated dataset: {generated_path}")
                st.rerun()

    st.header("Ollama generation loop")
    ollama_loop_count = st.number_input("Accepted rows target", min_value=1, value=100, step=1)
    ollama_loop_seed = st.number_input("Loop seed", min_value=1, value=2120, step=1)
    ollama_loop_name = st.text_input("Loop output name", value="ollama_loop_generated.jsonl")
    ollama_loop_rounds = st.number_input("Max rounds", min_value=1, value=6, step=1)
    ollama_loop_multiplier = st.number_input("Candidate multiplier", min_value=1.0, value=1.5, step=0.1)
    ollama_loop_batch = st.number_input("Min batch size", min_value=1, value=25, step=1)

    st.header("Ollama")
    base_url = st.text_input("Ollama URL", value=DEFAULT_OLLAMA_URL)
    model = st.text_input("Model name", value="gemma4:latest")
    if st.button("Fetch local models"):
        try:
            models = list_ollama_models(base_url)
            if models:
                st.success("Available models: " + ", ".join(models))
            else:
                st.warning("No Ollama models found.")
        except Exception as exc:
            st.error(str(exc))

    st.header("Run options")
    max_rows = st.number_input("Max rows", min_value=1, value=50, step=1)
    only_incorrect = st.checkbox("Export only incorrect rows", value=True)

input_path = None
if uploaded_file is not None:
    input_path = _save_upload(uploaded_file)
    st.session_state.selected_dataset = str(input_path)
elif selected_existing:
    input_path = Path(selected_existing)
    st.session_state.selected_dataset = selected_existing
elif st.session_state.selected_dataset:
    input_path = Path(st.session_state.selected_dataset)

if input_path:
    st.info(f"Selected dataset: {input_path}")
else:
    st.warning("Upload a JSONL file or choose one from the repo to start verification.")

if st.session_state.last_generated_dataset:
    with st.expander("Last generator output"):
        st.code(st.session_state.last_generated_dataset, language="text")

run_button = st.button("Run verification", type="primary", disabled=input_path is None)
run_ollama_loop_button = st.button("Run Ollama generation loop", disabled=not model.strip())

if run_button and input_path is not None:
    progress = st.progress(0.0)
    status = st.empty()

    def on_progress(current: int, total: int, result) -> None:
        progress.progress(current / max(total, 1))
        label = "correct" if result.is_correct else "issue found"
        status.write(f"Reviewed row {current}/{total}: {label}")

    try:
        results, csv_path = verify_dataset(
            input_path=input_path,
            model=model,
            base_url=base_url,
            max_rows=int(max_rows),
            only_incorrect=only_incorrect,
            progress_callback=on_progress,
        )
    except Exception as exc:
        st.error(str(exc))
    else:
        rows = []
        for result in results:
            for row in result.to_rows():
                rows.append(row)
        st.session_state.verification_rows = rows
        st.session_state.csv_path = str(csv_path)
        st.success(f"Verification finished. CSV saved to {csv_path}")

if run_ollama_loop_button:
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    loop_output_name = ollama_loop_name.strip() or "ollama_loop_generated.jsonl"
    if not loop_output_name.endswith(".jsonl"):
        loop_output_name = f"{loop_output_name}.jsonl"
    loop_output_path = output_dir / loop_output_name

    st.subheader("Ollama loop progress")
    overall_progress = st.progress(0.0)
    review_progress = st.progress(0.0)
    round_status = st.empty()
    review_status = st.empty()
    metrics_placeholder = st.empty()
    event_log_placeholder = st.empty()
    event_logs: list[str] = []

    def _refresh_logs() -> None:
        joined = "\n".join(event_logs[-12:])
        event_log_placeholder.code(joined or "Waiting for loop events...", language="text")

    def on_round_event(event: dict) -> None:
        event_type = event.get("event")
        if event_type == "round_start":
            round_number = event["round_number"]
            max_rounds = event["max_rounds"]
            overall_progress.progress((round_number - 1) / max(max_rounds, 1))
            round_status.info(
                f"Round {round_number}/{max_rounds}: generating {event['requested_candidates']} candidates "
                f"with seed {event['seed']} to fill {event['remaining']} remaining rows."
            )
            event_logs.append(
                f"Round {round_number} started: need {event['remaining']} rows, "
                f"generating {event['requested_candidates']} candidates."
            )
            _refresh_logs()
        elif event_type == "generation_complete":
            review_progress.progress(0.0)
            event_logs.append(
                f"Round {event['round_number']} generation complete: {event['generated_candidates']} candidates ready for LLM review."
            )
            _refresh_logs()
        elif event_type == "round_complete":
            summary = event["summary"]
            completed_round = summary["round_number"]
            overall_progress.progress(completed_round / max(int(ollama_loop_rounds), 1))
            metrics_placeholder.metric("Accepted rows", f"{event['accepted_total']}/{event['target_count']}")
            event_logs.append(
                f"Round {completed_round} complete: accepted {summary['accepted_in_round']}, "
                f"rejected {summary['rejected_in_round']}, total accepted {summary['accepted_total']}."
            )
            _refresh_logs()
        elif event_type == "loop_complete":
            overall_progress.progress(1.0)
            report = event["report"]
            metrics_placeholder.metric("Accepted rows", f"{report['accepted_count']}/{report['target_count']}")
            event_logs.append(
                f"Loop complete: accepted {report['accepted_count']} rows, rejected {report['rejected_count']} rows."
            )
            _refresh_logs()

    def on_review_event(event: dict) -> None:
        if event.get("event") != "review_progress":
            return
        review_progress.progress(event["review_index"] / max(event["review_total"], 1))
        label = "accepted" if event["is_correct"] else "rejected"
        review_status.write(
            f"Round {event['round_number']} review {event['review_index']}/{event['review_total']}: "
            f"{label} row {event['row_number']} ({event['tool_name']})"
        )
        event_logs.append(
            f"Reviewed row {event['row_number']} ({event['tool_name']}): {label}. {event['summary']}"
        )
        _refresh_logs()

    try:
        report = run_ollama_generation_loop(
            target_count=int(ollama_loop_count),
            output_path=loop_output_path,
            model=model,
            base_url=base_url,
            seed=int(ollama_loop_seed),
            max_rounds=int(ollama_loop_rounds),
            candidate_multiplier=float(ollama_loop_multiplier),
            min_batch_size=int(ollama_loop_batch),
            keep_candidates=True,
            round_callback=on_round_event,
            review_progress_callback=on_review_event,
        )
    except Exception as exc:
        st.error(str(exc))
    else:
        st.session_state.ollama_loop_report = report
        st.session_state.selected_dataset = str(loop_output_path)
        rejected_csv_path = Path(report["rejected_csv_path"])
        if rejected_csv_path.exists():
            st.session_state.ollama_loop_rejections = pd.read_csv(rejected_csv_path).to_dict(orient="records")
        else:
            st.session_state.ollama_loop_rejections = []
        st.success(f"Ollama generation loop finished. Accepted dataset: {loop_output_path}")
        st.info(f"Selected dataset updated to: {loop_output_path}")


if st.session_state.verification_rows:
    st.subheader("Verification results")
    df = pd.DataFrame(st.session_state.verification_rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    csv_path = Path(st.session_state.csv_path)
    if csv_path.exists():
        st.download_button(
            "Download CSV",
            data=csv_path.read_bytes(),
            file_name=csv_path.name,
            mime="text/csv",
        )

    incorrect_rows = df["row_number"].nunique() if not df.empty else 0
    st.metric("Rows flagged", incorrect_rows)

if st.session_state.ollama_loop_report:
    st.subheader("Ollama loop report")
    report = st.session_state.ollama_loop_report
    col1, col2, col3 = st.columns(3)
    col1.metric("Accepted", report.get("accepted_count", 0))
    col2.metric("Rejected", report.get("rejected_count", 0))
    col3.metric("Rounds", len(report.get("rounds", [])))

    rounds = report.get("rounds", [])
    if rounds:
        st.dataframe(pd.DataFrame(rounds), use_container_width=True, hide_index=True)

    with st.expander("Loop report JSON"):
        st.code(json.dumps(report, indent=2), language="json")

    if st.session_state.ollama_loop_rejections:
        st.subheader("Ollama rejected rows")
        st.dataframe(pd.DataFrame(st.session_state.ollama_loop_rejections), use_container_width=True, hide_index=True)
