import subprocess
import shlex
import time
from pathlib import Path

import pandas as pd
import streamlit as st

# =========================
# Page settings
# =========================
st.set_page_config(
    page_title="Ellipse Detection UI",
    layout="wide"
)

st.title("Ellipse Detection Project UI")
st.caption("Run each pipeline step with buttons, and preview outputs.")

# =========================
# Project root
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# =========================
# Helpers
# =========================
def run_command_live(command: str):
    """
    Run shell command and stream stdout/stderr to UI.
    """
    st.code(command, language="bash")

    stdout_placeholder = st.empty()
    stderr_placeholder = st.empty()
    status_placeholder = st.empty()

    stdout_lines = []
    stderr_lines = []

    with st.spinner("Command is running..."):
        process = subprocess.Popen(
            shlex.split(command),
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1
        )

        while True:
            stdout_line = process.stdout.readline() if process.stdout else ""
            stderr_line = process.stderr.readline() if process.stderr else ""

            if stdout_line:
                stdout_lines.append(stdout_line.rstrip())
                stdout_placeholder.subheader("STDOUT")
                stdout_placeholder.text("\n".join(stdout_lines[-50:]))

            if stderr_line:
                stderr_lines.append(stderr_line.rstrip())
                stderr_placeholder.subheader("STDERR")
                stderr_placeholder.text("\n".join(stderr_lines[-50:]))

            if stdout_line == "" and stderr_line == "" and process.poll() is not None:
                break

            time.sleep(0.05)

        return_code = process.poll()

    if return_code == 0:
        status_placeholder.success("Command finished successfully.")
    else:
        status_placeholder.error(f"Command failed with return code {return_code}")

    # flush remaining
    if process.stdout:
        rest_out = process.stdout.read()
        if rest_out:
            stdout_lines.extend(rest_out.splitlines())
    if process.stderr:
        rest_err = process.stderr.read()
        if rest_err:
            stderr_lines.extend(rest_err.splitlines())

    if stdout_lines:
        stdout_placeholder.subheader("STDOUT")
        stdout_placeholder.text("\n".join(stdout_lines[-100:]))

    if stderr_lines:
        stderr_placeholder.subheader("STDERR")
        stderr_placeholder.text("\n".join(stderr_lines[-100:]))

    return return_code


def section_title(text: str):
    st.markdown(f"## {text}")


def find_recent_images(folder: Path, limit: int = 6):
    if not folder.exists():
        return []

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".PNG", ".JPG", ".JPEG", ".BMP", ".WEBP"}
    files = [p for p in folder.rglob("*") if p.is_file() and p.suffix in exts]
    files = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
    return files[:limit]


def find_recent_csv(folder: Path):
    if not folder.exists():
        return None

    csv_files = list(folder.rglob("*.csv"))
    if not csv_files:
        return None

    csv_files = sorted(csv_files, key=lambda p: p.stat().st_mtime, reverse=True)
    return csv_files[0]


def show_image_preview(folder: Path, title: str, limit: int = 6):
    st.subheader(title)

    images = find_recent_images(folder, limit=limit)
    if not images:
        st.info(f"No images found in: {folder}")
        return

    cols = st.columns(3)
    for i, img_path in enumerate(images):
        with cols[i % 3]:
            st.image(str(img_path), caption=img_path.name, use_container_width=True)


def show_csv_preview(folder: Path, title: str, nrows: int = 20):
    st.subheader(title)

    csv_path = find_recent_csv(folder)
    if csv_path is None:
        st.info(f"No CSV found in: {folder}")
        return

    st.write(f"CSV: `{csv_path}`")
    try:
        df = pd.read_csv(csv_path)
        st.dataframe(df.head(nrows), use_container_width=True)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")


# =========================
# Sidebar
# =========================
st.sidebar.header("Common Parameters")

redenhance_version = st.sidebar.text_input("RedEnhance version", value="default")
classify_run_name = st.sidebar.text_input("Classify run name", value="clf_v001_on_default")
dataset_name = st.sidebar.text_input("Segmentation dataset name", value="seg_dataset_v001")
seg_train_run_name = st.sidebar.text_input("Segmentation train run name", value="seg_v001_on_seg_dataset_v001")
seg_infer_run_name = st.sidebar.text_input("Segmentation inference run name", value="seginf_v001_on_clf_v001_on_default")
ellipse_run_name = st.sidebar.text_input("Ellipse run name", value="ellipse_v001")
ellipse_compare_run_name = st.sidebar.text_input("Ellipse compare run name", value="ellipse_compare_v001")
pipeline_run_name = st.sidebar.text_input("Pipeline run name", value="pipeline_run_v001")

patient_ids_text = st.sidebar.text_input("Patient IDs (space separated)", value="01")
patient_ids = patient_ids_text.strip()

max_train_images = st.sidebar.number_input("Max train images", min_value=1, value=120)
max_val_images = st.sidebar.number_input("Max val images", min_value=1, value=40)
max_test_images = st.sidebar.number_input("Max test images", min_value=1, value=40)

batch_size = st.sidebar.number_input("Batch size", min_value=1, value=4)
epochs = st.sidebar.number_input("Epochs", min_value=1, value=30)
lr = st.sidebar.text_input("Learning rate", value="1e-3")

# =========================
# Layout
# =========================
tab_run, tab_preview = st.tabs(["Run Commands", "Preview Outputs"])

with tab_run:
    col1, col2 = st.columns(2)

    with col1:
        section_title("1. Preprocessing")

        if st.button("Run RedEnhance", use_container_width=True):
            cmd = "python -m src.preprocessing.redenhance"
            run_command_live(cmd)

        section_title("2. Classify")

        if st.button("Run Classify Inference", use_container_width=True):
            cmd = (
                f"python -m src.classify.infer_classifier "
                f"--redenhance_version {redenhance_version} "
                f"--run_name {classify_run_name}"
            )
            run_command_live(cmd)

        section_title("3. Segmentation Dataset")

        if st.button("Prepare Segmentation Dataset", use_container_width=True):
            cmd = (
                f"python -m src.segmentation.prepare_segmentation_images "
                f"--classify_run_name {classify_run_name} "
                f"--dataset_name {dataset_name} "
                f"--max_train_images {max_train_images} "
                f"--max_val_images {max_val_images} "
                f"--max_test_images {max_test_images}"
            )
            run_command_live(cmd)

        if st.button("Convert JSON to Mask", use_container_width=True):
            cmd = (
                f"python -m src.segmentation.json_to_mask "
                f"--dataset_name {dataset_name}"
            )
            run_command_live(cmd)

    with col2:
        section_title("4. Segmentation")

        if st.button("Train Segmentation", use_container_width=True):
            cmd = (
                f"python -m src.segmentation.train_segmentation "
                f"--dataset_name {dataset_name} "
                f"--run_name {seg_train_run_name} "
                f"--batch_size {batch_size} "
                f"--epochs {epochs} "
                f"--lr {lr}"
            )
            run_command_live(cmd)

        if st.button("Run Segmentation Inference", use_container_width=True):
            cmd = (
                f"python -m src.segmentation.infer_segmentation "
                f"--classify_run_name {classify_run_name} "
                f"--run_name {seg_infer_run_name}"
            )
            run_command_live(cmd)

        section_title("5. Ellipse")

        if st.button("Run Ellipse Fit", use_container_width=True):
            cmd = (
                f"python -m src.ellipse.fit_ellipse "
                f"--segmentation_run_name {seg_infer_run_name} "
                f"--redenhance_version {redenhance_version} "
                f"--run_name {ellipse_run_name}"
            )
            run_command_live(cmd)

        if st.button("Compare Ellipse GT vs Pred", use_container_width=True):
            cmd = (
                f"python -m src.ellipse.compare_ellipse_gt_pred "
                f"--segmentation_train_run_name {seg_train_run_name} "
                f"--redenhance_version {redenhance_version} "
                f"--run_name {ellipse_compare_run_name}"
            )
            run_command_live(cmd)

        section_title("6. Full Pipeline")

        if st.button("Run Full Pipeline", use_container_width=True):
            cmd = (
                f"python -m src.pipeline.main "
                f"--patient_ids {patient_ids} "
                f"--run_name {pipeline_run_name}"
            )
            run_command_live(cmd)

with tab_preview:
    st.markdown("## Quick Preview")

    preview_col1, preview_col2 = st.columns(2)

    with preview_col1:
        st.markdown("### RedEnhance")
        redenhance_folder = PROJECT_ROOT / "data" / "processed" / "redenhance" / redenhance_version
        show_image_preview(redenhance_folder, "Recent RedEnhance Images")
        show_csv_preview(redenhance_folder, "Recent RedEnhance CSV")

        st.markdown("### Classify")
        classify_folder = PROJECT_ROOT / "data" / "processed" / "classify_outputs" / classify_run_name
        show_image_preview(classify_folder, "Recent Classify Images")
        show_csv_preview(classify_folder, "Recent Classify CSV")

        st.markdown("### Segmentation Dataset")
        seg_dataset_folder = PROJECT_ROOT / "data" / "processed" / "segmentation_dataset" / dataset_name
        show_image_preview(seg_dataset_folder, "Recent Segmentation Dataset Images")
        show_csv_preview(seg_dataset_folder, "Recent Segmentation Dataset CSV")

    with preview_col2:
        st.markdown("### Segmentation Inference")
        seg_infer_folder = PROJECT_ROOT / "data" / "processed" / "segmentation_inference" / seg_infer_run_name
        show_image_preview(seg_infer_folder, "Recent Segmentation Inference Images")
        show_csv_preview(seg_infer_folder, "Recent Segmentation Inference CSV")

        st.markdown("### Ellipse Fit")
        ellipse_folder = PROJECT_ROOT / "data" / "processed" / "ellipse_outputs" / ellipse_run_name
        show_image_preview(ellipse_folder, "Recent Ellipse Images")
        show_csv_preview(ellipse_folder, "Recent Ellipse CSV")

        st.markdown("### Ellipse Compare")
        ellipse_compare_folder = PROJECT_ROOT / "data" / "processed" / "ellipse_outputs" / ellipse_compare_run_name
        show_image_preview(ellipse_compare_folder, "Recent Ellipse Compare Images")
        show_csv_preview(ellipse_compare_folder, "Recent Ellipse Compare CSV")

    st.markdown("### Full Pipeline")
    pipeline_folder = PROJECT_ROOT / "data" / "processed" / "pipeline_runs" / pipeline_run_name
    show_image_preview(pipeline_folder, "Recent Pipeline Images")
    show_csv_preview(pipeline_folder, "Recent Pipeline CSV")

# =========================
# Footer
# =========================
st.markdown("---")
st.subheader("How to start this UI")
st.code("streamlit run src/ui/app.py", language="bash")

st.subheader("Notes")
st.markdown(
    """
- This version shows live command output.
- It also previews recent output images and CSV files.
- Patient IDs should be entered with spaces, for example: `01 02 04`
- If preview looks empty, make sure that run name and version name match actual output folders.
"""
)