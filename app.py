import streamlit as st
st.set_page_config(page_title="Lumber Cut Optimizer", layout="wide")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import io
import csv
import json
import yaml
from fractions import Fraction

# ---- Helper Functions ----
def parse_measurement(value):
    if isinstance(value, (int, float)):
        return float(value)
    try:
        parts = str(value).strip().split()
        if len(parts) == 2:
            whole = float(parts[0])
            frac = float(Fraction(parts[1]))
            return whole + frac
        else:
            return float(Fraction(parts[0]))
    except:
        return None

def calculate_board_feet(length, width, quantity, thickness=0.75):
    area = length * width * thickness
    return (area * quantity) / 144

def generate_required_pieces(required_df):
    pieces = []
    for _, row in required_df.iterrows():
        try:
            quantity = int(parse_measurement(row.get('Quantity', 1)))
        except:
            quantity = 1
        length = parse_measurement(row.get('Length'))
        width = parse_measurement(row.get('Width'))
        project = row.get('Project', '')
        if length is None or width is None:
            continue
        for _ in range(quantity):
            pieces.append({
                'length': length,
                'width': width,
                'id': f"{length:.3f}x{width:.3f}",
                'project': project
            })
    return sorted(pieces, key=lambda x: max(x['length'], x['width']), reverse=True)

def expand_boards_by_quantity(boards_df):
    expanded = []
    for _, row in boards_df.iterrows():
        try:
            quantity = int(parse_measurement(row.get('Quantity', 1)))
        except:
            quantity = 1
        length = parse_measurement(row.get('Length'))
        width = parse_measurement(row.get('Width'))
        project = row.get('Project', '')
        if length is None or width is None:
            continue
        for _ in range(quantity):
            expanded.append({'length': length, 'width': width, 'project': project})
    return expanded

# --- Optimizing Packer ---
def try_place_pieces(board, pieces, kerf):
    free_rectangles = [{'x': 0, 'y': 0, 'length': board['length'], 'width': board['width']}]
    placements = []
    remaining = []

    for piece in pieces:
        placed = False
        for rect in free_rectangles:
            for rotated in [False, True]:
                p_length = piece['length'] + kerf
                p_width = piece['width'] + kerf
                if rotated:
                    p_length, p_width = p_width, p_length
                if p_length <= rect['length'] and p_width <= rect['width']:
                    placements.append({
                        'piece': piece,
                        'x': rect['x'],
                        'y': rect['y'],
                        'length': p_length - kerf,
                        'width': p_width - kerf,
                        'rotated': rotated
                    })
                    new_rects = [
                        {'x': rect['x'] + p_length, 'y': rect['y'], 'length': rect['length'] - p_length, 'width': p_width},
                        {'x': rect['x'], 'y': rect['y'] + p_width, 'length': rect['length'], 'width': rect['width'] - p_width}
                    ]
                    free_rectangles.remove(rect)
                    free_rectangles.extend([r for r in new_rects if r['length'] > 0 and r['width'] > 0])
                    placed = True
                    break
            if placed:
                break
        if not placed:
            remaining.append(piece)
    return placements, remaining

def fit_pieces_to_boards(boards_list, required_df, kerf):
    pieces = generate_required_pieces(required_df)
    cut_plan = []
    board_id = 1
    for board in boards_list:
        placements, pieces = try_place_pieces(board, pieces, kerf)
        used_area = sum(p['length'] * p['width'] for p in placements)
        total_area = board['length'] * board['width']
        waste_area = total_area - used_area
        cut_plan.append({
            'board_id': board_id,
            'board': board,
            'cuts': placements,
            'waste_area': waste_area
        })
        board_id += 1
        if not pieces:
            break
    return cut_plan, pieces

# ---- Plan Persistence Functions ----
def save_plan_to_json(plan, leftovers, boards_df, required_df):
    data = {
        'cut_plan': plan,
        'leftovers': leftovers,
        'boards_input': boards_df.to_dict(orient='records'),
        'required_input': required_df.to_dict(orient='records')
    }
    return json.dumps(data, indent=2)

def save_plan_to_yaml(plan, leftovers, boards_df, required_df):
    data = {
        'cut_plan': plan,
        'leftovers': leftovers,
        'boards_input': boards_df.to_dict(orient='records'),
        'required_input': required_df.to_dict(orient='records')
    }
    return yaml.dump(data)

# Sidebar: Cut Settings and Plan Persistence Options
st.sidebar.header("Cut Settings")
kerf = st.sidebar.number_input("Kerf Size (inches)", value=0.125, step=0.001, format="%.3f")
thickness = st.sidebar.number_input("Board Thickness (inches)", value=0.75, step=0.01)
cost_per_bf = st.sidebar.number_input("Cost per Board Foot ($)", value=5.00, step=0.01)

st.sidebar.header("Plan Persistence Options")
file_format = st.sidebar.radio("Select File Format", options=["JSON", "YAML"])
save_plan_button = st.sidebar.button("Save Plan")
load_file = st.sidebar.file_uploader("Load Plan File", type=["json", "yaml", "yml"])

# ---- Output Enhancements ----
def generate_csv(cut_plan):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Board ID', 'Project', 'Piece ID', 'Length', 'Width', 'Rotated', 'X', 'Y'])
    for board in cut_plan:
        project = board['board'].get('project', '')
        for cut in board['cuts']:
            writer.writerow([
                board['board_id'],
                project,
                cut['piece']['id'],
                f"{cut['length']:.3f}",
                f"{cut['width']:.3f}",
                cut['rotated'],
                round(cut['x'], 2),
                round(cut['y'], 2)
            ])
    output.seek(0)
    return output.getvalue()

# ---- PDF & Preview Drawing ----
def to_fraction_string(value):
    try:
        frac = Fraction(value).limit_denominator(16)
        if frac.denominator == 1:
            return f"{frac.numerator}"
        else:
            whole = frac.numerator // frac.denominator
            remainder = frac - whole
            if whole > 0 and remainder:
                return f"{whole} {remainder.numerator}/{remainder.denominator}"
            else:
                return f"{frac.numerator}/{frac.denominator}"
    except Exception:
        return str(value)

def create_board_preview(board, job_title=""):
    fig, ax = plt.subplots(figsize=(8.5, 11))
    b = board['board']
    ax.set_title(f"{job_title}\nBoard {board['board_id']} - Project: {b.get('project', '')}", fontsize=12)
    ax.set_xlim(0, b['length'])
    ax.set_ylim(0, b['width'])
    ax.set_aspect('equal')
    for cut in board['cuts']:
        rect = patches.Rectangle((cut['x'], cut['y']), cut['length'], cut['width'], edgecolor='black', facecolor='lightgrey')
        ax.add_patch(rect)
        ax.text(cut['x'] + cut['length']/2, cut['y'] + cut['width']/2,
                f"{to_fraction_string(cut['piece']['length'])} x {to_fraction_string(cut['piece']['width'])}",
                ha='center', va='center', fontsize=8, color='red')
    ax.axis('off')
    return fig

def generate_pdf(cut_plan, leftovers=None, job_title=""):
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        for board in cut_plan:
            fig = create_board_preview(board, job_title)
            pdf.savefig(fig)
            plt.close(fig)
        if leftovers:
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.set_title(f"{job_title}\nLeftover Pieces", fontsize=12)
            ax.axis('off')
            lines = [f"Length       Width", "-------------------"]
            for p in leftovers:
                lines.append(f"{to_fraction_string(p['length']).rjust(12)} {to_fraction_string(p['width']).rjust(8)}")
            ax.text(0, 1, "\n".join(lines), fontsize=10, family='monospace', va='top')
            pdf.savefig(fig)
            plt.close(fig)
    buffer.seek(0)
    return buffer

# ---- Streamlit UI ----
st.title("📐 Lumber Cut Optimizer")

job_title = st.text_input("Job Title", "My Woodworking Project")
st.markdown(f"### Job Title: {job_title}")

st.subheader("Available Lumber")
def default_board_df():
    return pd.DataFrame([{"Length": "96", "Width": "12", "Quantity": 1, "Project": ""}])
boards_df = st.data_editor(
    st.session_state.get('boards_df', default_board_df()),
    num_rows="dynamic", use_container_width=True
)

st.subheader("Required Cuts")
def default_cut_df():
    return pd.DataFrame([{"Length": "24", "Width": "6", "Quantity": 2, "Project": ""}])
required_df = st.data_editor(
    st.session_state.get('required_df', default_cut_df()),
    num_rows="dynamic", use_container_width=True)

if st.button("✂️ Optimize Cuts"):
    boards_list = expand_boards_by_quantity(boards_df)
    cut_plan, leftovers = fit_pieces_to_boards(boards_list, required_df, kerf=kerf)
    st.session_state.cut_plan = cut_plan
    st.session_state.leftovers = leftovers
    st.session_state.boards_df = boards_df
    st.session_state.required_df = required_df

    total_bf = sum(
        calculate_board_feet(b['board']['length'], b['board']['width'], 1, thickness)
        for b in cut_plan
    )
    total_cost = total_bf * cost_per_bf
    st.success(f"Optimization complete! 🧮 Total board feet: {total_bf:.2f}, Estimated Cost: ${total_cost:.2f}")

    csv_data = generate_csv(cut_plan)
    pdf_data = generate_pdf(cut_plan, leftovers, job_title=job_title)

    st.subheader("Cut Plan Image Previews")
    for board in cut_plan:
        preview_fig = create_board_preview(board, job_title)
        st.pyplot(preview_fig)

    st.download_button("📄 Download CSV", csv_data, file_name="cut_plan.csv", mime="text/csv")
    st.download_button("📄 Download PDF", pdf_data, file_name="cut_plan.pdf")
    if leftovers:
        st.warning("Some pieces could not be placed. Check the PDF for suggestions.")

if save_plan_button and 'cut_plan' in st.session_state:
    if file_format == "JSON":
        saved_data = save_plan_to_json(
            st.session_state.cut_plan,
            st.session_state.leftovers,
            st.session_state.boards_df,
            st.session_state.required_df
        )
        st.sidebar.download_button("Download JSON", saved_data, file_name="cut_plan.json", mime="application/json")
    else:
        saved_data = save_plan_to_yaml(
            st.session_state.cut_plan,
            st.session_state.leftovers,
            st.session_state.boards_df,
            st.session_state.required_df
        )
        st.sidebar.download_button("Download YAML", saved_data, file_name="cut_plan.yaml", mime="text/yaml")

if load_file:
    file_content = load_file.read().decode("utf-8")
    if load_file.name.endswith(".json"):
        cut_plan, leftovers, boards_df, required_df = json.loads(file_content).values()
    else:
        cut_plan, leftovers, boards_df, required_df = yaml.safe_load(file_content).values()
    st.session_state.cut_plan = cut_plan
    st.session_state.leftovers = leftovers
    st.session_state.boards_df = pd.DataFrame(boards_df)
    st.session_state.required_df = pd.DataFrame(required_df)
    st.rerun()  # Updated from experimental_rerun for compatibility
