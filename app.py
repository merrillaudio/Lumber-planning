import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import io
import csv
import json
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
        if length is None or width is None:
            continue
        for _ in range(quantity):
            pieces.append({
                'length': length,
                'width': width,
                'id': f"{length:.3f}x{width:.3f}"
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
        if length is None or width is None:
            continue
        for _ in range(quantity):
            expanded.append({'length': length, 'width': width})
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

                    # Subdivide remaining space
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

def to_fraction_string(value):
    """Converts a float value to a string representing a fraction (or mixed number)."""
    try:
        frac = Fraction(value).limit_denominator(16)
        # If the fraction is a whole number, return it without a denominator
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

def generate_pdf(cut_plan, leftovers=None):
    buffer = io.BytesIO()
    # Use letter-size pages (8.5" x 11")
    page_width, page_height = 8.5, 11
    with PdfPages(buffer) as pdf:
        for board in cut_plan:
            b = board['board']
            # Create a fixed page-size figure
            fig = plt.figure(figsize=(page_width, page_height))
            # Create a GridSpec: top for drawing, bottom for a textual table
            gs = fig.add_gridspec(2, 1, height_ratios=[3, 2], hspace=0.3)
            
            # ----- Top: Board Drawing -----
            ax_draw = fig.add_subplot(gs[0])
            board_title = (
                f"Board {board['board_id']} - "
                f"{to_fraction_string(b['length'])}\" x {to_fraction_string(b['width'])}\""
            )
            ax_draw.set_title(board_title, fontsize=10)
            ax_draw.set_xlim(0, b['length'])
            ax_draw.set_ylim(0, b['width'])
            ax_draw.set_xlabel("Inches", fontsize=8)
            ax_draw.set_ylabel("Inches", fontsize=8)
            ax_draw.set_aspect('equal', adjustable='box')
            
            # Draw each cut on the board
            for cut in board['cuts']:
                rect = patches.Rectangle(
                    (cut['x'], cut['y']),
                    cut['length'],
                    cut['width'],
                    linewidth=1,
                    edgecolor='black',
                    facecolor='lightgrey'
                )
                ax_draw.add_patch(rect)
                piece_label = (
                    f"{to_fraction_string(cut['piece']['length'])}\" x "
                    f"{to_fraction_string(cut['piece']['width'])}\""
                )
                ax_draw.text(
                    cut['x'] + cut['length'] / 2,
                    cut['y'] + cut['width'] / 2,
                    piece_label,
                    ha='center', va='center', fontsize=8
                )
            
            # ----- Bottom: List of Cuts -----
            ax_text = fig.add_subplot(gs[1])
            ax_text.axis('off')
            text_lines = []
            header = f"{'Piece ID':<12} {'X':>6} {'Y':>6} {'Length':>8} {'Width':>8} {'Rotated':>8}"
            text_lines.append(header)
            text_lines.append("-" * len(header))
            for cut in board['cuts']:
                piece_id = cut['piece']['id']
                x_str = to_fraction_string(cut['x'])
                y_str = to_fraction_string(cut['y'])
                length_str = to_fraction_string(cut['length'])
                width_str = to_fraction_string(cut['width'])
                rotated_str = "Yes" if cut.get('rotated', False) else "No"
                line = f"{piece_id:<12} {x_str:>6} {y_str:>6} {length_str:>8} {width_str:>8} {rotated_str:>8}"
                text_lines.append(line)
            text_block = "\n".join(text_lines)
            ax_text.text(0, 1, text_block, fontsize=8, family='monospace', va='top')
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        
        # Optionally, add an extra page for leftover pieces
        if leftovers:
            fig, ax = plt.subplots(figsize=(page_width, page_height))
            ax.axis('off')
            ax.set_title("Leftover Pieces", fontsize=10)
            text_lines = []
            header = f"{'Length':>8} {'Width':>8}"
            text_lines.append(header)
            text_lines.append("-" * len(header))
            for piece in leftovers:
                length_str = to_fraction_string(piece['length'])
                width_str = to_fraction_string(piece['width'])
                line = f"{length_str:>8} {width_str:>8}"
                text_lines.append(line)
            text_block = "\n".join(text_lines)
            ax.text(0, 1, text_block, fontsize=8, family='monospace', va='top')
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    buffer.seek(0)
    return buffer

def generate_csv(cut_plan):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Board ID', 'Piece ID', 'Length', 'Width', 'Rotated', 'X', 'Y'])
    for board in cut_plan:
        for cut in board['cuts']:
            writer.writerow([
                board['board_id'],
                cut['piece']['id'],
                f"{cut['length']:.3f}",
                f"{cut['width']:.3f}",
                cut['rotated'],
                round(cut['x'], 2),
                round(cut['y'], 2)
            ])
    output.seek(0)
    return output.getvalue()


# ---- Streamlit App UI ----
def save_plan_to_json(plan, leftovers, boards_df, required_df):
    data = {
        'cut_plan': plan,
        'leftovers': leftovers,
        'boards_input': boards_df.to_dict(orient='records'),
        'required_input': required_df.to_dict(orient='records')
    }
    return json.dumps(data, indent=2)

def load_plan_from_json(json_data):
    data = json.loads(json_data)
    return (
        data['cut_plan'],
        data.get('leftovers', []),
        pd.DataFrame(data.get('boards_input', [])),
        pd.DataFrame(data.get('required_input', [])),
    )


# ---- Streamlit App UI ----
st.set_page_config(page_title="Lumber Cut Optimizer", layout="wide")
st.title("📐 Lumber Cut Optimizer")

st.sidebar.header("Cut Settings")
kerf = st.sidebar.number_input("Kerf Size (inches)", value=0.125, step=0.001, format="%.3f")
thickness = st.sidebar.number_input("Board Thickness (inches)", value=0.75, step=0.01)
cost_per_bf = st.sidebar.number_input("Cost per Board Foot ($)", value=5.00, step=0.01)

st.subheader("Available Lumber")
def default_board_df():
    return pd.DataFrame([{"Length": "96", "Width": "12", "Quantity": 1}])

boards_df = st.data_editor(
    st.session_state.get('boards_df', default_board_df()),
    num_rows="dynamic", use_container_width=True
)

st.subheader("Required Cuts")
def default_cut_df():
    return pd.DataFrame([{"Length": "24", "Width": "6", "Quantity": 2}])

required_df = st.data_editor(
    st.session_state.get('required_df', default_cut_df()),
    num_rows="dynamic", use_container_width=True)

if st.button("✂️ Optimize Cuts"):
    boards_list = expand_boards_by_quantity(boards_df)
    cut_plan, leftovers = fit_pieces_to_boards(boards_list, required_df, kerf)

    st.session_state.cut_plan = cut_plan
    st.session_state.leftovers = leftovers
    st.session_state.boards_df = boards_df
    st.session_state.required_df = required_df

    boards_list = expand_boards_by_quantity(boards_df)
    cut_plan, leftovers = fit_pieces_to_boards(boards_list, required_df, kerf)

    total_bf = sum(
        calculate_board_feet(b['board']['length'], b['board']['width'], 1, thickness) for b in cut_plan
    )
    total_cost = total_bf * cost_per_bf

    st.success(f"Optimization complete! 🧮 Total board feet: {total_bf:.2f}, Estimated Cost: ${total_cost:.2f}")

    csv_data = generate_csv(cut_plan)
    pdf_data = generate_pdf(cut_plan, leftovers)

    st.download_button("📄 Download CSV", csv_data, file_name="cut_plan.csv", mime="text/csv")
    st.download_button("📄 Download PDF", pdf_data, file_name="cut_plan.pdf")

    if leftovers:
        st.warning("Some pieces could not be placed. Check the PDF for suggestions.")

st.subheader("💾 Save or Load Plan")

col1, col2 = st.columns(2)

with col1:
    if 'cut_plan' in st.session_state and st.button("💾 Save Plan as JSON"):
        saved_json = save_plan_to_json(
            st.session_state.cut_plan,
            st.session_state.leftovers,
            st.session_state.boards_df,
            st.session_state.required_df
        )
        st.download_button("📥 Download JSON", saved_json, file_name="cut_plan.json", mime="application/json")


with col2:
    uploaded_file = st.file_uploader("📤 Load Plan from JSON", type=["json"])
    if uploaded_file:
        content = uploaded_file.read().decode("utf-8")
        cut_plan, leftovers, boards_df, required_df = load_plan_from_json(content)
        st.session_state.cut_plan = cut_plan
        st.session_state.leftovers = leftovers
        st.session_state.boards_df = boards_df
        st.session_state.required_df = required_df
        st.rerun()


# --- rest of your app remains unchanged (fit, UI, etc.) ---
