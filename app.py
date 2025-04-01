import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import io
import csv
import json
from fractions import Fraction

# ---- Helpers ----

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

def try_place_pieces(board, pieces, kerf):
    placements = []
    remaining = []
    x_cursor = 0
    y_cursor = 0
    max_row_height = 0

    for piece in pieces:
        placed = False
        for rotate in [False, True]:
            p_length = piece['length']
            p_width = piece['width']
            if rotate:
                p_length, p_width = p_width, p_length
            if x_cursor + p_length + kerf > board['length']:
                x_cursor = 0
                y_cursor += max_row_height + kerf
                max_row_height = 0
            if y_cursor + p_width + kerf > board['width']:
                continue
            placements.append({
                'piece': piece,
                'x': x_cursor,
                'y': y_cursor,
                'length': p_length,
                'width': p_width,
                'rotated': rotate
            })
            x_cursor += p_length + kerf
            max_row_height = max(max_row_height, p_width)
            placed = True
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

def generate_pdf(cut_plan):
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        for board in cut_plan:
            fig, ax = plt.subplots()
            b = board['board']
            ax.set_title(f"Board {board['board_id']} - {b['length']}\" x {b['width']}\"")
            ax.set_xlim(0, b['length'])
            ax.set_ylim(0, b['width'])
            for cut in board['cuts']:
                rect = patches.Rectangle(
                    (cut['x'], cut['y']),
                    cut['length'],
                    cut['width'],
                    linewidth=1,
                    edgecolor='black',
                    facecolor='lightgrey'
                )
                ax.add_patch(rect)
                ax.text(
                    cut['x'] + cut['length'] / 2,
                    cut['y'] + cut['width'] / 2,
                    cut['piece']['id'],
                    ha='center', va='center', fontsize=8
                )
            plt.gca().set_aspect('equal', adjustable='box')
            plt.xlabel("Inches")
            plt.ylabel("Inches")
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

# ---- Streamlit App ----

st.title("🪚 Wood Cutting Optimizer (with Fractions, Save & Load)")

st.sidebar.header("Settings")
kerf_input = st.sidebar.text_input("Kerf (inches)", value="0.125")
kerf = parse_measurement(kerf_input) or 0.125

st.sidebar.header("📂 Load Previous Plan")
uploaded_json = st.sidebar.file_uploader("Upload saved plan (.json)", type=["json"])

# Use uploaded plan or show default editors
if uploaded_json:
    cut_plan, leftovers, boards_df, required_df = load_plan_from_json(uploaded_json.read())
    st.success("✅ Loaded previous plan!")
else:
    st.header("1. Enter Available Boards")
    boards_df = st.data_editor(pd.DataFrame([
        {"Length": "96", "Width": "48", "Quantity": "2"},
        {"Length": "48", "Width": "24", "Quantity": "1"},
    ]), num_rows="dynamic", key="boards")

    st.header("2. Enter Required Pieces")
    required_df = st.data_editor(pd.DataFrame([
        {"Length": "24", "Width": "24", "Quantity": "3"},
        {"Length": "18.5", "Width": "18", "Quantity": "4"},
        {"Length": "12 1/4", "Width": "12.25", "Quantity": "6"},
    ]), num_rows="dynamic", key="required")

if st.button("🧠 Optimize Cut Plan"):
    expanded_boards = expand_boards_by_quantity(boards_df)
    cut_plan, leftovers = fit_pieces_to_boards(expanded_boards, required_df, kerf)

    st.subheader("📋 Cut Plan Preview")
    for board in cut_plan:
        st.markdown(f"### Board {board['board_id']} - {board['board']['length']}\" x {board['board']['width']}\"")
        fig, ax = plt.subplots()
        ax.set_xlim(0, board['board']['length'])
        ax.set_ylim(0, board['board']['width'])
        for cut in board['cuts']:
            rect = patches.Rectangle(
                (cut['x'], cut['y']),
                cut['length'],
                cut['width'],
                linewidth=1,
                edgecolor='black',
                facecolor='lightgrey'
            )
            ax.add_patch(rect)
            ax.text(
                cut['x'] + cut['length'] / 2,
                cut['y'] + cut['width'] / 2,
                cut['piece']['id'],
                ha='center', va='center', fontsize=8
            )
        plt.gca().set_aspect('equal', adjustable='box')
        st.pyplot(fig)

    pdf_bytes = generate_pdf(cut_plan)
    st.download_button("📄 Download PDF", data=pdf_bytes, file_name="cut_plan.pdf", mime="application/pdf")

    csv_string = generate_csv(cut_plan)
    st.download_button("📄 Download CSV", data=csv_string, file_name="cut_plan.csv", mime="text/csv")

    json_data = save_plan_to_json(cut_plan, leftovers, boards_df, required_df)
    st.download_button("💾 Save Plan (.json)", data=json_data, file_name="wood_cut_plan.json", mime="application/json")

    if leftovers:
        st.warning("⚠️ Some pieces couldn't be placed:")
        for piece in leftovers:
            st.text(f"- {piece['length']}\" x {piece['width']}\"")
