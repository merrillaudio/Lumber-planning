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

def generate_pdf(cut_plan, leftovers=None):
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

        if leftovers:
            fig, ax = plt.subplots()
            ax.axis('off')
            ax.set_title("Suggested Additional Pieces")
            y = 1.0
            ax.text(0, y, "The following pieces could not be placed:", fontsize=12, ha='left')
            y -= 0.1
            for piece in leftovers:
                ax.text(0, y, f"- {piece['length']:.2f}\" x {piece['width']:.2f}\"", fontsize=10, ha='left')
                y -= 0.05
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
st.set_page_config(page_title="Lumber Cut Optimizer", layout="wide")
st.title("📐 Lumber Cut Optimizer")

st.sidebar.header("Cut Settings")
kerf = st.sidebar.number_input("Kerf Size (inches)", value=0.125, step=0.001, format="%.3f")
thickness = st.sidebar.number_input("Board Thickness (inches)", value=0.75, step=0.01)
cost_per_bf = st.sidebar.number_input("Cost per Board Foot ($)", value=5.00, step=0.01)

st.subheader("Available Lumber")
def default_board_df():
    return pd.DataFrame([{"Length": "96", "Width": "12", "Quantity": 1}])

boards_df = st.data_editor(default_board_df(), num_rows="dynamic", use_container_width=True)

st.subheader("Required Cuts")
def default_cut_df():
    return pd.DataFrame([{"Length": "24", "Width": "6", "Quantity": 2}])

required_df = st.data_editor(default_cut_df(), num_rows="dynamic", use_container_width=True)

if st.button("✂️ Optimize Cuts"):
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
