import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import io
import csv
import json
from fractions import Fraction
import re

# ---- Helper Functions ----
def parse_measurement(value):
    if isinstance(value, (int, float)):
        return float(value)

    try:
        s = str(value).strip().lower().replace('in', '').replace('″', '"')
        feet = 0

        # Parse formats like 6', 6 ft, 6' 4 1/2, etc.
        ft_match = re.match(r"(?:(\d+)[\s']*(?:ft)?)[\s,]*(.*)", s)
        if ft_match:
            feet = float(ft_match.group(1)) * 12
            s = ft_match.group(2)

        # Now parse the remaining inch/fraction part
        parts = s.strip().split()
        if len(parts) == 2:
            whole = float(parts[0])
            frac = float(Fraction(parts[1]))
            inches = whole + frac
        elif len(parts) == 1 and parts[0]:
            inches = float(Fraction(parts[0]))
        else:
            inches = 0

        return feet + inches

    except Exception:
        return None

def format_inches_as_fraction(inches):
    whole = int(inches)
    frac = Fraction(inches - whole).limit_denominator(16)
    if frac.numerator == 0:
        return f"{whole}\""
    if whole == 0:
        return f"{frac.numerator}/{frac.denominator}\""
    return f"{whole} {frac.numerator}/{frac.denominator}\""

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

# ---- Streamlit UI: Editable Tables with Checkboxes ----
def default_board_df():
    return pd.DataFrame([{"Length": "96", "Width": "12", "Quantity": 1}])

def default_cut_df():
    return pd.DataFrame([{"Length": "24", "Width": "6", "Quantity": 2}])

def display_editable_table(label, df):
    df = df.copy()
    df['❌ Delete'] = False
    edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True, key=f"table_{label}")
    cleaned_df = edited_df[~edited_df['❌ Delete']].drop(columns=['❌ Delete'])
    return cleaned_df.reset_index(drop=True)

st.subheader("Available Lumber")
st.session_state.boards_df = display_editable_table("boards", st.session_state.get("boards_df", default_board_df()))

st.subheader("Required Cuts")
st.session_state.required_df = display_editable_table("cuts", st.session_state.get("required_df", default_cut_df()))

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

def generate_pdf(cut_plan, leftovers=None):
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        for board in cut_plan:
            fig, ax = plt.subplots()
            b = board['board']
            ax.set_title(f"Board {board['board_id']} - {format_inches_as_fraction(b['length'])} x {format_inches_as_fraction(b['width'])}")
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
                    f"{format_inches_as_fraction(cut['length'])} x {format_inches_as_fraction(cut['width'])}",
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
                ax.text(0, y, f"- {format_inches_as_fraction(piece['length'])} x {format_inches_as_fraction(piece['width'])}", fontsize=10, ha='left')
                y -= 0.05
            pdf.savefig(fig)
            plt.close(fig)
    buffer.seek(0)
    return buffer

