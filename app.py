import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import io
import csv

# ---- Settings ----
KERF = 0.125  # Default kerf in inches

# ---- Helper Functions ----

def generate_required_pieces(required_df):
    pieces = []
    for _, row in required_df.iterrows():
        for _ in range(int(row['Quantity'])):
            pieces.append({
                'length': float(row['Length']),
                'width': float(row['Width']),
                'id': f"{float(row['Length'])}x{float(row['Width'])}"
            })
    return sorted(pieces, key=lambda x: max(x['length'], x['width']), reverse=True)

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


def fit_pieces_to_boards(boards_df, required_df, kerf):
    pieces = generate_required_pieces(required_df)
    cut_plan = []
    board_id = 1

    for _, board in boards_df.iterrows():
        board_data = {'length': float(board['Length']), 'width': float(board['Width'])}
        placements, pieces = try_place_pieces(board_data, pieces, kerf)

        used_area = sum(p['length'] * p['width'] for p in placements)
        total_area = board_data['length'] * board_data['width']
        waste_area = total_area - used_area

        cut_plan.append({
            'board_id': board_id,
            'board': board_data,
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
                cut['length'],
                cut['width'],
                cut['rotated'],
                round(cut['x'], 2),
                round(cut['y'], 2)
            ])
    output.seek(0)
    return output.getvalue()


# ---- Streamlit App ----

st.title("🪚 Wood Cutting Optimizer (Imperial)")

st.sidebar.header("Settings")
kerf = st.sidebar.number_input("Kerf (inches)", value=0.125, step=0.01)

st.header("1. Enter Available Boards")
default_boards = pd.DataFrame([
    {"Length": 96.0, "Width": 48.0},
    {"Length": 96.0, "Width": 48.0},
], dtype=float)
boards_df = st.data_editor(default_boards, num_rows="dynamic", key="boards")

st.header("2. Enter Required Pieces")
default_required = pd.DataFrame([
    {"Length": 24.0, "Width": 24.0, "Quantity": 3},
    {"Length": 18.5, "Width": 18.0, "Quantity": 4},
    {"Length": 12.25, "Width": 12.25, "Quantity": 6},
], dtype=float)
required_df = st.data_editor(default_required, num_rows="dynamic", key="required")

if st.button("🧠 Optimize Cut Plan"):
    cut_plan, leftovers = fit_pieces_to_boards(boards_df, required_df, kerf)

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

    # PDF & CSV Downloads
    pdf_bytes = generate_pdf(cut_plan)
    st.download_button("📄 Download PDF", data=pdf_bytes, file_name="cut_plan.pdf", mime="application/pdf")

    csv_string = generate_csv(cut_plan)
    st.download_button("📄 Download CSV", data=csv_string, file_name="cut_plan.csv", mime="text/csv")

    # Leftovers
    if leftovers:
        st.warning("⚠️ Some pieces couldn't be placed:")
        for piece in leftovers:
            st.text(f"- {piece['length']}\" x {piece['width']}\"")

