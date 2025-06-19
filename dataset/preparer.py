import os
import shutil
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

load_dotenv()

LAT_MIN, LAT_MAX = -27.843357, -27.374617
LNG_MIN, LNG_MAX = -48.611627, -48.35722
GRID_ROWS, GRID_COLS = 15, 5

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMGS_DIR = os.path.join(BASE_DIR, 'images')
MANIFEST_DIR = os.path.join(BASE_DIR, 'manifests')
TRAIN_DIR  = os.path.join(IMGS_DIR, "by_cell_train")
VALID_DIR  = os.path.join(IMGS_DIR, "by_cell_valid")

MANIFEST_PATH = os.path.join(MANIFEST_DIR, 'manifest.csv')
T_MANIFEST_PATH = os.path.join(MANIFEST_DIR, 'training_manifest.csv')
V_MANIFEST_PATH = os.path.join(MANIFEST_DIR, 'validation_manifest.csv')

def coord_to_label(lat, lng):
    '''
    Converte um conjunto de coordenadas (lat,lng) em um intervalo discreto delimitado pelo grid (GRID_ROWS, GRID_COLS).
    A label resultante é um valor inteiro que indica a célula (em 1D) a qual a coordenada pertence no grid.
    O intervalo da label é dado por 0..GRID_ROWS*GRID_COLS-1.
    '''
    lat = min(max(lat, LAT_MIN), LAT_MAX)
    lng = min(max(lng, LNG_MIN), LNG_MAX)

    # normaliza coordenadas em intervalo [0, 1]
    row_frac = (lat - LAT_MIN) / (LAT_MAX - LAT_MIN)
    col_frac = (lng - LNG_MIN) / (LNG_MAX - LNG_MIN)

    row = int(row_frac * GRID_ROWS)
    col = int(col_frac * GRID_COLS)

    if row == GRID_ROWS: row -= 1
    if col == GRID_COLS: col -= 1

    return row * GRID_COLS + col

def build_folders(split_data, out_dir):
    for cell in split_data['cell'].unique():
        os.makedirs(os.path.join(out_dir, str(cell)), exist_ok=True)
    for _, row in split_data.iterrows():
        src = row['file']
        dst = os.path.join(out_dir, str(row['cell']), os.path.basename(src))
        if os.path.exists(dst):
            continue
        try:
            os.symlink(os.path.abspath(src), dst)
        except OSError as e:
            print(f'Could not copy {src} to {dst}: {e}')

def main():
    data = pd.read_csv(MANIFEST_PATH)
    n_samples = len(data)
    print(f'Found {n_samples} entries in the manifest file')

    data['cell'] = data.apply(lambda r: coord_to_label(r.lat, r.lng), axis=1)
    # se houver, remove pontos fora dos limites
    data = data.dropna(subset=["cell"]).reset_index(drop=True)
    data["cell"] = data["cell"].astype(int)
    
    print('Converted coordinates to labels')
    cell_counts = data["cell"].value_counts().sort_index()
    for cell, count in cell_counts.items():
        print(f"Cell {cell:>3}: {count} images")

    train_data, valid_data = train_test_split(data, test_size=0.2, stratify=data['cell'], random_state=0)
    train_data.to_csv(T_MANIFEST_PATH, index=False)
    valid_data.to_csv(V_MANIFEST_PATH, index=False)

    build_folders(train_data, TRAIN_DIR)
    build_folders(valid_data, VALID_DIR)

if __name__ == '__main__':
    main()