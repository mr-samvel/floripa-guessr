import os
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

load_dotenv()

LAT_MIN, LAT_MAX = -27.843357, -27.374617
LNG_MIN, LNG_MAX = -48.611627, -48.35722
TARGET_CELLS = 30

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMGS_DIR = os.path.join(BASE_DIR, 'images')
MANIFEST_DIR = os.path.join(BASE_DIR, 'manifests')
TRAIN_DIR = os.path.join(IMGS_DIR, "by_cell_train")
VALID_DIR = os.path.join(IMGS_DIR, "by_cell_valid")

MANIFEST_PATH = os.path.join(MANIFEST_DIR, 'manifest.csv')
T_MANIFEST_PATH = os.path.join(MANIFEST_DIR, 'training_manifest.csv')
V_MANIFEST_PATH = os.path.join(MANIFEST_DIR, 'validation_manifest.csv')
GRID_BOUNDS_PATH = os.path.join(MANIFEST_DIR, 'grid_bounds.csv')

def create_balanced_grid(data, target_cells):
    coords = data[['lat', 'lng']].values
    
    lat_range = LAT_MAX - LAT_MIN
    lng_range = LNG_MAX - LNG_MIN
    coords_norm = coords.copy()
    coords_norm[:, 0] = (coords[:, 0] - LAT_MIN) / lat_range
    coords_norm[:, 1] = (coords[:, 1] - LNG_MIN) / lng_range
    
    kmeans = KMeans(n_clusters=target_cells, random_state=42, n_init=10)
    cell_assignments = kmeans.fit_predict(coords_norm)
    
    cell_bounds = []
    for cell_id in range(target_cells):
        cell_mask = cell_assignments == cell_id
        if not cell_mask.any():
            continue
            
        cell_coords = coords[cell_mask]
        bounds = {
            'cell_id': cell_id,
            'lat_min': cell_coords[:, 0].min(),
            'lat_max': cell_coords[:, 0].max(),
            'lng_min': cell_coords[:, 1].min(),
            'lng_max': cell_coords[:, 1].max(),
            'sample_count': cell_mask.sum()
        }
        cell_bounds.append(bounds)
    
    return cell_bounds, cell_assignments.tolist()

def coord_to_balanced_label(lat, lng, cell_bounds):
    lat = min(max(lat, LAT_MIN), LAT_MAX)
    lng = min(max(lng, LNG_MIN), LNG_MAX)
    
    for bounds in cell_bounds:
        if (bounds['lat_min'] <= lat <= bounds['lat_max'] and 
            bounds['lng_min'] <= lng <= bounds['lng_max']):
            return bounds['cell_id']
    
    min_dist = float('inf')
    closest_cell = 0
    
    for bounds in cell_bounds:
        center_lat = (bounds['lat_min'] + bounds['lat_max']) / 2
        center_lng = (bounds['lng_min'] + bounds['lng_max']) / 2
        dist = ((lat - center_lat) ** 2 + (lng - center_lng) ** 2) ** 0.5
        
        if dist < min_dist:
            min_dist = dist
            closest_cell = bounds['cell_id']
    
    return closest_cell

def main():
    data = pd.read_csv(MANIFEST_PATH)
    n_samples = len(data)
    print(f'Found {n_samples} entries in the manifest file')
    
    data = data.dropna(subset=['lat', 'lng'])
    data = data[(data['lat'] >= LAT_MIN) & (data['lat'] <= LAT_MAX)]
    data = data[(data['lng'] >= LNG_MIN) & (data['lng'] <= LNG_MAX)]
    data = data.reset_index(drop=True)
    
    print(f'Creating balanced grid with {TARGET_CELLS} cells...')
    
    cell_bounds, cell_assignments = create_balanced_grid(data, TARGET_CELLS)
    
    data['cell'] = cell_assignments
    
    bounds_df = pd.DataFrame(cell_bounds)
    bounds_df.to_csv(GRID_BOUNDS_PATH, index=False)
    print(f'Grid bounds saved to {GRID_BOUNDS_PATH}')
    
    print('\nCell distribution:')
    cell_counts = data['cell'].value_counts().sort_index()
    print(f'Mean: {cell_counts.mean():.1f}, Std: {cell_counts.std():.1f}')
    
    for cell, count in cell_counts.items():
        bounds = cell_bounds[cell]
        lat_size = bounds['lat_max'] - bounds['lat_min']
        lng_size = bounds['lng_max'] - bounds['lng_min']
        print(f"Cell {cell:>2}: {count:>3} samples, "
              f"size: {lat_size:.4f}° × {lng_size:.4f}°")
    
    train_data, valid_data = train_test_split(data, test_size=0.2, stratify=data['cell'], random_state=0)
    
    train_data.to_csv(T_MANIFEST_PATH, index=False)
    valid_data.to_csv(V_MANIFEST_PATH, index=False)
    
    print(f'\nTraining samples: {len(train_data)}')
    print(f'Validation samples: {len(valid_data)}')

if __name__ == '__main__':
    main()