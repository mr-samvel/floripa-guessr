import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

LAT_MIN, LAT_MAX = -27.843357, -27.374617
LNG_MIN, LNG_MAX = -48.611627, -48.35722

def coord_to_pixel(lat, lng, width, height):
    lat_norm = (lat - LAT_MIN) / (LAT_MAX - LAT_MIN)
    lng_norm = (lng - LNG_MIN) / (LNG_MAX - LNG_MIN)
    
    x = int(lng_norm * width)
    y = int((1 - lat_norm) * height)  # Flip Y axis
    
    return x, y

def create_balanced_grid_overlay(image_path, grid_bounds_path, output_path):
    image = Image.open(image_path)
    width, height = image.size
    grid_bounds = pd.read_csv(grid_bounds_path)
    
    overlay_image = image.copy()
    draw = ImageDraw.Draw(overlay_image)
    
    try:
        font = ImageFont.load_default()
    except:
        font = None
    
    for _, bounds in grid_bounds.iterrows():
        cell_id = bounds['cell_id']
        
        x1, y1 = coord_to_pixel(bounds['lat_min'], bounds['lng_min'], width, height)
        x2, y2 = coord_to_pixel(bounds['lat_max'], bounds['lng_max'], width, height)
        
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
        
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        text = str(int(cell_id))
        if font:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        else:
            text_width, text_height = 20, 10
        
        padding = 2
        draw.rectangle([
            center_x - text_width // 2 - padding,
            center_y - text_height // 2 - padding,
            center_x + text_width // 2 + padding,
            center_y + text_height // 2 + padding
        ], fill=(255, 255, 255))
        
        draw.text(
            (center_x - text_width // 2, center_y - text_height // 2),
            text, fill=(0, 0, 0), font=font
        )
    
    overlay_image.save(output_path)
    print(f"Balanced grid overlay saved to: {output_path}")
    print(f"Total cells: {len(grid_bounds)}")

def main():
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python balanced_grid_overlay.py <input_image>")
        print("Note: Requires grid_bounds.csv in manifests/ directory")
        return
    
    input_path = sys.argv[1]
    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_balanced_grid{ext}"
    
    grid_bounds_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'manifests', 'grid_bounds.csv')
    
    if not os.path.exists(grid_bounds_path):
        print(f"Error: {grid_bounds_path} not found")
        print("Run the balanced grid classification script first")
        return
    
    create_balanced_grid_overlay(input_path, grid_bounds_path, output_path)

if __name__ == '__main__':
    main()