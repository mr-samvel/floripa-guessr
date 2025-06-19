import os
from PIL import Image, ImageDraw, ImageFont

LAT_MIN, LAT_MAX = -27.843357, -27.374617
LNG_MIN, LNG_MAX = -48.611627, -48.35722
GRID_ROWS, GRID_COLS = 15, 5

def create_grid_overlay(image_path, output_path):
    image = Image.open(image_path)
    width, height = image.size
    
    overlay_image = image.copy()
    draw = ImageDraw.Draw(overlay_image)
    
    cell_width = width / GRID_COLS
    cell_height = height / GRID_ROWS
    
    for col in range(GRID_COLS + 1):
        x = int(col * cell_width)
        draw.line([(x, 0), (x, height)], fill=(255, 0, 0), width=2)
    
    for row in range(GRID_ROWS + 1):
        y = int(row * cell_height)
        draw.line([(0, y), (width, y)], fill=(255, 0, 0), width=2)
    
    try:
        font = ImageFont.load_default()
    except:
        font = None
    
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            cell_id = row * GRID_COLS + col
            center_x = int((col + 0.5) * cell_width)
            center_y = int((row + 0.5) * cell_height)
            
            text = str(cell_id)
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
    print(f"Grid overlay saved to: {output_path}")

def main():
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python grid_overlay.py <input_image>")
        return
    
    input_path = sys.argv[1]
    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_grid{ext}"
    
    create_grid_overlay(input_path, output_path)

if __name__ == '__main__':
    main()