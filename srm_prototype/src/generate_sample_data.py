#!/usr/bin/env python3
"""
DeepSRM - Sample Satellite Imagery & Data Synthesizer
-----------------------------------------------------
Generates realistic multi-spectral satellite-like imagery (RGB simulated)
for Krishna/Guntur agriculture, Urban frame survey, Disaster flood monitoring,
Defence perimeter monitoring, and Forest reserves without requiring external assets.
"""

import os
import math
import zlib
import struct

def write_png(path, w, h, pixels):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw_rows = []
    for y in range(h):
        row_bytes = bytearray([0]) # Filter 0
        for x in range(w):
            r, g, b = pixels[y][x]
            row_bytes.extend([min(255, max(0, int(r))), min(255, max(0, int(g))), min(255, max(0, int(b)))])
        raw_rows.append(bytes(row_bytes))
    
    compressed = zlib.compress(b''.join(raw_rows), 6)
    
    def make_chunk(chunk_type, data):
        length = len(data)
        crc = zlib.crc32(chunk_type + data) & 0xffffffff
        return struct.pack('>I', length) + chunk_type + data + struct.pack('>I', crc)

    with open(path, 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n')
        f.write(make_chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)))
        f.write(make_chunk(b'IDAT', compressed))
        f.write(make_chunk(b'IEND', b''))

def create_agriculture_scene(w=360, h=360):
    """
    Simulates Krishna/Guntur Delta agricultural landscape:
    Paddy fields with varied crop stages, irrigation canal network, and field bunds.
    """
    pixels = []
    # Grid of parcels
    num_cols = 6
    num_rows = 6
    col_w = w / num_cols
    row_h = h / num_rows

    # Color palettes for fields (shades of lush green, emerald paddy, dry harvest, red soil)
    parcel_colors = [
        (34, 139, 34),   # Forest/Lush Paddy
        (46, 125, 50),   # Mature rice
        (76, 175, 80),   # Vibrant green vegetative
        (139, 195, 74),  # Light green shoot
        (197, 160, 89),  # Dry stubble / ready for harvest
        (160, 82, 45),   # Brown / red fertile soil
        (56, 142, 60),   # Chili plantation
        (67, 160, 71),   # Sugarcane
    ]

    for y in range(h):
        row = []
        c_r = int(y / row_h)
        for x in range(w):
            c_c = int(x / col_w)
            
            # Irrigation canal cutting diagonally
            is_canal = abs((x * 0.85 + y * 0.3) - 180) < 6
            # Secondary canal
            is_branch_canal = (x > 180 and abs(y - 200) < 4)
            # Bund / bund roads separating parcels
            is_bund_x = (int(x) % int(col_w)) < 3 or (int(x) % int(col_w)) > int(col_w) - 3
            is_bund_y = (int(y) % int(row_h)) < 3 or (int(y) % int(row_h)) > int(row_h) - 3

            if is_canal or is_branch_canal:
                # Turbid canal water (deep blue-grey)
                r, g, b = 38, 70, 95
            elif is_bund_x or is_bund_y:
                # Earthen bunds / tracks (tan / beige)
                r, g, b = 185, 170, 140
            else:
                # Field interior with subtle soil noise
                base_color = parcel_colors[(c_r * 3 + c_c * 5) % len(parcel_colors)]
                noise = int(8 * math.sin(x * 0.15) * math.cos(y * 0.15))
                r = base_color[0] + noise
                g = base_color[1] + noise
                b = base_color[2] + noise
            row.append((r, g, b))
        pixels.append(row)
    return pixels

def create_urban_scene(w=360, h=360):
    """
    Simulates Guntur urban settlement with highway corridors, residential clusters,
    commercial roofs, and sparse tree lines.
    """
    pixels = []
    for y in range(h):
        row = []
        for x in range(w):
            # Main arterial highway
            is_highway = abs(x - 175) < 8 or abs(y - 185) < 8
            # Secondary streets
            is_street = (int(x) % 50 < 4) or (int(y) % 50 < 4)

            if is_highway:
                r, g, b = 75, 78, 82 # Dark asphalt
                if abs(x - 175) < 1 or abs(y - 185) < 1:
                    r, g, b = 220, 220, 120 # Road lane marking
            elif is_street:
                r, g, b = 110, 115, 120 # Concrete / paved street
            else:
                # Building blocks / Rooftops
                bx = int(x / 50)
                by = int(y / 50)
                roof_type = (bx * 7 + by * 11) % 4
                if roof_type == 0:
                    r, g, b = 180, 85, 65 # Red/Terracotta roof
                elif roof_type == 1:
                    r, g, b = 175, 180, 185 # Concrete roof terrace
                elif roof_type == 2:
                    r, g, b = 70, 120, 160 # Blue industrial sheet roof
                else:
                    r, g, b = 75, 125, 65 # Backyard green / vegetation
            row.append((r, g, b))
        pixels.append(row)
    return pixels

def create_disaster_scenes(w=360, h=360):
    """
    Simulates River Basin Pre-flood and Post-flood inundation.
    """
    # Pre-flood
    pixels_pre = []
    for y in range(h):
        row = []
        for x in range(w):
            # Normal meandering river channel
            river_center = 180 + int(45 * math.sin(y * 0.025))
            dist_to_river = abs(x - river_center)
            if dist_to_river < 14:
                r, g, b = 40, 90, 140 # Blue river water
            elif dist_to_river < 28:
                r, g, b = 190, 175, 140 # Sandy bank / floodplain
            else:
                # Surrounding villages and crops
                r, g, b = 60 + int((x%30)), 130 + int((y%20)), 50
            row.append((r, g, b))
        pixels_pre.append(row)

    # Post-flood (Extensive inundation zone)
    pixels_post = []
    for y in range(h):
        row = []
        for x in range(w):
            river_center = 180 + int(45 * math.sin(y * 0.025))
            dist_to_river = abs(x - river_center)
            # Flood expands river from width 14 to 65
            if dist_to_river < 68 + int(12 * math.sin(y * 0.08)):
                r, g, b = 65, 85, 110 # Turbid floodwaters
            elif dist_to_river < 85:
                r, g, b = 110, 100, 85 # Waterlogged muddy fields
            else:
                r, g, b = 60 + int((x%30)), 130 + int((y%20)), 50
            row.append((r, g, b))
        pixels_post.append(row)

    return pixels_pre, pixels_post

def create_defence_scene(w=360, h=360):
    """
    Simulates High-level non-sensitive terrain & perimeter monitoring.
    """
    pixels = []
    for y in range(h):
        row = []
        for x in range(w):
            # Perimeter fence / border road corridor
            is_patrol_track = abs((x * 0.6 + y * 0.8) - 220) < 5
            # Outpost / facility square
            is_compound = (110 <= x <= 165 and 110 <= y <= 165)
            is_compound_fence = is_compound and (x in (110, 165) or y in (110, 165))
            
            if is_compound_fence:
                r, g, b = 230, 230, 240
            elif is_compound:
                r, g, b = 150, 150, 155 # Paved compound
            elif is_patrol_track:
                r, g, b = 180, 165, 130 # Sandy graded track
            else:
                # Arid / rocky plateau landscape
                noise = int(10 * math.sin(x * 0.1) + 10 * math.cos(y * 0.1))
                r, g, b = 165 + noise, 145 + noise, 115 + noise
            row.append((r, g, b))
        pixels.append(row)
    return pixels

def create_forest_scene(w=360, h=360):
    """
    Simulates dense forest canopy, clearing edges, and reserve boundary fire lines.
    """
    pixels = []
    for y in range(h):
        row = []
        for x in range(w):
            # Reserve boundary line / fire cut
            is_fire_break = abs(x - (190 + int(20 * math.sin(y * 0.03)))) < 4
            # Degradation / clearing patch
            is_clearing = ((x - 260)**2 + (y - 120)**2) < 40**2

            if is_fire_break:
                r, g, b = 175, 160, 125 # Bare earth firebreak
            elif is_clearing:
                r, g, b = 150, 140, 80 # Scrub / degraded patch
            else:
                # Deep evergreen canopy
                canopy_variation = int(12 * math.sin(x * 0.3) * math.cos(y * 0.3))
                r = max(15, 25 + canopy_variation)
                g = max(40, 85 + canopy_variation)
                b = max(15, 30 + canopy_variation)
            row.append((r, g, b))
        pixels.append(row)
    return pixels

def main():
    print("[DeepSRM Setup] Generating synthetic high-realism medium resolution satellite imagery...")
    base_dir = "data/original"
    os.makedirs(base_dir, exist_ok=True)

    # 1. Agriculture
    agri_px = create_agriculture_scene()
    write_png(os.path.join(base_dir, "agri_krishna_sentinel2.png"), 360, 360, agri_px)
    print(" -> Created data/original/agri_krishna_sentinel2.png")

    # 2. Urban
    urban_px = create_urban_scene()
    write_png(os.path.join(base_dir, "urban_guntur.png"), 360, 360, urban_px)
    print(" -> Created data/original/urban_guntur.png")

    # 3. Disaster
    dis_pre, dis_post = create_disaster_scenes()
    write_png(os.path.join(base_dir, "disaster_flood_before.png"), 360, 360, dis_pre)
    write_png(os.path.join(base_dir, "disaster_flood_after.png"), 360, 360, dis_post)
    print(" -> Created data/original/disaster_flood_before.png & after.png")

    # 4. Defence
    def_px = create_defence_scene()
    write_png(os.path.join(base_dir, "defence_border.png"), 360, 360, def_px)
    print(" -> Created data/original/defence_border.png")

    # 5. Forest
    for_px = create_forest_scene()
    write_png(os.path.join(base_dir, "forest_reserve.png"), 360, 360, for_px)
    print(" -> Created data/original/forest_reserve.png")

    print("[DeepSRM Setup] Sample original imagery synthesis completed successfully.")

if __name__ == "__main__":
    main()
