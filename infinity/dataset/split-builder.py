import os
import json
import glob
import math
import multiprocessing
from tqdm import tqdm
from PIL import Image

MARKER=".finished"
# Paths
ROOT_DIR = "."        # Root directory that contains 'images'
IMAGES_DIR = os.path.join(ROOT_DIR, "images")
OUTPUT_DIR = os.path.join(ROOT_DIR, "split")
ratio2hws = {
    1.000: [(1,1),(2,2),(4,4),(6,6),(8,8),(12,12),(16,16),(20,20),(24,24),(32,32),(40,40),(48,48),(64,64)],
    1.250: [(1,1),(2,2),(3,3),(5,4),(10,8),(15,12),(20,16),(25,20),(30,24),(35,28),(45,36),(55,44),(70,56)],
    1.333: [(1,1),(2,2),(4,3),(8,6),(12,9),(16,12),(20,15),(24,18),(28,21),(36,27),(48,36),(60,45),(72,54)],
    1.500: [(1,1),(2,2),(3,2),(6,4),(9,6),(15,10),(21,14),(27,18),(33,22),(39,26),(48,32),(63,42),(78,52)],
    1.750: [(1,1),(2,2),(3,3),(7,4),(11,6),(14,8),(21,12),(28,16),(35,20),(42,24),(56,32),(70,40),(84,48)],
    2.000: [(1,1),(2,2),(4,2),(6,3),(10,5),(16,8),(22,11),(30,15),(38,19),(46,23),(60,30),(74,37),(90,45)],
    2.500: [(1,1),(2,2),(5,2),(10,4),(15,6),(20,8),(25,10),(30,12),(40,16),(50,20),(65,26),(80,32),(100,40)],
    3.000: [(1,1),(2,2),(6,2),(9,3),(15,5),(21,7),(27,9),(36,12),(45,15),(54,18),(72,24),(90,30),(111,37)],
}

def get_ratio(image_path):
    """
    Return the ratio h_div_w for the given image.
    """
    with Image.open(image_path) as im:
        w, h = im.size
    # Avoid division by zero
    if w == 0:
        return 0.0
    # get ratio from ratio2hws, if error > 0.1, set to 0 for indicating skip
    ratio = h / w
    nearest_ratio = min(ratio2hws.keys(), key=lambda x: abs(x - ratio))
    if abs(nearest_ratio - ratio) > 0.1:
        return 0.0
    return nearest_ratio

def process_subfolder(subfolder_path):
    """
    Pass 1:
    Process a single subfolder to create partial results.
      1) Check .finished -> skip if exists
      2) Find .webp & matching .txt
      3) Compute ratio, read caption
      4) Write partial results to a local file (partial.json)
      5) Mark subfolder as .finished
    """
    finished_marker = os.path.join(subfolder_path, MARKER)
    if os.path.exists(finished_marker):
        # Already processed
        return

    items = []
    
    # Gather all .webp files in the subfolder
    webp_files = glob.glob(os.path.join(subfolder_path, "*.webp"))
    for webp_path in webp_files:
        base_name = os.path.splitext(os.path.basename(webp_path))[0]
        txt_path = os.path.join(subfolder_path, base_name + ".txt")
        
        if not os.path.exists(txt_path):
            # No matching text file -> skip
            continue
        
        # Compute ratio
        ratio = get_ratio(webp_path)
        if ratio == 0.0:
            # Skip this image
            continue
        
        # Read long caption from .txt
        with open(txt_path, "r", encoding="utf-8") as f:
            long_caption = f.read().strip()

        # Create item (skip short_caption_type & text)
        item = {
            "image_path": webp_path,
            "h_div_w": ratio,
            "long_caption": long_caption,
            "long_caption_type": "tag"
        }
        items.append(item)
    
    # Write partial results to local file
    partial_path = os.path.join(subfolder_path, "partial.json")
    with open(partial_path, "w", encoding="utf-8") as pf:
        json.dump(items, pf, ensure_ascii=False)
    
    # Mark subfolder as finished
    with open(finished_marker, "w", encoding="utf-8") as fm:
        fm.write("done\n")

def gather_partial_results():
    """
    Pass 2:
    Read partial.json from all subfolders and create final JSONL outputs.
    """
    # 1) Collect all partial.json files
    partial_files = []
    for folder in os.listdir(IMAGES_DIR):
        subfolder_path = os.path.join(IMAGES_DIR, folder)
        if not os.path.isdir(subfolder_path):
            continue
        
        partial_path = os.path.join(subfolder_path, "partial.json")
        if os.path.exists(partial_path):
            partial_files.append(partial_path)

    ratio_map = {}

    # 2) Read each partial.json and group items by ratio
    for pf in tqdm(partial_files, desc="Gathering partial results"):
        with open(pf, "r", encoding="utf-8") as f:
            items = json.load(f)
            for it in items:
                ratio_rounded = round(it["h_div_w"], 3)
                if ratio_rounded not in ratio_map:
                    ratio_map[ratio_rounded] = []
                ratio_map[ratio_rounded].append(it)
    
    # 3) Write each ratio group to final JSONL in OUTPUT_DIR
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for ratio, items in ratio_map.items():
        # Example filename: 1.000_000002500.jsonl
        ratio_str = f"{ratio:.3f}"
        count = len(items)
        filename = f"{ratio_str}_{count:09d}.jsonl"
        output_path = os.path.join(OUTPUT_DIR, filename)

        # Write to JSONL (append or overwrite as needed)
        # Usually you'd just overwrite once, but if you want 
        # to do multiple passes, you could open with 'a' (append).
        with open(output_path, "w", encoding="utf-8") as of:
            for record in items:
                # Keep only required fields
                out_record = {
                    "image_path": record["image_path"],
                    "h_div_w": record["h_div_w"],
                    "long_caption": record["long_caption"],
                    "long_caption_type": record["long_caption_type"]
                }
                of.write(json.dumps(out_record, ensure_ascii=False) + "\n")
    
    print(f"Done! Created {len(ratio_map)} grouped JSONL files in {OUTPUT_DIR}")

def main():
    # --- PASS 1 ---
    # Process each subfolder in parallel to create partial results
    subfolders = [
        os.path.join(IMAGES_DIR, d)
        for d in os.listdir(IMAGES_DIR)
        if os.path.isdir(os.path.join(IMAGES_DIR, d))
    ]
    subfolders.sort()

    cpu_count = min(24, max(1, multiprocessing.cpu_count() - 1))
    pool = multiprocessing.Pool(cpu_count)

    # Multiprocessing with a progress bar
    with tqdm(total=len(subfolders), desc="Processing subfolders") as pbar:
        for _ in pool.imap_unordered(process_subfolder, subfolders):
            pbar.update(1)

    pool.close()
    pool.join()

    # --- PASS 2 ---
    # Now gather partial.json from each subfolder to produce final JSONLs
    gather_partial_results()


if __name__ == "__main__":
    main()
