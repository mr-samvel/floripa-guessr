import os
import csv
import random
import requests
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

load_dotenv()

MAPS_API_KEY = os.getenv('MAPS_API_KEY')
assert MAPS_API_KEY

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMGS_DIR = os.path.join(BASE_DIR, 'images')
MANIFEST_DIR = os.path.join(BASE_DIR, 'manifests')

MANIFEST_PATH = os.path.join(MANIFEST_DIR, 'manifest.csv')

LAT_MIN, LAT_MAX = -27.843357, -27.374617
LNG_MIN, LNG_MAX = -48.611627, -48.35722
NUM_SAMPLES = 2000
MAX_WORKERS = 10

def download():
    while True:
        lat = random.uniform(LAT_MIN, LAT_MAX)
        lng = random.uniform(LNG_MIN, LNG_MAX)

        try:
            meta = requests.get(
                "https://maps.googleapis.com/maps/api/streetview/metadata"
                f"?location={lat},{lng}"
                f"&key={MAPS_API_KEY}"
            ).json()
        except (requests.exceptions.ConnectionError, requests.exceptions.ConnectTimeout):
            continue

        if meta.get("status") != "OK":
            if meta.get("status") != 'ZERO_RESULTS':
                print(f"Error getting meta request: {meta}")
            continue
        
        pano_id = meta["pano_id"]
        if pano_id.startswith('CAoS'): # remove fotos tiradas por usuários
            continue

        data = []
        for heading in [0, 90, 180, 270]:
            resp = requests.get(
                "https://maps.googleapis.com/maps/api/streetview"
                f"?size=640x640"
                f"&pano={pano_id}"
                f"&heading={heading}"
                f"&fov=90"
                f"&pitch=0"
                f"&key={MAPS_API_KEY}"
            )
            if resp.status_code == 200:
                fname = f"{pano_id}_{heading}.jpg"
                fpath = os.path.join(IMGS_DIR, fname)
                with open(fpath, "wb") as img_file:
                    img_file.write(resp.content)

                data.append({
                    "pano_id": pano_id,
                    "lat": lat,
                    "lng": lng,
                    "heading": heading,
                    "file": fpath
                })
        if len(data) > 0:
            return data


def main():
    manifest_file = open(MANIFEST_PATH, mode="a", newline="", encoding='utf-8')
    fieldnames = ["pano_id", "lat", "lng", "heading", "file"]
    writer = csv.DictWriter(manifest_file, fieldnames=fieldnames)
    if not os.path.isfile(MANIFEST_PATH):
        writer.writeheader()

    print(f'Starting download of {NUM_SAMPLES} images using {MAX_WORKERS} threads')
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(download) for _ in range(NUM_SAMPLES)]
        n_downloaded = 0
        for future in as_completed(futures):
            try:
                data = future.result()
            except Exception as e:
                print('Worker error', e)
                continue
            for d in data:
                writer.writerow(d)
                manifest_file.flush()

            n_downloaded+=1
            if n_downloaded % 50 == 0 or n_downloaded == NUM_SAMPLES:
                print(f"Downloaded {n_downloaded}/{NUM_SAMPLES} images...")

    manifest_file.close()

    print(f"Downloaded {n_downloaded} images into {IMGS_DIR}")

if __name__ == '__main__':
    main()