import csv
import os
import argparse

def filter_csv(input_csv: str, output_csv: str):
    with open(input_csv, newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)

        valid_rows = []
        for row in reader:
            file_path = row['file']
            if os.path.exists(file_path):
                valid_rows.append(row)
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(valid_rows)

def filter_files(filtered_csv):
    valid_paths = set()
    with open(filtered_csv, newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            valid_paths.add(os.path.abspath(row['file']))
    if not valid_paths:
        return
    
    dir = os.path.dirname(next(iter(valid_paths)))
    if not os.path.isdir(dir):
        print(f'Directory not found: {dir}')
        return
    
    for file_name in os.listdir(dir):
        file_path = os.path.abspath(os.path.join(dir, file_name))
        if os.path.isdir(file_path):
            continue
        if file_path not in valid_paths:
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_csv')
    parser.add_argument('output_csv')
    args = parser.parse_args()
    
    filter_csv(args.input_csv, args.output_csv)
    filter_files(args.output_csv)

if __name__ == '__main__':
    main()
