#!/usr/bin/env python3
from fitparse import FitFile
from collections import defaultdict, Counter
from pathlib import Path
import json

FIT_DIR = Path(r"C:\SmakrykoDev\GitHubRepos\MS-Buddy-Fitness-App\utilities-tools\fit_test_files")

def enumerate_fit_fields(fit_path: Path):
    fitfile = FitFile(str(fit_path))
    msg_fields = defaultdict(set)
    for msg in fitfile.get_messages():
        if msg.name is None:
            continue
        for field in msg.fields:
            if field.name is not None:
                msg_fields[msg.name].add(field.name)
    return msg_fields

def main():
    if not FIT_DIR.exists():
        print(f"Directory not found: {FIT_DIR}")
        return

    all_files_stats = {}
    for fit_file in FIT_DIR.glob("*.fit"):
        print(f"\n{'='*60}")
        print(f"Analyzing: {fit_file.name}")
        print('='*60)
        
        msg_fields = enumerate_fit_fields(fit_file)
        file_stats = {}
        
        for msg_name in sorted(msg_fields.keys()):
            fields = sorted(msg_fields[msg_name])
            print(f"\n[{msg_name}] ({len(fields)} fields)")
            for field_name in fields:
                print(f"  - {field_name}")
            file_stats[msg_name] = fields
        
        all_files_stats[fit_file.name] = file_stats

    # Summary across all files
    print(f"\n{'='*60}")
    print("SUMMARY ACROSS ALL FIT FILES")
    print('='*60)
    
    all_msgs = Counter()
    for filename, stats in all_files_stats.items():
        print(f"\n{filename}:")
        for msg in stats:
            all_msgs[msg] += 1
            print(f"  {msg}: {len(stats[msg])} fields")
    
    print("\nMessage types present in files:")
    for msg, count in all_msgs.most_common():
        print(f"  {msg}: {count}/{len(all_files_stats)} files")

if __name__ == "__main__":
    main()
