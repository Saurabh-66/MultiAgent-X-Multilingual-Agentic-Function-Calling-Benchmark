#!/usr/bin/env python3
"""
Assemble final dataset from filtered records.
Produces train.jsonl, dev.jsonl, test.jsonl + stats.
"""

import json
import random
from pathlib import Path
from collections import defaultdict

INPUT_DIR = Path("/home/s2892267/multiagent-x/data/filtered")
OUTPUT_DIR = Path("/home/s2892267/multiagent-x/data/final")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)


def main():
    all_records = []

    for jsonl_path in sorted(INPUT_DIR.glob("*.jsonl")):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        all_records.append(json.loads(line))
                    except:
                        continue

    print(f"Total records loaded: {len(all_records)}")

    # Split
    train = [r for r in all_records if r.get("split") == "train"]
    dev   = [r for r in all_records if r.get("split") == "dev"]
    test  = [r for r in all_records if r.get("split") == "test"]

    # Shuffle train
    random.shuffle(train)

    # Save splits
    for split_name, split_data in [("train", train), ("dev", dev), ("test", test)]:
        out_path = OUTPUT_DIR / f"{split_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in split_data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{split_name}: {len(split_data)} records")

    # Statistics
    stats = {
        "total_records": len(all_records),
        "splits": {
            "train": len(train),
            "dev": len(dev),
            "test": len(test)
        },
        "languages": {},
        "domains": defaultdict(int),
        "example_types": defaultdict(int),
        "scripts": defaultdict(int),
        "resource_levels": defaultdict(int)
    }

    lang_stats = defaultdict(lambda: {"count": 0, "name": "", "script": "", "resource_level": ""})
    for r in all_records:
        lc = r.get("language_code", "?")
        lang_stats[lc]["count"] += 1
        lang_stats[lc]["name"] = r.get("language_name", "")
        lang_stats[lc]["script"] = r.get("script", "")
        lang_stats[lc]["resource_level"] = r.get("resource_level", "")
        stats["domains"][r.get("domain", "?")] += 1
        stats["example_types"][r.get("example_type", "?")] += 1
        stats["scripts"][r.get("script", "?")] += 1
        stats["resource_levels"][r.get("resource_level", "?")] += 1

    stats["languages"] = dict(lang_stats)
    stats["domains"] = dict(stats["domains"])
    stats["example_types"] = dict(stats["example_types"])
    stats["scripts"] = dict(stats["scripts"])
    stats["resource_levels"] = dict(stats["resource_levels"])

    stats_path = OUTPUT_DIR / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'='*60}")
    print(f"DATASET ASSEMBLED: {len(all_records)} total records")
    print(f"Languages covered: {len(lang_stats)}")
    print(f"\nPer-language counts:")
    for lc, data in sorted(lang_stats.items(), key=lambda x: -x[1]['count']):
        print(f"  {lc:6} {data['name']:20} {data['count']:5} records  [{data['script']}]")
    print(f"\nDomain distribution:")
    for domain, count in sorted(stats['domains'].items(), key=lambda x: -x[1]):
        print(f"  {domain:20} {count:5}")
    print(f"\nExample type distribution:")
    for et, count in sorted(stats['example_types'].items(), key=lambda x: -x[1]):
        print(f"  {et:25} {count:5}")
    print(f"\nStats saved: {stats_path}")


if __name__ == "__main__":
    main()
