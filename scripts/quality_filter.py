#!/usr/bin/env python3
"""
Quality filter multilingual records.
Removes: empty utterances, invalid function names, missing required args,
         identical utterance to English (failed translation), empty files.
Executed from hastings (cluster) after expansion completes (or partially completes).
"""

import json
import unicodedata
import re
from pathlib import Path
from collections import defaultdict

INPUT_DIR = Path("/home/s2892267/multiagent-x/data/multilingual")
OUTPUT_DIR = Path("/home/s2892267/multiagent-x/data/filtered")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = Path("/home/s2892267/multiagent-x/data/quality_report.json")

VALID_FUNCTIONS = {
    "book_clinic_appointment", "check_medication_availability",
    "request_ambulance", "get_vaccination_schedule", "report_disease_outbreak",
    "get_crop_disease_diagnosis", "get_market_price", "request_extension_worker",
    "get_weather_planting_advice", "register_cooperative_sale",
    "send_mobile_money", "check_balance", "pay_bill",
    "request_mobile_loan", "get_exchange_rate",
    "register_birth", "check_document_status",
    "report_infrastructure_issue", "apply_for_social_support",
    "report_emergency", "request_evacuation",
    "locate_nearest_shelter", "request_food_distribution",
    "ask_clarification"
}

REQUIRED_ARGS = {
    "book_clinic_appointment": ["facility_type", "urgency"],
    "request_ambulance": ["location_description", "emergency_type"],
    "send_mobile_money": ["recipient_number", "amount"],
    "report_emergency": ["emergency_type", "location", "affected_people"],
    "request_evacuation": ["location", "people_count", "urgency"],
    "locate_nearest_shelter": ["current_location", "people_count"],
    "get_crop_disease_diagnosis": ["crop_type", "symptoms"],
    "report_disease_outbreak": ["disease_type", "district"],
    "register_birth": ["child_name", "birth_date"],
    "check_document_status": ["document_type", "application_number"],
    "report_infrastructure_issue": ["issue_type", "location_description", "severity"],
    "apply_for_social_support": ["program_type", "district"],
    "ask_clarification": ["what_to_ask"],
}

NON_LATIN_SCRIPTS = {
    "am": ("\u1200", "\u137F"),  # Ethiopic
    "ti": ("\u1200", "\u137F"),
    "om": None,  # Latin
    "bn": ("\u0980", "\u09FF"),  # Bengali
    "or": ("\u0B00", "\u0B7F"),  # Odia
    "ne": ("\u0900", "\u097F"),  # Devanagari
    "hi": ("\u0900", "\u097F"),
    "pa": ("\u0A00", "\u0A7F"),  # Gurmukhi
    "si": ("\u0D80", "\u0DFF"),  # Sinhala
    "km": ("\u1780", "\u17FF"),  # Khmer
    "lo": ("\u0E80", "\u0EFF"),  # Lao
    "my": ("\u1000", "\u109F"),  # Myanmar
    "dz": ("\u0F00", "\u0FFF"),  # Tibetan
}


def check_record(record):
    errors = []
    utterance = record.get("user_utterance", "")
    example_type = record.get("example_type", "positive")
    expected = record.get("expected_output")
    lang_code = record.get("language_code", "")

    # 1. Empty utterance
    if not utterance or len(utterance.strip()) < 3:
        errors.append("empty_utterance")
        return errors

    # 2. Utterance identical to English reference (failed translation)
    ref = record.get("english_reference_utterance", "")
    if utterance.strip().lower() == ref.strip().lower() and lang_code != "en":
        errors.append("identical_to_english")

    # 3. Validate expected output
    if example_type == "negative":
        if expected is not None:
            errors.append("negative_should_be_null")
    elif expected is not None:
        if isinstance(expected, list):
            # Parallel call
            for call in expected:
                fn = call.get("function_name", "")
                if fn not in VALID_FUNCTIONS:
                    errors.append(f"invalid_function:{fn}")
        elif isinstance(expected, dict):
            fn = expected.get("function_name", "")
            if fn not in VALID_FUNCTIONS:
                errors.append(f"invalid_function:{fn}")
            elif fn in REQUIRED_ARGS:
                args = expected.get("arguments", {})
                for req in REQUIRED_ARGS[fn]:
                    if req not in args:
                        errors.append(f"missing_arg:{req}")

    # 4. Script check for non-Latin languages
    if lang_code in NON_LATIN_SCRIPTS and NON_LATIN_SCRIPTS[lang_code]:
        lo, hi = NON_LATIN_SCRIPTS[lang_code]
        has_script = any(lo <= c <= hi for c in utterance)
        if not has_script and len(utterance) > 10:
            errors.append(f"missing_native_script")

    return errors


def main():
    all_filtered = []
    report = {}

    jsonl_files = sorted(INPUT_DIR.glob("*.jsonl"))
    if not jsonl_files:
        print("No JSONL files found in multilingual dir")
        return

    for jsonl_path in jsonl_files:
        lang_code = jsonl_path.stem
        records = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        if not records:
            print(f"[{lang_code}] EMPTY — skipping")
            continue

        passed = []
        failed = []
        error_counts = defaultdict(int)

        for record in records:
            errors = check_record(record)
            if not errors:
                passed.append(record)
            else:
                failed.append((record, errors))
                for e in errors:
                    error_counts[e] += 1

        pass_rate = len(passed) / max(len(records), 1) * 100
        report[lang_code] = {
            "total": len(records),
            "passed": len(passed),
            "failed": len(failed),
            "pass_rate": round(pass_rate, 1),
            "errors": dict(error_counts)
        }

        # Save filtered
        out_path = OUTPUT_DIR / f"{lang_code}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in passed:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f"[{lang_code}] {len(passed)}/{len(records)} passed ({pass_rate:.1f}%) → {out_path.name}")
        all_filtered.extend(passed)

    # Save report
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    print(f"TOTAL FILTERED: {len(all_filtered)} records across {len(report)} languages")
    print(f"Quality report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
