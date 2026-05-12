#!/usr/bin/env python3
"""
Expand English seeds into 30 languages using Qwen3-14B (or Qwen3.6-27B-FP8).
Run from hastings after seeds are complete.

Usage:
    python scripts/expand_multilingual.py
"""

import json
import time
import re
import sys
from pathlib import Path
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT = Path("/home/s2892267/multiagent-x")
SEEDS_PATH = PROJECT / "data" / "seeds" / "all_seeds_en.json"
OUTPUT_DIR = PROJECT / "data" / "multilingual"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VLLM_HOST = "saxa"
VLLM_PORT = 8102
SERVED_MODEL_NAME = "qwen3-14b"

# ── 30 Target Languages ────────────────────────────────────────────────────────
LANGUAGES = [
    {"code": "am", "name": "Amharic", "script": "Ethiopic", "resource_level": "very_low",
     "context": "Ethiopia. Mobile money via Telebirr. Government via kebele offices. Healthcare via government hospitals and health centres. Crops: teff, maize, coffee, sorghum. Use Amharic script."},
    {"code": "ti", "name": "Tigrinya", "script": "Ethiopic", "resource_level": "very_low",
     "context": "Eritrea and northern Ethiopia (Tigray). Similar to Amharic context. Healthcare via community health posts. Use Tigrinya/Ethiopic script."},
    {"code": "so", "name": "Somali", "script": "Latin", "resource_level": "very_low",
     "context": "Somalia and Somaliland. Mobile money via Hormuud EVC+. Healthcare via NGO and government facilities. Livestock and agriculture common."},
    {"code": "ha", "name": "Hausa", "script": "Latin", "resource_level": "low",
     "context": "Northern Nigeria and Niger. Mobile money via OPay, MTN MoMo. Crops: groundnuts, millet, sorghum, cotton. Healthcare via PHCs (Primary Health Centres)."},
    {"code": "yo", "name": "Yoruba", "script": "Latin", "resource_level": "low",
     "context": "Southwestern Nigeria. Mobile money via OPay, Kuda. Crops: cassava, yam, cocoa. Healthcare via general hospitals."},
    {"code": "ig", "name": "Igbo", "script": "Latin", "resource_level": "low",
     "context": "Southeastern Nigeria. Mobile money via OPay. Crops: yam, cassava, palm oil. Strong trader culture."},
    {"code": "sw", "name": "Swahili", "script": "Latin", "resource_level": "medium",
     "context": "East Africa (Kenya, Tanzania, Uganda). Mobile money via M-Pesa. Crops: tea, coffee, maize. Healthcare via Level 4/5 hospitals."},
    {"code": "om", "name": "Oromo", "script": "Latin", "resource_level": "very_low",
     "context": "Ethiopia and Kenya. Largest Ethiopian language group. Crops: coffee (origin), teff, maize. Mobile money via Telebirr."},
    {"code": "wo", "name": "Wolof", "script": "Latin", "resource_level": "very_low",
     "context": "Senegal and Gambia. Mobile money via Orange Money, Wave. Crops: groundnuts, millet. Healthcare via district hospitals."},
    {"code": "ak", "name": "Twi (Akan)", "script": "Latin", "resource_level": "very_low",
     "context": "Ghana. Mobile money via MTN MoMo, Vodafone Cash. Crops: cocoa, yam, cassava. Healthcare via CHPS compounds and district hospitals."},
    {"code": "ln", "name": "Lingala", "script": "Latin", "resource_level": "very_low",
     "context": "Democratic Republic of Congo and Congo-Brazzaville. Mobile money via Airtel Money, Orange Money. Crops: cassava, plantain. Humanitarian NGO presence."},
    {"code": "rw", "name": "Kinyarwanda", "script": "Latin", "resource_level": "low",
     "context": "Rwanda and Burundi. Mobile money via MTN MoMo, Airtel. Government services via Irembo platform. Crops: coffee, tea, beans."},
    {"code": "mg", "name": "Malagasy", "script": "Latin", "resource_level": "very_low",
     "context": "Madagascar. Mobile money via MVola, Airtel Money. Crops: rice, vanilla, cloves. Healthcare via fokontany health posts."},
    {"code": "bn", "name": "Bengali", "script": "Bengali", "resource_level": "low",
     "context": "Bangladesh. Mobile money via bKash, Nagad, Rocket. Crops: rice, jute. Healthcare via upazila health complexes. Garment industry workers common. Use Bengali script."},
    {"code": "or", "name": "Odia", "script": "Odia", "resource_level": "very_low",
     "context": "Odisha state, India. Government services via MoSeva. Crops: rice, vegetables. Healthcare via PHCs and CHCs. Use Odia script."},
    {"code": "ne", "name": "Nepali", "script": "Devanagari", "resource_level": "low",
     "context": "Nepal. Mobile money via eSewa, Khalti. Crops: rice, maize, vegetables. Many migrant workers. Healthcare via health posts and district hospitals. Use Devanagari script."},
    {"code": "hi", "name": "Hindi", "script": "Devanagari", "resource_level": "medium",
     "context": "Rural India (UP, Bihar, MP, Rajasthan). Mobile money via PhonePe, Paytm. Government via DigiLocker, Common Service Centres. Crops: wheat, rice, sugarcane. Healthcare via PHCs. Use Devanagari script. Focus on rural/informal register not urban elite Hindi."},
    {"code": "pa", "name": "Punjabi", "script": "Gurmukhi", "resource_level": "low",
     "context": "Punjab (India and Pakistan). Mobile money via JazzCash, EasyPaisa (Pakistan) or PhonePe (India). Crops: wheat, rice. Large diaspora remittance culture. Use Gurmukhi script for Indian Punjabi."},
    {"code": "si", "name": "Sinhala", "script": "Sinhala", "resource_level": "low",
     "context": "Sri Lanka. Mobile money via eZ Cash, mCash. Crops: tea, rubber, coconut, rice. Healthcare via government hospitals. Use Sinhala script."},
    {"code": "km", "name": "Khmer", "script": "Khmer", "resource_level": "very_low",
     "context": "Cambodia. Mobile money via Wing, ABA. Crops: rice, cassava, sugarcane. Healthcare via referral hospitals. Use Khmer script."},
    {"code": "lo", "name": "Lao", "script": "Lao", "resource_level": "very_low",
     "context": "Laos. Mobile money via BCEL One, U-Money. Crops: rice, corn, sugarcane. Healthcare via district hospitals. Use Lao script."},
    {"code": "my", "name": "Burmese", "script": "Myanmar", "resource_level": "very_low",
     "context": "Myanmar. Mobile money via KBZPay, Wave Money. Crops: rice, beans, pulses. Healthcare access limited. Use Myanmar script."},
    {"code": "ht", "name": "Haitian Creole", "script": "Latin", "resource_level": "very_low",
     "context": "Haiti. Mobile money via MonCash, Lajan Cash. NGO healthcare common. Crops: mango, coffee, sugarcane. Creole is the first language of most Haitians, not French."},
    {"code": "qu", "name": "Quechua", "script": "Latin", "resource_level": "very_low",
     "context": "Peru, Bolivia, Ecuador (Andes). Government services via RENIEC (Peru). Crops: potato (origin), quinoa, maize. Healthcare via ESSALUD and MINSA health posts."},
    {"code": "gn", "name": "Guarani", "script": "Latin", "resource_level": "very_low",
     "context": "Paraguay. Both Spanish and Guarani used. Crops: soy, cassava, sugarcane. Mobile money via Tigo Money, Personal Pay."},
    {"code": "tl", "name": "Tagalog", "script": "Latin", "resource_level": "medium",
     "context": "Philippines. Mobile money via GCash, Maya. Crops: rice, corn, coconut, banana. Many OFW (overseas worker) remittances. Healthcare via barangay health centres and PhilHealth."},
    {"code": "ceb", "name": "Cebuano", "script": "Latin", "resource_level": "very_low",
     "context": "Visayas and Mindanao, Philippines. Mobile money via GCash, Maya. Crops: coconut, sugarcane, corn. Healthcare via rural health units."},
    {"code": "ps", "name": "Pashto", "script": "Arabic", "resource_level": "very_low",
     "context": "Afghanistan and northwest Pakistan. Healthcare via NGO clinics and government hospitals. Crops: wheat, corn, fruits. Humanitarian emergency context common. Use Arabic/Nastaliq script."},
    {"code": "dje", "name": "Zarma", "script": "Latin", "resource_level": "very_low",
     "context": "Niger and Mali. One of the most under-resourced languages. Mobile money via Orange Money. Crops: millet, sorghum, cowpea. Healthcare via district health centres (CSI)."},
    {"code": "dz", "name": "Dzongkha", "script": "Tibetan", "resource_level": "very_low",
     "context": "Bhutan. Government services via G2C (Government to Citizen) portal. Crops: rice, maize, cardamom. Healthcare via BHUs (Basic Health Units). Use Tibetan/Dzongkha script."},
]

# ── Adaptation prompt ──────────────────────────────────────────────────────────
ADAPT_PROMPT = """You are adapting AI agent training data into {language_name} for a multilingual benchmark.

Source English utterance:
"{english_utterance}"

Target language: {language_name} ({script} script)
Cultural context for this language:
{cultural_context}

Domain: {domain}
Example type: {example_type}

Original function call:
{original_output}

Your task:
1. Rewrite the user_utterance as a NATIVE {language_name} speaker would naturally say it
2. Adapt cultural references: names, places, services, crops, amounts to be locally authentic for {language_name} speakers
3. Keep the SEMANTIC INTENT identical — the same function should be called with equivalent arguments
4. Adapt argument VALUES to local context (e.g., use local currency, local service names, local place names)
5. Use natural colloquial {language_name}, not formal or literary language
6. For non-Latin scripts: MUST use the correct script ({script})

Output a single JSON object (no markdown, no explanation):
{{
  "user_utterance": "(natural {language_name} text in {script} script)",
  "cultural_context_note": "(brief note on what was culturally adapted)",
  "expected_output": {original_output},
  "reasoning_trace": "(one sentence explaining the function call in English)"
}}

Output ONLY the JSON object."""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=20))
def adapt_record(client, seed, lang):
    """Adapt a single English seed record to a target language."""
    original_output = json.dumps(seed.get("expected_output"), ensure_ascii=False)

    prompt = ADAPT_PROMPT.format(
        language_name=lang["name"],
        script=lang["script"],
        english_utterance=seed["user_utterance"],
        cultural_context=lang["context"],
        domain=seed["domain"],
        example_type=seed.get("example_type", "positive"),
        original_output=original_output
    )

    response = client.chat.completions.create(
        model=SERVED_MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "You are an expert multilingual dataset creator. Output ONLY valid JSON. No markdown. No explanation. /no_think"
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.4,
        max_tokens=1500,
    )

    text = response.choices[0].message.content.strip()
    # Strip thinking tokens
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    # Strip markdown fences
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        if text.strip().endswith("```"):
            text = text.strip()[:-3]

    return json.loads(text.strip())


def build_record(adapted, seed, lang, record_id):
    """Merge adapted content with full metadata."""
    return {
        "id": f"max_{seed['domain']}_{lang['code']}_{seed.get('example_type','pos')}_{record_id:06d}",
        "language_code": lang["code"],
        "language_name": lang["name"],
        "script": lang["script"],
        "resource_level": lang["resource_level"],
        "domain": seed["domain"],
        "example_type": seed.get("example_type", "positive"),
        "difficulty": seed.get("difficulty", "medium"),
        "ambiguity_type": seed.get("ambiguity_type", "none"),
        "user_utterance": adapted.get("user_utterance", ""),
        "english_reference_utterance": seed["user_utterance"],
        "cultural_context_note": adapted.get("cultural_context_note", ""),
        "expected_output": adapted.get("expected_output", seed.get("expected_output")),
        "reasoning_trace": adapted.get("reasoning_trace", seed.get("reasoning_trace", "")),
        "previous_context": seed.get("previous_context", ""),
        "split": assign_split(seed["id"]),
        "source": "synthetic_culturally_grounded",
        "created_with": "Adaptive Data by Adaption",
        "validated": False
    }


def assign_split(seed_id):
    """Deterministic 70/15/15 train/dev/test split."""
    h = hash(seed_id) % 100
    if h < 70: return "train"
    elif h < 85: return "dev"
    else: return "test"


def get_completed_languages():
    """Return set of language codes already processed."""
    completed = set()
    for f in OUTPUT_DIR.glob("*.jsonl"):
        completed.add(f.stem)
    return completed


def main():
    print(f"Loading seeds from {SEEDS_PATH}")
    seeds = json.load(open(SEEDS_PATH))
    print(f"Loaded {len(seeds)} English seeds")

    client = OpenAI(
        base_url=f"http://{VLLM_HOST}:{VLLM_PORT}/v1",
        api_key="not-needed"
    )

    # Connectivity check
    try:
        models = client.models.list()
        print(f"Connected to {VLLM_HOST}:{VLLM_PORT}. Models: {[m.id for m in models.data]}")
    except Exception as e:
        print(f"ERROR: Cannot connect to {VLLM_HOST}:{VLLM_PORT}: {e}")
        sys.exit(1)

    completed = get_completed_languages()
    if completed:
        print(f"Already completed languages: {completed}")

    total_records = 0
    record_id = 0

    for lang in LANGUAGES:
        lang_code = lang["code"]

        if lang_code in completed:
            print(f"[{lang_code}] Already done, skipping.")
            # Count existing records
            existing = sum(1 for _ in open(OUTPUT_DIR / f"{lang_code}.jsonl"))
            total_records += existing
            record_id += existing
            continue

        print(f"\n[{lang_code}] {lang['name']} ({lang['script']}) — processing {len(seeds)} seeds...")
        out_path = OUTPUT_DIR / f"{lang_code}.jsonl"
        errors = 0
        lang_count = 0

        with open(out_path, "w", encoding="utf-8") as out_f:
            for i, seed in enumerate(seeds):
                try:
                    adapted = adapt_record(client, seed, lang)
                    record = build_record(adapted, seed, lang, record_id)
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_f.flush()
                    lang_count += 1
                    record_id += 1

                    if (i + 1) % 100 == 0:
                        print(f"  [{lang_code}] {i+1}/{len(seeds)} done ({errors} errors)")

                except Exception as e:
                    errors += 1
                    if errors <= 5:
                        print(f"  [{lang_code}] Error on seed {i}: {e}")
                    continue

                time.sleep(0.05)  # small delay to avoid overwhelming server

        total_records += lang_count
        print(f"  [{lang_code}] DONE: {lang_count} records saved ({errors} errors) -> {out_path}")

    print(f"\n{'='*60}")
    print(f"EXPANSION COMPLETE: {total_records} total multilingual records")
    print(f"Files in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
