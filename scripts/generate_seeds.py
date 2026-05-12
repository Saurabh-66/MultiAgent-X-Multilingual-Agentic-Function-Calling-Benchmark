#!/usr/bin/env python3
"""
Generate English seed records using Qwen3.6-27B-FP8 via vLLM.
Run this after starting the vLLM server separately.

Usage:
    python generate_seeds.py
"""

import json
import time
import sys
from pathlib import Path
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT = Path("/home/s2892267/multiagent-x")
SCHEMA_PATH = PROJECT / "schemas" / "functions.json"
OUTPUT_DIR = PROJECT / "data" / "seeds"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VLLM_HOST = "saxa"
VLLM_PORT = 8101
SERVED_MODEL_NAME = "qwen3-32b"

# ── Domain definitions ─────────────────────────────────────────────────────────
DOMAINS = {
    "healthcare": {
        "functions": [
            "book_clinic_appointment", "check_medication_availability",
            "request_ambulance", "get_vaccination_schedule", "report_disease_outbreak"
        ],
        "context": (
            "Sub-Saharan African or South Asian government/community health system. "
            "Users are patients, caregivers, community health workers (CHWs), village leaders. "
            "Facilities are government hospitals, health centres, community clinics, mobile units. "
            "NOT private hospitals or insurance systems like ZocDoc. "
            "Use real place names: Addis Ababa, Oromia, Lagos, Kano, Dhaka, Sylhet, Kathmandu, Phnom Penh. "
            "Amounts in local currency. CHW referrals common in rural Ethiopia, Kenya, Bangladesh."
        )
    },
    "agriculture": {
        "functions": [
            "get_crop_disease_diagnosis", "get_market_price",
            "request_extension_worker", "get_weather_planting_advice",
            "register_cooperative_sale"
        ],
        "context": (
            "Smallholder farming in Sub-Saharan Africa or South Asia. "
            "Users are farmers, extension workers, cooperative members. "
            "Crops: maize, sorghum, cassava, rice, teff, millet, groundnuts, yam, coffee, tea, cotton. "
            "Markets are local/regional: Mercato Addis Ababa, Mile 12 Lagos, Kawran Bazar Dhaka. "
            "Extension workers are government agricultural officers who visit farms. "
            "Cooperatives handle bulk grain/coffee sales in Ethiopia, Kenya, Rwanda."
        )
    },
    "mobile_finance": {
        "functions": [
            "send_mobile_money", "check_balance", "pay_bill",
            "request_mobile_loan", "get_exchange_rate"
        ],
        "context": (
            "Mobile money ecosystem in Africa and South Asia. "
            "M-Pesa in Kenya/Tanzania, MTN MoMo in Uganda/Ghana/Cameroon, "
            "Telebirr in Ethiopia, EVC+ in Somalia, Orange Money in Senegal/Mali, "
            "bKash/Nagad in Bangladesh, eSewa/Khalti in Nepal, GCash in Philippines, "
            "OPay/Opay in Nigeria, Wave in Senegal/Gambia. "
            "Typical amounts: 50-5000 KES, 100-10000 ETB, 200-20000 BDT, 500-50000 NGN. "
            "Users often unbanked, using basic smartphones."
        )
    },
    "government": {
        "functions": [
            "register_birth", "check_document_status",
            "report_infrastructure_issue", "apply_for_social_support"
        ],
        "context": (
            "Low-income country government services accessed via mobile or local office. "
            "Countries: Ethiopia, Nigeria, Kenya, Ghana, Bangladesh, Nepal, Cambodia, Haiti, Philippines. "
            "Documents: birth certificates (kebele in Ethiopia), national IDs, land titles. "
            "Social programs: PSNP Ethiopia, BISP Pakistan, VUP Rwanda, 4Ps Philippines. "
            "Infrastructure issues: unpaved roads, broken boreholes, no electricity in village."
        )
    },
    "emergency": {
        "functions": [
            "report_emergency", "request_evacuation",
            "locate_nearest_shelter", "request_food_distribution"
        ],
        "context": (
            "Humanitarian emergency response in disaster-prone or conflict-affected regions. "
            "Flooding in Bangladesh/Nigeria/South Sudan, drought in Ethiopia/Somalia/Kenya, "
            "earthquakes in Nepal/Haiti, conflict in DRC/South Sudan/Myanmar. "
            "Users: community leaders, affected individuals, local NGO workers. "
            "Coordination with UNHCR, WFP, government disaster management agencies."
        )
    }
}

# ── Example type definitions ───────────────────────────────────────────────────
EXAMPLE_TYPES = {
    "positive_easy": {
        "count": 60,
        "desc": (
            "Clear unambiguous request. All required arguments directly extractable. "
            "User states exactly what they need with enough detail to call the function. "
            "Example: 'I need to book a routine checkup for my 8-month-old at the government health centre in Kirkos tomorrow.'"
        )
    },
    "positive_medium": {
        "count": 40,
        "desc": (
            "Intent clear but some arguments require inference or regional knowledge. "
            "Model must reason about context to fill arguments correctly. "
            "Example: 'My sorghum leaves are turning brown and curling at the edges near Bahir Dar, what is wrong?'"
        )
    },
    "positive_hard": {
        "count": 20,
        "desc": (
            "Indirect, culturally specific, or idiomatic phrasing. "
            "Requires deep contextual understanding to identify correct function and arguments. "
            "May use local expressions, proverbs, or indirect speech patterns. "
            "Example: 'The sky has been unkind this season. My fields in Jimma are waiting. Should I act now?'"
        )
    },
    "multi_turn_q": {
        "count": 20,
        "desc": (
            "Turn 1 of a 2-turn dialogue. Intent is clear but a REQUIRED argument is completely missing. "
            "The agent cannot call the function yet. Must use ask_clarification. "
            "The what_to_ask field must be a specific natural question to the user. "
            "Example user message: 'I want to send money to my brother.' "
            "Missing: recipient_number, amount, service_provider. "
            "what_to_ask: 'Sure, I can help with that. What is your brother's mobile number, how much would you like to send, and which service should I use?'"
        )
    },
    "multi_turn_a": {
        "count": 20,
        "desc": (
            "Turn 2 of a 2-turn dialogue. User has now provided the missing info from turn 1. "
            "The agent should NOW make the complete function call with all arguments. "
            "Include a 'previous_context' field describing what was asked in turn 1. "
            "Example user message: 'His number is 0712345678, send 500 shillings via M-Pesa.' "
            "Now all arguments are available: call send_mobile_money with full args."
        )
    },
    "negative": {
        "count": 20,
        "desc": (
            "User asks something that NONE of the available functions can address. "
            "expected_output must be JSON null. Do not force-fit to any function. "
            "Examples: asking for weather forecast, telling a story, asking for directions, "
            "asking what time it is, asking for recipe, casual conversation. "
            "These are genuine out-of-scope requests."
        )
    },
    "parallel": {
        "count": 10,
        "desc": (
            "User request genuinely requires calling EXACTLY TWO functions simultaneously. "
            "Both calls needed to satisfy the complete request. "
            "expected_output must be a JSON array containing exactly 2 function call objects. "
            "Example: 'Check the price of maize at Mercato AND book an extension worker for soil testing this week.' "
            "This requires get_market_price AND request_extension_worker both called."
        )
    },
    "ambiguous": {
        "count": 10,
        "desc": (
            "Intent is identifiable but a REQUIRED argument cannot be determined at all from the utterance. "
            "Use ask_clarification. The missing_fields array must list the missing argument names. "
            "Example: 'I want to report something broken in my community.' "
            "issue_type is missing (road? water? electricity?), location_description is missing. "
            "what_to_ask: 'I can report that for you. What type of infrastructure is damaged, and where exactly is it located?'"
        )
    }
}

# ── Prompt template ────────────────────────────────────────────────────────────
PROMPT = """You are building training data for a multilingual AI agent benchmark for Global South communities.

DOMAIN: {domain_name}
CONTEXT: {domain_context}

AVAILABLE FUNCTIONS: {function_names}

FUNCTION SCHEMAS:
{function_schemas}

TASK: Generate exactly {count} English examples of type "{example_type}".

TYPE DESCRIPTION:
{type_desc}

ABSOLUTE RULES:
1. user_utterance must sound like a real person speaking naturally, not an API query
2. Use authentic names, places, services for Sub-Saharan Africa or South Asia
3. For negative: expected_output = null (JSON null, not string)
4. For parallel: expected_output = array of exactly 2 function call objects
5. For ask_clarification types: function_name = "ask_clarification"
6. Amounts in local currency only (KES, ETB, BDT, NGN, NPR etc)
7. reasoning_trace: 1-2 sentences explaining the decision
8. cultural_context_note: explain what makes this regionally authentic
9. difficulty must be one of: easy, medium, hard
10. ambiguity_type must be one of: none, missing_arg, underspecified, no_matching_function, conflicting_intent

OUTPUT: Return ONLY a valid JSON array. No markdown. No explanation. No text before or after.

Schema for each element:
{{
  "user_utterance": "string",
  "cultural_context_note": "string",
  "expected_output": null | {{"function_name": "string", "arguments": {{...}}}} | [{{"function_name": "...", "arguments": {{...}}}}, {{"function_name": "...", "arguments": {{...}}}}],
  "reasoning_trace": "string",
  "difficulty": "easy|medium|hard",
  "ambiguity_type": "none|missing_arg|underspecified|no_matching_function|conflicting_intent",
  "example_type": "{example_type}",
  "domain": "{domain_name}",
  "previous_context": "string or empty string"
}}

Generate exactly {count} objects now:"""


# ── vLLM client ────────────────────────────────────────────────────────────────
@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=4, max=30))
def call_model(client, prompt_text):
    response = client.chat.completions.create(
        model=SERVED_MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert multilingual dataset creator. /no_think "
                    "You output ONLY valid JSON arrays with no markdown, no explanation, no preamble."
                )
            },
            {"role": "user", "content": prompt_text}
        ],
        temperature=0.75,
        max_tokens=6000,
    )
    return response.choices[0].message.content.strip()


def parse_response(text):
    """Strip thinking tokens, markdown fences and parse JSON."""
    # Strip <think>...</think> blocks
    import re
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    # Strip markdown fences
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        if text.strip().endswith("```"):
            text = text.strip()[:-3]
    return json.loads(text.strip())


def main():
    print(f"Loading function schemas from {SCHEMA_PATH}")
    all_schemas = json.load(open(SCHEMA_PATH))
    print(f"Loaded {len(all_schemas)} functions")

    client = OpenAI(
        base_url=f"http://{VLLM_HOST}:{VLLM_PORT}/v1",
        api_key="not-needed"
    )

    # Quick connectivity check
    try:
        models = client.models.list()
        print(f"vLLM server connected. Models: {[m.id for m in models.data]}")
    except Exception as e:
        print(f"ERROR: Cannot connect to vLLM server on port {VLLM_PORT}")
        print(f"  Start it first with: python scripts/serve_27b.py")
        print(f"  Error: {e}")
        sys.exit(1)

    all_seeds = []
    record_id = 0

    for domain_name, domain_config in DOMAINS.items():
        print(f"\n{'='*60}")
        print(f"DOMAIN: {domain_name.upper()}")
        print(f"{'='*60}")
        domain_seeds = []

        # Get relevant schemas for this domain
        relevant = {
            k: v for k, v in all_schemas.items()
            if k in domain_config["functions"] or k == "ask_clarification"
        }

        for ex_type, type_config in EXAMPLE_TYPES.items():
            count = type_config["count"]
            CHUNK = 20  # max examples per API call to stay within 8192 token limit
            collected = []
            chunks = [CHUNK] * (count // CHUNK) + ([count % CHUNK] if count % CHUNK else [])
            print(f"  [{ex_type}] Generating {count} examples ({len(chunks)} chunks)...", end=" ", flush=True)

            for chunk_size in chunks:
                prompt_text = PROMPT.format(
                    domain_name=domain_name,
                    domain_context=domain_config["context"],
                    function_names=", ".join(domain_config["functions"]),
                    function_schemas=json.dumps(relevant, indent=2),
                    count=chunk_size,
                    example_type=ex_type,
                    type_desc=type_config["desc"]
                )

                try:
                    raw = call_model(client, prompt_text)
                    examples = parse_response(raw)
                    collected.extend(examples)
                    time.sleep(0.3)
                except json.JSONDecodeError as e:
                    print(f"PARSE ERROR: {e}")
                    print(f"  Raw response snippet: {raw[:200]}")
                    continue
                except Exception as e:
                    print(f"ERROR: {e}")
                    continue

            for ex in collected:
                ex["id"] = f"seed_{domain_name}_en_{ex_type}_{record_id:05d}"
                ex["language_code"] = "en"
                ex["language_name"] = "English"
                ex["script"] = "Latin"
                ex["split"] = "seed"
                ex["source"] = "synthetic_culturally_grounded"
                ex["created_with"] = "Adaptive Data by Adaption"
                record_id += 1
                domain_seeds.append(ex)

            print(f"OK ({len(collected)} records)")
            time.sleep(0.5)

        # Save per domain immediately
        out_path = OUTPUT_DIR / f"seeds_{domain_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(domain_seeds, f, ensure_ascii=False, indent=2)
        print(f"\n  Saved {len(domain_seeds)} seeds -> {out_path}")
        all_seeds.extend(domain_seeds)

    # Save combined
    combined_path = OUTPUT_DIR / "all_seeds_en.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_seeds, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"COMPLETE: {len(all_seeds)} total seeds -> {combined_path}")

    # Validation summary
    has_utterance = sum(1 for s in all_seeds if s.get("user_utterance", "").strip())
    has_output = sum(1 for s in all_seeds if "expected_output" in s)
    has_cultural = sum(1 for s in all_seeds if s.get("cultural_context_note", "").strip())
    print(f"Validation: {has_utterance} have utterance, {has_output} have output, {has_cultural} have cultural note")


if __name__ == "__main__":
    main()
