"""Routing task from CLINC150 (Larson et al., 2019; CC BY 3.0): 150 intents in 10 domains plus out-of-scope.

Two question layouts are produced from every utterance:
  * `domain`: choice over the 10 domains, plus an "out of scope" option; the out-of-scope population is what the
    escalation gate should refuse, so records keep the OOS label (gold = index of "out of scope").
  * `intent150`: choice over all 150 intent names: the option-budget case (a single question with 150 options).
The state is the utterance; there is no shared long state in this task, which is why it is the small-S cell.
"""
from __future__ import annotations

from ..schema import DecisionRecord, Question

DATASET = ("clinc/clinc_oos", "plus")  # the "plus" configuration has 250 extra OOS training utterances
OOS = "out of scope"

# CLINC150 domain -> intents (from the dataset paper's appendix, 15 intents per domain)
DOMAINS: dict[str, list[str]] = {
    "banking": ["transfer", "transactions", "balance", "freeze_account", "pay_bill", "bill_balance", "bill_due",
                "interest_rate", "routing", "min_payment", "order_checks", "pin_change", "report_fraud",
                "account_blocked", "spending_history"],
    "credit cards": ["credit_score", "report_lost_card", "credit_limit", "rewards_balance", "new_card",
                     "application_status", "card_declined", "international_fees", "apr", "redeem_rewards",
                     "credit_limit_change", "damaged_card", "replacement_card_duration", "improve_credit_score",
                     "expiration_date"],
    "kitchen and dining": ["recipe", "restaurant_reviews", "calories", "nutrition_info", "restaurant_suggestion",
                           "ingredients_list", "ingredient_substitution", "cook_time", "food_last",
                           "meal_suggestion", "restaurant_reservation", "confirm_reservation", "how_busy",
                           "cancel_reservation", "accept_reservations"],
    "home": ["shopping_list", "shopping_list_update", "next_song", "play_music", "update_playlist", "todo_list",
             "todo_list_update", "calendar", "calendar_update", "what_song", "order", "order_status", "reminder",
             "reminder_update", "smart_home"],
    "auto and commute": ["traffic", "directions", "gas", "gas_type", "distance", "current_location", "mpg",
                         "oil_change_when", "oil_change_how", "jump_start", "uber", "schedule_maintenance",
                         "last_maintenance", "tire_pressure", "tire_change"],
    "travel": ["book_flight", "book_hotel", "car_rental", "travel_suggestion", "travel_alert", "travel_notification",
               "carry_on", "timezone", "vaccines", "translate", "flight_status", "international_visa",
               "lost_luggage", "plug_type", "exchange_rate"],
    "utility": ["time", "alarm", "share_location", "find_phone", "weather", "text", "spelling", "make_call",
                "timer", "date", "calculator", "measurement_conversion", "flip_coin", "roll_dice", "definition"],
    "work": ["direct_deposit", "pto_request", "taxes", "payday", "w2", "pto_balance", "pto_request_status",
             "next_holiday", "insurance", "insurance_change", "schedule_meeting", "pto_used", "meeting_schedule",
             "rollover_401k", "income"],
    "small talk": ["greeting", "goodbye", "tell_joke", "where_are_you_from", "how_old_are_you", "what_is_your_name",
                   "who_made_you", "thank_you", "what_can_i_ask_you", "what_are_your_hobbies", "do_you_have_pets",
                   "are_you_a_bot", "meaning_of_life", "who_do_you_work_for", "fun_fact"],
    "meta": ["change_ai_name", "change_user_name", "cancel", "user_name", "reset_settings", "whisper_mode", "repeat",
             "no", "yes", "maybe", "change_language", "change_accent", "change_volume", "change_speed", "sync_device"],
}
DOMAIN_NAMES = list(DOMAINS)
INTENT_TO_DOMAIN = {i: d for d, ints in DOMAINS.items() for i in ints}
INTENTS = sorted(INTENT_TO_DOMAIN)
DOMAIN_OPTIONS = DOMAIN_NAMES + [OOS]
INTENT_OPTIONS = [i.replace("_", " ") for i in INTENTS] + [OOS]

assert len(INTENTS) == 150 and len(DOMAIN_NAMES) == 10


def domain_question() -> Question:
    return Question("choice", "Which domain does this request belong to?", list(DOMAIN_OPTIONS), qid="domain")


def intent_question() -> Question:
    return Question("choice", "Which intent does this request express?", list(INTENT_OPTIONS), qid="intent150")


def records_from_rows(rows, label_names: list[str], split_name: str, layout: str = "domain") -> list[DecisionRecord]:
    """`rows` are dicts with `text` and `intent` (integer label into label_names, where 'oos' is out of scope)."""
    out = []
    for i, row in enumerate(rows):
        name = label_names[int(row["intent"])]
        text = row["text"]
        if layout == "domain":
            q = domain_question()
            gold = DOMAIN_OPTIONS.index(OOS) if name == "oos" else DOMAIN_OPTIONS.index(INTENT_TO_DOMAIN[name])
        elif layout == "intent150":
            q = intent_question()
            gold = INTENT_OPTIONS.index(OOS) if name == "oos" else INTENTS.index(name)
        else:
            raise ValueError(layout)
        out.append(DecisionRecord("routing", text, q, gold, "dataset", group_id=f"{split_name}:{i}",
                                  meta={"intent": name, "oos": name == "oos", "layout": layout}))
    return out


def load_rows(split: str):
    """Download-on-demand loader; kept separate so tests can feed rows directly."""
    from datasets import load_dataset

    ds = load_dataset(DATASET[0], DATASET[1], split=split)
    return list(ds), ds.features["intent"].names
