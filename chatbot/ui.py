import re
import streamlit as st

from api_client import get_required_features, predict_price
from extractor import extract_features, apply_defaults


CRITICAL_FIELDS = [
    "room type",
    "minimum nights",
    "availability 365",
    "number of reviews",
    "review rate number",
    "service fee",
]


st.set_page_config(page_title="Airbnb Price Chatbot", page_icon="🏠", layout="centered")
st.title("🏠 Airbnb Price Prediction Chatbot (US Dataset)")
st.caption("Chat → extract → ask missing details → predict")


if "booted" not in st.session_state:
    st.session_state.booted = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_features" not in st.session_state:
    st.session_state.pending_features = None


def parse_kv_reply(text: str) -> dict:
    """
    Parses replies like:
    room type: Entire home/apt, minimum nights: 2, availability 365: 200
    """
    out = {}
    parts = re.split(r",|\n", text)
    for p in parts:
        if ":" not in p:
            continue
        k, v = p.split(":", 1)
        key = k.strip().lower()
        val = v.strip()

        if key in {"room type", "room_type"}:
            val_low = val.lower()
            if "entire" in val_low:
                out["room type"] = "Entire home/apt"
            elif "private" in val_low:
                out["room type"] = "Private room"
            elif "shared" in val_low:
                out["room type"] = "Shared room"
            else:
                out["room type"] = val
            continue

        numeric_map = {
            "minimum nights": "minimum nights",
            "availability 365": "availability 365",
            "number of reviews": "number of reviews",
            "review rate number": "review rate number",
            "service fee": "service fee",
            "reviews per month": "reviews per month",
        }

        if key in numeric_map:
            try:
                out[numeric_map[key]] = float(val)
            except Exception:
                pass

    return out


def missing_critical(features: dict):
    return [f for f in CRITICAL_FIELDS if f not in features]


if not st.session_state.booted:
    cat, num, allf = get_required_features()
    st.session_state.categorical = cat
    st.session_state.numerical = num
    st.session_state.all_features = allf
    st.session_state.booted = True


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


user_msg = st.chat_input("Describe your listing...")


if user_msg:
    st.session_state.messages.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    with st.chat_message("assistant"):
        with st.spinner("Processing..."):

            # ✅ CASE 1: We are waiting for follow-up values
            if st.session_state.pending_features is not None:
                followup = parse_kv_reply(user_msg)
                merged = {**st.session_state.pending_features, **followup}

                missing = missing_critical(merged)

                if missing:
                    reply = (
                        "I still need a few more details:\n\n"
                        + "\n".join([f"- **{m}**" for m in missing])
                        + "\n\nReply like:\n"
                          "`room type: Entire home/apt, minimum nights: 2, availability 365: 200, number of reviews: 50, review rate number: 4.5, service fee: 25`"
                    )
                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    st.session_state.pending_features = merged
                else:
                    final_features = apply_defaults(
                        merged,
                        st.session_state.categorical,
                        st.session_state.numerical,
                    )
                    result = predict_price(final_features)

                    price = result.get("predicted_price")
                    currency = result.get("currency", "USD")

                    reply = (
                        f"✅ **Prediction Result**\n\n"
                        f"💰 **{currency} {price} / night**\n\n"
                        f"✅ **Merged extracted + followup**\n```json\n{merged}\n```\n\n"
                        f"✅ **Final payload**\n```json\n{final_features}\n```"
                    )
                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    st.session_state.pending_features = None

            # ✅ CASE 2: Fresh message → extract normally
            else:
                extracted = extract_features(
                    user_msg,
                    st.session_state.categorical,
                    st.session_state.numerical,
                )

                missing = missing_critical(extracted)

                if missing:
                    st.session_state.pending_features = extracted
                    reply = (
                        "I need a few more details to predict accurately:\n\n"
                        + "\n".join([f"- **{m}**" for m in missing])
                        + "\n\nReply with values like:\n"
                          "`room type: Entire home/apt, minimum nights: 2, availability 365: 200, number of reviews: 50, review rate number: 4.5, service fee: 25`"
                    )
                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                else:
                    final_features = apply_defaults(
                        extracted,
                        st.session_state.categorical,
                        st.session_state.numerical,
                    )
                    result = predict_price(final_features)

                    price = result.get("predicted_price")
                    currency = result.get("currency", "USD")

                    reply = (
                        f"✅ **Prediction Result**\n\n"
                        f"💰 **{currency} {price} / night**\n\n"
                        f"✅ **Extracted**\n```json\n{extracted}\n```\n\n"
                        f"✅ **Final payload**\n```json\n{final_features}\n```"
                    )
                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})


