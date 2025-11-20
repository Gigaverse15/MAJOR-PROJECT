import os
import re
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ============================================================
#                INITIALIZATION & CONFIGURATION
# ============================================================

DB_FAISS_PATH = "vectorstore/db_faiss"
DOCTORS_DATABASE = pd.read_csv("doctors_300_dataset.csv")


def init_session():
    if "bookings" not in st.session_state:
        st.session_state.bookings = []
    if "ai_booking_chat" not in st.session_state:
        st.session_state.ai_booking_chat = [
            {"role": "assistant", "content": "Hello! 👋 I can help you book an appointment. Where are you located?"}
        ]
    if "ai_doctors" not in st.session_state:
        st.session_state.ai_doctors = None
    if "ai_selected_doctor" not in st.session_state:
        st.session_state.ai_selected_doctor = None
    if "ai_slots" not in st.session_state:
        st.session_state.ai_slots = None


# ============================================================
#                   VECTORSTORE (RAG)
# ============================================================

@st.cache_resource
def load_vectorstore():
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        return db
    except:
        return None


# ============================================================
#                  SUPPORT FUNCTIONS
# ============================================================

def search_doctors(location=None, specialty=None):
    df = DOCTORS_DATABASE.copy()

    if location:
        df = df[df["location_area"].str.contains(location, case=False, na=False)]

    if specialty:
        df = df[df["sub_specialty"].str.contains(specialty, case=False, na=False) |
                df["specialty"].str.contains(specialty, case=False, na=False)]

    if len(df) == 0:
        return None

    df = df.sort_values("rating", ascending=False)
    return df.head(5)


def generate_time_slots(doctor_id, date):
    doctor = DOCTORS_DATABASE[DOCTORS_DATABASE["doctor_id"] == doctor_id].iloc[0]
    mins = int(doctor["consultation_duration_mins"])

    slots = []
    start = datetime.combine(date, datetime.strptime("09:00", "%H:%M").time())
    end = datetime.combine(date, datetime.strptime("17:00", "%H:%M").time())

    current = start
    while current < end:
        slot_end = current + timedelta(minutes=mins)
        is_booked = any(
            b["doctor_id"] == doctor_id and b["date"] == date and b["slot_start"] == current
            for b in st.session_state.bookings
        )

        slots.append({
            "start": current,
            "end": slot_end,
            "available": not is_booked,
            "display": current.strftime("%H:%M")
        })
        current = slot_end

    return slots


def book_appointment(doctor_id, date, slot_start, user_name, user_contact):
    doctor = DOCTORS_DATABASE[DOCTORS_DATABASE["doctor_id"] == doctor_id].iloc[0]

    # conflict
    for b in st.session_state.bookings:
        if b["doctor_id"] == doctor_id and b["date"] == date and b["slot_start"] == slot_start:
            return False, "Slot already booked."

    booking_id = f"BK{1000 + len(st.session_state.bookings) + 1}"

    booking = {
        "booking_id": booking_id,
        "doctor_id": doctor_id,
        "doctor_name": doctor["doctor_name_en"],
        "hospital_name": doctor["hospital_name"],
        "location": doctor["location_area"],
        "date": date,
        "slot_start": slot_start,
        "slot_end": slot_start + timedelta(minutes=int(doctor["consultation_duration_mins"])),
        "user_name": user_name,
        "user_contact": user_contact,
        "created_at": datetime.now()
    }

    st.session_state.bookings.append(booking)
    return True, booking


# ============================================================
#              MEDICAL Q&A RAG (Groq)
# ============================================================

def medical_answer(api_key, question):
    db = load_vectorstore()
    if not db:
        return "Vector DB not loaded."

    docs = db.similarity_search(question, k=3)
    context = "\n\n".join([d.page_content for d in docs])

    client = Groq(api_key=api_key)

    prompt = f"""
You are a cancer medical assistant.
You MUST answer using ONLY the context below.

If info not in context: reply ONLY with:
"Not in documents. Please consult cancer.gov or your doctor."

Be empathetic.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:
    """

    res = client.chat.completions.create(
        model="groq/compound",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )

    return res.choices[0].message.content


# ============================================================
#              BOOKING AGENT — INTENT PARSER
# ============================================================

def parse_intent(text):
    out = {}

    loc = re.search(r"(Noida|Indirapuram|Gurgaon|Delhi|Ghaziabad|Botanical Garden|Sector \d+)", text, re.I)
    if loc: out["location"] = loc.group(0)

    doc = re.search(r"Dr\.?\s+[A-Za-z ]+", text)
    if doc: out["doctor_name"] = doc.group(0)

    spec = re.search(r"(kidney|breast|lung|blood|oncology|cancer)", text, re.I)
    if spec: out["specialty"] = spec.group(0)

    date = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", text)
    if date:
        out["date"] = datetime.strptime(date.group(1), "%Y-%m-%d").date()

    time = re.search(r"\b(\d{2}:\d{2})\b", text)
    if time: out["slot"] = time.group(1)

    return out


# ============================================================
#      BOOKING AGENT — PYTHON LOGIC (NO HALLUCINATIONS)
# ============================================================

def process_booking(intent):
    response = ""

    # ---- STEP 1: location/specialty → doctor list ----
    if "location" in intent or "specialty" in intent:
        docs = search_doctors(intent.get("location"), intent.get("specialty"))
        if docs is not None:
            st.session_state.ai_doctors = docs
            response += "Here are top matching doctors:\n\n"
            for i, (_, d) in enumerate(docs.iterrows(), start=1):
                response += f"{i}. {d['doctor_name_en']} — {d['specialty']} ({d['location_area']})\n"
            response += "\nTell me the doctor name to book.\n"
        else:
            response += "No doctors found, try another specialty.\n"

    # ---- STEP 2: doctor selection ----
    if "doctor_name" in intent and st.session_state.ai_doctors is not None:
        dmatch = st.session_state.ai_doctors[
            st.session_state.ai_doctors["doctor_name_en"].str.contains(intent["doctor_name"], case=False)
        ]
        if len(dmatch) > 0:
            selected = dmatch.iloc[0]
            st.session_state.ai_selected_doctor = selected["doctor_id"]
            response += f"Doctor selected: {selected['doctor_name_en']}\n"
            response += "Tell me the date you want (YYYY-MM-DD).\n"
        else:
            response += "Doctor not in list.\n"

    # ---- STEP 3: date → show slots ----
    if "date" in intent and st.session_state.ai_selected_doctor:
        slots = generate_time_slots(st.session_state.ai_selected_doctor, intent["date"])
        available = [s for s in slots if s["available"]]

        if available:
            st.session_state.ai_slots = available
            response += f"Available slots on {intent['date']}:\n"
            for s in available:
                response += f"- {s['display']}\n"
            response += "\nTell me time like 10:30\n"
        else:
            response += "No slots available.\n"

    # ---- STEP 4: slot selection → REAL booking ----
    if "slot" in intent and "date" in intent and st.session_state.ai_selected_doctor:
        chosen = None
        for s in st.session_state.ai_slots:
            if s["display"].startswith(intent["slot"]):
                chosen = s
                break

        if not chosen:
            return "Invalid slot."

        success, booking = book_appointment(
            st.session_state.ai_selected_doctor,
            intent["date"],
            chosen["start"],
            "AI User",
            "9999999999"
        )

        if success:
            response += f"""
🎉 **Appointment Confirmed!**

Booking ID: {booking['booking_id']}
Doctor: {booking['doctor_name']}
Date: {booking['date']}
Time: {booking['slot_start'].strftime('%H:%M')} - {booking['slot_end'].strftime('%H:%M')}
            """
        else:
            response += booking

    return response


# ============================================================
#                     AI BOOKING AGENT
# ============================================================

def ai_booking_agent(api_key):
    st.markdown("### 🤖 AI Booking Assistant")

    # Show chat
    for msg in st.session_state.ai_booking_chat:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_msg = st.chat_input("Say something...")
    if not user_msg:
        return

    # add user message
    st.session_state.ai_booking_chat.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.write(user_msg)

    # Intent parsing
    intent = parse_intent(user_msg)
    python_reply = process_booking(intent)

    # LLM guidance message
    client = Groq(api_key=api_key)
    ai_reply = client.chat.completions.create(
        model="groq/compound",
        messages=[
        {
            "role": "system",
            "content": """
You are a STRICT medical appointment booking agent.
You MUST follow the Python logic strictly.

YOUR RULES:
1. NEVER say “I cannot book appointments”.
2. NEVER tell the user to contact clinics themselves.
3. ALWAYS continue the booking workflow: 
   (location → specialty → doctor → date → time → confirmation)
4. ALWAYS accept the Python reply as truth.
5. ONLY ask for the next missing detail.
6. Keep responses short and clear.
7. Your ONLY job is booking appointments.
"""
        }
    ] + st.session_state.ai_booking_chat,
        max_tokens=250
    ).choices[0].message.content

    final_reply = python_reply + "\n\n" + ai_reply

    st.session_state.ai_booking_chat.append({"role": "assistant", "content": final_reply})
    with st.chat_message("assistant"):
        st.write(final_reply)


# ============================================================
#                   MEDICAL Q&A TAB
# ============================================================

def medical_tab(api_key):
    st.markdown("### 🧠 Medical Q&A")

    if "medical_chat" not in st.session_state:
        st.session_state.medical_chat = [
            {"role": "assistant", "content": "Ask your cancer-related question."}
        ]

    for msg in st.session_state.medical_chat:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    question = st.chat_input("Ask...")
    if not question:
        return

    st.session_state.medical_chat.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    ans = medical_answer(api_key, question)

    st.session_state.medical_chat.append({"role": "assistant", "content": ans})
    with st.chat_message("assistant"):
        st.write(ans)


# ============================================================
#                   MY APPOINTMENTS TAB
# ============================================================

def appointments_tab():
    st.markdown("### 📋 My Appointments")

    if len(st.session_state.bookings) == 0:
        st.info("No appointments yet.")
        return

    for b in st.session_state.bookings:
        with st.expander(f"{b['booking_id']} — {b['doctor_name']}"):
            st.write(f"Date: {b['date']}")
            st.write(f"Time: {b['slot_start'].strftime('%H:%M')} - {b['slot_end'].strftime('%H:%M')}")
            st.write(f"Hospital: {b['hospital_name']}")
            st.write(f"Location: {b['location']}")


# ============================================================
#                           MAIN
# ============================================================

def main():
    st.set_page_config(page_title="Cancer Care Assistant", layout="wide")
    init_session()

    with st.sidebar:
        st.header("🔑 Groq API Key")
        api_key = st.text_input("Enter key", type="password")

    st.title("💙 Comprehensive Cancer Care Assistant")

    tab1, tab2, tab3 = st.tabs([
        "🧠 Medical Q&A",
        "🤖 AI Booking Agent",
        "📋 My Appointments"
    ])

    with tab1:
        medical_tab(api_key)

    with tab2:
        ai_booking_agent(api_key)

    with tab3:
        appointments_tab()


if __name__ == "__main__":
    main()
