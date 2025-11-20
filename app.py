# ============================================================
# PART 1 — Imports, Config, Doctors DB, Vector DB, Utilities
# ============================================================

import os
import re
import json
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ----------------------------
# CONFIG
# ----------------------------

DB_FAISS_PATH = "vectorstore/db_faiss"
DOCTORS_DATABASE = pd.read_csv("doctors_300_dataset.csv")

def init_session_state():
    """Initialize session state variables"""
    if 'bookings' not in st.session_state:
        st.session_state.bookings = []

    if 'selected_doctor' not in st.session_state:
        st.session_state.selected_doctor = None

    if 'user_language' not in st.session_state:
        st.session_state.user_language = "English"

    if 'ai_booking_chat' not in st.session_state:
        st.session_state.ai_booking_chat = [
            {"role": "assistant", "content": "Hello! 👋 I can help you book an appointment. Where are you located?"}
        ]

    if "ai_doctors" not in st.session_state:
        st.session_state.ai_doctors = None

    if "ai_selected_doctor" not in st.session_state:
        st.session_state.ai_selected_doctor = None


# ----------------------------
# VECTOR STORE (for RAG)
# ----------------------------

@st.cache_resource
def get_vectorstore():
    """Loads FAISS vector store"""
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        st.error(f"FAISS load error: {e}")
        return None


# ----------------------------
# DOCTOR SEARCH UTILITIES
# ----------------------------

def search_doctors(location=None, specialty=None, disease=None, top_n=5):
    """Ranking + filtering system for doctor recommendations"""
    df = DOCTORS_DATABASE.copy()

    if location:
        df = df[df["location_area"].str.contains(location, case=False, na=False)]

    if specialty:
        df = df[
            df["specialty"].str.contains(specialty, case=False, na=False) |
            df["sub_specialty"].str.contains(specialty, case=False, na=False)
        ]

    if disease:
        df = df[df["diseases_treated"].str.contains(disease, case=False, na=False)]

    if len(df) > 0:
        df["score"] = (
            0.45 * (df["rating"] / 5.0) +
            0.25 * (1.0 if specialty or disease else 0.5) +
            0.20 * (1.0 if location else 0.5) +
            0.10 * (df["reviews_count"] / df["reviews_count"].max())
        )
        df = df.sort_values("score", ascending=False)

    return df.head(top_n)


# ----------------------------
# TIME SLOT GENERATOR
# ----------------------------

def generate_time_slots(doctor_id, date):
    """Generate all 10-15 min slots from 9am to 5pm"""
    doctor = DOCTORS_DATABASE[DOCTORS_DATABASE["doctor_id"] == doctor_id].iloc[0]
    duration = int(doctor["consultation_duration_mins"])

    slots = []
    start = datetime.combine(date, datetime.strptime("09:00", "%H:%M").time())
    end = datetime.combine(date, datetime.strptime("17:00", "%H:%M").time())

    current = start
    while current < end:
        slot_end = current + timedelta(minutes=duration)

        is_booked = any(
            b["doctor_id"] == doctor_id and b["date"] == date and b["slot_start"] == current
            for b in st.session_state.bookings
        )

        slots.append({
            "start": current,
            "end": slot_end,
            "available": not is_booked,
            "display": current.strftime("%I:%M %p")
        })

        current = slot_end

    return slots


# ----------------------------
# BOOKING LOGIC
# ----------------------------

def book_appointment(doctor_id, date, slot_start, user_name, user_contact):
    """Registers appointment with queue number logic"""
    doctor = DOCTORS_DATABASE[DOCTORS_DATABASE["doctor_id"] == doctor_id].iloc[0]

    # Check conflicts
    conflict = [
        b for b in st.session_state.bookings
        if b["doctor_id"] == doctor_id and b["date"] == date and b["slot_start"] == slot_start
    ]

    if conflict:
        return False, "Slot already booked."

    booking_id = f"BK{1000 + len(st.session_state.bookings) + 1}"

    hour_start = slot_start.replace(minute=0, second=0)
    hour_bookings = [
        b for b in st.session_state.bookings
        if b["doctor_id"] == doctor_id and
        b["date"] == date and
        hour_start <= b["slot_start"] < hour_start + timedelta(hours=1)
    ]

    queue_number = len(hour_bookings) + 1

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
        "queue_position": queue_number,
        "created_at": datetime.now(),
        "status": "confirmed"
    }

    st.session_state.bookings.append(booking)
    return True, booking

# ============================================================
# PART 2 — AI Agents (Medical Q&A + Booking Agent) + Intent Parser
# ============================================================

# -------------------------------------------------------------
# MEDICAL Q&A AGENT (RAG + Groq Compound Model)
# -------------------------------------------------------------

def medical_query_groq(api_key, user_query, docs):
    """Uses Groq model with medical safety rules + citations"""

    try:
        client = Groq(api_key=api_key)

        context = "\n\n".join([doc.page_content for doc in docs])

        system_prompt = """
You are a highly reliable medical assistant for cancer patients.
Your job is to answer questions ONLY using the provided medical documents.

RULES:
- Use ONLY the context provided.
- If information is missing: reply EXACTLY with 
  "Not in documents. Please consult cancer.gov or your doctor."
- Never hallucinate.
- Never give medical advice, only explain what the documents say.
- ALWAYS be empathetic and clear.
- Cite page numbers when mentioned in metadata.
"""

        full_prompt = f"""
CONTEXT:
{context}

QUESTION: {user_query}

ANSWER:
"""

        response = client.chat.completions.create(
            model="groq/compound",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.2,
            max_tokens=800
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error: {e}"


# -------------------------------------------------------------
# INTENT PARSER FOR APPOINTMENT BOOKING
# -------------------------------------------------------------

def parse_booking_intent(text):
    """Extracts meaning from user messages for booking agent."""

    intent = {}

    # ✔ Location
    loc = re.search(
        r"(Noida|Indirapuram|Botanical Garden|Gurgaon|Delhi|Ghaziabad|Faridabad|Sector \d+)",
        text,
        re.I
    )
    if loc:
        intent["location"] = loc.group(0)

    # ✔ Doctor name
    doc = re.search(r"Dr\.?\s+[A-Za-z]+(?:\s+[A-Za-z]+)*", text)
    if doc:
        intent["doctor_name"] = doc.group(0)

    # ✔ Specialty / Disease
    spec = re.search(
        r"(oncology|cancer|renal|kidney|lung|breast|blood|pediatric|sarcoma|leukemia|lymphoma)",
        text,
        re.I
    )
    if spec:
        intent["specialty"] = spec.group(0)

    # ✔ Time (10:30, 3:15, etc.)
    tm = re.search(r"\b(\d{1,2}:\d{2})\b", text)
    if tm:
        intent["slot"] = tm.group(1)

    # ✔ Date (2025-11-20)
    date = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", text)
    if date:
        try:
            intent["date"] = datetime.strptime(date.group(1), "%Y-%m-%d").date()
        except:
            pass

    return intent


# -------------------------------------------------------------
# APPOINTMENT BOOKING CONVERSATIONAL AI AGENT
# -------------------------------------------------------------

def appointment_agent_response(api_key, conversation_history):
    """LLM generates next message for booking assistant."""

    client = Groq(api_key=api_key)

    system_prompt = """
You are an intelligent appointment-booking assistant.
Your ONLY job is to help users:

1. Give location
2. Give specialty / doctor choice
3. Show top 5 recommended doctors (Python will handle)
4. Ask for date
5. Show time slots (Python will handle)
6. Confirm booking

RULES:
- Do NOT answer medical questions.
- Do NOT use medical documents.
- DO NOT say 'Not in documents'.
- Always ask for next needed detail.
- Be friendly and quick.
- If user already gave enough info → proceed.
- Keep responses short.
"""

    response = client.chat.completions.create(
        model="groq/compound",
        messages=[
            {"role": "system", "content": system_prompt},
            *conversation_history
        ],
        temperature=0.4,
        max_tokens=300
    )

    return response.choices[0].message.content


# -------------------------------------------------------------
# PROCESS BOOKING INTENT → TRIGGER PYTHON STEPS
# -------------------------------------------------------------

def process_booking_logic(intent):
    """Connects parsed intent with Python actions: search, select, timeslots, booking."""
    
    response_text = ""

    # ----------------------------------------------
    # STEP 1 — Search doctors based on location/specialty
    # ----------------------------------------------

    if "location" in intent or "specialty" in intent:
        docs = search_doctors(
            location=intent.get("location"),
            specialty=intent.get("specialty")
        )

        if len(docs) > 0:
            st.session_state.ai_doctors = docs
            response_text += "Here are the top doctors I found:\n\n"
            for idx, (_, d) in enumerate(docs.iterrows(), start=1):
                response_text += f"{idx}. {d['doctor_name_en']} — {d['specialty']} ({d['location_area']})\n"
        else:
            response_text += "I couldn't find any matching doctors.\n"

    # ----------------------------------------------
    # STEP 2 — Select doctor
    # ----------------------------------------------

    if "doctor_name" in intent and st.session_state.ai_doctors is not None:
        dmatch = st.session_state.ai_doctors[
            st.session_state.ai_doctors["doctor_name_en"].str.contains(intent["doctor_name"], case=False)
        ]

        if len(dmatch) > 0:
            selected = dmatch.iloc[0]
            st.session_state.ai_selected_doctor = selected["doctor_id"]
            response_text += f"\nDoctor selected: {selected['doctor_name_en']}\n"
            response_text += "Please tell me a date (YYYY-MM-DD).\n"

    # ----------------------------------------------
    # STEP 3 — Time slot generation
    # ----------------------------------------------

    if "date" in intent and st.session_state.ai_selected_doctor:
        date = intent["date"]
        doctor_id = st.session_state.ai_selected_doctor
        slots = generate_time_slots(doctor_id, date)
        available = [s for s in slots if s["available"]]

        if available:
            response_text += f"\nAvailable slots on {date}:\n"
            for s in available[:10]:
                response_text += f"- {s['display']}\n"

            response_text += "\nPlease choose a time slot (e.g., 12:30).\n"
        else:
            response_text += "No slots available. Please choose another date.\n"

    # ----------------------------------------------
    # STEP 4 — Final Booking
    # ----------------------------------------------

    if "slot" in intent and st.session_state.ai_selected_doctor and "date" in intent:
        date = intent["date"]
        time = datetime.strptime(intent["slot"], "%H:%M").time()
        doctor_id = st.session_state.ai_selected_doctor
        slot_dt = datetime.combine(date, time)

        success, booking = book_appointment(
            doctor_id, date, slot_dt,
            user_name="AI User",
            user_contact="9999999999"
        )

        if success:
            response_text += f"\n🎉 Appointment Confirmed!\n\n"
            response_text += f"**Booking ID:** {booking['booking_id']}\n"
            response_text += f"**Doctor:** {booking['doctor_name']}\n"
            response_text += f"**Date:** {date}\n"
            response_text += f"**Time:** {intent['slot']}\n"
            response_text += f"**Queue:** {booking['queue_position']}\n"
        else:
            response_text += f"⚠️ {booking}\n"

    return response_text

# ============================================================
# PART 3 — Streamlit UI (5 Tabs) + Main App
# ============================================================

# -------------------------------------------------------------
# TAB 1 — MEDICAL Q&A (RAG)
# -------------------------------------------------------------

def medical_qa_tab(api_key):
    st.markdown("### 🧠 Medical Q&A")
    st.markdown("*Ask anything about cancer, symptoms, treatments — based on real medical documents.*")

    if not api_key:
        st.warning("⚠️ Please enter Groq API key in the left sidebar.")
        return

    if "medical_chat" not in st.session_state:
        st.session_state.medical_chat = [
            {"role": "assistant", "content": "Hello! Ask your medical question anytime."}
        ]

    # Show conversation
    for msg in st.session_state.medical_chat:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    query = st.chat_input("Type your medical question…")
    if not query:
        return

    st.session_state.medical_chat.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching medical documents…"):
            vector_db = get_vectorstore()
            if vector_db:
                docs = vector_db.similarity_search(query, k=3)
                answer = medical_query_groq(api_key, query, docs)
            else:
                answer = "Vector database not available."

            st.write(answer)
            st.session_state.medical_chat.append({"role": "assistant", "content": answer})


# -------------------------------------------------------------
# TAB 2 — AI APPOINTMENT BOOKING AGENT
# -------------------------------------------------------------

def ai_booking_tab(api_key):
    st.markdown("### 🤖 Smart Appointment Booking Assistant")
    st.markdown("Chat naturally — I’ll guide you step-by-step to book a doctor appointment.")

    if not api_key:
        st.warning("⚠️ Please add Groq API key.")
        return

    # Show chat history
    for msg in st.session_state.ai_booking_chat:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_msg = st.chat_input("Say something…")
    if not user_msg:
        return

    st.session_state.ai_booking_chat.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.write(user_msg)

    # ------------------------------
    # Parse intent
    # ------------------------------
    intent = parse_booking_intent(user_msg)
    python_action_reply = process_booking_logic(intent)

    # ------------------------------
    # AI Agent Response (LLM)
    # ------------------------------
    llm_reply = appointment_agent_response(api_key, st.session_state.ai_booking_chat)

    final_response = python_action_reply + "\n\n" + llm_reply

    st.session_state.ai_booking_chat.append({"role": "assistant", "content": final_response})

    with st.chat_message("assistant"):
        st.write(final_response)


# -------------------------------------------------------------
# TAB 3 — FIND DOCTORS (Manual Search)
# -------------------------------------------------------------

def find_doctors_tab():
    st.markdown("### 🔍 Find Doctors")

    col1, col2, col3 = st.columns(3)
    with col1:
        loc = st.text_input("Location")
    with col2:
        spec = st.text_input("Specialty")
    with col3:
        dis = st.text_input("Disease")

    if st.button("Search"):
        results = search_doctors(location=loc, specialty=spec, disease=dis)

        if len(results) == 0:
            st.warning("No matching doctors found.")
            return

        for idx, (_, d) in enumerate(results.iterrows(), start=1):
            with st.expander(f"{idx}. {d['doctor_name_en']} — {d['rating']}/5"):
                st.write(f"**Specialty:** {d['specialty']}")
                st.write(f"**Hospital:** {d['hospital_name']}")
                st.write(f"**Location:** {d['location_area']}")
                st.write(f"**Languages:** {d['languages']}")
                st.write(f"**Treats:** {d['diseases_treated']}")
                st.write(f"**Fee:** ₹{d['fee']}")
                st.write(f"**Experience:** {d['experience_years']} yrs")

                if st.button(f"Select {d['doctor_id']}", key=f"choose_{d['doctor_id']}"):
                    st.session_state.selected_doctor = d["doctor_id"]
                    st.success("Doctor selected. Proceed to Book Appointment tab.")


# -------------------------------------------------------------
# TAB 4 — BOOK APPOINTMENT (Manual)
# -------------------------------------------------------------

def manual_booking_tab():
    st.markdown("### 📅 Book Appointment")

    if not st.session_state.selected_doctor:
        st.info("Select a doctor from 'Find Doctors' first.")
        return

    doctor = DOCTORS_DATABASE[
        DOCTORS_DATABASE["doctor_id"] == st.session_state.selected_doctor
    ].iloc[0]

    st.write(f"**Doctor:** {doctor['doctor_name_en']}")
    st.write(f"**Hospital:** {doctor['hospital_name']}")
    st.write(f"**Location:** {doctor['location_area']}")

    date = st.date_input("Choose date", min_value=datetime.today().date())

    slots = generate_time_slots(st.session_state.selected_doctor, date)
    available_slots = [s for s in slots if s["available"]]

    slot_choices = [s["display"] for s in available_slots]
    chosen = st.selectbox("Available slots", slot_choices)

    name = st.text_input("Your Name")
    phone = st.text_input("Phone Number")

    if st.button("Confirm"):
        if name and phone:
            slot_obj = next(s for s in available_slots if s["display"] == chosen)

            ok, booking = book_appointment(
                st.session_state.selected_doctor,
                date,
                slot_obj["start"],
                name,
                phone
            )

            if ok:
                st.success(f"Booking Confirmed! ID: {booking['booking_id']}")
                st.balloons()
        else:
            st.warning("Please fill name and phone.")


# -------------------------------------------------------------
# TAB 5 — MY APPOINTMENTS
# -------------------------------------------------------------

def appointments_tab():
    st.markdown("### 📋 My Appointments")

    if len(st.session_state.bookings) == 0:
        st.info("No appointments yet.")
        return

    for b in st.session_state.bookings:
        with st.expander(f"{b['booking_id']} — {b['doctor_name']}"):
            st.write(f"**Date:** {b['date']}")
            st.write(f"**Time:** {b['slot_start'].strftime('%I:%M %p')}")
            st.write(f"**Hospital:** {b['hospital_name']}")
            st.write(f"**Queue:** {b['queue_position']}")
            st.write(f"**Phone:** {b['user_contact']}")


# -------------------------------------------------------------
# MAIN STREAMLIT APP
# -------------------------------------------------------------

def main():
    st.set_page_config(page_title="Cancer Care Assistant", page_icon="💙", layout="wide")
    init_session_state()

    st.title("💙 Comprehensive Cancer Care Assistant")

    with st.sidebar:
        st.header("🔑 Groq API Key")
        api_key = st.text_input("Enter key", type="password")
        st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🧠 Medical Q&A",
        "🤖 AI Booking Agent",
        "🔍 Find Doctors",
        "📅 Book Appointment",
        "📋 My Appointments"
    ])

    with tab1:
        medical_qa_tab(api_key)

    with tab2:
        ai_booking_tab(api_key)

    with tab3:
        find_doctors_tab()

    with tab4:
        manual_booking_tab()

    with tab5:
        appointments_tab()


# -------------------------------------------------------------
# BOOTSTRAP
# -------------------------------------------------------------

if __name__ == "__main__":
    main()
