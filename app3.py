# app.py
# Comprehensive Cancer Care Assistant - single file (fixed AI booking agent + appointments persist)
# Requirements: streamlit, pandas, groq, langchain_huggingface, langchain_community, sentence-transformers
# Run: streamlit run app.py

import os
import re
import json
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, time
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ----------------------------
# CONFIG
# ----------------------------
DB_FAISS_PATH = "vectorstore/db_faiss"   # keep if you use vectorstore for Medical Q&A
DOCTORS_CSV = "doctors_300_dataset.csv"  # your CSV file path (must exist)
DEFAULT_USER_NAME = "AI User"
DEFAULT_USER_CONTACT = "9999999999"

# Load doctors dataset (safe fallback to small sample if CSV missing)
if os.path.exists(DOCTORS_CSV):
    DOCTORS_DATABASE = pd.read_csv(DOCTORS_CSV)
else:
    # minimal sample fallback
    DOCTORS_DATABASE = pd.DataFrame([
        {"doctor_id": "D001", "doctor_name_en": "Dr. Amit Sharma", "doctor_name_hi": "डॉ. अमित शर्मा",
         "specialty": "Oncology", "sub_specialty": "Kidney Cancer", "occupation": "Senior Consultant",
         "hospital_name": "Noida Cancer Center", "location_area": "Noida Sector 62", "languages": "English,Hindi",
         "diseases_treated": "Kidney Cancer,Bladder Cancer,Prostate Cancer", "rating": 4.8, "reviews_count": 420,
         "consultation_duration_mins": 15, "telemedicine": True, "fee": 1500, "experience_years": 15},
    ])

# ----------------------------
# SESSION STATE INIT
# ----------------------------
def init_session_state():
    if 'bookings' not in st.session_state:
        st.session_state.bookings = []  # list of booking dicts
    if 'selected_doctor' not in st.session_state:
        st.session_state.selected_doctor = None  # used by manual booking tab
    if 'user_language' not in st.session_state:
        st.session_state.user_language = "English"
    if 'medical_chat' not in st.session_state:
        st.session_state.medical_chat = [{"role": "assistant", "content": "Hello! Ask your medical question anytime."}]
    if 'ai_booking_chat' not in st.session_state:
        st.session_state.ai_booking_chat = [{"role": "assistant", "content": "Hello! 👋 I can help you book an appointment. Where are you located?"}]
    if 'ai_doctors' not in st.session_state:
        st.session_state.ai_doctors = None
    if 'ai_selected_doctor' not in st.session_state:
        st.session_state.ai_selected_doctor = None
    if 'groq_api_key' not in st.session_state:
        st.session_state.groq_api_key = None

# ----------------------------
# Vectorstore loader (for Medical Q&A)
# ----------------------------
@st.cache_resource
def get_vectorstore():
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        return db
    except Exception:
        return None

# ----------------------------
# DOCTOR SEARCH / RANKING
# ----------------------------
def search_doctors(location=None, specialty=None, disease=None, top_n=5):
    df = DOCTORS_DATABASE.copy()
    if location and isinstance(location, str) and location.strip():
        df = df[df['location_area'].str.contains(location, case=False, na=False)]
    if specialty and isinstance(specialty, str) and specialty.strip():
        df = df[
            df['specialty'].str.contains(specialty, case=False, na=False) |
            df['sub_specialty'].str.contains(specialty, case=False, na=False)
        ]
    if disease and isinstance(disease, str) and disease.strip():
        df = df[df['diseases_treated'].str.contains(disease, case=False, na=False)]
    if len(df) > 0:
        df['score'] = (
            0.45 * (df['rating'].fillna(3.0) / 5.0) +
            0.25 * (1.0 if (specialty or disease) else 0.5) +
            0.20 * (1.0 if location else 0.5) +
            0.10 * (df['reviews_count'].fillna(10) / df['reviews_count'].fillna(10).max())
        )
        df = df.sort_values('score', ascending=False)
    return df.head(top_n)

# ----------------------------
# TIME SLOT GENERATOR
# ----------------------------
def generate_time_slots(doctor_id, date: datetime.date):
    """
    Generate slots for a given doctor and date.
    doctor consultation duration used from dataset.
    slots between 09:00 and 17:00.
    Returns list of dicts with 'start','end','available','display'
    """
    try:
        doctor = DOCTORS_DATABASE[DOCTORS_DATABASE['doctor_id'] == doctor_id].iloc[0]
    except Exception:
        return []
    duration = int(doctor.get('consultation_duration_mins', 15))
    start_dt = datetime.combine(date, time(hour=9, minute=0))
    end_dt = datetime.combine(date, time(hour=17, minute=0))
    current = start_dt
    slots = []
    while current + timedelta(minutes=duration) <= end_dt:
        slot_end = current + timedelta(minutes=duration)
        is_booked = any(
            b['doctor_id'] == doctor_id and b['date'] == date and b['slot_start'] == current
            for b in st.session_state.bookings
        )
        slots.append({
            'start': current,
            'end': slot_end,
            'available': not is_booked,
            'display': current.strftime("%H:%M")  # we let user respond with HH:MM
        })
        current = slot_end
    return slots

# ----------------------------
# BOOKING LOGIC
# ----------------------------
def book_appointment(doctor_id, date, slot_start_dt, user_name=DEFAULT_USER_NAME, user_contact=DEFAULT_USER_CONTACT):
    # check conflict
    conflicts = [b for b in st.session_state.bookings if b['doctor_id'] == doctor_id and b['date'] == date and b['slot_start'] == slot_start_dt]
    if conflicts:
        return False, "Slot already booked"
    # get doctor data
    doctor = DOCTORS_DATABASE[DOCTORS_DATABASE['doctor_id'] == doctor_id].iloc[0]
    booking_id = f"BK{1000 + len(st.session_state.bookings) + 1}"
    hour_start = slot_start_dt.replace(minute=0, second=0, microsecond=0)
    hour_bookings = [
        b for b in st.session_state.bookings
        if b['doctor_id'] == doctor_id and b['date'] == date and hour_start <= b['slot_start'] < hour_start + timedelta(hours=1)
    ]
    queue_pos = len(hour_bookings) + 1
    booking = {
        'booking_id': booking_id,
        'doctor_id': doctor_id,
        'doctor_name': doctor['doctor_name_en'],
        'doctor_name_hi': doctor.get('doctor_name_hi', ''),
        'hospital_name': doctor['hospital_name'],
        'location': doctor['location_area'],
        'date': date,
        'slot_start': slot_start_dt,
        'slot_end': slot_start_dt + timedelta(minutes=int(doctor.get('consultation_duration_mins', 15))),
        'user_name': user_name,
        'user_contact': user_contact,
        'queue_position': queue_pos,
        'created_at': datetime.now(),
        'status': 'confirmed'
    }
    st.session_state.bookings.append(booking)
    return True, booking

# ----------------------------
# PARSING USER INTENT (booking agent)
# ----------------------------
def parse_booking_intent(text: str):
    """Return dict with any of: location, specialty, doctor_name, date (date object), slot (HH:MM)"""
    intent = {}
    text = text.strip()
    # location simple heuristics (extend as needed)
    loc_match = re.search(r"(Noida|Indirapuram|Botanical Garden|Ghaziabad|Gurgaon|Sector\s*\d+|Greater Noida|Delhi)", text, re.I)
    if loc_match:
        intent['location'] = loc_match.group(0)
    # doctor name: "Dr X Y"
    doc_match = re.search(r"(Dr\.?\s*[A-Z][a-zA-Z]+\s*[A-Za-z]*)", text)
    if doc_match:
        intent['doctor_name'] = doc_match.group(0).replace(".", "").strip()
    # specialty keywords
    spec_match = re.search(r"(kidney|renal|lung|breast|blood|oncology|nephrology|urology|surgery|pediatric|sarcoma|leukemia|lymphoma)", text, re.I)
    if spec_match:
        intent['specialty'] = spec_match.group(0)
    # date YYYY-MM-DD
    date_match = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", text)
    if date_match:
        try:
            intent['date'] = datetime.strptime(date_match.group(1), "%Y-%m-%d").date()
        except:
            pass
    # common words: today / tomorrow / next week
    if 'today' in text.lower() and 'date' not in intent:
        intent['date'] = datetime.now().date()
    if 'tomorrow' in text.lower() and 'date' not in intent:
        intent['date'] = (datetime.now() + timedelta(days=1)).date()
    # time like 09:30 or 9:30 or 14:00
    time_match = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", text)
    if time_match:
        hhmm = f"{int(time_match.group(1)):02d}:{time_match.group(2)}"
        intent['slot'] = hhmm
    # user asking for "available dates" or "availability"
    if re.search(r"available dates|availability|available slots|what dates", text, re.I):
        intent['ask_availability'] = True
    # user confirming like "book" or "confirm"
    if re.search(r"\b(book|confirm|reserve)\b", text, re.I):
        intent['confirm'] = True
    return intent

# ----------------------------
# PROCESS INTENT (Python actions)
# ----------------------------
def process_booking_intent_and_act(intent: dict):
    """
    This function performs the authoritative python-side actions:
    - search doctors
    - select doctor
    - list next available dates/time slots
    - perform final booking
    Returns a text reply (string) describing the action / results to show to the user.
    """
    reply = ""

    # 1) Search by location/specialty if provided
    if 'location' in intent or 'specialty' in intent:
        docs = search_doctors(location=intent.get('location'), specialty=intent.get('specialty'), top_n=10)
        if len(docs) == 0:
            reply += "I couldn't find doctors matching that location/specialty. Try a nearby location or another specialty.\n"
        else:
            st.session_state.ai_doctors = docs  # save list for selection
            reply += "Here are the top doctors I found:\n"
            for i, (_, d) in enumerate(docs.iterrows(), start=1):
                reply += f"{i}. {d['doctor_name_en']} — {d['specialty']} ({d['location_area']})\n"
            reply += "\nYou can pick a doctor by replying with the doctor number (e.g., '1') or 'Book Dr. Name'.\n"

    # 2) If user provided a doctor name (Dr X) or picked a number, try select
    if 'doctor_name' in intent:
        # search in previously loaded ai_doctors (if present) else global dataset
        target = intent['doctor_name']
        selected = None
        if st.session_state.ai_doctors is not None:
            # match within displayed subset
            matches = st.session_state.ai_doctors[st.session_state.ai_doctors['doctor_name_en'].str.contains(target, case=False, na=False)]
            if len(matches) > 0:
                selected = matches.iloc[0]
        if selected is None:
            # try to match in full dataset by partial name
            matches_all = DOCTORS_DATABASE[DOCTORS_DATABASE['doctor_name_en'].str.contains(target, case=False, na=False)]
            if len(matches_all) > 0:
                selected = matches_all.iloc[0]
        if selected is not None:
            st.session_state.ai_selected_doctor = selected['doctor_id']
            reply += f"Doctor selected: {selected['doctor_name_en']} — {selected['hospital_name']} ({selected['location_area']}).\n"
            # auto-provide next available dates (next 7 days with at least 1 free slot)
            results = []
            today = datetime.now().date()
            for delta in range(0, 14):  # look ahead 14 days
                d = today + timedelta(days=delta)
                slots = generate_time_slots(selected['doctor_id'], d)
                available = [s for s in slots if s['available']]
                if available:
                    results.append((d, available))
                if len(results) >= 5:
                    break
            if results:
                reply += "Next available dates and example times:\n"
                for d, available in results:
                    # show up to 4 times per date
                    times = ", ".join(s['display'] for s in available[:4])
                    reply += f"- {d} : {times}\n"
                reply += "\nPlease reply with date (YYYY-MM-DD) — I'll then show all available times for that date.\n"
            else:
                reply += "No available slots found in the next two weeks for this doctor. Try another doctor or date.\n"
        else:
            reply += "Doctor not found in the dataset. Please provide exact name or choose from the list.\n"

    # 3) If user asked for availability explicitly (and a doctor is already selected), show upcoming dates:
    if intent.get('ask_availability') and st.session_state.ai_selected_doctor:
        doctor_id = st.session_state.ai_selected_doctor
        doc_row = DOCTORS_DATABASE[DOCTORS_DATABASE['doctor_id'] == doctor_id].iloc[0]
        reply += f"Availability for {doc_row['doctor_name_en']} ({doc_row['hospital_name']}):\n"
        today = datetime.now().date()
        result_dates = []
        for delta in range(0, 14):
            d = today + timedelta(days=delta)
            slots = generate_time_slots(doctor_id, d)
            available = [s for s in slots if s['available']]
            if available:
                result_dates.append((d, available))
            if len(result_dates) >= 7:
                break
        if result_dates:
            for d, available in result_dates:
                times = ", ".join(s['display'] for s in available[:6])
                reply += f"- {d}: {times}\n"
            reply += "\nPick a date (YYYY-MM-DD) to see full slots.\n"
        else:
            reply += "No slots available in the next two weeks for this doctor.\n"

    # 4) If user provided a date, show slots for selected doctor
    if 'date' in intent and st.session_state.ai_selected_doctor:
        doctor_id = st.session_state.ai_selected_doctor
        date = intent['date']
        slots = generate_time_slots(doctor_id, date)
        available = [s for s in slots if s['available']]
        if not available:
            reply += f"No available slots on {date}. Try another date.\n"
        else:
            reply += f"Available slots on {date}:\n"
            for s in available:
                reply += f"- {s['display']}\n"
            reply += "\nReply with the time (HH:MM) you'd like (e.g., '10:30').\n"

    # 5) If user provided slot and date and doctor => try final booking
    if 'slot' in intent and 'date' in intent and st.session_state.ai_selected_doctor:
        doctor_id = st.session_state.ai_selected_doctor
        date = intent['date']
        try:
            hh, mm = [int(x) for x in intent['slot'].split(':')]
            slot_dt = datetime.combine(date, time(hour=hh, minute=mm))
        except Exception:
            reply += "Couldn't parse the time you gave. Use HH:MM (e.g., 10:30).\n"
            return reply
        ok, result = book_appointment(doctor_id, date, slot_dt, user_name=DEFAULT_USER_NAME, user_contact=DEFAULT_USER_CONTACT)
        if ok:
            reply += f"🎉 Appointment confirmed!\nBooking ID: {result['booking_id']}\nDoctor: {result['doctor_name']}\nDate: {result['date']}\nTime: {result['slot_start'].strftime('%H:%M')} - {result['slot_end'].strftime('%H:%M')}\nQueue position: {result['queue_position']}\n"
            # reset selection after booking
            st.session_state.ai_selected_doctor = None
            st.session_state.ai_doctors = None
        else:
            reply += f"⚠️ Could not book: {result}\n"

    return reply

# ----------------------------
# Groq wrappers (light: just for language/phrasing if key provided)
# ----------------------------
def appointment_agent_llm_reply(api_key, conv_history):
    try:
        if not api_key:
            return ""
        client = Groq(api_key=api_key)
        sys = {"role":"system","content": "You are a friendly appointment assistant. Keep answers short and ask next needed question only."}
        messages = [sys] + conv_history[-8:]
        resp = client.chat.completions.create(model="groq/compound", messages=messages, temperature=0.4, max_tokens=180)
        return resp.choices[0].message.content
    except Exception:
        return ""

def medical_query_groq(api_key, user_query, context_docs):
    # minimal wrapper: call groq with documents (if you want RAG)
    try:
        if not api_key:
            return "Groq API key not provided. Please add key in the sidebar."
        client = Groq(api_key=api_key)
        context = "\n\n".join([getattr(d, "page_content", str(d)) for d in context_docs]) if context_docs else ""
        system_prompt = (
            "You are a careful medical assistant. ONLY use the provided context. "
            "If missing data say: 'Not in documents. Please consult cancer.gov or your doctor.'"
        )
        full = f"CONTEXT:\n{context}\n\nQUESTION: {user_query}\n\nANSWER:"
        res = client.chat.completions.create(model="groq/compound", messages=[{"role":"system","content":system_prompt},{"role":"user","content":full}], temperature=0.2, max_tokens=400)
        return res.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

# ----------------------------
# STREAMLIT TABS
# ----------------------------
def medical_qa_tab(api_key):
    st.markdown("### 🧠 Medical Q&A")
    st.markdown("*Ask medical questions; answers are retrieved from documents.*")

    for msg in st.session_state.medical_chat:
        with st.chat_message(msg['role']):
            st.write(msg['content'])

    query = st.chat_input("Type your medical question…")
    if not query:
        return

    st.session_state.medical_chat.append({'role':'user','content':query})
    with st.chat_message('user'):
        st.write(query)
    with st.chat_message('assistant'):
        with st.spinner("Searching..."):
            vector_db = get_vectorstore()
            if vector_db:
                docs = vector_db.similarity_search(query, k=3)
            else:
                docs = []
            answer = medical_query_groq(api_key, query, docs)
            st.write(answer)
            st.session_state.medical_chat.append({'role':'assistant','content':answer})

def ai_booking_tab(api_key):
    st.markdown("### 🤖 AI Booking Agent")
    st.markdown("Chat naturally — I’ll guide you step-by-step to book an appointment.")

    # show chat history
    for msg in st.session_state.ai_booking_chat:
        with st.chat_message(msg['role']):
            st.write(msg['content'])

    user_msg = st.chat_input("Say something…")
    if not user_msg:
        return

    # append user message
    st.session_state.ai_booking_chat.append({'role':'user','content':user_msg})
    with st.chat_message('user'):
        st.write(user_msg)

    # parse intent & do authoritative python actions
    intent = parse_booking_intent(user_msg)
    python_reply = process_booking_intent_and_act(intent)

    # get LLM phrasing if you like, but DO NOT let it override python actions.
    # We'll call it to generate a short friendly follow-up only when python_reply is empty.
    llm_followup = ""
    if not python_reply:
        # ask LLM to suggest next question (only if python didn't handle it)
        llm_followup = appointment_agent_llm_reply(api_key, st.session_state.ai_booking_chat)

    final_text = python_reply.strip()
    if llm_followup:
        final_text = (final_text + "\n\n" + llm_followup).strip()

    if not final_text:
        final_text = "I couldn't interpret that. Try: location, specialty, 'Book Dr. Name', 'available dates', a date (YYYY-MM-DD) or a time (HH:MM)."

    st.session_state.ai_booking_chat.append({'role':'assistant','content':final_text})
    with st.chat_message('assistant'):
        st.write(final_text)

def find_doctors_tab():
    st.markdown("### 🔍 Find Doctors (manual)")
    col1, col2, col3 = st.columns(3)
    with col1:
        loc = st.text_input("Location (e.g., Noida Sector 62)")
    with col2:
        spec = st.text_input("Specialty (e.g., Kidney Cancer)")
    with col3:
        dis = st.text_input("Disease (e.g., Leukemia)")

    if st.button("Search"):
        results = search_doctors(location=loc, specialty=spec, disease=dis, top_n=20)
        if len(results) == 0:
            st.warning("No matching doctors found.")
            return
        for idx, (_, d) in enumerate(results.iterrows(), start=1):
            with st.expander(f"{idx}. {d['doctor_name_en']} — {d.get('rating', 'NA')}/5", expanded=False):
                st.write(f"**Specialty:** {d['specialty']} / {d['sub_specialty']}")
                st.write(f"**Hospital:** {d['hospital_name']}")
                st.write(f"**Location:** {d['location_area']}")
                st.write(f"**Languages:** {d.get('languages','')}")
                st.write(f"**Treats:** {d.get('diseases_treated','')}")
                st.write(f"**Fee:** ₹{d.get('fee','')}")
                st.write(f"**Experience:** {d.get('experience_years','')} yrs")
                if st.button(f"Select {d['doctor_id']}", key=f"choose_{d['doctor_id']}"):
                    st.session_state.selected_doctor = d['doctor_id']
                    st.success("Doctor selected. Go to 'Book Appointment' tab to continue.")

def manual_booking_tab():
    st.markdown("### 📅 Book Appointment (manual)")
    if not st.session_state.selected_doctor:
        st.info("Select a doctor from 'Find Doctors' first.")
        return
    doctor = DOCTORS_DATABASE[DOCTORS_DATABASE['doctor_id'] == st.session_state.selected_doctor].iloc[0]
    st.write(f"**Doctor:** {doctor['doctor_name_en']} — {doctor['hospital_name']} ({doctor['location_area']})")
    date = st.date_input("Pick a date", min_value=datetime.now().date(), value=datetime.now().date())
    slots = generate_time_slots(doctor['doctor_id'], date)
    available = [s for s in slots if s['available']]
    if not available:
        st.warning("No slots available for this date. Try another date.")
        return
    slot_display = [s['display'] for s in available]
    chosen = st.selectbox("Available slots", slot_display)
    name = st.text_input("Your name")
    phone = st.text_input("Phone number")
    if st.button("Confirm booking"):
        if not name or not phone:
            st.warning("Please enter name and phone.")
        else:
            slot_obj = next(s for s in available if s['display'] == chosen)
            ok, booking = book_appointment(doctor['doctor_id'], date, slot_obj['start'], user_name=name, user_contact=phone)
            if ok:
                st.success(f"Booking confirmed — ID: {booking['booking_id']}")
                st.balloons()

def appointments_tab():
    st.markdown("### 📋 My Appointments")
    if len(st.session_state.bookings) == 0:
        st.info("No appointments yet.")
        return
    for b in sorted(st.session_state.bookings, key=lambda x: (x['date'], x['slot_start'])):
        with st.expander(f"{b['booking_id']} — {b['doctor_name']}"):
            st.write(f"**Date:** {b['date']}")
            st.write(f"**Time:** {b['slot_start'].strftime('%H:%M')} - {b['slot_end'].strftime('%H:%M')}")
            st.write(f"**Hospital:** {b['hospital_name']}")
            st.write(f"**Location:** {b['location']}")
            st.write(f"**Booked At:** {b['created_at'].strftime('%Y-%m-%d %H:%M:%S')}")
            st.write(f"**Queue position:** {b['queue_position']}")
            st.write(f"**Contact:** {b['user_contact']}")
            if st.button(f"Cancel {b['booking_id']}", key=f"cancel_{b['booking_id']}"):
                st.session_state.bookings.remove(b)
                st.success("Booking cancelled.")
                st.experimental_rerun()

# ----------------------------
# MAIN
# ----------------------------
def main():
    st.set_page_config(page_title="Cancer Care Assistant", page_icon="💙", layout="wide")
    init_session_state()

    st.title("💙 Comprehensive Cancer Care Assistant")

    # Sidebar
    with st.sidebar:
        st.header("🔑 Groq API Key")
        key = st.text_input("Enter key", type="password")
        if key:
            st.session_state.groq_api_key = key.strip()
            st.success("API key saved (session).")
        st.markdown("---")
        st.header("Language")
        lang = st.radio("Select", ["English", "हिंदी"], index=0)
        st.session_state.user_language = lang

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🧠 Medical Q&A",
        "🤖 AI Booking Agent",
        "🔍 Find Doctors",
        "📅 Book Appointment",
        "📋 My Appointments"
    ])

    with tab1:
        medical_qa_tab(st.session_state.groq_api_key)
    with tab2:
        ai_booking_tab(st.session_state.groq_api_key)
    with tab3:
        find_doctors_tab()
    with tab4:
        manual_booking_tab()
    with tab5:
        appointments_tab()

if __name__ == "__main__":
    main()
