# app.py
import os
import re
import json
import calendar
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ============================================================
#  CONFIG & DATA
# ============================================================
DB_FAISS_PATH = "vectorstore/db_faiss"
DOCTORS_CSV = "doctors_300_dataset.csv"

# Load doctors CSV safely
if os.path.exists(DOCTORS_CSV):
    DOCTORS_DATABASE = pd.read_csv(DOCTORS_CSV)
else:
    # If missing, create a very small fallback to avoid crashes
    DOCTORS_DATABASE = pd.DataFrame([
        {"doctor_id": "D001", "doctor_name_en": "Dr. Amit Sharma", "doctor_name_hi": "डॉ. अमित शर्मा",
         "specialty": "Oncology", "sub_specialty": "Kidney Cancer", "occupation": "Senior Consultant",
         "hospital_name": "Noida Cancer Center", "location_area": "Noida Sector 62", "languages": "English,Hindi",
         "diseases_treated": "Kidney Cancer,Bladder Cancer,Prostate Cancer", "rating": 4.8, "reviews_count": 420,
         "consultation_duration_mins": 15, "telemedicine": True, "fee": 1500, "experience_years": 15},
    ])

# ============================================================
#  SESSION STATE INIT
# ============================================================
def init_session_state():
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
    if 'ai_doctors' not in st.session_state:
        st.session_state.ai_doctors = None
    if 'ai_selected_doctor' not in st.session_state:
        st.session_state.ai_selected_doctor = None
    if 'medical_chat' not in st.session_state:
        st.session_state.medical_chat = [
            {"role": "assistant", "content": "Hello! Ask your medical question (I will use documents to answer)."}
        ]

# ============================================================
#  VECTORSTORE LOADER (FAISS)
# ============================================================
@st.cache_resource
def get_vectorstore():
    try:
        embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        # Don't crash app if FAISS not available
        return None

# ============================================================
#  UTILITIES: date/time parsing (friendly), doctor search, slots
# ============================================================
def parse_natural_date(text):
    """Small natural date parser that supports many common formats."""
    text = text.strip()
    # 1) ISO 2025-11-18
    m_iso = re.search(r"\b(20\d{2})[/-](\d{1,2})[/-](\d{1,2})\b", text)
    if m_iso:
        y, mo, d = map(int, m_iso.groups())
        try:
            return datetime(y, mo, d).date()
        except:
            return None

    # 2) dd/mm/yyyy or dd-mm-yyyy
    m_dmy = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b", text)
    if m_dmy:
        d, mo, y = m_dmy.groups()
        y = int(y)
        if y < 100:
            y += 2000
        try:
            return datetime(int(y), int(mo), int(d)).date()
        except:
            return None

    # 3) "18 Nov 2025" or "18 Nov"
    m = re.search(r"\b(\d{1,2})\s*([A-Za-z]{3,9})(?:\s*(\d{4}))?\b", text)
    if m:
        d = int(m.group(1))
        mon_str = m.group(2)[:3].title()
        y = int(m.group(3)) if m.group(3) else datetime.now().year
        try:
            month_number = list(calendar.month_abbr).index(mon_str)
            return datetime(y, month_number, d).date()
        except Exception:
            return None

    # 4) "Nov 18" or "November 18 2025"
    m2 = re.search(r"\b([A-Za-z]{3,9})\s*(\d{1,2})(?:\s*(\d{4}))?\b", text)
    if m2:
        mon_str = m2.group(1)[:3].title()
        d = int(m2.group(2))
        y = int(m2.group(3)) if m2.group(3) else datetime.now().year
        try:
            month_number = list(calendar.month_abbr).index(mon_str)
            return datetime(y, month_number, d).date()
        except Exception:
            return None

    # 5) "today" / "tomorrow" support
    if re.search(r"\btoday\b", text, re.I):
        return datetime.now().date()
    if re.search(r"\btomorrow\b", text, re.I):
        return (datetime.now() + timedelta(days=1)).date()

    return None

def parse_natural_time(text):
    """Parse time like 10:30, 10.30, 10am, 10 pm, 9 -> 'HH:MM'"""
    text = text.strip()
    m = re.search(r"\b(\d{1,2})[:.](\d{2})\b", text)
    if m:
        h = int(m.group(1)) % 24
        mm = int(m.group(2)) % 60
        return f"{h:02d}:{mm:02d}"
    m2 = re.search(r"\b(\d{1,2})(?:\s*(am|pm))\b", text, re.I)
    if m2:
        h = int(m2.group(1))
        ampm = m2.group(2).lower()
        if ampm == "pm" and h != 12:
            h = (h + 12) % 24
        if ampm == "am" and h == 12:
            h = 0
        return f"{h:02d}:00"
    # plain hour e.g., "10" -> 10:00
    m3 = re.search(r"\b(\d{1,2})\b", text)
    if m3:
        h = int(m3.group(1)) % 24
        return f"{h:02d}:00"
    return None

def search_doctors(location=None, specialty=None, disease=None, top_n=5):
    df = DOCTORS_DATABASE.copy()
    if location:
        df = df[df['location_area'].str.contains(location, case=False, na=False)]
    if specialty:
        df = df[df['specialty'].str.contains(specialty, case=False, na=False) |
                df['sub_specialty'].str.contains(specialty, case=False, na=False)]
    if disease:
        df = df[df['diseases_treated'].str.contains(disease, case=False, na=False)]
    if len(df) > 0:
        # compute simple score
        df['score'] = (
            0.45 * (df['rating'] / 5.0) +
            0.25 * (1.0 if specialty or disease else 0.5) +
            0.20 * (1.0 if location else 0.5) +
            0.10 * (df['reviews_count'] / df['reviews_count'].max())
        )
        df = df.sort_values('score', ascending=False)
    return df.head(top_n)

def generate_time_slots(doctor_id, date):
    doctor = DOCTORS_DATABASE[DOCTORS_DATABASE['doctor_id'] == doctor_id].iloc[0]
    duration = int(doctor['consultation_duration_mins'])
    slots = []
    start = datetime.combine(date, datetime.strptime("09:00", "%H:%M").time())
    end = datetime.combine(date, datetime.strptime("17:00", "%H:%M").time())
    current = start
    while current < end:
        slot_end = current + timedelta(minutes=duration)
        is_booked = any(
            b['doctor_id'] == doctor_id and b['date'] == date and b['slot_start'] == current
            for b in st.session_state.bookings
        )
        slots.append({
            'start': current,
            'end': slot_end,
            'available': not is_booked,
            'display': current.strftime("%H:%M")
        })
        current = slot_end
    return slots

def book_appointment(doctor_id, date, slot_start, user_name, user_contact):
    doctor = DOCTORS_DATABASE[DOCTORS_DATABASE['doctor_id'] == doctor_id].iloc[0]
    conflicts = [b for b in st.session_state.bookings
                 if b['doctor_id'] == doctor_id and b['date'] == date and b['slot_start'] == slot_start]
    if conflicts:
        return False, "Slot already booked"
    booking_id = f"BK{len(st.session_state.bookings) + 1001}"
    hour_start = slot_start.replace(minute=0, second=0)
    hour_bookings = [b for b in st.session_state.bookings
                     if b['doctor_id'] == doctor_id and b['date'] == date and
                     b['slot_start'] >= hour_start and b['slot_start'] < hour_start + timedelta(hours=1)]
    queue_position = len(hour_bookings) + 1
    booking = {
        'booking_id': booking_id,
        'doctor_id': doctor_id,
        'doctor_name': doctor['doctor_name_en'],
        'hospital_name': doctor['hospital_name'],
        'location': doctor['location_area'],
        'date': date,
        'slot_start': slot_start,
        'slot_end': slot_start + timedelta(minutes=int(doctor['consultation_duration_mins'])),
        'user_name': user_name,
        'user_contact': user_contact,
        'queue_position': queue_position,
        'created_at': datetime.now(),
        'status': 'confirmed'
    }
    st.session_state.bookings.append(booking)
    return True, booking

# ============================================================
#  INTENT PARSING (booking) - uses natural parsers above
# ============================================================
def parse_booking_intent(text):
    intent = {}
    # location
    loc = re.search(r"(Noida|Indirapuram|Botanical Garden|Gurgaon|Ghaziabad|Delhi|Sector \d+)", text, re.I)
    if loc:
        intent['location'] = loc.group(0)
    # doctor name
    doc = re.search(r"Dr\.?\s+[A-Za-z]+(?:\s+[A-Za-z]+)*", text)
    if doc:
        intent['doctor_name'] = doc.group(0)
    # specialty/disease
    spec = re.search(r"(kidney|renal|breast|lung|blood|oncology|pediatric|sarcoma|leukemia|lymphoma)", text, re.I)
    if spec:
        intent['specialty'] = spec.group(0)
    # date (flexible)
    dt = parse_natural_date(text)
    if dt:
        intent['date'] = dt
    # time (flexible)
    tm = parse_natural_time(text)
    if tm:
        intent['slot'] = tm
    return intent

# ============================================================
#  GROQ: Medical Q&A & Booking Agent helper calls
# ============================================================
def medical_query_groq(api_key, user_query, docs):
    try:
        client = Groq(api_key=api_key)
        context = "\n\n".join([d.page_content for d in docs]) if docs else ""
        system_prompt = """
You are a highly reliable medical assistant for cancer patients.
You must only use the provided context to answer.
If info not present, reply EXACTLY:
"Not in documents. Please consult cancer.gov or your doctor."
Be empathetic and cite page numbers when available.
"""
        full_prompt = f"CONTEXT:\n{context}\n\nQUESTION: {user_query}\n\nANSWER:"
        res = client.chat.completions.create(
            model="groq/compound",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.2,
            max_tokens=800
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

def appointment_agent_llm(api_key, conversation_history):
    try:
        client = Groq(api_key=api_key)
        system_prompt = """
You are an appointment-booking assistant. Your job: help the user proceed step-by-step
(location → specialty → doctor selection → date → time → confirm). 
You MUST not refuse to continue, and you must ask for the next missing detail.
Keep responses short and polite.
"""
        res = client.chat.completions.create(
            model="groq/compound",
            messages=[{"role": "system", "content": system_prompt}] + conversation_history,
            temperature=0.3,
            max_tokens=300
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"(LLM error: {e})"

# ============================================================
#  PROCESS BOOKING LOGIC (Python-side deterministic actions)
# ============================================================
def process_booking_logic(intent):
    """Performs search/select/slot/booking based on parsed intent and session state."""
    response_text = ""

    # Step 1: search doctors
    if 'location' in intent or 'specialty' in intent:
        docs = search_doctors(location=intent.get('location'), specialty=intent.get('specialty'))
        if docs is not None and len(docs) > 0:
            st.session_state.ai_doctors = docs
            response_text += "Top doctors:\n"
            for i, (_, d) in enumerate(docs.iterrows(), start=1):
                response_text += f"{i}. {d['doctor_name_en']} — {d['specialty']} ({d['location_area']})\n"
            response_text += "\nPlease tell me which doctor (name) you want, or say the number.\n"
        else:
            response_text += "No doctors found. Try another location or specialty.\n"

    # Step 2: select by doctor name or number
    if 'doctor_name' in intent and st.session_state.ai_doctors is not None:
        # try name match
        name = intent['doctor_name']
        # allow inputs like "Dr. Amit Sharma" or a partial name
        dmatch = st.session_state.ai_doctors[
            st.session_state.ai_doctors['doctor_name_en'].str.contains(name.replace("Dr.", "").strip(), case=False, na=False)
        ]
        # also check if user sent a number like "1"
        if len(dmatch) == 0:
            mnum = re.search(r"\b([1-9]\d?)\b", name)
            if mnum:
                idx = int(mnum.group(1)) - 1
                if 0 <= idx < len(st.session_state.ai_doctors):
                    selected = st.session_state.ai_doctors.iloc[idx]
                    st.session_state.ai_selected_doctor = selected['doctor_id']
                    response_text += f"Doctor selected: {selected['doctor_name_en']}\nPlease provide a date (YYYY-MM-DD or '18 Nov').\n"
                    return response_text
        if len(dmatch) > 0:
            selected = dmatch.iloc[0]
            st.session_state.ai_selected_doctor = selected['doctor_id']
            response_text += f"Doctor selected: {selected['doctor_name_en']}\nPlease provide a date (YYYY-MM-DD or '18 Nov').\n"
        else:
            response_text += "Doctor not found in the current list. Please provide exact name or pick a number.\n"

    # Step 3: show slots for given date
    if 'date' in intent and st.session_state.ai_selected_doctor:
        date = intent['date']
        slots = generate_time_slots(st.session_state.ai_selected_doctor, date)
        available = [s for s in slots if s['available']]
        if available:
            response_text += f"Available slots on {date}:\n"
            for s in available[:20]:
                response_text += f"- {s['display']}\n"
            response_text += "Please choose a time (e.g., 10:30 or 10:30 AM).\n"
        else:
            response_text += "No slots available on that date. Try another date.\n"

    # Step 4: final booking if slot provided
    if 'slot' in intent and 'date' in intent and st.session_state.ai_selected_doctor:
        # convert slot string ('HH:MM') to datetime
        try:
            time_obj = datetime.strptime(intent['slot'], "%H:%M").time()
        except:
            response_text += "Invalid time format. Use HH:MM.\n"
            return response_text
        slot_dt = datetime.combine(intent['date'], time_obj)
        success, booking_or_msg = book_appointment(
            st.session_state.ai_selected_doctor,
            intent['date'],
            slot_dt,
            user_name="AI User",
            user_contact="9999999999"
        )
        if success:
            b = booking_or_msg
            response_text += f"🎉 Appointment Confirmed!\nBooking ID: {b['booking_id']}\nDoctor: {b['doctor_name']}\nDate: {b['date']}\nTime: {b['slot_start'].strftime('%H:%M')} - {b['slot_end'].strftime('%H:%M')}\nQueue: {b['queue_position']}\n"
        else:
            response_text += f"⚠️ {booking_or_msg}\n"

    return response_text

# ============================================================
#  STREAMLIT UI: Tabs
# ============================================================
def medical_qa_tab(api_key):
    st.markdown("### 🧠 Medical Q&A")
    st.markdown("*Ask cancer-related questions. Answers are produced using your uploaded medical docs and the vector DB.*")

    if not api_key:
        st.warning("⚠️ Enter Groq API key in the sidebar.")
        return

    for msg in st.session_state.medical_chat:
        with st.chat_message(msg['role']):
            st.write(msg['content'])

    q = st.chat_input("Type your medical question...")
    if not q:
        return
    st.session_state.medical_chat.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.write(q)

    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            vector_db = get_vectorstore()
            if vector_db:
                docs = vector_db.similarity_search(q, k=3)
                ans = medical_query_groq(api_key, q, docs)
            else:
                ans = "Vector DB not available. Please run ingestion first."
            st.write(ans)
            st.session_state.medical_chat.append({"role": "assistant", "content": ans})

def ai_booking_tab(api_key):
    st.markdown("### 🤖 AI Booking Agent")
    st.markdown("Chat naturally — the agent will guide you to book an appointment.")

    if not api_key:
        st.warning("⚠️ Enter Groq API key in the sidebar.")
        return

    # Show existing chat
    for msg in st.session_state.ai_booking_chat:
        with st.chat_message(msg['role']):
            st.write(msg['content'])

    user_msg = st.chat_input("Say something...")
    if not user_msg:
        return

    # Save user message
    st.session_state.ai_booking_chat.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.write(user_msg)

    # Parse intent and run python actions
    intent = parse_booking_intent(user_msg)
    python_reply = process_booking_logic(intent)

    # If python confirmed booking -> append python confirmation and rerun to update My Appointments
    if any(k in python_reply for k in ["Appointment Confirmed", "Booking ID:", "🎉", "Appointment Confirmed!"]):
        # Append Python reply directly (truth)
        st.session_state.ai_booking_chat.append({"role": "assistant", "content": python_reply})
        with st.chat_message("assistant"):
            st.write(python_reply)
        # reload app so My Appointments refreshes
        st.experimental_rerun()
        return

    # Else call LLM to continue the conversation (with strict system prompt)
    llm_reply = appointment_agent_llm(api_key, st.session_state.ai_booking_chat)
    final_reply = (python_reply + "\n\n" + llm_reply).strip()

    st.session_state.ai_booking_chat.append({"role": "assistant", "content": final_reply})
    with st.chat_message("assistant"):
        st.write(final_reply)

def find_doctors_tab():
    st.markdown("### 🔍 Find Doctors (Manual Search)")
    col1, col2, col3 = st.columns(3)
    with col1:
        loc = st.text_input("Location (e.g., Noida, Indirapuram)")
    with col2:
        spec = st.text_input("Specialty (e.g., Kidney Cancer)")
    with col3:
        dis = st.text_input("Disease (optional)")
    if st.button("Search"):
        results = search_doctors(location=loc, specialty=spec, disease=dis)
        if results is None or len(results) == 0:
            st.warning("No matching doctors found.")
            return
        for idx, (_, d) in enumerate(results.iterrows(), start=1):
            with st.expander(f"{idx}. {d['doctor_name_en']} — {d['rating']}/5"):
                st.write(f"**Specialty:** {d['specialty']} / {d['sub_specialty']}")
                st.write(f"**Hospital:** {d['hospital_name']}")
                st.write(f"**Location:** {d['location_area']}")
                st.write(f"**Languages:** {d['languages']}")
                st.write(f"**Treats:** {d['diseases_treated']}")
                st.write(f"**Fee:** ₹{d['fee']} | Experience: {d['experience_years']} yrs")
                if st.button(f"Select {d['doctor_id']}", key=f"choose_{d['doctor_id']}"):
                    st.session_state.selected_doctor = d['doctor_id']
                    st.success("Doctor selected. Go to 'Book Appointment' tab to confirm.")

def manual_booking_tab():
    st.markdown("### 📅 Book Appointment (Manual)")

    if not st.session_state.selected_doctor:
        st.info("Select a doctor from 'Find Doctors' first.")
        return

    doctor = DOCTORS_DATABASE[DOCTORS_DATABASE['doctor_id'] == st.session_state.selected_doctor].iloc[0]
    st.write(f"**Doctor:** {doctor['doctor_name_en']}")
    st.write(f"**Hospital:** {doctor['hospital_name']} • {doctor['location_area']}")
    date = st.date_input("Choose date", min_value=datetime.now().date())
    slots = generate_time_slots(st.session_state.selected_doctor, date)
    available = [s for s in slots if s['available']]
    if not available:
        st.warning("No slots available.")
        return
    choice = st.selectbox("Available slots", [s['display'] for s in available])
    name = st.text_input("Your name")
    phone = st.text_input("Phone number")
    if st.button("Confirm Booking"):
        if not name or not phone:
            st.warning("Please provide name and phone.")
        else:
            slot_obj = next(s for s in available if s['display'] == choice)
            ok, booking = book_appointment(st.session_state.selected_doctor, date, slot_obj['start'], name, phone)
            if ok:
                st.success(f"Booking confirmed! ID: {booking['booking_id']}")
                st.balloons()
                # ensure My Appointments shows the booking
                st.experimental_rerun()
            else:
                st.error(booking)

def appointments_tab():
    st.markdown("### 📋 My Appointments")
    if len(st.session_state.bookings) == 0:
        st.info("No appointments yet.")
        return
    sorted_b = sorted(st.session_state.bookings, key=lambda x: (x['date'], x['slot_start']))
    for b in sorted_b:
        with st.expander(f"{b['booking_id']} — {b['doctor_name']}"):
            date_str = b['date'].strftime('%d %b %Y') if hasattr(b['date'], 'strftime') else str(b['date'])
            st.write(f"**Date:** {date_str}")
            st.write(f"**Time:** {b['slot_start'].strftime('%I:%M %p')} - {b['slot_end'].strftime('%I:%M %p')}")
            st.write(f"**Hospital:** {b['hospital_name']}")
            st.write(f"**Location:** {b['location']}")
            st.write(f"**Queue position:** {b.get('queue_position', '-')}")
            st.write(f"**Phone:** {b['user_contact']}")

# ============================================================
#  MAIN
# ============================================================
def main():
    st.set_page_config(page_title="Cancer Care Assistant", layout="wide")
    init_session_state()

    with st.sidebar:
        st.header("🔑 Groq API Key")
        api_key = st.text_input("Enter key", type="password")
        st.markdown("---")
        st.header("Language")
        lang = st.radio("Select:", ["English", "हिंदी"], key="lang")
        st.session_state.user_language = lang

    st.title("💙 Comprehensive Cancer Care Assistant")
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🧠 Medical Q&A",
        "🤖 AI Booking Agent",
        "🔍 Find Doctors",
        "📅 Book Appointment",
        "📋 My Appointments"
    ])

    with tab1:
        medical_qa_tab(api_key if api_key else None)
    with tab2:
        ai_booking_tab(api_key if api_key else None)
    with tab3:
        find_doctors_tab()
    with tab4:
        manual_booking_tab()
    with tab5:
        appointments_tab()

if __name__ == "__main__":
    main()
