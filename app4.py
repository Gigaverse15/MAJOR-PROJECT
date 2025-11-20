# app.py
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
# CONFIG / DATA
# ----------------------------
DB_FAISS_PATH = "vectorstore/db_faiss"
DOCTORS_CSV = "doctors_300_dataset.csv"
DEFAULT_USER_NAME = "AI User"
DEFAULT_USER_CONTACT = "9999999999"

# load doctors CSV (fallback to minimal if missing)
if os.path.exists(DOCTORS_CSV):
    DOCTORS_DATABASE = pd.read_csv(DOCTORS_CSV)
else:
    # minimal fallback example row (so app doesn't crash)
    DOCTORS_DATABASE = pd.DataFrame([{
        "doctor_id":"D001","doctor_name_en":"Dr. Amit Sharma","doctor_name_hi":"डॉ. अमित शर्मा",
        "specialty":"Oncology","sub_specialty":"Kidney Cancer","occupation":"Senior Consultant",
        "hospital_name":"Noida Cancer Center","location_area":"Noida Sector 62",
        "languages":"English,Hindi","diseases_treated":"Kidney Cancer,Bladder Cancer,Prostate Cancer",
        "rating":4.8,"reviews_count":420,"consultation_duration_mins":15,"telemedicine":True,
        "fee":1500,"experience_years":15
    }])

# ----------------------------
# STREAMLIT SESSION INIT
# ----------------------------
def init_session_state():
    if 'bookings' not in st.session_state:
        st.session_state.bookings = []
    if 'selected_doctor' not in st.session_state:
        st.session_state.selected_doctor = None
    if 'user_language' not in st.session_state:
        st.session_state.user_language = "English"
    if 'ai_booking_chat' not in st.session_state:
        st.session_state.ai_booking_chat = [
            {"role":"assistant","content":"Hello! 👋 I can help you book an appointment. Where are you located?"}
        ]
    if 'ai_doctors' not in st.session_state:
        st.session_state.ai_doctors = None
    if 'ai_selected_doctor' not in st.session_state:
        st.session_state.ai_selected_doctor = None

# ----------------------------
# VECTORSTORE loader (RAG)
# ----------------------------
@st.cache_resource
def get_vectorstore():
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        return None

# ----------------------------
# DOCTOR SEARCH & SLOTS
# ----------------------------
def search_doctors(location=None, specialty=None, disease=None, top_n=10):
    df = DOCTORS_DATABASE.copy()
    if location:
        df = df[df['location_area'].str.contains(location, case=False, na=False)]
    if specialty:
        df = df[
            df['specialty'].str.contains(specialty, case=False, na=False) |
            df['sub_specialty'].str.contains(specialty, case=False, na=False)
        ]
    if disease:
        df = df[df['diseases_treated'].str.contains(disease, case=False, na=False)]
    if len(df) > 0:
        # simple score: rating + reviews influence
        max_reviews = df['reviews_count'].max() if df['reviews_count'].max() > 0 else 1
        df['score'] = 0.6*(df['rating']/5.0) + 0.4*(df['reviews_count']/max_reviews)
        df = df.sort_values('score', ascending=False)
    return df.head(top_n)

def generate_time_slots(doctor_id, date):
    """Create slots between 09:00 and 17:00 using doctor's consultation_duration_mins"""
    doctor = DOCTORS_DATABASE[DOCTORS_DATABASE['doctor_id'] == doctor_id]
    if doctor.empty:
        return []
    doctor = doctor.iloc[0]
    duration = int(doctor.get('consultation_duration_mins', 15))
    start_dt = datetime.combine(date, time(hour=9, minute=0))
    end_dt = datetime.combine(date, time(hour=17, minute=0))
    current = start_dt
    slots = []
    while current + timedelta(minutes=duration) <= end_dt:
        # check booking conflicts
        is_booked = any(
            (b['doctor_id'] == doctor_id and b['date'] == date and b['slot_start'] == current)
            for b in st.session_state.bookings
        )
        slots.append({
            'start': current,
            'end': current + timedelta(minutes=duration),
            'display': current.strftime("%H:%M"),
            'available': not is_booked
        })
        current += timedelta(minutes=duration)
    return slots

def next_n_dates(n=7):
    today = datetime.now().date()
    return [today + timedelta(days=i) for i in range(0, n)]

# ----------------------------
# BOOKING LOGIC (local)
# ----------------------------
def book_appointment(doctor_id, date, slot_start, user_name=DEFAULT_USER_NAME, user_contact=DEFAULT_USER_CONTACT):
    # conflict check
    conflict = [b for b in st.session_state.bookings if b['doctor_id']==doctor_id and b['date']==date and b['slot_start']==slot_start]
    if conflict:
        return False, "Slot already booked"
    doctor = DOCTORS_DATABASE[DOCTORS_DATABASE['doctor_id']==doctor_id].iloc[0]
    booking_id = f"BK{1000 + len(st.session_state.bookings) + 1}"
    # compute queue within hour
    hour_start = slot_start.replace(minute=0, second=0, microsecond=0)
    hour_bookings = [b for b in st.session_state.bookings if b['doctor_id']==doctor_id and b['date']==date and hour_start <= b['slot_start'] < hour_start + timedelta(hours=1)]
    queue_pos = len(hour_bookings) + 1
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
        'queue_position': queue_pos,
        'created_at': datetime.now(),
        'status': 'confirmed'
    }
    st.session_state.bookings.append(booking)
    return True, booking

# ----------------------------
# INTENT PARSING (improved)
# ----------------------------
def parse_booking_intent(text):
    intent = {}
    text = text.strip()

    # numeric pick (e.g., "1" or "pick 2")
    m_num = re.search(r"\b(?:pick|choose|select|number|#)?\s*(\d{1,2})\b", text, re.I)
    if m_num:
        intent['select_number'] = int(m_num.group(1))

    # doctor by "Dr. Name" or "book Dr. Name"
    m_doc = re.search(r"(Dr\.?\s+[A-Za-z][A-Za-z\.\-']+(?:\s+[A-Za-z\.\-']+){0,3})", text, re.I)
    if m_doc:
        intent['doctor_name'] = m_doc.group(1).strip()

    # location keywords (allow common tokens)
    m_loc = re.search(r"(Noida|Noida Sector \d+|Sector \d+|Indirapuram|Botanical Garden|Greater Noida|Gurugram|Ghaziabad|Delhi)", text, re.I)
    if m_loc:
        intent['location'] = m_loc.group(0)

    # specialty/disease
    m_spec = re.search(r"(kidney|renal|nephrology|oncology|lung|breast|leukemia|lymphoma|urology|pediatric)", text, re.I)
    if m_spec:
        intent['specialty'] = m_spec.group(1)

    # date: accept YYYY-MM-DD or natural forms like "9 Oct", "Oct 9", "9/10/2025"
    m_iso = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", text)
    if m_iso:
        try:
            intent['date'] = datetime.strptime(m_iso.group(1), "%Y-%m-%d").date()
        except:
            pass
    else:
        # detect dd Mon (e.g., 9 Oct or 9 October)
        m_natural = re.search(r"\b(\d{1,2})\s*(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)\b", text, re.I)
        if m_natural:
            day = int(m_natural.group(1))
            mon = m_natural.group(2)[:3]
            try:
                candidate = datetime.strptime(f"{day} {mon} {datetime.now().year}", "%d %b %Y").date()
                intent['date'] = candidate
            except:
                pass

    # time: 10:30, 14:00, 2pm, 2:30pm
    m_time = re.search(r"\b(\d{1,2}[:\.]\d{2}|\d{1,2}\s*(?:am|pm))\b", text, re.I)
    if m_time:
        t_raw = m_time.group(1).replace('.',':').strip()
        try:
            if 'am' in t_raw.lower() or 'pm' in t_raw.lower():
                t_parsed = datetime.strptime(t_raw.lower(), "%I%p").time() if ':' not in t_raw else datetime.strptime(t_raw.lower(), "%I:%M%p").time()
            else:
                t_parsed = datetime.strptime(t_raw, "%H:%M").time()
            intent['slot'] = t_parsed.strftime("%H:%M")
        except:
            pass

    return intent

# ----------------------------
# PROCESS INTENT -> ACTIONS
# ----------------------------
def process_booking_logic(intent):
    """This function both updates session state and returns text reply used by chat UI"""
    reply_parts = []

    # 1) user provided location/specialty -> search and show top doctors
    if 'location' in intent or 'specialty' in intent:
        docs = search_doctors(location=intent.get('location'), specialty=intent.get('specialty'))
        if docs is None or len(docs) == 0:
            reply_parts.append("I couldn't find matching doctors for that location/specialty.")
        else:
            st.session_state.ai_doctors = docs.reset_index(drop=True)
            # show numbered list (1..N)
            lines = ["Here are the top doctors I found:"]
            for idx, (_, d) in enumerate(st.session_state.ai_doctors.iterrows(), start=1):
                lines.append(f"{idx}. {d['doctor_name_en']} — {d['specialty']} ({d['location_area']})")
            lines.append("\nYou can pick a doctor by replying with the number (e.g., '1') or 'Book Dr. Name'.")
            reply_parts.append("\n".join(lines))

    # 2) selection by number
    if 'select_number' in intent and st.session_state.get('ai_doctors') is not None:
        n = intent['select_number']
        doctors_list = st.session_state.get('ai_doctors')
        if 1 <= n <= len(doctors_list):
            row = doctors_list.iloc[n-1]
            st.session_state.ai_selected_doctor = row['doctor_id']
            reply_parts.append(f"Doctor selected: {row['doctor_name_en']}. I will show available dates and time slots.")
            # auto-show next available dates
            dates = next_n_dates(7)
            date_lines = ["Next available dates (choose one or reply with a date):"]
            for d in dates:
                date_lines.append(f"- {d.isoformat()} ({d.strftime('%A')})")
            reply_parts.append("\n".join(date_lines))
        else:
            reply_parts.append("Number out of range. Please choose a number from the list.")

    # 3) selection by doctor name text
    if 'doctor_name' in intent and intent.get('doctor_name'):
        name = intent['doctor_name'].replace("Dr.", "").strip()
        matches = DOCTORS_DATABASE[DOCTORS_DATABASE['doctor_name_en'].str.contains(name, case=False, na=False)]
        if len(matches) > 0:
            selected = matches.iloc[0]
            st.session_state.ai_selected_doctor = selected['doctor_id']
            reply_parts.append(f"Doctor selected: {selected['doctor_name_en']}. I'll list dates and slots.")
            # list next dates
            dates = next_n_dates(7)
            date_lines = ["Available dates (next 7 days):"]
            for d in dates:
                date_lines.append(f"- {d.isoformat()} ({d.strftime('%A')})")
            reply_parts.append("\n".join(date_lines))
        else:
            if st.session_state.get('ai_doctors') is not None:
                reply_parts.append("Doctor not found in the dataset. Please provide an exact name or choose a number from the list.")
            else:
                reply_parts.append("Doctor not found. Try searching first with location or specialty.")

    # 4) user provided a date -> show time slots automatically (we pick next slots using generate_time_slots)
    if 'date' in intent and st.session_state.get('ai_selected_doctor'):
        date = intent['date']
        doc_id = st.session_state.ai_selected_doctor
        slots = generate_time_slots(doc_id, date)
        available = [s for s in slots if s['available']]
        if available:
            lines = [f"Available slots on {date.isoformat()}:"]
            for s in available[:20]:
                lines.append(f"- {s['display']}")
            lines.append("Please pick a time (e.g., '10:30' or '10:30 AM').")
            reply_parts.append("\n".join(lines))
        else:
            reply_parts.append("No slots available on that date. Try another date.")

    # 5) user provided a time slot (slot) AND date previously chosen -> attempt booking
    if 'slot' in intent and 'date' in intent and st.session_state.get('ai_selected_doctor'):
        # parse time robustly
        dt_time = None
        slot_raw = intent.get('slot')
        if isinstance(slot_raw, str):
            sr = slot_raw.strip().lower()
            # try H:M
            try:
                dt_time = datetime.strptime(sr, "%H:%M").time()
            except:
                pass
            # try 12hr with am/pm
            if dt_time is None:
                try:
                    dt_time = datetime.strptime(sr.replace(" ", ""), "%I:%M%p").time()
                except:
                    try:
                        dt_time = datetime.strptime(sr.replace(" ", ""), "%I%p").time()
                    except:
                        dt_time = None
        elif isinstance(slot_raw, time):
            dt_time = slot_raw

        if dt_time is not None:
            dt_combined = datetime.combine(intent['date'], dt_time)
            ok, result = book_appointment(st.session_state.ai_selected_doctor, intent['date'], dt_combined)
            if ok:
                # booking already appended by book_appointment()
                reply_parts.append("🎉 Appointment confirmed!")
                reply_parts.append(f"Booking ID: {result['booking_id']}")
                reply_parts.append(f"Doctor: {result['doctor_name']}")
                reply_parts.append(f"Date: {result['date'].isoformat()}")
                reply_parts.append(f"Time: {result['slot_start'].strftime('%I:%M %p')}")
                reply_parts.append(f"Queue position: {result['queue_position']}")
                # clear selected doctor so user can make another booking if needed
                st.session_state.ai_selected_doctor = None

                # force a re-run so UI (My Appointments) immediately shows the new booking
                try:
                    st.experimental_rerun()
                except Exception:
                    # in some contexts rerun may be disallowed; ignore safely
                    pass
            else:
                reply_parts.append(f"⚠️ {result}")
        else:
            reply_parts.append("Couldn't parse that time — send like '10:30' or '10:30 AM'.")

    # fallback
    if not reply_parts:
        reply_parts.append("I didn't get enough detail. Please tell me location / specialty to start, or pick a doctor number from the list, or give a date/time to book.")

    return "\n\n".join(reply_parts)

# def process_booking_logic(intent):
#     """This function both updates session state and returns text reply used by chat UI"""
#     reply_parts = []

#     # 1) user provided location/specialty -> search and show top doctors
#     if 'location' in intent or 'specialty' in intent:
#         docs = search_doctors(location=intent.get('location'), specialty=intent.get('specialty'))
#         if docs is None or len(docs) == 0:
#             reply_parts.append("I couldn't find matching doctors for that location/specialty.")
#         else:
#             st.session_state.ai_doctors = docs.reset_index(drop=True)
#             # show numbered list (1..N)
#             lines = ["Here are the top doctors I found:"]
#             for idx, (_, d) in enumerate(st.session_state.ai_doctors.iterrows(), start=1):
#                 lines.append(f"{idx}. {d['doctor_name_en']} — {d['specialty']} ({d['location_area']})")
#             lines.append("\nYou can pick a doctor by replying with the number (e.g., '1') or 'Book Dr. Name'.")
#             reply_parts.append("\n".join(lines))

#     # 2) selection by number
#     if 'select_number' in intent and st.session_state.ai_doctors is not None:
#         n = intent['select_number']
#         if 1 <= n <= len(st.session_state.ai_doctors):
#             row = st.session_state.ai_doctors.iloc[n-1]
#             st.session_state.ai_selected_doctor = row['doctor_id']
#             reply_parts.append(f"Doctor selected: {row['doctor_name_en']}. I will show available dates and time slots.")
#             # auto-show next available dates
#             dates = next_n_dates(7)
#             date_lines = ["Next available dates (choose one or reply with a date):"]
#             for d in dates:
#                 date_lines.append(f"- {d.isoformat()} ({d.strftime('%A')})")
#             reply_parts.append("\n".join(date_lines))
#         else:
#             reply_parts.append("Number out of range. Please choose a number from the list.")

#     # 3) selection by doctor name text
#     if 'doctor_name' in intent:
#         name = intent['doctor_name']
#         # match case-insensitive name substring against 'doctor_name_en'
#         matches = DOCTORS_DATABASE[DOCTORS_DATABASE['doctor_name_en'].str.contains(name.replace('Dr.','').strip(), case=False, na=False)]
#         if len(matches) > 0:
#             selected = matches.iloc[0]
#             st.session_state.ai_selected_doctor = selected['doctor_id']
#             reply_parts.append(f"Doctor selected: {selected['doctor_name_en']}. I'll list dates and slots.")
#             # list next dates
#             dates = next_n_dates(7)
#             date_lines = ["Available dates (next 7 days):"]
#             for d in dates:
#                 date_lines.append(f"- {d.isoformat()} ({d.strftime('%A')})")
#             reply_parts.append("\n".join(date_lines))
#         else:
#             # If we had a previous ai_doctors list, suggest to pick from it
#             if st.session_state.ai_doctors is not None:
#                 reply_parts.append("Doctor not found in the dataset. Please provide an exact name or choose a number from the list.")
#             else:
#                 reply_parts.append("Doctor not found. Try searching first with location or specialty.")

#     # 4) user provided a date -> show time slots automatically (we pick next slots using generate_time_slots)
#     if 'date' in intent and st.session_state.ai_selected_doctor:
#         date = intent['date']
#         doc_id = st.session_state.ai_selected_doctor
#         slots = generate_time_slots(doc_id, date)
#         available = [s for s in slots if s['available']]
#         if available:
#             lines = [f"Available slots on {date.isoformat()}:"]
#             for s in available[:20]:
#                 lines.append(f"- {s['display']}")
#             lines.append("Please pick a time (e.g., '10:30' or '10:30 AM').")
#             reply_parts.append("\n".join(lines))
#         else:
#             reply_parts.append("No slots available on that date. Try another date.")

#     # 5) user provided a time slot (slot) AND date previously chosen -> attempt booking
#     if 'slot' in intent and 'date' in intent and st.session_state.ai_selected_doctor:
#         # build datetime
#         try:
#             dt_time = datetime.strptime(intent['slot'], "%H:%M").time()
#         except:
#             try:
#                 dt_time = datetime.strptime(intent['slot'], "%I:%M%p").time()
#             except:
#                 dt_time = None
#         if dt_time is not None:
#             dt_combined = datetime.combine(intent['date'], dt_time)
#             ok, result = book_appointment(st.session_state.ai_selected_doctor, intent['date'], dt_combined)
#             if ok:
#                 reply_parts.append("🎉 Appointment confirmed!")
#                 reply_parts.append(f"Booking ID: {result['booking_id']}")
#                 reply_parts.append(f"Doctor: {result['doctor_name']}")
#                 reply_parts.append(f"Date: {result['date'].isoformat()}")
#                 reply_parts.append(f"Time: {result['slot_start'].strftime('%I:%M %p')}")
#                 reply_parts.append(f"Queue position: {result['queue_position']}")
#                 # clear selected doctor so user can make another booking if needed
#                 st.session_state.ai_selected_doctor = None
#             else:
#                 reply_parts.append(f"⚠️ {result}")
#         else:
#             reply_parts.append("Couldn't parse that time — send like '10:30' or '10:30 AM'.")

#     # fallback if nothing handled
#     if not reply_parts:
#         reply_parts.append("I didn't get enough detail. Please tell me location / specialty to start, or pick a doctor number from the list, or give a date/time to book.")
#     return "\n\n".join(reply_parts)

# ----------------------------
# Groq wrappers (medical Q&A + small booking assistant prompts if you want)
# ----------------------------
def medical_query_groq(api_key, user_query, docs):
    if not api_key:
        return "Groq key not provided. Set it in the sidebar."
    try:
        client = Groq(api_key=api_key)
        context = "\n\n".join([doc.page_content for doc in docs]) if docs else ""
        system_prompt = """You are a medical assistant. Use ONLY the provided documents. If info missing reply: "Not in documents. Please consult cancer.gov or your doctor."""
        full_prompt = f"CONTEXT:\n{context}\n\nQUESTION: {user_query}\n\nANSWER:"
        response = client.chat.completions.create(
            model="groq/compound",
            messages=[{"role":"system","content":system_prompt},{"role":"user","content":full_prompt}],
            temperature=0.2,
            max_tokens=800
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error calling Groq: {e}"

def appointment_agent_response(api_key, conversation_history):
    """Optional: produce friendly text using LLM, but python logic handles slot selection and booking"""
    if not api_key:
        # fallback to simple local prompt
        return "I'm your booking assistant. Please provide location/specialty to start."
    try:
        client = Groq(api_key=api_key)
        system_prompt = "You are a friendly booking assistant that asks for location, specialty, doctor choice, date and time. Keep messages short."
        response = client.chat.completions.create(
            model="groq/compound",
            messages=[{"role":"system","content":system_prompt}, *conversation_history],
            temperature=0.5,
            max_tokens=200
        )
        return response.choices[0].message.content
    except Exception as e:
        return "LLM error (booking assistant)."

# ----------------------------
# STREAMLIT TABS
# ----------------------------
def medical_qa_tab(api_key):
    st.markdown("### 🧠 Medical Q&A")
    if not api_key:
        st.warning("Enter Groq API key to enable medical RAG answers (sidebar).")
    if 'medical_chat' not in st.session_state:
        st.session_state.medical_chat = [{"role":"assistant","content":"Ask your medical question (based on uploaded documents)."}]
    for m in st.session_state.medical_chat:
        with st.chat_message(m['role']):
            st.write(m['content'])
    q = st.chat_input("Type a medical question...")
    if not q:
        return
    st.session_state.medical_chat.append({"role":"user","content":q})
    with st.chat_message("assistant"):
        with st.spinner("Searching docs and composing answer..."):
            vector_db = get_vectorstore()
            if vector_db:
                docs = vector_db.similarity_search(q, k=3)
            else:
                docs = []
            ans = medical_query_groq(st.session_state.get('groq_api_key'), q, docs)
            st.write(ans)
            st.session_state.medical_chat.append({"role":"assistant","content":ans})
def ai_booking_tab(api_key):
    st.markdown("### 🤖 AI Booking Agent")
    st.markdown("Chat naturally — I'll guide you. Start by saying where you are (e.g., 'Noida Sector 62') or what specialty you need.")

    # ensure chat exists
    if 'ai_booking_chat' not in st.session_state:
        st.session_state.ai_booking_chat = [{"role":"assistant","content":"Hello! 👋 I can help you book an appointment. Where are you located?"}]

    for msg in st.session_state.ai_booking_chat:
        with st.chat_message(msg['role']):
            st.write(msg['content'])

    user_msg = st.chat_input("Say something…")
    if not user_msg:
        return

    # record user message
    st.session_state.ai_booking_chat.append({"role":"user","content":user_msg})
    with st.chat_message("user"):
        st.write(user_msg)

    # parse + python actions
    intent = parse_booking_intent(user_msg)
    python_reply = process_booking_logic(intent)

    # optional: LLM for friendlier tone (kept small)
    llm_reply = appointment_agent_response(api_key, st.session_state.ai_booking_chat) if api_key else ""

    # prefer python reply first (actionable) then a short LLM follow-up (if any)
    final = python_reply
    if llm_reply:
        # keep LLM reply short so python actions are visible first
        final += "\n\n" + llm_reply.strip()

    # save assistant reply and show it
    st.session_state.ai_booking_chat.append({"role":"assistant","content":final})
    with st.chat_message("assistant"):
        st.write(final)

# def ai_booking_tab(api_key):
#     st.markdown("### 🤖 AI Booking Agent")
#     st.markdown("Chat naturally — I'll guide you. Start by saying where you are (e.g., 'Noida Sector 62') or what specialty you need.")
#     for msg in st.session_state.ai_booking_chat:
#         with st.chat_message(msg['role']):
#             st.write(msg['content'])
#     user_msg = st.chat_input("Say something…")
#     if not user_msg:
#         return
#     st.session_state.ai_booking_chat.append({"role":"user","content":user_msg})
#     with st.chat_message("user"):
#         st.write(user_msg)

#     # parse + python actions
#     intent = parse_booking_intent(user_msg)
#     python_reply = process_booking_logic(intent)

#     # allow LLM to add conversational tone (optional)
#     llm_reply = appointment_agent_response(api_key, st.session_state.ai_booking_chat)

#     # prefer python reply first (actionable) then a short LLM follow-up (if any)
#     final = python_reply
#     if llm_reply:
#         final += "\n\n" + llm_reply

#     st.session_state.ai_booking_chat.append({"role":"assistant","content":final})
#     with st.chat_message("assistant"):
#         st.write(final)

def find_doctors_tab():
    st.markdown("### 🔍 Find Doctors")
    c1,c2,c3 = st.columns(3)
    with c1:
        loc = st.text_input("Location", placeholder="Noida, Indirapuram, Sector 62")
    with c2:
        spec = st.text_input("Specialty", placeholder="Kidney Cancer, Oncology")
    with c3:
        dis = st.text_input("Disease", placeholder="Leukemia, RCC")
    if st.button("Search"):
        res = search_doctors(location=loc, specialty=spec, disease=dis)
        if res is None or len(res)==0:
            st.warning("No doctors found.")
            return
        for idx, (_, d) in enumerate(res.reset_index(drop=True).iterrows(), start=1):
            with st.expander(f"{idx}. {d['doctor_name_en']} — {d['rating']}/5"):
                st.write(f"**Specialty:** {d['specialty']}")
                st.write(f"**Hospital:** {d['hospital_name']}")
                st.write(f"**Location:** {d['location_area']}")
                st.write(f"**Languages:** {d['languages']}")
                st.write(f"**Fee:** ₹{d['fee']}")
                if st.button(f"Select {d['doctor_id']}", key=f"sel_{d['doctor_id']}"):
                    st.session_state.selected_doctor = d['doctor_id']
                    st.success("Selected — go to Book Appointment tab.")

def manual_booking_tab():
    st.markdown("### 📅 Book Appointment (Manual)")
    if not st.session_state.selected_doctor:
        st.info("Select a doctor from 'Find Doctors' tab or use the AI Booking Agent.")
        return
    doctor = DOCTORS_DATABASE[DOCTORS_DATABASE['doctor_id']==st.session_state.selected_doctor].iloc[0]
    st.write(f"**Doctor:** {doctor['doctor_name_en']}")
    date = st.date_input("Date", min_value=datetime.today().date())
    slots = generate_time_slots(st.session_state.selected_doctor, date)
    available = [s for s in slots if s['available']]
    if not available:
        st.warning("No slots available. Try another date.")
        return
    slot_display = [s['display'] for s in available]
    chosen = st.selectbox("Choose slot", slot_display)
    name = st.text_input("Your name")
    phone = st.text_input("Phone number")
    if st.button("Confirm booking"):
        if not name or not phone:
            st.warning("Enter name and phone")
            return
        slot_obj = next(s for s in available if s['display']==chosen)
        ok, booking = book_appointment(st.session_state.selected_doctor, date, slot_obj['start'], user_name=name, user_contact=phone)
        if ok:
            st.success(f"Confirmed — ID {booking['booking_id']}")
            st.balloons()
            # clear selection
            st.session_state.selected_doctor = None

def appointments_tab():
    st.markdown("### 📋 My Appointments")

    bookings = st.session_state.get('bookings', [])
    if not bookings:
        st.info("No appointments yet.")
        return

    # sort by date & time
    try:
        bookings_sorted = sorted(bookings, key=lambda x: (x['date'], x['slot_start']))
    except Exception:
        bookings_sorted = bookings

    for b in bookings_sorted:
        # format date/time safely
        date_display = b['date'].isoformat() if hasattr(b['date'], 'isoformat') else str(b['date'])
        start_display = b['slot_start'].strftime('%I:%M %p') if hasattr(b['slot_start'], 'strftime') else str(b.get('slot_start'))
        end_display = b['slot_end'].strftime('%I:%M %p') if hasattr(b.get('slot_end'), 'strftime') else str(b.get('slot_end'))

        with st.expander(f"{b['booking_id']} — {b.get('doctor_name','Unknown')} ({date_display})", expanded=False):
            st.write(f"**Date:** {date_display}")
            st.write(f"**Time:** {start_display} - {end_display}")
            st.write(f"**Doctor:** {b.get('doctor_name')}")
            st.write(f"**Hospital:** {b.get('hospital_name')}")
            st.write(f"**Location:** {b.get('location')}")
            st.write(f"**Queue position:** {b.get('queue_position')}")
            st.write(f"**Phone:** {b.get('user_contact')}")
            if st.button(f"Cancel {b['booking_id']}", key=f"cancel_{b['booking_id']}"):
                st.session_state.bookings.remove(b)
                st.success("Booking cancelled")
                try:
                    st.experimental_rerun()
                except Exception:
                    pass


# ----------------------------
# MAIN
# ----------------------------
def main():
    st.set_page_config(layout="wide", page_title="Cancer Care Assistant")
    init_session_state()
    st.title("💙 Comprehensive Cancer Care Assistant")

    with st.sidebar:
        st.header("🔑 Groq API Key")
        key = st.text_input("Enter key (optional for bookings)", type="password")
        if key:
            st.session_state.groq_api_key = key
            st.success("API key saved (session).")
        st.markdown("---")
        st.header("Language")
        st.radio("Select", ["English","हिंदी"], index=0, key="user_language")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🧠 Medical Q&A","🤖 AI Booking Agent","🔍 Find Doctors","📅 Book Appointment","📋 My Appointments"])
    with tab1:
        medical_qa_tab(st.session_state.get('groq_api_key'))
    with tab2:
        ai_booking_tab(st.session_state.get('groq_api_key'))
    with tab3:
        find_doctors_tab()
    with tab4:
        manual_booking_tab()
    with tab5:
        appointments_tab()

if __name__ == "__main__":
    main()
