import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
import json
import re

# Configuration
DB_FAISS_PATH = "vectorstore/db_faiss"

# Hospital data
HOSPITALS_DATA = {
    "cancer_centers": [
        {
            "id": "hospital_1",
            "name": "Memorial Cancer Center",
            "address": "123 Medical Plaza, New York, NY 10001",
            "phone": "+1 (212) 555-0100",
            "email": "info@memorialcancer.org",
            "distance": "2.5 miles",
            "rating": 4.8,
            "specialties": ["Kidney Cancer", "Stomach Cancer", "Breast Cancer", "Lung Cancer"],
            "doctors": [
                {
                    "id": "doc_1",
                    "name": "Dr. Sarah Johnson",
                    "specialty": "Oncology - Kidney Cancer",
                    "experience": "15 years",
                    "phone": "+1 (212) 555-0101",
                    "email": "s.johnson@memorialcancer.org",
                    "available_days": ["Monday", "Wednesday", "Friday"],
                    "next_available": "Monday, Nov 18, 2025 at 10:00 AM"
                },
                {
                    "id": "doc_2",
                    "name": "Dr. Michael Chen",
                    "specialty": "Surgical Oncology",
                    "experience": "12 years",
                    "phone": "+1 (212) 555-0102",
                    "email": "m.chen@memorialcancer.org",
                    "available_days": ["Tuesday", "Thursday"],
                    "next_available": "Tuesday, Nov 19, 2025 at 2:00 PM"
                }
            ]
        },
        {
            "id": "hospital_2",
            "name": "City Oncology Hospital",
            "address": "456 Health Avenue, New York, NY 10002",
            "phone": "+1 (212) 555-0200",
            "email": "contact@cityoncology.org",
            "distance": "3.8 miles",
            "rating": 4.6,
            "specialties": ["Stomach Cancer", "Colon Cancer", "Liver Cancer"],
            "doctors": [
                {
                    "id": "doc_3",
                    "name": "Dr. Emily Rodriguez",
                    "specialty": "Gastric Oncology",
                    "experience": "10 years",
                    "phone": "+1 (212) 555-0201",
                    "email": "e.rodriguez@cityoncology.org",
                    "available_days": ["Monday", "Tuesday", "Thursday"],
                    "next_available": "Thursday, Nov 21, 2025 at 9:00 AM"
                }
            ]
        },
        {
            "id": "hospital_3",
            "name": "Advanced Cancer Treatment Center",
            "address": "789 Care Boulevard, New York, NY 10003",
            "phone": "+1 (212) 555-0300",
            "email": "info@advancedcancer.org",
            "distance": "5.2 miles",
            "rating": 4.9,
            "specialties": ["All Cancer Types", "Clinical Trials"],
            "doctors": [
                {
                    "id": "doc_4",
                    "name": "Dr. David Kumar",
                    "specialty": "Medical Oncology",
                    "experience": "20 years",
                    "phone": "+1 (212) 555-0301",
                    "email": "d.kumar@advancedcancer.org",
                    "available_days": ["Monday", "Wednesday", "Friday"],
                    "next_available": "Wednesday, Nov 20, 2025 at 11:00 AM"
                }
            ]
        }
    ]
}

# Simulated appointments
if 'appointments' not in st.session_state:
    st.session_state.appointments = []

@st.cache_resource
def get_vectorstore():
    """Load the FAISS vector store"""
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2'
        )
        db = FAISS.load_local(
            DB_FAISS_PATH, 
            embedding_model, 
            allow_dangerous_deserialization=True
        )
        return db
    except Exception as e:
        return None

def search_medical_docs(query):
    """Search medical documents"""
    try:
        vectorstore = get_vectorstore()
        if vectorstore:
            docs = vectorstore.similarity_search(query, k=2)
            if docs:
                result = ""
                for doc in docs:
                    result += doc.page_content[:400] + "\n\n"
                return result
        return None
    except:
        return None

def find_doctors_for_specialty(specialty):
    """Find doctors by specialty"""
    specialty_lower = specialty.lower()
    doctors = []
    
    for hospital in HOSPITALS_DATA["cancer_centers"]:
        for doctor in hospital["doctors"]:
            if specialty_lower in doctor["specialty"].lower() or specialty_lower in str(hospital["specialties"]).lower():
                doctors.append({
                    "doctor": doctor,
                    "hospital": hospital
                })
    
    return doctors

def analyze_user_intent(user_input):
    """Analyze what the user wants"""
    user_lower = user_input.lower()
    
    intent = {
        "needs_medical_info": False,
        "needs_doctor": False,
        "needs_appointment": False,
        "needs_hospital": False,
        "cancer_type": None,
        "has_symptoms": False
    }
    
    # Check for medical info needs
    symptoms = ["symptom", "pain", "blood", "fever", "weight loss", "fatigue", "what is", "what are"]
    if any(word in user_lower for word in symptoms):
        intent["needs_medical_info"] = True
        intent["has_symptoms"] = True
    
    # Check for doctor/hospital needs
    doctor_keywords = ["doctor", "specialist", "physician", "oncologist"]
    hospital_keywords = ["hospital", "clinic", "center", "treatment"]
    
    if any(word in user_lower for word in doctor_keywords):
        intent["needs_doctor"] = True
    
    if any(word in user_lower for word in hospital_keywords):
        intent["needs_hospital"] = True
    
    # Check for appointment needs
    appointment_keywords = ["appointment", "book", "schedule", "visit", "see a doctor"]
    if any(word in user_lower for word in appointment_keywords):
        intent["needs_appointment"] = True
        intent["needs_doctor"] = True
    
    # Detect cancer type
    cancer_types = ["kidney", "stomach", "breast", "lung", "colon", "liver", "gastric"]
    for cancer in cancer_types:
        if cancer in user_lower:
            intent["cancer_type"] = cancer.capitalize() + " Cancer"
            break
    
    return intent

def create_intelligent_response(user_input):
    """Create an intelligent multi-step response"""
    intent = analyze_user_intent(user_input)
    response_parts = []
    
    # Step 1: Acknowledge and empathize
    response_parts.append("I understand you need help. Let me assist you with that. 💙\n")
    
    # Step 2: Search medical docs if needed
    if intent["needs_medical_info"] or intent["has_symptoms"]:
        response_parts.append("🔍 **Searching medical documents...**\n")
        doc_info = search_medical_docs(user_input)
        if doc_info:
            response_parts.append(f"**What the medical documents say:**\n{doc_info[:500]}...\n")
        response_parts.append("\n---\n")
    
    # Step 3: Find doctors if needed
    if intent["needs_doctor"] and intent["cancer_type"]:
        response_parts.append(f"👨‍⚕️ **Finding {intent['cancer_type']} specialists...**\n")
        doctors = find_doctors_for_specialty(intent["cancer_type"])
        
        if doctors:
            response_parts.append(f"I found {len(doctors)} specialist(s) for you:\n\n")
            for item in doctors[:2]:  # Show top 2
                doc = item["doctor"]
                hosp = item["hospital"]
                response_parts.append(f"**{doc['name']}**\n")
                response_parts.append(f"- Specialty: {doc['specialty']}\n")
                response_parts.append(f"- Experience: {doc['experience']}\n")
                response_parts.append(f"- Hospital: {hosp['name']}\n")
                response_parts.append(f"- Phone: {doc['phone']}\n")
                response_parts.append(f"- Next Available: {doc['next_available']}\n\n")
        response_parts.append("\n---\n")
    
    # Step 4: Offer appointment booking
    if intent["needs_appointment"]:
        response_parts.append("📅 **Ready to book an appointment?**\n")
        response_parts.append("To book an appointment, please provide:\n")
        response_parts.append("1. Your full name\n")
        response_parts.append("2. Contact email or phone\n")
        response_parts.append("3. Preferred doctor (from the list above)\n\n")
        response_parts.append("Type: `Book appointment with [Doctor Name]` and I'll take care of it!\n")
    
    # Step 5: Provide guidance
    if intent["has_symptoms"]:
        response_parts.append("\n💡 **Important:** If you're experiencing concerning symptoms, I recommend:\n")
        response_parts.append("1. Consulting with a healthcare provider as soon as possible\n")
        response_parts.append("2. Calling one of the specialists listed above\n")
        response_parts.append("3. For emergencies, call 911 immediately\n")
    
    return "".join(response_parts)

def handle_appointment_booking(user_input, doctor_name):
    """Handle appointment booking"""
    # Extract patient info
    result = f"✅ **Appointment Booking Request**\n\n"
    
    # Find the doctor
    doctor_found = None
    hospital_found = None
    
    for hospital in HOSPITALS_DATA["cancer_centers"]:
        for doctor in hospital["doctors"]:
            if doctor_name.lower() in doctor["name"].lower():
                doctor_found = doctor
                hospital_found = hospital
                break
        if doctor_found:
            break
    
    if doctor_found:
        # Create appointment
        apt_id = f"APT{len(st.session_state.appointments) + 1001}"
        appointment = {
            "id": apt_id,
            "doctor": doctor_found["name"],
            "specialty": doctor_found["specialty"],
            "hospital": hospital_found["name"],
            "date_time": doctor_found["next_available"],
            "status": "Confirmed"
        }
        
        st.session_state.appointments.append(appointment)
        
        result += f"**Appointment Successfully Booked!** 🎉\n\n"
        result += f"📋 Appointment ID: **{apt_id}**\n"
        result += f"👨‍⚕️ Doctor: **{doctor_found['name']}**\n"
        result += f"🏥 Hospital: **{hospital_found['name']}**\n"
        result += f"📍 Address: {hospital_found['address']}\n"
        result += f"📅 Date & Time: **{doctor_found['next_available']}**\n"
        result += f"📞 Contact: {doctor_found['phone']}\n\n"
        result += f"✉️ Confirmation will be sent to your email.\n"
        result += f"⚠️ Please arrive 15 minutes early.\n"
    else:
        result += f"❌ Could not find doctor: {doctor_name}\n"
        result += "Please check the spelling or choose from the list above."
    
    return result

def intelligent_chat_interface():
    """Intelligent chat interface with pseudo-agentic behavior"""
    st.markdown("### 🤖 Intelligent Medical Assistant")
    st.markdown("*Smart assistant that autonomously searches, finds doctors, and books appointments*")
    
    # Initialize chat
    if 'smart_messages' not in st.session_state:
        st.session_state.smart_messages = []
        st.session_state.smart_messages.append({
            'role': 'assistant',
            'content': '''👋 **Hello! I'm your Intelligent Medical Assistant.**

I can help you with:
- 🔍 **Medical information** from your documents
- 🏥 **Finding hospitals** and specialists
- 👨‍⚕️ **Doctor recommendations** based on your needs
- 📅 **Booking appointments** automatically

**Example requests:**
- "I have blood in my urine, what should I do?"
- "Find me a kidney cancer specialist"
- "Book appointment with Dr. Johnson"
- "I need help with stomach pain"

What can I help you with today? 💙'''
        })
    
    # Display messages
    for message in st.session_state.smart_messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
    
    # Chat input
    if prompt := st.chat_input("Tell me what you need... 🤖"):
        # Add user message
        st.session_state.smart_messages.append({'role': 'user', 'content': prompt})
        with st.chat_message('user'):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message('assistant'):
            with st.spinner('🤖 Analyzing and processing your request...'):
                try:
                    # Check if booking appointment
                    if "book appointment" in prompt.lower():
                        # Extract doctor name
                        match = re.search(r'with (.+?)(?:\.|$)', prompt, re.IGNORECASE)
                        if match:
                            doctor_name = match.group(1).strip()
                            response_text = handle_appointment_booking(prompt, doctor_name)
                        else:
                            response_text = "Please specify which doctor you'd like to book with. Example: 'Book appointment with Dr. Johnson'"
                    else:
                        # Generate intelligent response
                        response_text = create_intelligent_response(prompt)
                    
                    st.markdown(response_text)
                    st.session_state.smart_messages.append({
                        'role': 'assistant',
                        'content': response_text
                    })
                
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}\n\nMake sure Ollama is running: `ollama serve`"
                    st.error(error_msg)

def appointments_tab():
    """View appointments"""
    st.markdown("### 📅 My Appointments")
    
    if not st.session_state.appointments:
        st.info("📋 No appointments yet. Use the chat to book appointments!")
    else:
        st.success(f"✅ You have {len(st.session_state.appointments)} appointment(s)")
        
        for apt in st.session_state.appointments:
            with st.expander(f"🆔 {apt['id']} - {apt['doctor']}", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"👨‍⚕️ **Doctor:** {apt['doctor']}")
                    st.write(f"🩺 **Specialty:** {apt['specialty']}")
                    st.write(f"🏥 **Hospital:** {apt['hospital']}")
                
                with col2:
                    st.write(f"📅 **Date & Time:** {apt['date_time']}")
                    st.write(f"✅ **Status:** {apt['status']}")
                
                st.success("Appointment confirmed! Check your email for details.")

def main():
    st.set_page_config(
        page_title="Intelligent Medical AI",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Intelligent Medical Assistant")
    st.markdown("**Smart AI that autonomously helps with medical information, doctors, and appointments** 💙")
    
    # Sidebar
    with st.sidebar:
        st.header("🤖 System Info")
        st.success("**Status:** Online ✅")
        
        st.markdown("---")
        
        st.header("🧠 Capabilities")
        st.markdown("""
        **What I can do:**
        - 🔍 Search medical documents
        - 🏥 Find hospitals & specialists
        - 👨‍⚕️ Recommend doctors
        - 📅 Book appointments
        - 💬 Provide medical guidance
        
        **How I work:**
        1. Analyze your request
        2. Search relevant information
        3. Find appropriate doctors
        4. Provide complete guidance
        5. Book appointments if needed
        """)
        
        st.markdown("---")
        
        st.header("💡 Quick Actions")
        if st.button("🏥 View All Doctors"):
            st.session_state.show_all_doctors = True
        
        if st.button("📍 View All Hospitals"):
            st.session_state.show_all_hospitals = True
        
        st.markdown("---")
        
        st.header("📊 Stats")
        st.metric("Appointments Booked", len(st.session_state.appointments))
        st.metric("Available Doctors", "7")
        st.metric("Partner Hospitals", "3")
    
    # Main tabs
    tab1, tab2 = st.tabs(["🤖 Smart Chat", "📅 My Appointments"])
    
    with tab1:
        intelligent_chat_interface()
    
    with tab2:
        appointments_tab()

if __name__ == "__main__":
    if not os.path.exists(DB_FAISS_PATH):
        st.warning("⚠️ Medical documents not loaded. Some features may be limited.")
    main()
# import os
# import streamlit as st
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA
# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import PromptTemplate
# from langchain_community.llms import Ollama
# from langchain.agents import AgentExecutor, create_react_agent, Tool
# from langchain.memory import ConversationBufferMemory
# from langchain_core.prompts import PromptTemplate as AgentPromptTemplate
# import json
# from datetime import datetime, timedelta
# import re

# # Configuration
# DB_FAISS_PATH = "vectorstore/db_faiss"

# # Hospital data
# HOSPITALS_DATA = {
#     "cancer_centers": [
#         {
#             "id": "hospital_1",
#             "name": "Memorial Cancer Center",
#             "address": "123 Medical Plaza, New York, NY 10001",
#             "phone": "+1 (212) 555-0100",
#             "email": "info@memorialcancer.org",
#             "website": "www.memorialcancer.org",
#             "distance": "2.5 miles",
#             "rating": 4.8,
#             "specialties": ["Kidney Cancer", "Stomach Cancer", "Breast Cancer", "Lung Cancer"],
#             "doctors": [
#                 {
#                     "id": "doc_1",
#                     "name": "Dr. Sarah Johnson",
#                     "specialty": "Oncology - Kidney Cancer",
#                     "experience": "15 years",
#                     "phone": "+1 (212) 555-0101",
#                     "email": "s.johnson@memorialcancer.org",
#                     "available_days": ["Monday", "Wednesday", "Friday"],
#                     "timings": "9:00 AM - 5:00 PM",
#                     "next_available": "Monday, Nov 18, 2025 at 10:00 AM"
#                 },
#                 {
#                     "id": "doc_2",
#                     "name": "Dr. Michael Chen",
#                     "specialty": "Surgical Oncology",
#                     "experience": "12 years",
#                     "phone": "+1 (212) 555-0102",
#                     "email": "m.chen@memorialcancer.org",
#                     "available_days": ["Tuesday", "Thursday"],
#                     "timings": "10:00 AM - 4:00 PM",
#                     "next_available": "Tuesday, Nov 19, 2025 at 2:00 PM"
#                 }
#             ]
#         },
#         {
#             "id": "hospital_2",
#             "name": "City Oncology Hospital",
#             "address": "456 Health Avenue, New York, NY 10002",
#             "phone": "+1 (212) 555-0200",
#             "email": "contact@cityoncology.org",
#             "website": "www.cityoncology.org",
#             "distance": "3.8 miles",
#             "rating": 4.6,
#             "specialties": ["Stomach Cancer", "Colon Cancer", "Liver Cancer"],
#             "doctors": [
#                 {
#                     "id": "doc_3",
#                     "name": "Dr. Emily Rodriguez",
#                     "specialty": "Gastric Oncology",
#                     "experience": "10 years",
#                     "phone": "+1 (212) 555-0201",
#                     "email": "e.rodriguez@cityoncology.org",
#                     "available_days": ["Monday", "Tuesday", "Thursday"],
#                     "timings": "8:00 AM - 3:00 PM",
#                     "next_available": "Thursday, Nov 21, 2025 at 9:00 AM"
#                 }
#             ]
#         }
#     ]
# }

# # Simulated appointment storage
# if 'appointments' not in st.session_state:
#     st.session_state.appointments = []

# @st.cache_resource
# def get_vectorstore():
#     """Load the FAISS vector store with embeddings"""
#     try:
#         embedding_model = HuggingFaceEmbeddings(
#             model_name='sentence-transformers/all-MiniLM-L6-v2'
#         )
#         db = FAISS.load_local(
#             DB_FAISS_PATH, 
#             embedding_model, 
#             allow_dangerous_deserialization=True
#         )
#         return db
#     except Exception as e:
#         return None

# # ==================== AGENT TOOLS ====================

# def search_medical_documents(query: str) -> str:
#     """
#     Search through medical PDF documents to find relevant information.
#     Use this when patient asks about medical conditions, symptoms, or treatments.
#     """
#     try:
#         vectorstore = get_vectorstore()
#         if vectorstore is None:
#             return "Error: Medical documents not available."
        
#         docs = vectorstore.similarity_search(query, k=2)
        
#         if not docs:
#             return "No relevant information found in medical documents."
        
#         result = "Found relevant information:\n\n"
#         for i, doc in enumerate(docs, 1):
#             source = doc.metadata.get('source', 'Unknown')
#             page = doc.metadata.get('page', 'Unknown')
#             content = doc.page_content[:300]
#             result += f"Source {i}: {os.path.basename(source)} (Page {page})\n{content}...\n\n"
        
#         return result
#     except Exception as e:
#         return f"Error searching documents: {str(e)}"

# def find_hospitals_by_specialty(specialty: str) -> str:
#     """
#     Find cancer hospitals that specialize in a specific cancer type.
#     Use this when patient needs hospital recommendations.
#     Specialty examples: "Kidney Cancer", "Stomach Cancer", "Breast Cancer"
#     """
#     specialty = specialty.strip().title()
#     hospitals = []
    
#     for hospital in HOSPITALS_DATA["cancer_centers"]:
#         if specialty in hospital["specialties"] or "All Cancer Types" in hospital["specialties"]:
#             hospitals.append(hospital)
    
#     if not hospitals:
#         return f"No hospitals found specializing in {specialty}."
    
#     result = f"Found {len(hospitals)} hospitals specializing in {specialty}:\n\n"
#     for hospital in hospitals:
#         result += f"🏥 {hospital['name']}\n"
#         result += f"   Address: {hospital['address']}\n"
#         result += f"   Phone: {hospital['phone']}\n"
#         result += f"   Rating: {hospital['rating']}/5.0\n"
#         result += f"   Distance: {hospital['distance']}\n\n"
    
#     return result

# def find_doctors_by_specialty(specialty: str) -> str:
#     """
#     Find doctors who specialize in treating specific types of cancer.
#     Use this when patient needs to find a specialist doctor.
#     Specialty examples: "Kidney Cancer", "Surgical Oncology", "Gastric Oncology"
#     """
#     specialty_lower = specialty.lower()
#     doctors_found = []
    
#     for hospital in HOSPITALS_DATA["cancer_centers"]:
#         for doctor in hospital["doctors"]:
#             if specialty_lower in doctor["specialty"].lower():
#                 doctors_found.append({
#                     "doctor": doctor,
#                     "hospital": hospital
#                 })
    
#     if not doctors_found:
#         return f"No doctors found specializing in {specialty}."
    
#     result = f"Found {len(doctors_found)} specialist doctors:\n\n"
#     for item in doctors_found:
#         doc = item["doctor"]
#         hosp = item["hospital"]
#         result += f"👨‍⚕️ {doc['name']}\n"
#         result += f"   Specialty: {doc['specialty']}\n"
#         result += f"   Experience: {doc['experience']}\n"
#         result += f"   Hospital: {hosp['name']}\n"
#         result += f"   Phone: {doc['phone']}\n"
#         result += f"   Email: {doc['email']}\n"
#         result += f"   Next Available: {doc['next_available']}\n\n"
    
#     return result

# def book_appointment(doctor_name: str, patient_name: str, contact: str) -> str:
#     """
#     Book an appointment with a specific doctor.
#     Use this when patient wants to schedule an appointment.
#     Args:
#         doctor_name: Full name of the doctor (e.g., "Dr. Sarah Johnson")
#         patient_name: Patient's full name
#         contact: Patient's phone or email
#     """
#     # Find the doctor
#     doctor_found = None
#     hospital_found = None
    
#     for hospital in HOSPITALS_DATA["cancer_centers"]:
#         for doctor in hospital["doctors"]:
#             if doctor_name.lower() in doctor["name"].lower():
#                 doctor_found = doctor
#                 hospital_found = hospital
#                 break
#         if doctor_found:
#             break
    
#     if not doctor_found:
#         return f"Error: Doctor '{doctor_name}' not found. Please check the name and try again."
    
#     # Create appointment
#     appointment = {
#         "id": f"APT{len(st.session_state.appointments) + 1001}",
#         "patient_name": patient_name,
#         "doctor": doctor_found["name"],
#         "specialty": doctor_found["specialty"],
#         "hospital": hospital_found["name"],
#         "date_time": doctor_found["next_available"],
#         "contact": contact,
#         "status": "Confirmed",
#         "confirmation_sent": True
#     }
    
#     st.session_state.appointments.append(appointment)
    
#     result = f"✅ Appointment Successfully Booked!\n\n"
#     result += f"Appointment ID: {appointment['id']}\n"
#     result += f"Patient: {patient_name}\n"
#     result += f"Doctor: {doctor_found['name']}\n"
#     result += f"Specialty: {doctor_found['specialty']}\n"
#     result += f"Hospital: {hospital_found['name']}\n"
#     result += f"Date & Time: {doctor_found['next_available']}\n"
#     result += f"Location: {hospital_found['address']}\n\n"
#     result += f"📧 Confirmation email sent to: {contact}\n"
#     result += f"📞 You can call {doctor_found['phone']} for any questions.\n\n"
#     result += f"⚠️ Please arrive 15 minutes early for registration."
    
#     return result

# def get_doctor_contact_info(doctor_name: str) -> str:
#     """
#     Get contact information (phone, email) for a specific doctor.
#     Use this when patient asks for a doctor's contact details.
#     """
#     for hospital in HOSPITALS_DATA["cancer_centers"]:
#         for doctor in hospital["doctors"]:
#             if doctor_name.lower() in doctor["name"].lower():
#                 result = f"📇 Contact Information for {doctor['name']}:\n\n"
#                 result += f"Phone: {doctor['phone']}\n"
#                 result += f"Email: {doctor['email']}\n"
#                 result += f"Hospital: {hospital['name']}\n"
#                 result += f"Hospital Phone: {hospital['phone']}\n"
#                 result += f"Available: {', '.join(doctor['available_days'])}\n"
#                 result += f"Timings: {doctor['timings']}\n"
#                 return result
    
#     return f"Doctor '{doctor_name}' not found in our database."

# def send_email_to_doctor(doctor_name: str, message: str, patient_name: str) -> str:
#     """
#     Send an email to a doctor on behalf of the patient.
#     Use this when patient wants to contact a doctor via email.
#     """
#     for hospital in HOSPITALS_DATA["cancer_centers"]:
#         for doctor in hospital["doctors"]:
#             if doctor_name.lower() in doctor["name"].lower():
#                 result = f"📧 Email sent successfully!\n\n"
#                 result += f"To: {doctor['name']} ({doctor['email']})\n"
#                 result += f"From: {patient_name}\n"
#                 result += f"Subject: Patient Inquiry\n\n"
#                 result += f"Message:\n{message}\n\n"
#                 result += f"You should receive a response within 24-48 hours."
#                 return result
    
#     return f"Error: Could not find email address for Dr. {doctor_name}"

# def check_appointment_status(appointment_id: str) -> str:
#     """
#     Check the status of a booked appointment.
#     Use this when patient wants to verify their appointment.
#     """
#     for apt in st.session_state.appointments:
#         if apt["id"].lower() == appointment_id.lower():
#             result = f"📋 Appointment Status:\n\n"
#             result += f"ID: {apt['id']}\n"
#             result += f"Status: {apt['status']}\n"
#             result += f"Patient: {apt['patient_name']}\n"
#             result += f"Doctor: {apt['doctor']}\n"
#             result += f"Date & Time: {apt['date_time']}\n"
#             result += f"Hospital: {apt['hospital']}\n"
#             return result
    
#     return f"No appointment found with ID: {appointment_id}"

# # ==================== AGENT SETUP ====================

# def create_agentic_system():
#     """Create the agentic AI system with tools"""
    
#     # Define tools
#     tools = [
#         Tool(
#             name="SearchMedicalDocuments",
#             func=search_medical_documents,
#             description="Search through medical PDF documents to find information about symptoms, conditions, or treatments. Input should be a medical question or symptom description."
#         ),
#         Tool(
#             name="FindHospitals",
#             func=find_hospitals_by_specialty,
#             description="Find cancer hospitals by specialty. Input should be the cancer type like 'Kidney Cancer', 'Stomach Cancer', etc."
#         ),
#         Tool(
#             name="FindDoctors",
#             func=find_doctors_by_specialty,
#             description="Find specialist doctors by their area of expertise. Input should be the specialty like 'Kidney Cancer', 'Surgical Oncology', etc."
#         ),
#         Tool(
#             name="BookAppointment",
#             func=lambda x: book_appointment(*x.split("|")),
#             description="Book an appointment with a doctor. Input should be in format: 'Doctor Name|Patient Name|Contact'. Example: 'Dr. Sarah Johnson|John Doe|john@email.com'"
#         ),
#         Tool(
#             name="GetDoctorContact",
#             func=get_doctor_contact_info,
#             description="Get contact information for a specific doctor. Input should be the doctor's name."
#         ),
#         Tool(
#             name="SendEmailToDoctor",
#             func=lambda x: send_email_to_doctor(*x.split("|")),
#             description="Send an email to a doctor. Input format: 'Doctor Name|Message|Patient Name'"
#         ),
#         Tool(
#             name="CheckAppointment",
#             func=check_appointment_status,
#             description="Check the status of a booked appointment. Input should be the appointment ID."
#         )
#     ]
    
#     # Agent prompt template
#     agent_prompt = """You are an empathetic and intelligent medical assistant agent. You can autonomously use various tools to help patients with their medical needs.

# You have access to the following tools:
# {tools}

# Use the following format:

# Question: the input question or request from the patient
# Thought: think about what action to take and which tool(s) to use
# Action: the action to take, should be one of [{tool_names}]
# Action Input: the input to the action
# Observation: the result of the action
# ... (this Thought/Action/Action Input/Observation can repeat N times)
# Thought: I now have enough information to provide a complete answer
# Final Answer: the final compassionate answer to the patient

# IMPORTANT GUIDELINES:
# - Always be warm, empathetic, and supportive in your responses
# - When a patient describes symptoms, FIRST search medical documents, THEN suggest finding hospitals/doctors
# - For appointment booking, make sure you have all required information (doctor name, patient name, contact)
# - Always confirm actions before executing them when possible
# - Provide complete information including contact details and next steps
# - If you don't have enough information, politely ask the patient for clarification
# - Chain multiple tools together when needed (e.g., search docs → find hospitals → find doctors → book appointment)

# Previous conversation:
# {chat_history}

# Question: {input}
# {agent_scratchpad}"""
    
#     # Initialize LLM
#     try:
#         llm = Ollama(
#             model="mistral:7b",
#             temperature=0.2
#         )
        
#         # Create agent
#         prompt = AgentPromptTemplate.from_template(agent_prompt)
#         agent = create_react_agent(llm, tools, prompt)
        
#         # Create memory
#         memory = ConversationBufferMemory(
#             memory_key="chat_history",
#             return_messages=True
#         )
        
#         # Create agent executor
#         agent_executor = AgentExecutor(
#             agent=agent,
#             tools=tools,
#             memory=memory,
#             verbose=True,
#             max_iterations=10,
#             handle_parsing_errors=True
#         )
        
#         return agent_executor
    
#     except Exception as e:
#         st.error(f"Error creating agent: {str(e)}")
#         return None

# # ==================== STREAMLIT UI ====================

# def agentic_chat_tab():
#     """Agentic AI chat interface"""
#     st.markdown("### 🤖 Agentic AI Assistant")
#     st.markdown("*An intelligent agent that can autonomously search docs, find hospitals, book appointments, and more!*")
    
#     # Initialize session state
#     if 'agent_messages' not in st.session_state:
#         st.session_state.agent_messages = []
#         st.session_state.agent_messages.append({
#             'role': 'assistant',
#             'content': '''👋 Hello! I'm your **Agentic AI Medical Assistant**. 

# I can autonomously:
# - 🔍 Search your medical documents
# - 🏥 Find hospitals and specialists
# - 📅 Book appointments for you
# - 📧 Contact doctors on your behalf
# - ℹ️ Get doctor contact information
# - ✅ Check appointment status

# Just tell me what you need, and I'll handle it! For example:
# - "I have blood in my urine, what should I do?"
# - "Find me a kidney cancer specialist and book an appointment"
# - "I need to see a doctor for stomach pain urgently"

# How can I help you today? 💙'''
#         })
    
#     if 'agent_executor' not in st.session_state:
#         st.session_state.agent_executor = None
    
#     # Display chat history
#     for message in st.session_state.agent_messages:
#         with st.chat_message(message['role']):
#             st.markdown(message['content'])
    
#     # Chat input
#     if prompt := st.chat_input("Tell me what you need... I'll take care of it! 🤖"):
#         # Add user message
#         st.session_state.agent_messages.append({'role': 'user', 'content': prompt})
#         with st.chat_message('user'):
#             st.markdown(prompt)
        
#         # Generate agent response
#         with st.chat_message('assistant'):
#             with st.spinner('🤖 Agent is thinking and taking actions...'):
#                 try:
#                     # Initialize agent if needed
#                     if st.session_state.agent_executor is None:
#                         st.session_state.agent_executor = create_agentic_system()
#                         if st.session_state.agent_executor:
#                             st.success("✅ Agentic AI initialized!")
                    
#                     if st.session_state.agent_executor is None:
#                         response_text = "❌ Failed to initialize agent. Make sure Ollama is running: `ollama serve`"
#                     else:
#                         # Run agent
#                         with st.status("🔄 Agent working...", expanded=True) as status:
#                             st.write("🧠 Analyzing your request...")
#                             st.write("🔧 Selecting appropriate tools...")
#                             st.write("⚙️ Executing actions...")
                            
#                             response = st.session_state.agent_executor.invoke({"input": prompt})
#                             response_text = response['output']
                            
#                             status.update(label="✅ Complete!", state="complete")
                    
#                     st.markdown(response_text)
#                     st.session_state.agent_messages.append({
#                         'role': 'assistant',
#                         'content': response_text
#                     })
                
#                 except Exception as e:
#                     error_msg = f"❌ Error: {str(e)}\n\n💡 Make sure Ollama is running: `ollama serve`"
#                     st.error(error_msg)
#                     st.session_state.agent_messages.append({
#                         'role': 'assistant',
#                         'content': error_msg
#                     })

# def appointments_tab():
#     """View booked appointments"""
#     st.markdown("### 📅 My Appointments")
    
#     if not st.session_state.appointments:
#         st.info("📋 No appointments booked yet. Use the Agentic AI Assistant to book appointments!")
#     else:
#         st.success(f"✅ You have {len(st.session_state.appointments)} appointment(s)")
        
#         for apt in st.session_state.appointments:
#             with st.container():
#                 col1, col2, col3 = st.columns([2, 2, 1])
                
#                 with col1:
#                     st.markdown(f"**🆔 {apt['id']}**")
#                     st.write(f"👤 Patient: {apt['patient_name']}")
#                     st.write(f"👨‍⚕️ Doctor: {apt['doctor']}")
                
#                 with col2:
#                     st.write(f"🏥 {apt['hospital']}")
#                     st.write(f"📅 {apt['date_time']}")
                
#                 with col3:
#                     if apt['status'] == 'Confirmed':
#                         st.success("✅ Confirmed")
#                     else:
#                         st.warning("⏳ Pending")
                
#                 st.markdown("---")

# def main():
#     st.set_page_config(
#         page_title="Agentic Medical AI",
#         page_icon="🤖",
#         layout="wide"
#     )
    
#     st.title("🤖 Agentic Medical AI Assistant")
#     st.markdown("**Autonomous AI that thinks, plans, and acts to help you** 🚀")
    
#     # Sidebar
#     with st.sidebar:
#         st.header("🤖 Agentic AI System")
#         st.success("**Status:** Active & Ready")
        
#         st.markdown("---")
        
#         st.header("🧠 Agent Capabilities")
#         st.markdown("""
#         **Autonomous Actions:**
#         - 🔍 Search medical documents
#         - 🏥 Find hospitals
#         - 👨‍⚕️ Find specialist doctors
#         - 📅 Book appointments
#         - 📧 Send emails to doctors
#         - 📞 Get contact information
#         - ✅ Check appointment status
        
#         **Intelligence:**
#         - Multi-step reasoning
#         - Tool selection
#         - Action chaining
#         - Context awareness
#         """)
        
#         st.markdown("---")
        
#         st.header("💡 Example Requests")
#         st.markdown("""
#         *"I have kidney problems, help me find a specialist and book an appointment"*
        
#         *"Find hospitals near me for stomach cancer treatment"*
        
#         *"Send an email to Dr. Johnson about my symptoms"*
        
#         *"Check my appointment APT1001"*
#         """)
        
#         st.markdown("---")
        
#         st.header("🔧 Model")
#         st.info("**Llama 3.2 1B** (Ollama)")
        
#         if st.button("🔄 Reset Agent"):
#             st.session_state.agent_executor = None
#             st.success("Agent reset!")
    
#     # Main tabs
#     tab1, tab2 = st.tabs(["🤖 Agentic AI Chat", "📅 My Appointments"])
    
#     with tab1:
#         agentic_chat_tab()
    
#     with tab2:
#         appointments_tab()

# if __name__ == "__main__":
#     if not os.path.exists(DB_FAISS_PATH):
#         st.warning(f"⚠️ Vector store not found. Some features may be limited.")
#     main()