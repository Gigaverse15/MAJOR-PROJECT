import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime, timedelta
import pandas as pd
from groq import Groq

# Configuration
DB_FAISS_PATH = "vectorstore/db_faiss"

# Doctor Database (same as before)
DOCTORS_DATABASE = pd.read_csv("doctors_300_dataset.csv")


def init_session_state():
    if 'bookings' not in st.session_state:
        st.session_state.bookings = []
    if 'selected_doctor' not in st.session_state:
        st.session_state.selected_doctor = None
    if 'user_language' not in st.session_state:
        st.session_state.user_language = 'English'

@st.cache_resource
def get_vectorstore():
    try:
        embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def query_with_groq(api_key, prompt, context_docs):
    """Query Groq with document context"""
    try:
        client = Groq(api_key=api_key)
        
        # Combine context from documents
        context = "\n\n".join([doc.page_content for doc in context_docs[:3]])
        
        # Create prompt
        full_prompt = f"""You are a compassionate medical assistant for cancer patients.

Based on the medical documents below, provide a clear, empathetic answer.

RULES:
- Use ONLY information from the context below
- Be warm and supportive
- Cite page numbers when available
- If info not in context, say "Not in documents. Consult cancer.gov or your doctor."
- Never diagnose, just explain what documents say

CONTEXT:
{context}

QUESTION: {prompt}

ANSWER:"""

        # Call Groq
        response = client.chat.completions.create(
            model="groq/compound",  # Very accurate, fast!
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.3,
            max_tokens=1024
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        return f"Error: {str(e)}"

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
    duration = doctor['consultation_duration_mins']
    
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
            'display': current.strftime("%I:%M %p")
        })
        current = slot_end
    
    return slots

def book_appointment(doctor_id, date, slot_start, user_name, user_contact):
    conflicts = [b for b in st.session_state.bookings 
                 if b['doctor_id'] == doctor_id and b['date'] == date and b['slot_start'] == slot_start]
    
    if conflicts:
        return False, "⚠️ Slot already booked"
    
    doctor = DOCTORS_DATABASE[DOCTORS_DATABASE['doctor_id'] == doctor_id].iloc[0]
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
        'doctor_name_hi': doctor['doctor_name_hi'],
        'hospital_name': doctor['hospital_name'],
        'location': doctor['location_area'],
        'date': date,
        'slot_start': slot_start,
        'slot_end': slot_start + timedelta(minutes=doctor['consultation_duration_mins']),
        'user_name': user_name,
        'user_contact': user_contact,
        'queue_position': queue_position,
        'created_at': datetime.now(),
        'status': 'confirmed'
    }
    
    st.session_state.bookings.append(booking)
    return True, booking

def main():
    st.set_page_config(page_title="Cancer Care Assistant", page_icon="🏥", layout="wide")
    init_session_state()
    
    st.title("🏥 Comprehensive Cancer Care Assistant")
    st.markdown("**Medical Info + Find Doctors + Book Appointments + Multilingual** 💙")
    
    with st.sidebar:
        st.header("🔑 Groq API Key")
        st.markdown("**FREE: 14,400 requests/day!**")
        st.info("Get key: https://console.groq.com/keys")
        api_key = st.text_input("Paste API key:", type="password", key="api")
        if api_key:
            st.session_state.groq_api_key = api_key
            st.success("✅ API Key saved!")
        
        st.markdown("---")
        st.header("🌐 Language")
        lang = st.radio("Select:", ["English", "हिंदी"], key="lang")
        st.session_state.user_language = lang
        
        st.markdown("---")
        st.header("📊 Bookings")
        if st.session_state.bookings:
            st.success(f"✅ {len(st.session_state.bookings)} appointment(s)")
        else:
            st.info("No bookings yet")
        
        st.markdown("---")
        st.header("⚡ Groq Benefits")
        st.success("""
        **30x faster than Gemini!**
        - 30 req/min (vs 15)
        - 14,400/day (vs 1,500)
        - Ultra-fast responses
        """)
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "💬 Medical Q&A",
        "🔍 Find Doctors",
        "📅 Book Appointment",
        "📋 My Appointments"
    ])
    
    with tab1:
        medical_qa_tab(api_key if api_key else None)
    
    with tab2:
        find_doctors_tab()
    
    with tab3:
        booking_tab()
    
    with tab4:
        appointments_tab()

def medical_qa_tab(api_key):
    st.markdown("### 💬 Ask Medical Questions")
    st.markdown("*Powered by Groq - Ultra-fast AI! ⚡*")
    
    if not api_key:
        st.warning("⚠️ Enter Groq API key in sidebar")
        st.info("Get FREE key: https://console.groq.com/keys (14,400 requests/day!)")
        return
    
    if 'messages' not in st.session_state:
        st.session_state.messages = [{'role': 'assistant', 'content': '👋 Ask me about cancer, symptoms, treatments!'}]
    
    for msg in st.session_state.messages:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])
    
    if prompt := st.chat_input("Your question..."):
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        
        with st.chat_message('user'):
            st.markdown(prompt)
        
        with st.chat_message('assistant'):
            with st.spinner('🤖 Groq AI thinking...'):
                try:
                    vectorstore = get_vectorstore()
                    if vectorstore:
                        docs = vectorstore.similarity_search(prompt, k=3)
                        answer = query_with_groq(api_key, prompt, docs)
                        
                        response = answer
                        if docs:
                            response += "\n\n---\n**📄 Sources:**\n"
                            for i, doc in enumerate(docs, 1):
                                response += f"{i}. {os.path.basename(doc.metadata.get('source', 'Doc'))} (Page {doc.metadata.get('page', '?')})\n"
                        
                        response += "\n\n📚 [cancer.gov](https://cancer.gov) | [cancer.org](https://cancer.org)"
                        
                        st.markdown(response)
                        st.session_state.messages.append({'role': 'assistant', 'content': response})
                except Exception as e:
                    st.error(f"Error: {str(e)}")

def find_doctors_tab():
    st.markdown("### 🔍 Find Cancer Specialists")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        location = st.text_input("📍 Location:", placeholder="Noida, Botanical Garden")
    with col2:
        specialty = st.text_input("🩺 Specialty:", placeholder="Kidney Cancer")
    with col3:
        disease = st.text_input("🔬 Disease:", placeholder="Leukemia")
    
    if st.button("🔍 Search", type="primary"):
        doctors = search_doctors(
            location=location if location else None,
            specialty=specialty if specialty else None,
            disease=disease if disease else None
        )
        
        if len(doctors) > 0:
            st.success(f"✅ Found {len(doctors)} specialist(s)")
            
            for idx, (_, doc) in enumerate(doctors.iterrows(), 1):
                lang = st.session_state.user_language
                doc_name = doc['doctor_name_hi'] if 'हिंदी' in lang else doc['doctor_name_en']
                
                with st.expander(f"#{idx} ⭐ {doc_name} - {doc['rating']}/5.0", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write(f"**🩺 Specialty:** {doc['specialty']}")
                        st.write(f"**📌 Focus:** {doc['sub_specialty']}")
                        st.write(f"**🏥 Hospital:** {doc['hospital_name']}")
                        st.write(f"**📍 Location:** {doc['location_area']}")
                    
                    with col2:
                        st.write(f"**⭐ Rating:** {doc['rating']}/5.0")
                        st.write(f"**🎓 Experience:** {doc['experience_years']} years")
                        st.write(f"**⏱️ Duration:** {doc['consultation_duration_mins']} mins")
                        st.write(f"**💰 Fee:** ₹{doc['fee']}")
                    
                    with col3:
                        st.write(f"**🗣️ Languages:** {doc['languages']}")
                        st.write(f"**💻 Telemedicine:** {'✅' if doc['telemedicine'] else '❌'}")
                        st.write(f"**🏥 Treats:** {doc['diseases_treated'][:40]}...")
                    
                    if st.button(f"📅 Book", key=f"book_{doc['doctor_id']}"):
                        st.session_state.selected_doctor = doc['doctor_id']
                        st.success("✅ Go to 'Book Appointment' tab")
        else:
            st.warning("❌ No doctors found")

def booking_tab():
    st.markdown("### 📅 Book Appointment")
    
    if not st.session_state.selected_doctor:
        st.info("👈 Select doctor from 'Find Doctors' first")
        return
    
    doctor = DOCTORS_DATABASE[DOCTORS_DATABASE['doctor_id'] == st.session_state.selected_doctor].iloc[0]
    lang = st.session_state.user_language
    doc_name = doctor['doctor_name_hi'] if 'हिंदी' in lang else doctor['doctor_name_en']
    
    st.success(f"**Booking:** {doc_name}")
    st.info(f"🏥 {doctor['hospital_name']}, {doctor['location_area']}")
    
    st.markdown("#### 📆 Choose Date")
    today = datetime.now().date()
    date = st.date_input("Date:", min_value=today, max_value=today + timedelta(days=30))
    
    st.markdown("#### ⏰ Choose Time")
    slots = generate_time_slots(doctor['doctor_id'], date)
    available = [s for s in slots if s['available']]
    
    if not available:
        st.warning("❌ No slots. Choose another date.")
    else:
        slot_map = {s['display']: s for s in available}
        selected_display = st.selectbox("Available:", list(slot_map.keys()))
        selected_slot = slot_map[selected_display]
        
        st.info(f"⏱️ {selected_slot['display']} - {selected_slot['end'].strftime('%I:%M %p')}")
        
        st.markdown("#### 👤 Your Details")
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Name:")
        with col2:
            contact = st.text_input("Phone:")
        
        if st.button("✅ Confirm", type="primary"):
            if name and contact:
                success, result = book_appointment(doctor['doctor_id'], date, selected_slot['start'], name, contact)
                
                if success:
                    st.balloons()
                    st.success("🎉 **Confirmed!**")
                    st.markdown(f"""
**Details:**
- 🆔 ID: {result['booking_id']}
- 👨‍⚕️ Doctor: {result['doctor_name']}
- 📅 Date: {result['date'].strftime('%d %B %Y')}
- ⏰ Time: {result['slot_start'].strftime('%I:%M %p')} - {result['slot_end'].strftime('%I:%M %p')}
- 🎫 Queue: {result['queue_position']}
                    """)
                    st.session_state.selected_doctor = None
                else:
                    st.error(result)
            else:
                st.warning("Fill all details")

def appointments_tab():
    st.markdown("### 📋 My Appointments")
    
    if not st.session_state.bookings:
        st.info("📭 No appointments")
        return
    
    st.success(f"✅ {len(st.session_state.bookings)} appointment(s)")
    
    for booking in sorted(st.session_state.bookings, key=lambda x: x['date']):
        lang = st.session_state.user_language
        doc_name = booking['doctor_name_hi'] if 'हिंदी' in lang else booking['doctor_name']
        
        with st.expander(f"🆔 {booking['booking_id']} - {doc_name}", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**👨‍⚕️ Doctor:** {doc_name}")
                st.write(f"**🏥 Hospital:** {booking['hospital_name']}")
                st.write(f"**📅 Date:** {booking['date'].strftime('%d %B %Y')}")
            
            with col2:
                st.write(f"**⏰ Time:** {booking['slot_start'].strftime('%I:%M %p')} - {booking['slot_end'].strftime('%I:%M %p')}")
                st.write(f"**🎫 Queue:** {booking['queue_position']}")
                st.write(f"**📞 Phone:** {booking['user_contact']}")
            
            if st.button(f"❌ Cancel", key=f"cancel_{booking['booking_id']}"):
                st.session_state.bookings.remove(booking)
                st.success("Cancelled")
                st.rerun()

if __name__ == "__main__":
    if not os.path.exists(DB_FAISS_PATH):
        st.error("❌ Run main.py first!")
    else:
        main()# import os
# import streamlit as st
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA
# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import PromptTemplate
# from langchain_community.llms import Ollama

# # Configuration
# DB_FAISS_PATH = "vectorstore/db_faiss"

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
#         st.error(f"Error loading vector store: {str(e)}")
#         return None

# def set_custom_prompt(custom_prompt_template):
#     """Create a custom prompt template"""
#     prompt = PromptTemplate(
#         template=custom_prompt_template, 
#         input_variables=["context", "question"]
#     )
#     return prompt

# def initialize_qa_chain():
#     """Initialize the QA chain with local Ollama LLM"""
#     try:
#         vectorstore = get_vectorstore()
#         if vectorstore is None:
#             return None
        
#         # BALANCED PROMPT - Uses documents but doesn't hallucinate
#         CUSTOM_PROMPT_TEMPLATE = """You are a compassionate medical assistant helping patients understand their medical documents.

# Based on the context below, answer the patient's question in a warm, empathetic way.

# RULES:
# - Use the information from the Context to answer the question
# - Be specific and quote relevant parts from the Context
# - If the Context has relevant information, explain it clearly to the patient
# - Only say "information not available" if the Context truly has nothing related to the question
# - Be warm and supportive in your language
# - Encourage them to discuss with healthcare provider

# Context from your medical documents:
# {context}

# Patient Question: {question}

# Answer (use the context information above to help the patient):"""
        
#         llm = Ollama(
#             model="llama3.2:1b",
#             temperature=0.3
#         )
        
#         qa_chain = RetrievalQA.from_chain_type(
#             llm=llm,
#             chain_type="stuff",
#             retriever=vectorstore.as_retriever(
#                 search_type="similarity",
#                 search_kwargs={'k': 5}  # Get 5 most relevant chunks
#             ),
#             return_source_documents=True,
#             chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
#         )
        
#         return qa_chain
    
#     except Exception as e:
#         st.error(f"Error initializing QA chain: {str(e)}")
#         import traceback
#         st.error(traceback.format_exc())
#         return None

# def format_sources(source_documents):
#     """Format source documents for display"""
#     if not source_documents:
#         return ""
    
#     sources_text = "\n\n---\n**📄 Information from Your Documents:**\n"
#     for i, doc in enumerate(source_documents, 1):
#         source = doc.metadata.get('source', 'Unknown')
#         page = doc.metadata.get('page', 'Unknown')
#         sources_text += f"\n**Document {i}:** {os.path.basename(source)} (Page {page})\n"
#         content_preview = doc.page_content[:300].replace('\n', ' ')
#         sources_text += f"*Content: {content_preview}...*\n"
    
#     return sources_text

# def text_chat_tab():
#     """Regular text chat interface"""
#     st.markdown("### 💬 Text Chat")
#     st.markdown("*Ask questions about your medical documents + trusted cancer sources*")
    
#     # Initialize session state for text chat
#     if 'text_messages' not in st.session_state:
#         st.session_state.text_messages = []
#         st.session_state.text_messages.append({
#             'role': 'assistant', 
#             'content': '''👋 **Hello! I'm here to help you understand your medical documents with care and compassion.**

# 📄 **I can explain:**
# - What conditions and symptoms mean
# - Treatment options mentioned in your documents
# - Medical terms and terminology
# - What the documents say about your health concerns

# 💙 **I'll provide:**
# - Clear explanations from your uploaded PDFs
# - Empathetic, supportive guidance
# - References to where information came from
# - Links to trusted medical sources for more details

# **Just ask me anything about your medical documents!** 

# For example: "What is kidney cancer?" or "What symptoms should I watch for?"'''
#         })
    
#     if 'qa_chain' not in st.session_state:
#         st.session_state.qa_chain = None
    
#     # Display chat history
#     for message in st.session_state.text_messages:
#         with st.chat_message(message['role']):
#             st.markdown(message['content'])
    
#     # Chat input
#     if prompt := st.chat_input("Ask about symptoms, treatments, or conditions... 💙"):
#         # Add user message
#         st.session_state.text_messages.append({'role': 'user', 'content': prompt})
#         with st.chat_message('user'):
#             st.markdown(prompt)
        
#         # Generate response
#         with st.chat_message('assistant'):
#             with st.spinner('🔍 Searching documents and trusted medical sources...'):
#                 try:
#                     if st.session_state.qa_chain is None:
#                         st.session_state.qa_chain = initialize_qa_chain()
#                         if st.session_state.qa_chain:
#                             st.success("✅ Using local Ollama model: llama3.2:1b")
                    
#                     if st.session_state.qa_chain is None:
#                         response_text = "❌ Failed to initialize the chatbot. Make sure Ollama is running!"
#                     else:
#                         response = st.session_state.qa_chain.invoke({'query': prompt})
#                         result = response["result"]
#                         source_documents = response.get("source_documents", [])
                        
#                         response_text = result
#                         if source_documents:
#                             response_text += format_sources(source_documents)
#                         else:
#                             response_text += "\n\n⚠️ No relevant documents found."
                        
#                         # Add trusted sources for additional information
#                         response_text += "\n\n---\n📚 **For more information from trusted sources:**"
#                         response_text += "\n- National Cancer Institute: https://www.cancer.gov"
#                         response_text += "\n- American Cancer Society: https://www.cancer.org"
#                         response_text += "\n- Mayo Clinic: https://www.mayoclinic.org"
#                         response_text += "\n\n⚠️ **Important:** Always consult your healthcare provider for medical decisions."
                    
#                     st.markdown(response_text)
#                     st.session_state.text_messages.append({
#                         'role': 'assistant', 
#                         'content': response_text
#                     })
                
#                 except Exception as e:
#                     error_msg = f"❌ Error: {str(e)}\n\n💡 **Tip:** Make sure Ollama is running. Check if it's already running with Task Manager."
#                     st.error(error_msg)
#                     st.session_state.text_messages.append({
#                         'role': 'assistant', 
#                         'content': error_msg
#                     })

# def voice_chat_tab():
#     """Voice and multilingual chat interface"""
#     st.markdown("### 🎙️ Voice & Multilingual Chat")
#     st.markdown("*Coming soon! Voice input and 100+ languages*")
    
#     st.info("📦 **Voice features require additional packages.**")
#     st.code("pip install SpeechRecognition audio-recorder-streamlit gTTS googletrans==4.0.0rc1 pyaudio")
    
#     st.markdown("---")
#     st.markdown("**For now, please use the Text Chat tab.** 💙")

# def main():
#     st.set_page_config(
#         page_title="Medical Chatbot",
#         page_icon="🏥",
#         layout="wide"
#     )
    
#     st.title("🏥 Trusted Cancer Information Assistant")
#     st.markdown("**Reliable information from your documents + NCI, ACS, Mayo Clinic** 💙")
    
#     # Trust indicator
#     st.info("✅ **All information sourced from authenticated medical organizations and your documents**")
    
#     # Sidebar
#     with st.sidebar:
#         st.header("ℹ️ About")
#         st.info(
#             "This chatbot uses **local Ollama** to answer questions using:\n"
#             "- Your medical documents\n"
#             "- Trusted cancer sources (NCI, ACS, Mayo Clinic)"
#         )
        
#         st.header("🏥 Trusted Sources")
#         st.markdown("""
#         **We reference only:**
#         - National Cancer Institute (cancer.gov)
#         - American Cancer Society (cancer.org)
#         - Mayo Clinic (mayoclinic.org)
#         - NCCN Guidelines
#         - Your uploaded documents
#         """)
        
#         st.header("🤖 Model Info")
#         st.markdown("""
#         **Model:** Llama 3.2 1B
#         **Provider:** Ollama (Local!)
#         **Privacy:** 100% Private
#         **No API Key Required** ✅
#         """)
        
#         st.header("⚠️ Disclaimer")
#         st.warning("""
#         This chatbot provides educational 
#         information from trusted sources but 
#         does NOT replace professional medical advice.
        
#         Always consult your healthcare provider.
#         """)
        
#         if st.button("🔄 Reload Vector Store"):
#             st.cache_resource.clear()
#             st.success("Cache cleared!")
    
#     # Main content with tabs
#     tab1, tab2 = st.tabs(["💬 Text Chat", "🎙️ Voice (Coming Soon)"])
    
#     with tab1:
#         text_chat_tab()
    
#     with tab2:
#         voice_chat_tab()

# if __name__ == "__main__":
#     if not os.path.exists(DB_FAISS_PATH):
#         st.error(f"❌ Vector store not found at {DB_FAISS_PATH}")
#         st.info("💡 Please run main.py first to process your PDF documents!")
#     else:
#         main()# import os
# import streamlit as st
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA
# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import PromptTemplate
# from langchain_community.llms import Ollama

# # Configuration
# DB_FAISS_PATH = "vectorstore/db_faiss"

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
#         st.error(f"Error loading vector store: {str(e)}")
#         return None

# def set_custom_prompt(custom_prompt_template):
#     """Create a custom prompt template"""
#     prompt = PromptTemplate(
#         template=custom_prompt_template, 
#         input_variables=["context", "question"]
#     )
#     return prompt

# def initialize_qa_chain():
#     """Initialize the QA chain with local Ollama LLM"""
#     try:
#         vectorstore = get_vectorstore()
#         if vectorstore is None:
#             return None
        
#         # FIXED: Shorter prompt that works with the QA chain
#         CUSTOM_PROMPT_TEMPLATE = """You are a compassionate medical assistant specializing in cancer information from trusted sources.

# GUIDELINES:
# - Use warm, supportive, empathetic language
# - Share relevant information from the context documents AND trusted cancer websites (National Cancer Institute cancer.gov, American Cancer Society cancer.org, Mayo Clinic mayoclinic.org)
# - Clearly cite sources: "According to the National Cancer Institute..." or "Your documents indicate..."
# - When patients describe symptoms, explain what documents and trusted sources say about those symptoms
# - You CAN provide medical information but CANNOT diagnose
# - Say "according to documents and cancer.gov, these symptoms are associated with..." NOT "you have cancer"
# - Never refuse to help - share what documents and trusted sources say
# - Always encourage discussion with healthcare provider for proper evaluation
# - Include disclaimer: "Information from trusted sources but does not replace professional medical advice"

# Context: {context}

# Question: {question}

# Provide a compassionate, evidence-based answer with source citations:"""
        
#         llm = Ollama(
#             model="llama3.2:1b",
#             temperature=0.3
#         )
        
#         qa_chain = RetrievalQA.from_chain_type(
#             llm=llm,
#             chain_type="stuff",
#             retriever=vectorstore.as_retriever(
#                 search_type="similarity",
#                 search_kwargs={'k': 3}
#             ),
#             return_source_documents=True,
#             chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
#         )
        
#         return qa_chain
    
#     except Exception as e:
#         st.error(f"Error initializing QA chain: {str(e)}")
#         import traceback
#         st.error(traceback.format_exc())
#         return None

# def format_sources(source_documents):
#     """Format source documents for display"""
#     if not source_documents:
#         return ""
    
#     sources_text = "\n\n---\n**📚 Source Documents:**\n"
#     for i, doc in enumerate(source_documents, 1):
#         source = doc.metadata.get('source', 'Unknown')
#         page = doc.metadata.get('page', 'Unknown')
#         sources_text += f"\n{i}. **{os.path.basename(source)}** (Page {page})\n"
#         content_preview = doc.page_content[:200].replace('\n', ' ') + "..."
#         sources_text += f"   *Preview:* {content_preview}\n"
    
#     sources_text += "\n\n✅ **Verified by trusted medical sources:** cancer.gov, cancer.org, mayoclinic.org"
#     return sources_text

# def text_chat_tab():
#     """Regular text chat interface"""
#     st.markdown("### 💬 Text Chat")
#     st.markdown("*Ask questions about your medical documents + trusted cancer sources*")
    
#     # Initialize session state for text chat
#     if 'text_messages' not in st.session_state:
#         st.session_state.text_messages = []
#         st.session_state.text_messages.append({
#             'role': 'assistant', 
#             'content': '''👋 **Hello! I'm here to help you understand medical information from trusted sources.**

# I provide information from:
# - 📄 Your uploaded medical documents
# - 🏥 Trusted cancer organizations (NCI, ACS, Mayo Clinic)
# - 📚 Verified medical guidelines

# Ask me anything about symptoms, treatments, or conditions. I'll provide information from reliable sources and help you understand what to discuss with your healthcare provider. 💙'''
#         })
    
#     if 'qa_chain' not in st.session_state:
#         st.session_state.qa_chain = None
    
#     # Display chat history
#     for message in st.session_state.text_messages:
#         with st.chat_message(message['role']):
#             st.markdown(message['content'])
    
#     # Chat input
#     if prompt := st.chat_input("Ask about symptoms, treatments, or conditions... 💙"):
#         # Add user message
#         st.session_state.text_messages.append({'role': 'user', 'content': prompt})
#         with st.chat_message('user'):
#             st.markdown(prompt)
        
#         # Generate response
#         with st.chat_message('assistant'):
#             with st.spinner('🔍 Searching documents and trusted medical sources...'):
#                 try:
#                     if st.session_state.qa_chain is None:
#                         st.session_state.qa_chain = initialize_qa_chain()
#                         if st.session_state.qa_chain:
#                             st.success("✅ Using local Ollama model: llama3.2:1b")
                    
#                     if st.session_state.qa_chain is None:
#                         response_text = "❌ Failed to initialize the chatbot. Make sure Ollama is running!"
#                     else:
#                         response = st.session_state.qa_chain.invoke({'query': prompt})
#                         result = response["result"]
#                         source_documents = response.get("source_documents", [])
                        
#                         response_text = result
#                         if source_documents:
#                             response_text += format_sources(source_documents)
                        
#                         # Add trusted sources reminder
#                         response_text += "\n\n---\n⚠️ **Important:** This information is for educational purposes. Always consult your healthcare provider for medical decisions."
                    
#                     st.markdown(response_text)
#                     st.session_state.text_messages.append({
#                         'role': 'assistant', 
#                         'content': response_text
#                     })
                
#                 except Exception as e:
#                     error_msg = f"❌ Error: {str(e)}\n\n💡 **Tip:** Make sure Ollama is running. Check if it's already running with Task Manager."
#                     st.error(error_msg)
#                     st.session_state.text_messages.append({
#                         'role': 'assistant', 
#                         'content': error_msg
#                     })

# def voice_chat_tab():
#     """Voice and multilingual chat interface"""
#     st.markdown("### 🎙️ Voice & Multilingual Chat")
#     st.markdown("*Coming soon! Voice input and 100+ languages*")
    
#     st.info("📦 **Voice features require additional packages.**")
#     st.code("pip install SpeechRecognition audio-recorder-streamlit gTTS googletrans==4.0.0rc1 pyaudio")
    
#     st.markdown("---")
#     st.markdown("**For now, please use the Text Chat tab.** 💙")

# def main():
#     st.set_page_config(
#         page_title="Medical Chatbot",
#         page_icon="🏥",
#         layout="wide"
#     )
    
#     st.title("🏥 Trusted Cancer Information Assistant")
#     st.markdown("**Reliable information from your documents + NCI, ACS, Mayo Clinic** 💙")
    
#     # Trust indicator
#     st.info("✅ **All information sourced from authenticated medical organizations and your documents**")
    
#     # Sidebar
#     with st.sidebar:
#         st.header("ℹ️ About")
#         st.info(
#             "This chatbot uses **local Ollama** to answer questions using:\n"
#             "- Your medical documents\n"
#             "- Trusted cancer sources (NCI, ACS, Mayo Clinic)"
#         )
        
#         st.header("🏥 Trusted Sources")
#         st.markdown("""
#         **We reference only:**
#         - National Cancer Institute (cancer.gov)
#         - American Cancer Society (cancer.org)
#         - Mayo Clinic (mayoclinic.org)
#         - NCCN Guidelines
#         - Your uploaded documents
#         """)
        
#         st.header("🤖 Model Info")
#         st.markdown("""
#         **Model:** Llama 3.2 1B
#         **Provider:** Ollama (Local!)
#         **Privacy:** 100% Private
#         **No API Key Required** ✅
#         """)
        
#         st.header("⚠️ Disclaimer")
#         st.warning("""
#         This chatbot provides educational 
#         information from trusted sources but 
#         does NOT replace professional medical advice.
        
#         Always consult your healthcare provider.
#         """)
        
#         if st.button("🔄 Reload Vector Store"):
#             st.cache_resource.clear()
#             st.success("Cache cleared!")
    
#     # Main content with tabs
#     tab1, tab2 = st.tabs(["💬 Text Chat", "🎙️ Voice (Coming Soon)"])
    
#     with tab1:
#         text_chat_tab()
    
#     with tab2:
#         voice_chat_tab()

# if __name__ == "__main__":
#     if not os.path.exists(DB_FAISS_PATH):
#         st.error(f"❌ Vector store not found at {DB_FAISS_PATH}")
#         st.info("💡 Please run main.py first to process your PDF documents!")
#     else:
#         main()# import os
# import streamlit as st
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA
# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import PromptTemplate
# from langchain_community.llms import Ollama

# # Configuration
# DB_FAISS_PATH = "vectorstore/db_faiss"

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
#         st.error(f"Error loading vector store: {str(e)}")
#         return None

# def set_custom_prompt(custom_prompt_template):
#     """Create a custom prompt template"""
#     prompt = PromptTemplate(
#         template=custom_prompt_template, 
#         input_variables=["context", "question"]
#     )
#     return prompt

# def initialize_qa_chain():
#     """Initialize the QA chain with local Ollama LLM"""
#     try:
#         vectorstore = get_vectorstore()
#         if vectorstore is None:
#             return None
        
#         CUSTOM_PROMPT_TEMPLATE = """
# You are a compassionate and knowledgeable medical assistant specializing in cancer information. 
# You provide reliable, evidence-based information from trusted sources to help patients understand their health.

# INFORMATION SOURCES:
# - Medical documents provided by the patient (PDFs, guidelines, reports)
# - Authentic cancer information websites: National Cancer Institute (cancer.gov), American Cancer Society (cancer.org), Mayo Clinic (mayoclinic.org), NCCN Guidelines, MD Anderson, Memorial Sloan Kettering
# - Only use information from verified, reputable medical organizations

# IMPORTANT GUIDELINES:

# Communication Style:
# - Use warm, supportive, and reassuring language
# - Acknowledge the patient's concerns with empathy and understanding
# - Show that you care about their emotional state, not just medical facts
# - Be encouraging and supportive while remaining accurate and honest

# Providing Information:
# - ALWAYS share relevant information from BOTH the documents AND trusted websites
# - Clearly cite your sources (e.g., "According to the National Cancer Institute..." or "Your medical documents indicate...")
# - Explain what the medical documents and trusted sources say about symptoms, conditions, and treatments
# - When patients describe symptoms, tell them what the documents AND authentic websites say about those symptoms
# - If the patient lists symptoms, check if those symptoms are mentioned in documents/trusted sources and explain what they say
# - Help patients understand their medical information clearly with references to reliable sources

# Medical Boundaries:
# - You CAN and SHOULD provide medical information from documents and authentic sources
# - You CANNOT diagnose (don't say "you have cancer")
# - You CAN say "according to the documents and trusted medical sources like cancer.gov, these symptoms are associated with..." or "the National Cancer Institute states that..."
# - Never refuse to help or say "I cannot provide medical advice" - instead, share what the documents and authentic websites say
# - Always include source attribution (document name, website name, organization)

# Guidance and Support:
# - ALWAYS end by encouraging them to discuss findings with their healthcare provider for proper evaluation
# - Provide links to the authentic sources you reference when available
# - If information isn't in the documents, search trusted cancer websites (cancer.gov, cancer.org, mayoclinic.org)
# - Emphasize that information is from reputable sources but still requires professional medical interpretation

# Trust and Verification:
# - Make it clear when information comes from your documents vs. external trusted sources
# - Only reference information from verified cancer organizations (NCI, ACS, Mayo Clinic, NCCN, etc.)
# - Never use information from unverified or questionable sources
# - Include medical disclaimer: "This information is from trusted medical sources but does not replace professional medical advice"

# Example Response Format:
# "I understand your concern about [symptom]. Let me share what I found from reliable sources:

# 📄 From your medical documents:
# [Document information]

# 🏥 From trusted medical sources:
# According to the National Cancer Institute (cancer.gov): [information]
# The American Cancer Society (cancer.org) notes that: [information]

# This information suggests [explanation], but it's important to discuss these findings with your healthcare provider for proper evaluation and personalized guidance. Would you like me to help you find specialists or book an appointment? 💙"

# Remember: You are a trusted bridge between medical information and patients, helping them understand complex health information from reliable sources while emphasizing the importance of professional medical care.
# """
#         llm = Ollama(
#             model="llama3.2:1b",
#             temperature=0.3
#         )
        
#         qa_chain = RetrievalQA.from_chain_type(
#             llm=llm,
#             chain_type="stuff",
#             retriever=vectorstore.as_retriever(
#                 search_type="similarity",
#                 search_kwargs={'k': 3}
#             ),
#             return_source_documents=True,
#             chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
#         )
        
#         return qa_chain
    
#     except Exception as e:
#         st.error(f"Error initializing QA chain: {str(e)}")
#         return None

# def format_sources(source_documents):
#     """Format source documents for display"""
#     if not source_documents:
#         return ""
    
#     sources_text = "\n\n---\n**📚 Source Documents:**\n"
#     for i, doc in enumerate(source_documents, 1):
#         source = doc.metadata.get('source', 'Unknown')
#         page = doc.metadata.get('page', 'Unknown')
#         sources_text += f"\n{i}. **{os.path.basename(source)}** (Page {page})\n"
#         content_preview = doc.page_content[:200].replace('\n', ' ') + "..."
#         sources_text += f"   *Preview:* {content_preview}\n"
    
#     return sources_text

# def text_chat_tab():
#     """Regular text chat interface"""
#     st.markdown("### 💬 Text Chat")
#     st.markdown("*Ask your questions by typing below*")
    
#     # Initialize session state for text chat
#     if 'text_messages' not in st.session_state:
#         st.session_state.text_messages = []
#         st.session_state.text_messages.append({
#             'role': 'assistant', 
#             'content': '👋 Hello! I\'m here to help you understand your medical documents with care and compassion. Please feel free to ask me anything about your medical PDFs! 💙'
#         })
    
#     if 'qa_chain' not in st.session_state:
#         st.session_state.qa_chain = None
    
#     # Display chat history
#     for message in st.session_state.text_messages:
#         with st.chat_message(message['role']):
#             st.markdown(message['content'])
    
#     # Chat input
#     if prompt := st.chat_input("Ask me anything about your medical documents... 💙"):
#         # Add user message
#         st.session_state.text_messages.append({'role': 'user', 'content': prompt})
#         with st.chat_message('user'):
#             st.markdown(prompt)
        
#         # Generate response
#         with st.chat_message('assistant'):
#             with st.spinner('🔍 Looking through your documents...'):
#                 try:
#                     if st.session_state.qa_chain is None:
#                         st.session_state.qa_chain = initialize_qa_chain()
#                         if st.session_state.qa_chain:
#                             st.success("✅ Using local Ollama model: llama3.2:1b")
                    
#                     if st.session_state.qa_chain is None:
#                         response_text = "❌ Failed to initialize the chatbot."
#                     else:
#                         response = st.session_state.qa_chain.invoke({'query': prompt})
#                         result = response["result"]
#                         source_documents = response.get("source_documents", [])
                        
#                         response_text = result
#                         if source_documents:
#                             response_text += format_sources(source_documents)
                    
#                     st.markdown(response_text)
#                     st.session_state.text_messages.append({
#                         'role': 'assistant', 
#                         'content': response_text
#                     })
                
#                 except Exception as e:
#                     error_msg = f"❌ Error: {str(e)}"
#                     st.error(error_msg)
#                     st.session_state.text_messages.append({
#                         'role': 'assistant', 
#                         'content': error_msg
#                     })

# def voice_chat_tab():
#     """Voice and multilingual chat interface"""
#     st.markdown("### 🎙️ Voice & Multilingual Chat")
#     st.markdown("*Speak or type in any language*")
    
#     # Check if required packages are installed
#     try:
#         from audio_recorder_streamlit import audio_recorder
#         import speech_recognition as sr
#         from googletrans import Translator
#         from gtts import gTTS
#         import tempfile
#         from io import BytesIO
        
#         voice_available = True
#     except ImportError as e:
#         st.error("❌ Voice features not available. Please install required packages:")
#         st.code("pip install SpeechRecognition audio-recorder-streamlit gTTS googletrans==4.0.0rc1 pyaudio")
#         st.info("After installing, restart the app to use voice features.")
#         voice_available = False
#         return
    
#     # Initialize translator
#     translator = Translator()
    
#     # Language selection
#     col1, col2 = st.columns(2)
    
#     with col1:
#         languages = {
#             'English': 'en', 'Spanish': 'es', 'French': 'fr', 'German': 'de',
#             'Italian': 'it', 'Portuguese': 'pt', 'Russian': 'ru', 'Japanese': 'ja',
#             'Korean': 'ko', 'Chinese': 'zh-cn', 'Arabic': 'ar', 'Hindi': 'hi',
#             'Bengali': 'bn', 'Urdu': 'ur', 'Turkish': 'tr'
#         }
        
#         selected_lang = st.selectbox(
#             "🌐 Select Your Language:",
#             options=list(languages.keys()),
#             index=0
#         )
#         user_lang = languages[selected_lang]
    
#     with col2:
#         enable_voice_output = st.checkbox("🔊 Enable Voice Responses", value=True)
    
#     st.markdown("---")
    
#     # Initialize session state for voice chat
#     if 'voice_messages' not in st.session_state:
#         st.session_state.voice_messages = []
#         welcome_en = "👋 Hello! I'm your multilingual medical assistant. Ask me anything in your language!"
        
#         try:
#             welcome_translated = translator.translate(welcome_en, dest=user_lang).text if user_lang != 'en' else welcome_en
#         except:
#             welcome_translated = welcome_en
        
#         st.session_state.voice_messages.append({
#             'role': 'assistant', 
#             'content': f"{welcome_en}\n\n*{welcome_translated}*" if user_lang != 'en' else welcome_en
#         })
    
#     # Display chat history
#     for message in st.session_state.voice_messages:
#         with st.chat_message(message['role']):
#             st.markdown(message['content'])
    
#     # Voice input
#     st.markdown("#### 🎙️ Voice Input")
#     audio_bytes = audio_recorder(
#         text="Click to record your question",
#         recording_color="#e74c3c",
#         neutral_color="#3498db",
#         icon_size="2x"
#     )
    
#     prompt = None
#     if audio_bytes:
#         st.audio(audio_bytes, format="audio/wav")
#         with st.spinner("🎧 Converting speech to text..."):
#             try:
#                 with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
#                     tmp_file.write(audio_bytes)
#                     tmp_file_path = tmp_file.name
                
#                 recognizer = sr.Recognizer()
#                 with sr.AudioFile(tmp_file_path) as source:
#                     audio_data = recognizer.record(source)
#                     prompt = recognizer.recognize_google(audio_data)
                
#                 os.unlink(tmp_file_path)
#                 st.success(f"📝 You said: {prompt}")
#             except Exception as e:
#                 st.error(f"❌ Could not understand audio: {str(e)}")
#                 prompt = None
    
#     # Text input
#     if not prompt:
#         st.markdown("#### 💬 Or Type Your Question")
#         prompt = st.chat_input("Type in any language... / اكتب / 在这里输入 / Escribe...")
    
#     if prompt:
#         # Detect and translate
#         try:
#             detected = translator.detect(prompt)
#             detected_lang = detected.lang
            
#             if detected_lang != 'en':
#                 prompt_en = translator.translate(prompt, dest='en').text
#                 st.info(f"🌐 Detected: {detected_lang.upper()} → Translating to English")
#             else:
#                 prompt_en = prompt
#         except:
#             prompt_en = prompt
#             detected_lang = 'en'
        
#         # Display user message
#         st.session_state.voice_messages.append({'role': 'user', 'content': prompt})
#         with st.chat_message('user'):
#             st.markdown(prompt)
#             if detected_lang != 'en':
#                 st.caption(f"*English: {prompt_en}*")
        
#         # Generate response
#         with st.chat_message('assistant'):
#             with st.spinner('🔍 Searching documents...'):
#                 try:
#                     if st.session_state.qa_chain is None:
#                         st.session_state.qa_chain = initialize_qa_chain()
                    
#                     if st.session_state.qa_chain is None:
#                         response_text = "❌ Failed to initialize chatbot."
#                     else:
#                         response = st.session_state.qa_chain.invoke({'query': prompt_en})
#                         result_en = response["result"]
#                         source_documents = response.get("source_documents", [])
                        
#                         response_text_en = result_en
#                         if source_documents:
#                             response_text_en += format_sources(source_documents)
                        
#                         # Show English
#                         if detected_lang != 'en':
#                             st.markdown("**📄 English Response:**")
#                             st.markdown(response_text_en)
#                             st.markdown(f"\n---\n**🌐 Translation to {selected_lang}:**")
                            
#                             try:
#                                 result_translated = translator.translate(result_en, dest=detected_lang).text
#                                 st.markdown(result_translated)
#                                 response_text = f"{response_text_en}\n\n---\n**{selected_lang}:**\n{result_translated}"
#                             except:
#                                 st.warning("Translation failed, showing English only")
#                                 response_text = response_text_en
#                         else:
#                             st.markdown(response_text_en)
#                             response_text = response_text_en
                        
#                         # Voice output
#                         if enable_voice_output:
#                             with st.spinner("🔊 Generating voice..."):
#                                 try:
#                                     voice_text = result_translated if detected_lang != 'en' else result_en
#                                     voice_lang = detected_lang if detected_lang != 'en' else 'en'
                                    
#                                     tts = gTTS(text=voice_text, lang=voice_lang, slow=False)
#                                     fp = BytesIO()
#                                     tts.write_to_fp(fp)
#                                     fp.seek(0)
#                                     st.audio(fp, format='audio/mp3')
#                                 except Exception as e:
#                                     st.warning(f"Voice generation failed: {str(e)}")
                    
#                     st.session_state.voice_messages.append({
#                         'role': 'assistant', 
#                         'content': response_text
#                     })
                
#                 except Exception as e:
#                     error_msg = f"❌ Error: {str(e)}"
#                     st.error(error_msg)

# def main():
#     st.set_page_config(
#         page_title="Medical Chatbot",
#         page_icon="🏥",
#         layout="wide"
#     )
    
#     st.title("🏥 Compassionate Medical Assistant")
#     st.markdown("**A caring AI assistant to help you understand your medical documents** 💙")
    
#     # Sidebar
#     with st.sidebar:
#         st.header("ℹ️ About")
#         st.info(
#             "This chatbot uses **Meta's Llama 3.2** model running locally via Ollama "
#             "to answer questions about your medical documents with empathy and care."
#         )
        
#         st.header("🤖 Model Info")
#         st.markdown("""
#         **Model:** Llama 3.2 1B
#         **Provider:** Ollama (Local!)
#         **No API Key Required** ✅
#         **Fast & Private** 🔒
#         **Empathetic Responses** 💙
#         """)
        
#         st.header("📋 Features")
#         st.markdown("""
#         **💬 Text Chat:**
#         - Type your questions
#         - Fast text responses
#         - Source citations
        
#         **🎙️ Voice & Multilingual:**
#         - Speak your questions
#         - 100+ languages supported
#         - Voice responses
#         - Auto translation
#         """)
        
#         st.header("⚙️ Available Models")
#         st.markdown("""
#         Change model in code line 77:
#         - `llama3.2:1b` (current)
#         - `mistral:7b` (more detailed)
#         - `gemma3:latest`
#         """)
        
#         if st.button("🔄 Reload Vector Store"):
#             st.cache_resource.clear()
#             st.success("Cache cleared!")
    
#     # Main content with tabs
#     tab1, tab2 = st.tabs(["💬 Text Chat", "🎙️ Voice & Multilingual"])
    
#     with tab1:
#         text_chat_tab()
    
#     with tab2:
#         voice_chat_tab()

# if __name__ == "__main__":
#     if not os.path.exists(DB_FAISS_PATH):
#         st.error(f"❌ Vector store not found at {DB_FAISS_PATH}")
#         st.info("💡 Please run main.py first to process your PDF documents!")
#     else:
#         main()# import os
# import streamlit as st
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA
# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import PromptTemplate
# from langchain_community.llms import Ollama

# # Configuration
# DB_FAISS_PATH = "vectorstore/db_faiss"

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
#         st.error(f"Error loading vector store: {str(e)}")
#         return None

# def set_custom_prompt(custom_prompt_template):
#     """Create a custom prompt template"""
#     prompt = PromptTemplate(
#         template=custom_prompt_template, 
#         input_variables=["context", "question"]
#     )
#     return prompt

# def initialize_qa_chain():
#     """Initialize the QA chain with local Ollama LLM"""
#     try:
#         # Load vector store
#         vectorstore = get_vectorstore()
#         if vectorstore is None:
#             return None
        
#         # Empathetic and compassionate prompt template
#         CUSTOM_PROMPT_TEMPLATE = """You are a caring and empathetic medical assistant. Your role is to help patients understand their medical documents with compassion and support.

# Important guidelines:
# - Use warm, supportive, and reassuring language
# - Acknowledge the patient's concerns with empathy and understanding
# - ALWAYS share relevant information from the context/documents provided
# - Explain what the medical documents say about symptoms, conditions, and treatments
# - Help patients understand their medical information clearly
# - When patients describe symptoms, tell them what the documents say about those symptoms
# - If the patient lists symptoms, check if those symptoms are mentioned in the documents and explain what the documents say
# - You CAN and SHOULD provide medical information from the documents
# - You CANNOT diagnose (don't say "you have cancer"), but you CAN say "according to the documents, these symptoms are associated with..."
# - Never refuse to help or say "I cannot provide medical advice" - instead, share what the documents say and encourage them to discuss with their doctor
# - Be encouraging and supportive while remaining accurate and honest
# - Use compassionate phrases like:
#   * "I understand your concern, let me check what the documents say"
#   * "Based on the medical documents, here's what I found"
#   * "The documents mention that these symptoms..."
#   * "I'm here to help you understand what's in your medical documents"
#   * "Your health and wellbeing matter"
# - ALWAYS end by encouraging them to discuss findings with their healthcare provider for proper evaluation
# - Show that you care about their emotional state, not just the medical facts

# Context: {context}

# Question: {question}

# Please provide a compassionate, empathetic, and informative answer based on what the documents say:"""
        
#         # Initialize Ollama LLM with your local model
#         llm = Ollama(
#             model="llama3.2:1b",  # Using your local Ollama model
#             temperature=0.3  # Slightly higher for more natural, empathetic responses
#         )
        
#         st.success(f"✅ Using local Ollama model: llama3.2:1b")
        
#         # Create QA chain
#         qa_chain = RetrievalQA.from_chain_type(
#             llm=llm,
#             chain_type="stuff",
#             retriever=vectorstore.as_retriever(
#                 search_type="similarity",
#                 search_kwargs={'k': 3}
#             ),
#             return_source_documents=True,
#             chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
#         )
        
#         return qa_chain
    
#     except Exception as e:
#         st.error(f"Error initializing QA chain: {str(e)}")
#         import traceback
#         st.error(traceback.format_exc())
#         return None

# def format_sources(source_documents):
#     """Format source documents for display"""
#     if not source_documents:
#         return ""
    
#     sources_text = "\n\n---\n**📚 Source Documents:**\n"
#     for i, doc in enumerate(source_documents, 1):
#         source = doc.metadata.get('source', 'Unknown')
#         page = doc.metadata.get('page', 'Unknown')
#         sources_text += f"\n{i}. **{os.path.basename(source)}** (Page {page})\n"
#         # Add a snippet of the content
#         content_preview = doc.page_content[:200].replace('\n', ' ') + "..."
#         sources_text += f"   *Preview:* {content_preview}\n"
    
#     return sources_text

# def main():
#     # Page configuration
#     st.set_page_config(
#         page_title="Medical Chatbot",
#         page_icon="🏥",
#         layout="wide"
#     )
    
#     # Title and description
#     st.title("🏥 Compassionate Medical Assistant")
#     st.markdown("**A caring AI assistant to help you understand your medical documents** 💙")
    
#     # Sidebar with information
#     with st.sidebar:
#         st.header("ℹ️ About")
#         st.info(
#             "This chatbot uses **Meta's Llama 3.2** model running locally via Ollama "
#             "to answer questions about your medical documents with empathy and care."
#         )
        
#         st.header("🤖 Model Info")
#         st.markdown("""
#         **Model:** Llama 3.2 1B
#         **Provider:** Ollama (Local!)
#         **No API Key Required** ✅
#         **Fast & Private** 🔒
#         **Empathetic Responses** 💙
#         """)
        
#         st.header("📋 How to Use")
#         st.markdown("""
#         1. Type your question in the chat input
#         2. The AI will search through your documents
#         3. You'll get a compassionate answer with sources
#         4. Feel free to ask follow-up questions
#         """)
        
#         st.header("💭 Remember")
#         st.markdown("""
#         - This chatbot is here to help you understand your medical information
#         - It provides supportive guidance based on your documents
#         - For medical advice, always consult your healthcare provider
#         - Your concerns are valid and important
#         """)
        
#         st.header("⚙️ Available Models")
#         st.markdown("""
#         Change model in medibot.py line 63:
#         - `llama3.2:1b` (current)
#         - `mistral:7b` (more detailed)
#         - `gemma3:latest`
#         """)
        
#         if st.button("🔄 Reload Vector Store"):
#             st.cache_resource.clear()
#             st.success("Cache cleared! Vector store will reload on next query.")
    
#     # Initialize session state for chat history
#     if 'messages' not in st.session_state:
#         st.session_state.messages = []
#         # Add welcome message
#         st.session_state.messages.append({
#             'role': 'assistant', 
#             'content': '👋 Hello! I\'m here to help you understand your medical documents with care and compassion. I know health concerns can be worrying, and I\'m here to support you. Please feel free to ask me anything about your medical PDFs - there are no silly questions, and I\'m here to listen. 💙'
#         })
    
#     if 'qa_chain' not in st.session_state:
#         st.session_state.qa_chain = None
    
#     # Display chat history
#     for message in st.session_state.messages:
#         with st.chat_message(message['role']):
#             st.markdown(message['content'])
    
#     # Chat input
#     if prompt := st.chat_input("Ask me anything about your medical documents... I'm here to help 💙"):
#         # Add user message to chat
#         st.session_state.messages.append({'role': 'user', 'content': prompt})
#         with st.chat_message('user'):
#             st.markdown(prompt)
        
#         # Generate response
#         with st.chat_message('assistant'):
#             with st.spinner('🔍 Looking through your documents to help answer your question...'):
#                 try:
#                     # Initialize QA chain if not already done
#                     if st.session_state.qa_chain is None:
#                         st.session_state.qa_chain = initialize_qa_chain()
                    
#                     if st.session_state.qa_chain is None:
#                         response_text = "❌ I'm having trouble accessing the medical documents right now. Please make sure everything is set up correctly, and I'll be ready to help you."
#                     else:
#                         # Get response from QA chain
#                         response = st.session_state.qa_chain.invoke({'query': prompt})
                        
#                         # Format the response
#                         result = response["result"]
#                         source_documents = response.get("source_documents", [])
                        
#                         # Combine result with formatted sources
#                         response_text = result
#                         if source_documents:
#                             response_text += format_sources(source_documents)
                    
#                     # Display response
#                     st.markdown(response_text)
                    
#                     # Add assistant response to chat history
#                     st.session_state.messages.append({
#                         'role': 'assistant', 
#                         'content': response_text
#                     })
                
#                 except Exception as e:
#                     error_msg = f"I'm sorry, I encountered an issue while trying to help you: {str(e)}\n\n"
#                     error_msg += "**Here's what you can try:**\n"
#                     error_msg += "1. Make sure Ollama is running (it should be!)\n"
#                     error_msg += "2. Try asking your question again\n"
#                     error_msg += "3. If the problem persists, please let someone know\n\n"
#                     error_msg += "I'm here to help once things are working again! 💙"
#                     st.error(error_msg)
#                     st.session_state.messages.append({
#                         'role': 'assistant', 
#                         'content': error_msg
#                     })

# if __name__ == "__main__":
#     # Check if vector store exists
#     if not os.path.exists(DB_FAISS_PATH):
#         st.error(f"❌ I can't find the medical documents database at {DB_FAISS_PATH}")
#         st.info("💡 Please run main.py first to process your PDF documents, then I'll be ready to help you!")
#     else:
#         main()# import os
# import streamlit as st
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA
# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import PromptTemplate
# from langchain_community.llms import Ollama

# # Configuration
# DB_FAISS_PATH = "vectorstore/db_faiss"

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
#         st.error(f"Error loading vector store: {str(e)}")
#         return None

# def set_custom_prompt(custom_prompt_template):
#     """Create a custom prompt template"""
#     prompt = PromptTemplate(
#         template=custom_prompt_template, 
#         input_variables=["context", "question"]
#     )
#     return prompt

# def initialize_qa_chain():
#     """Initialize the QA chain with local Ollama LLM"""
#     try:
#         # Load vector store
#         vectorstore = get_vectorstore()
#         if vectorstore is None:
#             return None
        
#         # Empathetic and compassionate prompt template
#         CUSTOM_PROMPT_TEMPLATE = """You are a caring and empathetic medical assistant. Your role is to help patients understand their medical documents with compassion and support.

# Important guidelines:
# - Use warm, supportive, and reassuring language
# - Acknowledge the patient's concerns with empathy and understanding
# - Only use information from the context provided
# - If the information isn't in the documents, gently guide them to consult their healthcare provider
# - Never make the patient feel dismissed, ignored, or unnecessarily worried
# - Be encouraging and supportive while remaining accurate and honest
# - Use compassionate phrases like:
#   * "I understand your concern"
#   * "It's completely natural to worry about this"
#   * "I'm here to help you understand"
#   * "Thank you for sharing this with me"
#   * "Your health and wellbeing matter"
# - End responses with reassurance and encouragement when appropriate
# - Show that you care about their emotional state, not just the medical facts
# - If discussing symptoms, acknowledge that experiencing them can be difficult or scary

# Context: {context}

# Question: {question}

# Please provide a compassionate, empathetic, and informative answer:"""
        
#         # Initialize Ollama LLM with your local model
#         llm = Ollama(
#             model="llama3.2:1b",  # Using your local Ollama model
#             temperature=0.3  # Slightly higher for more natural, empathetic responses
#         )
        
#         st.success(f"✅ Using local Ollama model: llama3.2:1b")
        
#         # Create QA chain
#         qa_chain = RetrievalQA.from_chain_type(
#             llm=llm,
#             chain_type="stuff",
#             retriever=vectorstore.as_retriever(
#                 search_type="similarity",
#                 search_kwargs={'k': 3}
#             ),
#             return_source_documents=True,
#             chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
#         )
        
#         return qa_chain
    
#     except Exception as e:
#         st.error(f"Error initializing QA chain: {str(e)}")
#         import traceback
#         st.error(traceback.format_exc())
#         return None

# def format_sources(source_documents):
#     """Format source documents for display"""
#     if not source_documents:
#         return ""
    
#     sources_text = "\n\n---\n**📚 Source Documents:**\n"
#     for i, doc in enumerate(source_documents, 1):
#         source = doc.metadata.get('source', 'Unknown')
#         page = doc.metadata.get('page', 'Unknown')
#         sources_text += f"\n{i}. **{os.path.basename(source)}** (Page {page})\n"
#         # Add a snippet of the content
#         content_preview = doc.page_content[:200].replace('\n', ' ') + "..."
#         sources_text += f"   *Preview:* {content_preview}\n"
    
#     return sources_text

# def main():
#     # Page configuration
#     st.set_page_config(
#         page_title="Medical Chatbot",
#         page_icon="🏥",
#         layout="wide"
#     )
    
#     # Title and description
#     st.title("🏥 Compassionate Medical Assistant")
#     st.markdown("**A caring AI assistant to help you understand your medical documents** 💙")
    
#     # Sidebar with information
#     with st.sidebar:
#         st.header("ℹ️ About")
#         st.info(
#             "This chatbot uses **Meta's Llama 3.2** model running locally via Ollama "
#             "to answer questions about your medical documents with empathy and care."
#         )
        
#         st.header("🤖 Model Info")
#         st.markdown("""
#         **Model:** Llama 3.2 1B
#         **Provider:** Ollama (Local!)
#         **No API Key Required** ✅
#         **Fast & Private** 🔒
#         **Empathetic Responses** 💙
#         """)
        
#         st.header("📋 How to Use")
#         st.markdown("""
#         1. Type your question in the chat input
#         2. The AI will search through your documents
#         3. You'll get a compassionate answer with sources
#         4. Feel free to ask follow-up questions
#         """)
        
#         st.header("💭 Remember")
#         st.markdown("""
#         - This chatbot is here to help you understand your medical information
#         - It provides supportive guidance based on your documents
#         - For medical advice, always consult your healthcare provider
#         - Your concerns are valid and important
#         """)
        
#         st.header("⚙️ Available Models")
#         st.markdown("""
#         Change model in medibot.py line 63:
#         - `llama3.2:1b` (current)
#         - `mistral:7b` (more detailed)
#         - `gemma3:latest`
#         """)
        
#         if st.button("🔄 Reload Vector Store"):
#             st.cache_resource.clear()
#             st.success("Cache cleared! Vector store will reload on next query.")
    
#     # Initialize session state for chat history
#     if 'messages' not in st.session_state:
#         st.session_state.messages = []
#         # Add welcome message
#         st.session_state.messages.append({
#             'role': 'assistant', 
#             'content': '👋 Hello! I\'m here to help you understand your medical documents with care and compassion. I know health concerns can be worrying, and I\'m here to support you. Please feel free to ask me anything about your medical PDFs - there are no silly questions, and I\'m here to listen. 💙'
#         })
    
#     if 'qa_chain' not in st.session_state:
#         st.session_state.qa_chain = None
    
#     # Display chat history
#     for message in st.session_state.messages:
#         with st.chat_message(message['role']):
#             st.markdown(message['content'])
    
#     # Chat input
#     if prompt := st.chat_input("Ask me anything about your medical documents... I'm here to help 💙"):
#         # Add user message to chat
#         st.session_state.messages.append({'role': 'user', 'content': prompt})
#         with st.chat_message('user'):
#             st.markdown(prompt)
        
#         # Generate response
#         with st.chat_message('assistant'):
#             with st.spinner('🔍 Looking through your documents to help answer your question...'):
#                 try:
#                     # Initialize QA chain if not already done
#                     if st.session_state.qa_chain is None:
#                         st.session_state.qa_chain = initialize_qa_chain()
                    
#                     if st.session_state.qa_chain is None:
#                         response_text = "❌ I'm having trouble accessing the medical documents right now. Please make sure everything is set up correctly, and I'll be ready to help you."
#                     else:
#                         # Get response from QA chain
#                         response = st.session_state.qa_chain.invoke({'query': prompt})
                        
#                         # Format the response
#                         result = response["result"]
#                         source_documents = response.get("source_documents", [])
                        
#                         # Combine result with formatted sources
#                         response_text = result
#                         if source_documents:
#                             response_text += format_sources(source_documents)
                    
#                     # Display response
#                     st.markdown(response_text)
                    
#                     # Add assistant response to chat history
#                     st.session_state.messages.append({
#                         'role': 'assistant', 
#                         'content': response_text
#                     })
                
#                 except Exception as e:
#                     error_msg = f"I'm sorry, I encountered an issue while trying to help you: {str(e)}\n\n"
#                     error_msg += "**Here's what you can try:**\n"
#                     error_msg += "1. Make sure Ollama is running (it should be!)\n"
#                     error_msg += "2. Try asking your question again\n"
#                     error_msg += "3. If the problem persists, please let someone know\n\n"
#                     error_msg += "I'm here to help once things are working again! 💙"
#                     st.error(error_msg)
#                     st.session_state.messages.append({
#                         'role': 'assistant', 
#                         'content': error_msg
#                     })

# if __name__ == "__main__":
#     # Check if vector store exists
#     if not os.path.exists(DB_FAISS_PATH):
#         st.error(f"❌ I can't find the medical documents database at {DB_FAISS_PATH}")
#         st.info("💡 Please run main.py first to process your PDF documents, then I'll be ready to help you!")
#     else:
#         main()# import os
# import streamlit as st
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA
# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import PromptTemplate
# from langchain_community.llms import Ollama

# # Configuration
# DB_FAISS_PATH = "vectorstore/db_faiss"

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
#         st.error(f"Error loading vector store: {str(e)}")
#         return None

# def set_custom_prompt(custom_prompt_template):
#     """Create a custom prompt template"""
#     prompt = PromptTemplate(
#         template=custom_prompt_template, 
#         input_variables=["context", "question"]
#     )
#     return prompt

# def initialize_qa_chain():
#     """Initialize the QA chain with local Ollama LLM"""
#     try:
#         # Load vector store
#         vectorstore = get_vectorstore()
#         if vectorstore is None:
#             return None
        
#         # Custom prompt template
#         CUSTOM_PROMPT_TEMPLATE = """You are a helpful medical assistant. Use the provided context to answer the user's question accurately.

# Important guidelines:
# - Only use information from the context provided
# - If you don't know the answer, say "I don't have enough information to answer that question"
# - Do not make up information
# - Be concise and clear
# - If the question is not related to the medical documents, politely say so

# Context: {context}

# Question: {question}

# Answer:"""
        
#         # Initialize Ollama LLM with your local model
#         llm = Ollama(
#             model="llama3.2:1b",  # Using your local Ollama model
#             temperature=0.1
#         )
        
#         st.success(f"✅ Using local Ollama model: llama3.2:1b")
        
#         # Create QA chain
#         qa_chain = RetrievalQA.from_chain_type(
#             llm=llm,
#             chain_type="stuff",
#             retriever=vectorstore.as_retriever(
#                 search_type="similarity",
#                 search_kwargs={'k': 3}
#             ),
#             return_source_documents=True,
#             chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
#         )
        
#         return qa_chain
    
#     except Exception as e:
#         st.error(f"Error initializing QA chain: {str(e)}")
#         import traceback
#         st.error(traceback.format_exc())
#         return None

# def format_sources(source_documents):
#     """Format source documents for display"""
#     if not source_documents:
#         return ""
    
#     sources_text = "\n\n---\n**📚 Source Documents:**\n"
#     for i, doc in enumerate(source_documents, 1):
#         source = doc.metadata.get('source', 'Unknown')
#         page = doc.metadata.get('page', 'Unknown')
#         sources_text += f"\n{i}. **{os.path.basename(source)}** (Page {page})\n"
#         # Add a snippet of the content
#         content_preview = doc.page_content[:200].replace('\n', ' ') + "..."
#         sources_text += f"   *Preview:* {content_preview}\n"
    
#     return sources_text

# def main():
#     # Page configuration
#     st.set_page_config(
#         page_title="Medical Chatbot",
#         page_icon="🏥",
#         layout="wide"
#     )
    
#     # Title and description
#     st.title("🏥 Medical Document Chatbot")
#     st.markdown("Ask questions about your medical documents using **Local Ollama (Llama 3.2)**!")
    
#     # Sidebar with information
#     with st.sidebar:
#         st.header("ℹ️ About")
#         st.info(
#             "This chatbot uses **Meta's Llama 3.2** model running locally via Ollama "
#             "to answer questions based on your medical PDF documents."
#         )
        
#         st.header("🤖 Model Info")
#         st.markdown("""
#         **Model:** Llama 3.2 1B
#         **Provider:** Ollama (Local!)
#         **No API Key Required** ✅
#         **Fast & Private** 🔒
#         """)
        
#         st.header("📋 Instructions")
#         st.markdown("""
#         1. Type your question in the chat input
#         2. The AI will search through your documents
#         3. You'll get an answer with source references
#         """)
        
#         st.header("⚙️ Available Models")
#         st.markdown("""
#         You can change the model in medibot.py:
#         - `llama3.2:1b` (current)
#         - `mistral:7b`
#         - `gemma3:latest`
#         """)
        
#         if st.button("🔄 Reload Vector Store"):
#             st.cache_resource.clear()
#             st.success("Cache cleared! Vector store will reload on next query.")
    
#     # Initialize session state for chat history
#     if 'messages' not in st.session_state:
#         st.session_state.messages = []
#         # Add welcome message
#         st.session_state.messages.append({
#             'role': 'assistant', 
#             'content': '👋 Hello! I\'m your medical document assistant powered by **Llama 3.2 running locally**. Ask me anything about your medical PDFs!'
#         })
    
#     if 'qa_chain' not in st.session_state:
#         st.session_state.qa_chain = None
    
#     # Display chat history
#     for message in st.session_state.messages:
#         with st.chat_message(message['role']):
#             st.markdown(message['content'])
    
#     # Chat input
#     if prompt := st.chat_input("Ask a question about your medical documents..."):
#         # Add user message to chat
#         st.session_state.messages.append({'role': 'user', 'content': prompt})
#         with st.chat_message('user'):
#             st.markdown(prompt)
        
#         # Generate response
#         with st.chat_message('assistant'):
#             with st.spinner('🔍 Searching through documents and generating answer...'):
#                 try:
#                     # Initialize QA chain if not already done
#                     if st.session_state.qa_chain is None:
#                         st.session_state.qa_chain = initialize_qa_chain()
                    
#                     if st.session_state.qa_chain is None:
#                         response_text = "❌ Failed to initialize the chatbot. Please check your configuration."
#                     else:
#                         # Get response from QA chain
#                         response = st.session_state.qa_chain.invoke({'query': prompt})
                        
#                         # Format the response
#                         result = response["result"]
#                         source_documents = response.get("source_documents", [])
                        
#                         # Combine result with formatted sources
#                         response_text = result
#                         if source_documents:
#                             response_text += format_sources(source_documents)
                    
#                     # Display response
#                     st.markdown(response_text)
                    
#                     # Add assistant response to chat history
#                     st.session_state.messages.append({
#                         'role': 'assistant', 
#                         'content': response_text
#                     })
                
#                 except Exception as e:
#                     error_msg = f"❌ Error: {str(e)}\n\n"
#                     error_msg += "**Tip:** Make sure Ollama is running:\n"
#                     error_msg += "1. Open a new terminal\n"
#                     error_msg += "2. Run: `ollama serve`\n"
#                     error_msg += "3. Try your question again"
#                     st.error(error_msg)
#                     st.session_state.messages.append({
#                         'role': 'assistant', 
#                         'content': error_msg
#                     })

# if __name__ == "__main__":
#     # Check if vector store exists
#     if not os.path.exists(DB_FAISS_PATH):
#         st.error(f"❌ Vector store not found at {DB_FAISS_PATH}")
#         st.info("Please run main.py first to create the vector store from your PDF documents.")
#     else:
#         main()# import os
# import streamlit as st
# from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
# from langchain.chains import RetrievalQA  # Updated import
# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import PromptTemplate
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# # Configuration
# DB_FAISS_PATH = "vectorstore/db_faiss"

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
#         st.error(f"Error loading vector store: {str(e)}")
#         return None

# def set_custom_prompt(custom_prompt_template):
#     """Create a custom prompt template"""
#     prompt = PromptTemplate(
#         template=custom_prompt_template, 
#         input_variables=["context", "question"]
#     )
#     return prompt

# def initialize_qa_chain():
#     """Initialize the QA chain with HuggingFace LLM"""
#     try:
#         # Check for API token
#         hf_token = os.environ.get("HF_TOKEN")
#         if not hf_token:
#             st.warning("⚠️ HF_TOKEN not found. Using public HuggingFace API (rate limited)")
#             st.info("For better performance, get a free token from: https://huggingface.co/settings/tokens")
#             hf_token = None  # Will use public API
#         else:
#             st.success(f"✅ HuggingFace token found: {hf_token[:10]}...")
        
#         # Load vector store
#         vectorstore = get_vectorstore()
#         if vectorstore is None:
#             return None
        
#         # Custom prompt template
#         CUSTOM_PROMPT_TEMPLATE = """You are a helpful medical assistant. Use the provided context to answer the user's question accurately.

# Important guidelines:
# - Only use information from the context provided
# - If you don't know the answer, say "I don't have enough information to answer that question"
# - Do not make up information
# - Be concise and clear
# - If the question is not related to the medical documents, politely say so

# Context: {context}

# Question: {question}

# Answer:"""
        
#         # Initialize HuggingFace LLM with Llama model
#         llm = HuggingFaceEndpoint(
#             repo_id="meta-llama/Llama-3.2-3B-Instruct",  # Free Llama 3.2 model
#             temperature=0.1,
#             max_new_tokens=512,
#             huggingfacehub_api_token=hf_token,
#         )
        
#         st.info(f"🤖 Using model: meta-llama/Llama-3.2-3B-Instruct")
        
#         # Create QA chain
#         qa_chain = RetrievalQA.from_chain_type(
#             llm=llm,
#             chain_type="stuff",
#             retriever=vectorstore.as_retriever(
#                 search_type="similarity",
#                 search_kwargs={'k': 3}
#             ),
#             return_source_documents=True,
#             chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
#         )
        
#         return qa_chain
    
#     except Exception as e:
#         st.error(f"Error initializing QA chain: {str(e)}")
#         import traceback
#         st.error(traceback.format_exc())
#         return None

# def format_sources(source_documents):
#     """Format source documents for display"""
#     if not source_documents:
#         return ""
    
#     sources_text = "\n\n---\n**📚 Source Documents:**\n"
#     for i, doc in enumerate(source_documents, 1):
#         source = doc.metadata.get('source', 'Unknown')
#         page = doc.metadata.get('page', 'Unknown')
#         sources_text += f"\n{i}. **{source}** (Page {page})\n"
#         # Add a snippet of the content
#         content_preview = doc.page_content[:200].replace('\n', ' ') + "..."
#         sources_text += f"   *Preview:* {content_preview}\n"
    
#     return sources_text

# def main():
#     # Page configuration
#     st.set_page_config(
#         page_title="Medical Chatbot",
#         page_icon="🏥",
#         layout="wide"
#     )
    
#     # Title and description
#     st.title("🏥 Medical Document Chatbot")
#     st.markdown("Ask questions about your medical documents using **Llama 3.2**!")
    
#     # Sidebar with information
#     with st.sidebar:
#         st.header("ℹ️ About")
#         st.info(
#             "This chatbot uses **Meta's Llama 3.2** model via HuggingFace to answer "
#             "questions based on your medical PDF documents."
#         )
        
#         st.header("🤖 Model Info")
#         st.markdown("""
#         **Model:** Llama 3.2 3B Instruct
#         **Provider:** HuggingFace (Free!)
#         **No API Key Required** (but recommended for better rate limits)
#         """)
        
#         st.header("📋 Instructions")
#         st.markdown("""
#         1. Type your question in the chat input
#         2. The AI will search through your documents
#         3. You'll get an answer with source references
#         """)
        
#         st.header("⚙️ Optional: Get Free Token")
#         st.markdown("""
#         For unlimited usage:
#         1. Visit [HuggingFace](https://huggingface.co/settings/tokens)
#         2. Create a free account
#         3. Generate an access token
#         4. Add to .env file: `HF_TOKEN=your_token`
#         """)
        
#         if st.button("🔄 Reload Vector Store"):
#             st.cache_resource.clear()
#             st.success("Cache cleared! Vector store will reload on next query.")
    
#     # Initialize session state for chat history
#     if 'messages' not in st.session_state:
#         st.session_state.messages = []
#         # Add welcome message
#         st.session_state.messages.append({
#             'role': 'assistant', 
#             'content': '👋 Hello! I\'m your medical document assistant powered by **Llama 3.2**. Ask me anything about your medical PDFs!'
#         })
    
#     if 'qa_chain' not in st.session_state:
#         st.session_state.qa_chain = None
    
#     # Display chat history
#     for message in st.session_state.messages:
#         with st.chat_message(message['role']):
#             st.markdown(message['content'])
    
#     # Chat input
#     if prompt := st.chat_input("Ask a question about your medical documents..."):
#         # Add user message to chat
#         st.session_state.messages.append({'role': 'user', 'content': prompt})
#         with st.chat_message('user'):
#             st.markdown(prompt)
        
#         # Generate response
#         with st.chat_message('assistant'):
#             with st.spinner('🔍 Searching through documents and generating answer...'):
#                 try:
#                     # Initialize QA chain if not already done
#                     if st.session_state.qa_chain is None:
#                         st.session_state.qa_chain = initialize_qa_chain()
                    
#                     if st.session_state.qa_chain is None:
#                         response_text = "❌ Failed to initialize the chatbot. Please check your configuration."
#                     else:
#                         # Get response from QA chain
#                         response = st.session_state.qa_chain.invoke({'query': prompt})
                        
#                         # Format the response
#                         result = response["result"]
#                         source_documents = response.get("source_documents", [])
                        
#                         # Combine result with formatted sources
#                         response_text = result
#                         if source_documents:
#                             response_text += format_sources(source_documents)
                    
#                     # Display response
#                     st.markdown(response_text)
                    
#                     # Add assistant response to chat history
#                     st.session_state.messages.append({
#                         'role': 'assistant', 
#                         'content': response_text
#                     })
                
#                 except Exception as e:
#                     error_msg = f"❌ Error: {str(e)}\n\n"
#                     error_msg += "**Tip:** If you see a rate limit error, you can:\n"
#                     error_msg += "1. Wait a few minutes and try again\n"
#                     error_msg += "2. Get a free HuggingFace token from https://huggingface.co/settings/tokens\n"
#                     error_msg += "3. Add it to your .env file as: `HF_TOKEN=your_token`"
#                     st.error(error_msg)
#                     st.session_state.messages.append({
#                         'role': 'assistant', 
#                         'content': error_msg
#                     })

# if __name__ == "__main__":
#     # Check if vector store exists
#     if not os.path.exists(DB_FAISS_PATH):
#         st.error(f"❌ Vector store not found at {DB_FAISS_PATH}")
#         st.info("Please run main.py first to create the vector store from your PDF documents.")
#     else:
#         main()# import os
# import streamlit as st
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA
# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import PromptTemplate
# from langchain_groq import ChatGroq
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# # Configuration
# DB_FAISS_PATH = "vectorstore/db_faiss"

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
#         st.error(f"Error loading vector store: {str(e)}")
#         return None

# def set_custom_prompt(custom_prompt_template):
#     """Create a custom prompt template"""
#     prompt = PromptTemplate(
#         template=custom_prompt_template, 
#         input_variables=["context", "question"]
#     )
#     return prompt

# def initialize_qa_chain():
#     """Initialize the QA chain with Groq LLM"""
#     try:
#         # Check for API key
#         groq_api_key = os.environ.get("GROQ_API_KEY")
#         if not groq_api_key:
#             st.error("⚠️ GROQ_API_KEY not found! Please add it to your .env file")
#             st.info("Get your API key from: https://console.groq.com/")
#             return None
        
#         # Load vector store
#         vectorstore = get_vectorstore()
#         if vectorstore is None:
#             return None
        
#         # Custom prompt template
#         CUSTOM_PROMPT_TEMPLATE = """
# You are a helpful medical assistant. Use the provided context to answer the user's question accurately.

# Important guidelines:
# - Only use information from the context provided
# - If you don't know the answer, say "I don't have enough information to answer that question"
# - Do not make up information
# - Be concise and clear
# - If the question is not related to the medical documents, politely say so

# Context: {context}

# Question: {question}

# Answer:"""
        
#         # Initialize Groq LLM with correct model name
#         llm = ChatGroq(
#             model="llama-3.3-70b-versatile",  # Correct free Groq model
#             temperature=0.1,
#             groq_api_key=groq_api_key,
#             max_tokens=1024
#         )
        
#         # Create QA chain
#         qa_chain = RetrievalQA.from_chain_type(
#             llm=llm,
#             chain_type="stuff",
#             retriever=vectorstore.as_retriever(
#                 search_type="similarity",
#                 search_kwargs={'k': 3}
#             ),
#             return_source_documents=True,
#             chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
#         )
        
#         return qa_chain
    
#     except Exception as e:
#         st.error(f"Error initializing QA chain: {str(e)}")
#         return None

# def format_sources(source_documents):
#     """Format source documents for display"""
#     if not source_documents:
#         return ""
    
#     sources_text = "\n\n---\n**📚 Source Documents:**\n"
#     for i, doc in enumerate(source_documents, 1):
#         source = doc.metadata.get('source', 'Unknown')
#         page = doc.metadata.get('page', 'Unknown')
#         sources_text += f"\n{i}. **{source}** (Page {page})\n"
#         # Add a snippet of the content
#         content_preview = doc.page_content[:200].replace('\n', ' ') + "..."
#         sources_text += f"   *Preview:* {content_preview}\n"
    
#     return sources_text

# def main():
#     # Page configuration
#     st.set_page_config(
#         page_title="Medical Chatbot",
#         page_icon="🏥",
#         layout="wide"
#     )
    
#     # Title and description
#     st.title("🏥 Medical Document Chatbot")
#     st.markdown("Ask questions about your medical documents!")
    
#     # Sidebar with information
#     with st.sidebar:
#         st.header("ℹ️ About")
#         st.info(
#             "This chatbot uses AI to answer questions based on your medical PDF documents. "
#             "The documents have been processed and stored in a vector database for quick retrieval."
#         )
        
#         st.header("📋 Instructions")
#         st.markdown("""
#         1. Type your question in the chat input
#         2. The AI will search through your documents
#         3. You'll get an answer with source references
#         """)
        
#         st.header("⚙️ Settings")
#         if st.button("🔄 Reload Vector Store"):
#             st.cache_resource.clear()
#             st.success("Cache cleared! Vector store will reload on next query.")
    
#     # Initialize session state for chat history
#     if 'messages' not in st.session_state:
#         st.session_state.messages = []
#         # Add welcome message
#         st.session_state.messages.append({
#             'role': 'assistant', 
#             'content': '👋 Hello! I\'m your medical document assistant. Ask me anything about your medical PDFs!'
#         })
    
#     if 'qa_chain' not in st.session_state:
#         st.session_state.qa_chain = None
    
#     # Display chat history
#     for message in st.session_state.messages:
#         with st.chat_message(message['role']):
#             st.markdown(message['content'])
    
#     # Chat input
#     if prompt := st.chat_input("Ask a question about your medical documents..."):
#         # Add user message to chat
#         st.session_state.messages.append({'role': 'user', 'content': prompt})
#         with st.chat_message('user'):
#             st.markdown(prompt)
        
#         # Generate response
#         with st.chat_message('assistant'):
#             with st.spinner('🔍 Searching through documents...'):
#                 try:
#                     # Initialize QA chain if not already done
#                     if st.session_state.qa_chain is None:
#                         st.session_state.qa_chain = initialize_qa_chain()
                    
#                     if st.session_state.qa_chain is None:
#                         response_text = "❌ Failed to initialize the chatbot. Please check your configuration."
#                     else:
#                         # Get response from QA chain
#                         response = st.session_state.qa_chain.invoke({'query': prompt})
                        
#                         # Format the response
#                         result = response["result"]
#                         source_documents = response.get("source_documents", [])
                        
#                         # Combine result with formatted sources
#                         response_text = result
#                         if source_documents:
#                             response_text += format_sources(source_documents)
                    
#                     # Display response
#                     st.markdown(response_text)
                    
#                     # Add assistant response to chat history
#                     st.session_state.messages.append({
#                         'role': 'assistant', 
#                         'content': response_text
#                     })
                
#                 except Exception as e:
#                     error_msg = f"❌ Error: {str(e)}"
#                     st.error(error_msg)
#                     st.session_state.messages.append({
#                         'role': 'assistant', 
#                         'content': error_msg
#                     })

# if __name__ == "__main__":
#     # Check if vector store exists
#     if not os.path.exists(DB_FAISS_PATH):
#         st.error(f"❌ Vector store not found at {DB_FAISS_PATH}")
#         st.info("Please run main.py first to create the vector store from your PDF documents.")
#     else:
#         main()