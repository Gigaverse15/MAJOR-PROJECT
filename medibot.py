import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama

# Configuration
DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    """Load the FAISS vector store with embeddings"""
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
        st.error(f"Error loading vector store: {str(e)}")
        return None

def set_custom_prompt(custom_prompt_template):
    """Create a custom prompt template"""
    prompt = PromptTemplate(
        template=custom_prompt_template, 
        input_variables=["context", "question"]
    )
    return prompt

def initialize_qa_chain():
    """Initialize the QA chain with local Ollama LLM"""
    try:
        vectorstore = get_vectorstore()
        if vectorstore is None:
            return None
        
        CUSTOM_PROMPT_TEMPLATE = """You are a caring and empathetic medical assistant. Your role is to help patients understand their medical documents with compassion and support.

Important guidelines:
- Use warm, supportive, and reassuring language
- Acknowledge the patient's concerns with empathy and understanding
- ALWAYS share relevant information from the context/documents provided
- Explain what the medical documents say about symptoms, conditions, and treatments
- Help patients understand their medical information clearly
- When patients describe symptoms, tell them what the documents say about those symptoms
- If the patient lists symptoms, check if those symptoms are mentioned in the documents and explain what the documents say
- You CAN and SHOULD provide medical information from the documents
- You CANNOT diagnose (don't say "you have cancer"), but you CAN say "according to the documents, these symptoms are associated with..."
- Never refuse to help or say "I cannot provide medical advice" - instead, share what the documents say and encourage them to discuss with their doctor
- Be encouraging and supportive while remaining accurate and honest
- ALWAYS end by encouraging them to discuss findings with their healthcare provider for proper evaluation
- Show that you care about their emotional state, not just the medical facts

Context: {context}

Question: {question}

Please provide a compassionate, empathetic, and informative answer based on what the documents say:"""
        
        llm = Ollama(
            model="llama3.2:1b",
            temperature=0.3
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={'k': 3}
            ),
            return_source_documents=True,
            chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
        )
        
        return qa_chain
    
    except Exception as e:
        st.error(f"Error initializing QA chain: {str(e)}")
        return None

def format_sources(source_documents):
    """Format source documents for display"""
    if not source_documents:
        return ""
    
    sources_text = "\n\n---\n**📚 Source Documents:**\n"
    for i, doc in enumerate(source_documents, 1):
        source = doc.metadata.get('source', 'Unknown')
        page = doc.metadata.get('page', 'Unknown')
        sources_text += f"\n{i}. **{os.path.basename(source)}** (Page {page})\n"
        content_preview = doc.page_content[:200].replace('\n', ' ') + "..."
        sources_text += f"   *Preview:* {content_preview}\n"
    
    return sources_text

def text_chat_tab():
    """Regular text chat interface"""
    st.markdown("### 💬 Text Chat")
    st.markdown("*Ask your questions by typing below*")
    
    # Initialize session state for text chat
    if 'text_messages' not in st.session_state:
        st.session_state.text_messages = []
        st.session_state.text_messages.append({
            'role': 'assistant', 
            'content': '👋 Hello! I\'m here to help you understand your medical documents with care and compassion. Please feel free to ask me anything about your medical PDFs! 💙'
        })
    
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None
    
    # Display chat history
    for message in st.session_state.text_messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about your medical documents... 💙"):
        # Add user message
        st.session_state.text_messages.append({'role': 'user', 'content': prompt})
        with st.chat_message('user'):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message('assistant'):
            with st.spinner('🔍 Looking through your documents...'):
                try:
                    if st.session_state.qa_chain is None:
                        st.session_state.qa_chain = initialize_qa_chain()
                        if st.session_state.qa_chain:
                            st.success("✅ Using local Ollama model: llama3.2:1b")
                    
                    if st.session_state.qa_chain is None:
                        response_text = "❌ Failed to initialize the chatbot."
                    else:
                        response = st.session_state.qa_chain.invoke({'query': prompt})
                        result = response["result"]
                        source_documents = response.get("source_documents", [])
                        
                        response_text = result
                        if source_documents:
                            response_text += format_sources(source_documents)
                    
                    st.markdown(response_text)
                    st.session_state.text_messages.append({
                        'role': 'assistant', 
                        'content': response_text
                    })
                
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.text_messages.append({
                        'role': 'assistant', 
                        'content': error_msg
                    })

def voice_chat_tab():
    """Voice and multilingual chat interface"""
    st.markdown("### 🎙️ Voice & Multilingual Chat")
    st.markdown("*Speak or type in any language*")
    
    # Check if required packages are installed
    try:
        from audio_recorder_streamlit import audio_recorder
        import speech_recognition as sr
        from googletrans import Translator
        from gtts import gTTS
        import tempfile
        from io import BytesIO
        
        voice_available = True
    except ImportError as e:
        st.error("❌ Voice features not available. Please install required packages:")
        st.code("pip install SpeechRecognition audio-recorder-streamlit gTTS googletrans==4.0.0rc1 pyaudio")
        st.info("After installing, restart the app to use voice features.")
        voice_available = False
        return
    
    # Initialize translator
    translator = Translator()
    
    # Language selection
    col1, col2 = st.columns(2)
    
    with col1:
        languages = {
            'English': 'en', 'Spanish': 'es', 'French': 'fr', 'German': 'de',
            'Italian': 'it', 'Portuguese': 'pt', 'Russian': 'ru', 'Japanese': 'ja',
            'Korean': 'ko', 'Chinese': 'zh-cn', 'Arabic': 'ar', 'Hindi': 'hi',
            'Bengali': 'bn', 'Urdu': 'ur', 'Turkish': 'tr'
        }
        
        selected_lang = st.selectbox(
            "🌐 Select Your Language:",
            options=list(languages.keys()),
            index=0
        )
        user_lang = languages[selected_lang]
    
    with col2:
        enable_voice_output = st.checkbox("🔊 Enable Voice Responses", value=True)
    
    st.markdown("---")
    
    # Initialize session state for voice chat
    if 'voice_messages' not in st.session_state:
        st.session_state.voice_messages = []
        welcome_en = "👋 Hello! I'm your multilingual medical assistant. Ask me anything in your language!"
        
        try:
            welcome_translated = translator.translate(welcome_en, dest=user_lang).text if user_lang != 'en' else welcome_en
        except:
            welcome_translated = welcome_en
        
        st.session_state.voice_messages.append({
            'role': 'assistant', 
            'content': f"{welcome_en}\n\n*{welcome_translated}*" if user_lang != 'en' else welcome_en
        })
    
    # Display chat history
    for message in st.session_state.voice_messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
    
    # Voice input
    st.markdown("#### 🎙️ Voice Input")
    audio_bytes = audio_recorder(
        text="Click to record your question",
        recording_color="#e74c3c",
        neutral_color="#3498db",
        icon_size="2x"
    )
    
    prompt = None
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        with st.spinner("🎧 Converting speech to text..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(audio_bytes)
                    tmp_file_path = tmp_file.name
                
                recognizer = sr.Recognizer()
                with sr.AudioFile(tmp_file_path) as source:
                    audio_data = recognizer.record(source)
                    prompt = recognizer.recognize_google(audio_data)
                
                os.unlink(tmp_file_path)
                st.success(f"📝 You said: {prompt}")
            except Exception as e:
                st.error(f"❌ Could not understand audio: {str(e)}")
                prompt = None
    
    # Text input
    if not prompt:
        st.markdown("#### 💬 Or Type Your Question")
        prompt = st.chat_input("Type in any language... / اكتب / 在这里输入 / Escribe...")
    
    if prompt:
        # Detect and translate
        try:
            detected = translator.detect(prompt)
            detected_lang = detected.lang
            
            if detected_lang != 'en':
                prompt_en = translator.translate(prompt, dest='en').text
                st.info(f"🌐 Detected: {detected_lang.upper()} → Translating to English")
            else:
                prompt_en = prompt
        except:
            prompt_en = prompt
            detected_lang = 'en'
        
        # Display user message
        st.session_state.voice_messages.append({'role': 'user', 'content': prompt})
        with st.chat_message('user'):
            st.markdown(prompt)
            if detected_lang != 'en':
                st.caption(f"*English: {prompt_en}*")
        
        # Generate response
        with st.chat_message('assistant'):
            with st.spinner('🔍 Searching documents...'):
                try:
                    if st.session_state.qa_chain is None:
                        st.session_state.qa_chain = initialize_qa_chain()
                    
                    if st.session_state.qa_chain is None:
                        response_text = "❌ Failed to initialize chatbot."
                    else:
                        response = st.session_state.qa_chain.invoke({'query': prompt_en})
                        result_en = response["result"]
                        source_documents = response.get("source_documents", [])
                        
                        response_text_en = result_en
                        if source_documents:
                            response_text_en += format_sources(source_documents)
                        
                        # Show English
                        if detected_lang != 'en':
                            st.markdown("**📄 English Response:**")
                            st.markdown(response_text_en)
                            st.markdown(f"\n---\n**🌐 Translation to {selected_lang}:**")
                            
                            try:
                                result_translated = translator.translate(result_en, dest=detected_lang).text
                                st.markdown(result_translated)
                                response_text = f"{response_text_en}\n\n---\n**{selected_lang}:**\n{result_translated}"
                            except:
                                st.warning("Translation failed, showing English only")
                                response_text = response_text_en
                        else:
                            st.markdown(response_text_en)
                            response_text = response_text_en
                        
                        # Voice output
                        if enable_voice_output:
                            with st.spinner("🔊 Generating voice..."):
                                try:
                                    voice_text = result_translated if detected_lang != 'en' else result_en
                                    voice_lang = detected_lang if detected_lang != 'en' else 'en'
                                    
                                    tts = gTTS(text=voice_text, lang=voice_lang, slow=False)
                                    fp = BytesIO()
                                    tts.write_to_fp(fp)
                                    fp.seek(0)
                                    st.audio(fp, format='audio/mp3')
                                except Exception as e:
                                    st.warning(f"Voice generation failed: {str(e)}")
                    
                    st.session_state.voice_messages.append({
                        'role': 'assistant', 
                        'content': response_text
                    })
                
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)

def main():
    st.set_page_config(
        page_title="Medical Chatbot",
        page_icon="🏥",
        layout="wide"
    )
    
    st.title("🏥 Compassionate Medical Assistant")
    st.markdown("**A caring AI assistant to help you understand your medical documents** 💙")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.info(
            "This chatbot uses **Meta's Llama 3.2** model running locally via Ollama "
            "to answer questions about your medical documents with empathy and care."
        )
        
        st.header("🤖 Model Info")
        st.markdown("""
        **Model:** Llama 3.2 1B
        **Provider:** Ollama (Local!)
        **No API Key Required** ✅
        **Fast & Private** 🔒
        **Empathetic Responses** 💙
        """)
        
        st.header("📋 Features")
        st.markdown("""
        **💬 Text Chat:**
        - Type your questions
        - Fast text responses
        - Source citations
        
        **🎙️ Voice & Multilingual:**
        - Speak your questions
        - 100+ languages supported
        - Voice responses
        - Auto translation
        """)
        
        st.header("⚙️ Available Models")
        st.markdown("""
        Change model in code line 77:
        - `llama3.2:1b` (current)
        - `mistral:7b` (more detailed)
        - `gemma3:latest`
        """)
        
        if st.button("🔄 Reload Vector Store"):
            st.cache_resource.clear()
            st.success("Cache cleared!")
    
    # Main content with tabs
    tab1, tab2 = st.tabs(["💬 Text Chat", "🎙️ Voice & Multilingual"])
    
    with tab1:
        text_chat_tab()
    
    with tab2:
        voice_chat_tab()

if __name__ == "__main__":
    if not os.path.exists(DB_FAISS_PATH):
        st.error(f"❌ Vector store not found at {DB_FAISS_PATH}")
        st.info("💡 Please run main.py first to process your PDF documents!")
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