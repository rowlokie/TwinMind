import os
import io
import json
import uuid
import tempfile
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from mirror.engine import MirrorEngine

app = FastAPI(
    title="MirrorGPT",
    description="Autonomous AI Digital Twin & Personal Alignment Engine",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine instance
engine = MirrorEngine(subject_name="Piyush", title="AI/ML Engineer & Agentic Systems Developer")

# Request Models
class ChatRequest(BaseModel):
    message: str
    provider: Optional[str] = "demo"
    api_key: Optional[str] = None
    model_name: Optional[str] = None

class ProfileUpdateRequest(BaseModel):
    subject_name: str
    title: str
    bio: str
    speaking_style: str

class FactCreateRequest(BaseModel):
    fact: str
    category: Optional[str] = "manual"

class ExtractTextRequest(BaseModel):
    text: str
    provider: Optional[str] = "demo"
    api_key: Optional[str] = None

class InterviewAnswerRequest(BaseModel):
    answer: str

class TTSRequest(BaseModel):
    text: str
    api_key: Optional[str] = None
    voice_id: Optional[str] = None


@app.get("/api/profile")
async def get_profile():
    return {
        "subject_name": engine.subject_name,
        "title": engine.title,
        "bio": engine.bio,
        "speaking_style": engine.speaking_style,
        "total_facts": len(engine.facts)
    }

@app.post("/api/profile")
async def update_profile(req: ProfileUpdateRequest):
    engine.subject_name = req.subject_name
    engine.title = req.title
    engine.bio = req.bio
    engine.speaking_style = req.speaking_style
    return {"status": "success", "profile": await get_profile()}

@app.post("/api/chat")
async def chat_with_mirror(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    result = engine.generate_response(
        user_message=req.message,
        api_key=req.api_key,
        provider=req.provider or "demo",
        model_name=req.model_name
    )
    return result

@app.get("/api/facts")
async def list_facts(q: Optional[str] = None):
    if q:
        return engine.search_facts(q, top_k=20)
    return engine.facts

@app.post("/api/facts")
async def add_fact(req: FactCreateRequest):
    if not req.fact.strip():
        raise HTTPException(status_code=400, detail="Fact cannot be empty")
    created = engine.add_fact(req.fact, category=req.category or "manual", source="manual_entry")
    return {"status": "success", "fact": created, "total_facts": len(engine.facts)}

@app.delete("/api/facts/{fact_id}")
async def delete_fact(fact_id: str):
    success = engine.delete_fact(fact_id)
    if not success:
        raise HTTPException(status_code=404, detail="Fact not found")
    return {"status": "success", "total_facts": len(engine.facts)}

@app.post("/api/facts/clear")
async def clear_facts():
    engine.clear_facts()
    return {"status": "success", "total_facts": 0}

@app.post("/api/upload")
async def upload_document(
    file: UploadFile = File(...),
    provider: Optional[str] = Form("demo"),
    api_key: Optional[str] = Form(None)
):
    filename = file.filename.lower()
    content_bytes = await file.read()
    extracted_text = ""

    if filename.endswith(".pdf"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(content_bytes)
            tmp_path = tmp.name
        try:
            extracted_text = engine.parse_pdf_file(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    elif filename.endswith((".txt", ".md", ".json", ".csv")):
        extracted_text = content_bytes.decode("utf-8", errors="ignore")
    else:
        raise HTTPException(status_code=400, detail="Supported formats: .pdf, .txt, .md, .json, .csv")

    if not extracted_text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from document")

    facts = engine.extract_facts_from_text(
        text=extracted_text,
        source=file.filename,
        api_key=api_key,
        provider=provider or "demo"
    )

    return {
        "status": "success",
        "filename": file.filename,
        "facts_extracted": len(facts),
        "total_facts": len(engine.facts),
        "sample_facts": facts[:5]
    }

@app.post("/api/extract_text")
async def extract_text(req: ExtractTextRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    facts = engine.extract_facts_from_text(
        text=req.text,
        source="pasted_text",
        api_key=req.api_key,
        provider=req.provider or "demo"
    )
    return {
        "status": "success",
        "facts_extracted": len(facts),
        "total_facts": len(engine.facts),
        "sample_facts": facts[:5]
    }

@app.get("/api/interview")
async def get_interview_status():
    has_more = engine.current_interview_idx < len(engine.interview_questions)
    current_q = engine.interview_questions[engine.current_interview_idx] if has_more else None
    return {
        "current_step": engine.current_interview_idx + 1,
        "total_steps": len(engine.interview_questions),
        "question": current_q,
        "completed": not has_more,
        "total_facts": len(engine.facts)
    }

@app.post("/api/interview/answer")
async def answer_interview(req: InterviewAnswerRequest):
    result = engine.process_interview_answer(req.answer)
    return result

@app.post("/api/interview/reset")
async def reset_interview():
    engine.reset_interview()
    return await get_interview_status()

@app.post("/api/preset/{preset_name}")
async def switch_preset(preset_name: str):
    engine.load_preset(preset_name)
    return {
        "status": "success",
        "preset": preset_name,
        "profile": await get_profile()
    }


# HTML Frontend
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>MirrorGPT — Autonomous AI Digital Twin</title>
  <!-- Tailwind CSS -->
  <script src="https://cdn.tailwindcss.com"></script>
  <!-- Lucide Icons -->
  <script src="https://unpkg.com/lucide@latest"></script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    body {
      font-family: 'Plus Jakarta Sans', sans-serif;
      background-color: #090d16;
    }
    .glass-card {
      background: rgba(16, 23, 42, 0.75);
      backdrop-filter: blur(16px);
      border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .glass-card-hover:hover {
      border-color: rgba(99, 102, 241, 0.4);
    }
    .custom-scroll::-webkit-scrollbar {
      width: 6px;
    }
    .custom-scroll::-webkit-scrollbar-track {
      background: rgba(15, 23, 42, 0.5);
    }
    .custom-scroll::-webkit-scrollbar-thumb {
      background: rgba(99, 102, 241, 0.3);
      border-radius: 9999px;
    }
    .glow-accent {
      box-shadow: 0 0 25px -5px rgba(99, 102, 241, 0.3);
    }
    .pulse-mic {
      animation: pulse-ring 1.5s cubic-bezier(0.215, 0.61, 0.355, 1) infinite;
    }
    @keyframes pulse-ring {
      0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
      70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
      100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }
  </style>
</head>
<body class="text-slate-100 min-h-screen flex flex-col antialiased selection:bg-indigo-500 selection:text-white">

  <!-- Header -->
  <header class="border-b border-slate-800/80 bg-slate-950/60 sticky top-0 z-40 backdrop-blur-md">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
      <div class="flex items-center space-x-3">
        <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 via-indigo-500 to-purple-500 flex items-center justify-center text-white shadow-lg shadow-indigo-500/25">
          <i data-lucide="sparkles" class="w-5 h-5"></i>
        </div>
        <div>
          <div class="flex items-center space-x-2">
            <span class="text-xl font-extrabold tracking-tight bg-gradient-to-r from-white via-slate-200 to-indigo-300 bg-clip-text text-transparent">MirrorGPT</span>
            <span class="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded-full bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">v2.0</span>
          </div>
          <p class="text-xs text-slate-400">Autonomous AI Digital Twin Engine</p>
        </div>
      </div>

      <!-- Presets & Controls -->
      <div class="flex items-center space-x-3">
        <div class="hidden sm:flex items-center space-x-2 bg-slate-900/90 p-1 rounded-xl border border-slate-800">
          <button onclick="switchPreset('piyush')" id="btn-preset-piyush" class="px-3 py-1 text-xs font-semibold rounded-lg bg-indigo-600 text-white shadow-sm transition">
            Piyush (AI Dev)
          </button>
          <button onclick="switchPreset('thomas')" id="btn-preset-thomas" class="px-3 py-1 text-xs font-semibold rounded-lg text-slate-400 hover:text-white transition">
            Thomas (Manager)
          </button>
        </div>

        <button onclick="openSettingsModal()" class="flex items-center space-x-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold bg-slate-900 border border-slate-700/70 hover:bg-slate-800 text-slate-200 transition">
          <i data-lucide="sliders-horizontal" class="w-3.5 h-3.5"></i>
          <span>Settings</span>
        </button>
      </div>
    </div>
  </header>

  <!-- Main Container -->
  <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 flex-1 w-full grid grid-cols-1 lg:grid-cols-12 gap-6">

    <!-- Left Column: Active Persona Overview -->
    <aside class="lg:col-span-4 flex flex-col space-y-5">
      
      <!-- Persona Card -->
      <div class="glass-card rounded-2xl p-5 relative overflow-hidden">
        <div class="absolute -right-10 -top-10 w-32 h-32 bg-indigo-500/10 rounded-full blur-2xl pointer-events-none"></div>
        <div class="flex items-start justify-between">
          <div class="flex items-center space-x-3">
            <div id="avatar-container" class="w-14 h-14 rounded-2xl bg-gradient-to-tr from-indigo-500 to-purple-600 flex items-center justify-center text-xl font-black text-white shadow-md shadow-indigo-500/30">
              P
            </div>
            <div>
              <h2 id="card-subject-name" class="text-lg font-bold text-white">Piyush</h2>
              <p id="card-subject-title" class="text-xs text-indigo-400 font-medium">AI/ML Engineer</p>
            </div>
          </div>
          <span class="inline-flex items-center px-2 py-0.5 rounded-md text-[11px] font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
            <span class="w-1.5 h-1.5 rounded-full bg-emerald-400 mr-1.5 animate-pulse"></span>
            Mirror Active
          </span>
        </div>

        <p id="card-subject-bio" class="text-xs text-slate-300 mt-4 leading-relaxed">
          Specializing in autonomous LLM agents, RAG architectures, and intelligent systems.
        </p>

        <!-- Stats Grid -->
        <div class="grid grid-cols-2 gap-3 mt-4 pt-4 border-t border-slate-800">
          <div class="bg-slate-900/60 p-2.5 rounded-xl border border-slate-800/80">
            <span class="text-[10px] text-slate-400 uppercase tracking-wider font-semibold">Indexed Facts</span>
            <p id="card-fact-count" class="text-lg font-bold text-white mt-0.5">7</p>
          </div>
          <div class="bg-slate-900/60 p-2.5 rounded-xl border border-slate-800/80">
            <span class="text-[10px] text-slate-400 uppercase tracking-wider font-semibold">Inference Model</span>
            <p id="card-active-model" class="text-xs font-semibold text-indigo-300 mt-1 truncate">Local Neural</p>
          </div>
        </div>

        <div class="mt-4 flex items-center justify-between text-xs text-slate-400 bg-slate-950/40 px-3 py-2 rounded-xl border border-slate-800/60">
          <div class="flex items-center space-x-2">
            <i data-lucide="volume-2" class="w-3.5 h-3.5 text-indigo-400"></i>
            <span>Voice Mode</span>
          </div>
          <span id="voice-indicator" class="text-emerald-400 font-medium">Browser Native</span>
        </div>
      </div>

      <!-- Quick Test Questions -->
      <div class="glass-card rounded-2xl p-5">
        <h3 class="text-xs uppercase font-bold text-slate-400 tracking-wider flex items-center space-x-1.5">
          <i data-lucide="help-circle" class="w-3.5 h-3.5 text-indigo-400"></i>
          <span>Sample Inquiries</span>
        </h3>
        <p class="text-[11px] text-slate-400 mt-1">Click any chip to prompt the mirror in first person:</p>
        <div class="flex flex-col space-y-2 mt-3">
          <button onclick="sendQuickPrompt('Who are you and what do you do?')" class="text-left text-xs p-2.5 rounded-xl bg-slate-900/80 hover:bg-indigo-600/20 border border-slate-800 hover:border-indigo-500/40 text-slate-300 hover:text-white transition">
            💬 "Who are you and what do you do?"
          </button>
          <button onclick="sendQuickPrompt('What are your top technical skills and frameworks?')" class="text-left text-xs p-2.5 rounded-xl bg-slate-900/80 hover:bg-indigo-600/20 border border-slate-800 hover:border-indigo-500/40 text-slate-300 hover:text-white transition">
            🛠️ "What are your top technical skills?"
          </button>
          <button onclick="sendQuickPrompt('Tell me about a major project you built.')" class="text-left text-xs p-2.5 rounded-xl bg-slate-900/80 hover:bg-indigo-600/20 border border-slate-800 hover:border-indigo-500/40 text-slate-300 hover:text-white transition">
            🚀 "Tell me about a major project you built."
          </button>
          <button onclick="sendQuickPrompt('What is your communication and work philosophy?')" class="text-left text-xs p-2.5 rounded-xl bg-slate-900/80 hover:bg-indigo-600/20 border border-slate-800 hover:border-indigo-500/40 text-slate-300 hover:text-white transition">
            🧠 "What is your work philosophy?"
          </button>
        </div>
      </div>

    </aside>

    <!-- Right Column: Interactive Studio & Tabs -->
    <section class="lg:col-span-8 flex flex-col space-y-4">
      
      <!-- Tab Navigation -->
      <div class="flex items-center space-x-2 bg-slate-950/80 p-1.5 rounded-2xl border border-slate-800/80">
        <button onclick="switchTab('chat')" id="tab-btn-chat" class="flex-1 py-2 px-3 text-xs font-semibold rounded-xl bg-indigo-600 text-white shadow-sm flex items-center justify-center space-x-2 transition">
          <i data-lucide="message-square" class="w-4 h-4"></i>
          <span>Live Chat & Voice</span>
        </button>
        <button onclick="switchTab('teach')" id="tab-btn-teach" class="flex-1 py-2 px-3 text-xs font-semibold rounded-xl text-slate-400 hover:text-white flex items-center justify-center space-x-2 transition">
          <i data-lucide="book-open" class="w-4 h-4"></i>
          <span>Teach & Ingest</span>
        </button>
        <button onclick="switchTab('memory')" id="tab-btn-memory" class="flex-1 py-2 px-3 text-xs font-semibold rounded-xl text-slate-400 hover:text-white flex items-center justify-center space-x-2 transition">
          <i data-lucide="database" class="w-4 h-4"></i>
          <span>Knowledge Bank (<span id="tab-fact-badge">7</span>)</span>
        </button>
      </div>

      <!-- VIEW 1: LIVE CHAT & VOICE -->
      <div id="view-chat" class="glass-card rounded-2xl p-4 flex flex-col h-[620px]">
        
        <!-- Chat History -->
        <div id="chat-messages" class="flex-1 overflow-y-auto custom-scroll space-y-4 pr-2">
          <!-- Welcome message -->
          <div class="flex items-start space-x-3">
            <div class="w-8 h-8 rounded-xl bg-indigo-600 flex items-center justify-center text-xs font-bold text-white flex-shrink-0 mt-0.5">
              🪞
            </div>
            <div class="bg-slate-900 border border-slate-800 p-3.5 rounded-2xl rounded-tl-sm max-w-[85%]">
              <p class="text-xs text-slate-200 leading-relaxed">
                Hi! I am your personalized AI Mirror. I speak in the first person representing your thoughts, background, and knowledge. Ask me anything, or try testing my voice output!
              </p>
              <div class="mt-2.5 flex items-center space-x-2">
                <button onclick="speakText(this.closest('.bg-slate-900').querySelector('p').innerText)" class="px-2 py-1 rounded-md text-[11px] bg-slate-800 hover:bg-slate-700 text-indigo-300 flex items-center space-x-1 transition">
                  <i data-lucide="volume-2" class="w-3 h-3"></i>
                  <span>Listen</span>
                </button>
              </div>
            </div>
          </div>
        </div>

        <!-- Chat Input Form -->
        <div class="mt-4 pt-3 border-t border-slate-800/80">
          <div class="relative flex items-center">
            <input 
              type="text" 
              id="chat-input" 
              placeholder="Ask the Mirror anything about its background or skills..." 
              onkeydown="if(event.key === 'Enter') sendMessage()"
              class="w-full bg-slate-900/90 border border-slate-700/80 rounded-2xl pl-4 pr-28 py-3.5 text-xs text-slate-100 placeholder-slate-500 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition"
            />
            
            <div class="absolute right-2 flex items-center space-x-1.5">
              <!-- Voice Input Button -->
              <button 
                id="mic-btn" 
                onclick="toggleSpeechRecognition()" 
                title="Speak to Mirror (Microphone)"
                class="w-8 h-8 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white flex items-center justify-center transition"
              >
                <i data-lucide="mic" class="w-4 h-4"></i>
              </button>
              
              <!-- Send Button -->
              <button 
                onclick="sendMessage()" 
                id="send-btn"
                class="px-3.5 py-1.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold flex items-center space-x-1 shadow-md shadow-indigo-600/30 transition"
              >
                <span>Send</span>
                <i data-lucide="send" class="w-3 h-3"></i>
              </button>
            </div>
          </div>
          <div class="flex items-center justify-between text-[11px] text-slate-500 mt-2 px-1">
            <span class="flex items-center space-x-1">
              <i data-lucide="shield-check" class="w-3 h-3 text-emerald-400"></i>
              <span>First-person alignment active</span>
            </span>
            <label class="flex items-center space-x-1.5 cursor-pointer text-slate-400 hover:text-slate-300">
              <input type="checkbox" id="auto-speak-toggle" class="rounded bg-slate-800 border-slate-700 text-indigo-600 focus:ring-0">
              <span>Auto-play audio response</span>
            </label>
          </div>
        </div>

      </div>

      <!-- VIEW 2: TEACH & INGEST -->
      <div id="view-teach" class="glass-card rounded-2xl p-6 hidden space-y-6">
        <div>
          <h3 class="text-base font-bold text-white flex items-center space-x-2">
            <i data-lucide="file-up" class="w-4 h-4 text-indigo-400"></i>
            <span>Upload Documents (Resume / Bio / PDF)</span>
          </h3>
          <p class="text-xs text-slate-400 mt-1">Upload your resume, LinkedIn PDF export, or notes. The system extracts atomic facts into memory automatically.</p>
        </div>

        <!-- Drag & Drop Upload Zone -->
        <div 
          id="dropzone" 
          onclick="document.getElementById('file-input').click()"
          class="border-2 border-dashed border-slate-700 hover:border-indigo-500/60 rounded-2xl p-8 flex flex-col items-center justify-center text-center cursor-pointer bg-slate-900/40 hover:bg-indigo-950/10 transition"
        >
          <input type="file" id="file-input" class="hidden" accept=".pdf,.txt,.md,.json" onchange="handleFileUpload(event)">
          <div class="w-12 h-12 rounded-2xl bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center text-indigo-400 mb-3">
            <i data-lucide="upload-cloud" class="w-6 h-6"></i>
          </div>
          <p class="text-xs font-semibold text-slate-200">Click or drag & drop documents here</p>
          <p class="text-[11px] text-slate-500 mt-1">Supports PDF, TXT, MD, JSON (Max 10MB)</p>
          <span id="upload-status" class="text-xs text-indigo-400 font-medium mt-3 hidden">Extracting facts...</span>
        </div>

        <!-- Quick Paste Text -->
        <div class="space-y-2">
          <label class="text-xs font-semibold text-slate-300">Or Paste Raw Bio / Experience Text:</label>
          <textarea 
            id="paste-text-input" 
            rows="3" 
            placeholder="e.g. Worked at Google as Machine Learning Intern. Graduated from Stanford with MS in AI. Love autonomous agents..."
            class="w-full bg-slate-900/90 border border-slate-800 rounded-xl p-3 text-xs text-slate-100 placeholder-slate-500 focus:outline-none focus:border-indigo-500"
          ></textarea>
          <button onclick="extractFromPastedText()" class="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-slate-200 flex items-center space-x-1.5 transition">
            <i data-lucide="zap" class="w-3.5 h-3.5 text-indigo-400"></i>
            <span>Extract & Store Facts</span>
          </button>
        </div>

        <!-- Interactive Interview Wizard -->
        <div class="border-t border-slate-800 pt-5 space-y-3">
          <div class="flex items-center justify-between">
            <h4 class="text-xs uppercase font-bold text-slate-300 tracking-wider flex items-center space-x-1.5">
              <i data-lucide="message-circle" class="w-3.5 h-3.5 text-indigo-400"></i>
              <span>Interactive Interview Mode</span>
            </h4>
            <span id="interview-step-badge" class="text-[11px] font-semibold text-indigo-400 bg-indigo-500/10 px-2 py-0.5 rounded-full border border-indigo-500/20">Step 1 of 4</span>
          </div>
          
          <div class="bg-slate-900/80 p-4 rounded-xl border border-slate-800 space-y-3">
            <p id="interview-question" class="text-xs text-slate-200 font-medium">
              Loading question...
            </p>
            <div class="flex items-center space-x-2">
              <input 
                type="text" 
                id="interview-answer-input" 
                placeholder="Type your response to teach the mirror..." 
                onkeydown="if(event.key === 'Enter') submitInterviewAnswer()"
                class="flex-1 bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs text-slate-100 focus:outline-none focus:border-indigo-500"
              />
              <button onclick="submitInterviewAnswer()" class="px-3.5 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-xs font-semibold text-white transition">
                Submit
              </button>
            </div>
          </div>
        </div>

      </div>

      <!-- VIEW 3: KNOWLEDGE BANK / MEMORY -->
      <div id="view-memory" class="glass-card rounded-2xl p-5 hidden flex-col h-[620px]">
        <div class="flex items-center justify-between pb-3 border-b border-slate-800">
          <div>
            <h3 class="text-sm font-bold text-white flex items-center space-x-2">
              <i data-lucide="database" class="w-4 h-4 text-indigo-400"></i>
              <span>Memory Bank</span>
            </h3>
            <p class="text-xs text-slate-400 mt-0.5">All atomic facts currently indexed for personal alignment</p>
          </div>
          <div class="flex items-center space-x-2">
            <button onclick="exportFactsJson()" class="px-3 py-1.5 rounded-xl bg-slate-900 hover:bg-slate-800 text-xs text-slate-300 border border-slate-700/60 flex items-center space-x-1.5 transition">
              <i data-lucide="download" class="w-3.5 h-3.5"></i>
              <span>Export</span>
            </button>
            <button onclick="clearAllFacts()" class="px-3 py-1.5 rounded-xl bg-rose-500/10 hover:bg-rose-500/20 text-xs text-rose-400 border border-rose-500/30 flex items-center space-x-1.5 transition">
              <i data-lucide="trash-2" class="w-3.5 h-3.5"></i>
              <span>Clear</span>
            </button>
          </div>
        </div>

        <!-- Add Manual Fact -->
        <div class="mt-3 flex items-center space-x-2">
          <input 
            type="text" 
            id="manual-fact-input" 
            placeholder="Add a new custom memory statement (e.g. 'I won 1st place in the 2026 GenAI Hackathon')..."
            onkeydown="if(event.key === 'Enter') addManualFact()"
            class="flex-1 bg-slate-900/90 border border-slate-800 rounded-xl px-3 py-2 text-xs text-slate-100 placeholder-slate-500 focus:outline-none focus:border-indigo-500"
          />
          <button onclick="addManualFact()" class="px-3 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-xs font-semibold text-white transition">
            Add Fact
          </button>
        </div>

        <!-- Facts List -->
        <div id="facts-container" class="mt-4 flex-1 overflow-y-auto custom-scroll space-y-2 pr-1">
          <!-- Dynamic fact cards -->
        </div>
      </div>

    </section>

  </main>

  <!-- Settings Modal -->
  <div id="settings-modal" class="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 hidden flex items-center justify-center p-4">
    <div class="glass-card rounded-2xl max-w-lg w-full p-6 space-y-4 border border-slate-700">
      <div class="flex items-center justify-between pb-3 border-b border-slate-800">
        <h3 class="text-sm font-bold text-white flex items-center space-x-2">
          <i data-lucide="sliders" class="w-4 h-4 text-indigo-400"></i>
          <span>Persona & Model Settings</span>
        </h3>
        <button onclick="closeSettingsModal()" class="text-slate-400 hover:text-white">
          <i data-lucide="x" class="w-4 h-4"></i>
        </button>
      </div>

      <div class="space-y-3 text-xs">
        <div>
          <label class="font-semibold text-slate-300">Subject Name:</label>
          <input type="text" id="setting-name" class="mt-1 w-full bg-slate-900 border border-slate-800 rounded-xl px-3 py-2 text-slate-100 focus:outline-none focus:border-indigo-500" value="Piyush">
        </div>

        <div>
          <label class="font-semibold text-slate-300">Title / Role:</label>
          <input type="text" id="setting-title" class="mt-1 w-full bg-slate-900 border border-slate-800 rounded-xl px-3 py-2 text-slate-100 focus:outline-none focus:border-indigo-500" value="AI/ML Engineer & Agentic Systems Developer">
        </div>

        <div>
          <label class="font-semibold text-slate-300">Bio:</label>
          <textarea id="setting-bio" rows="2" class="mt-1 w-full bg-slate-900 border border-slate-800 rounded-xl px-3 py-2 text-slate-100 focus:outline-none focus:border-indigo-500">Specializing in autonomous LLM agents, RAG architectures, and intelligent systems.</textarea>
        </div>

        <div>
          <label class="font-semibold text-slate-300">Model Provider:</label>
          <select id="setting-provider" onchange="toggleApiKeyFields()" class="mt-1 w-full bg-slate-900 border border-slate-800 rounded-xl px-3 py-2 text-slate-100 focus:outline-none focus:border-indigo-500">
            <option value="demo">Local Neural Engine (Zero API Key Required)</option>
            <option value="openai">OpenAI (GPT-4o-mini / GPT-4o)</option>
            <option value="gemini">Google Gemini (Gemini 1.5 Flash)</option>
          </select>
        </div>

        <div id="openai-key-group" class="hidden">
          <label class="font-semibold text-slate-300">OpenAI API Key:</label>
          <input type="password" id="setting-openai-key" placeholder="sk-..." class="mt-1 w-full bg-slate-900 border border-slate-800 rounded-xl px-3 py-2 text-slate-100 focus:outline-none focus:border-indigo-500">
        </div>

        <div id="gemini-key-group" class="hidden">
          <label class="font-semibold text-slate-300">Google Gemini API Key:</label>
          <input type="password" id="setting-gemini-key" placeholder="AIza..." class="mt-1 w-full bg-slate-900 border border-slate-800 rounded-xl px-3 py-2 text-slate-100 focus:outline-none focus:border-indigo-500">
        </div>
      </div>

      <div class="pt-3 border-t border-slate-800 flex justify-end space-x-2">
        <button onclick="closeSettingsModal()" class="px-3.5 py-1.5 rounded-xl bg-slate-800 text-xs text-slate-300 hover:bg-slate-700">Cancel</button>
        <button onclick="saveSettings()" class="px-4 py-1.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-xs font-semibold text-white">Save Changes</button>
      </div>
    </div>
  </div>

  <!-- JavaScript Client Logic -->
  <script>
    lucide.createIcons();

    let currentPreset = 'piyush';
    let isRecording = false;
    let recognition = null;

    // Load state from localStorage
    function loadSavedKeys() {
      const savedProvider = localStorage.getItem('mirror_provider') || 'demo';
      document.getElementById('setting-provider').value = savedProvider;
      document.getElementById('setting-openai-key').value = localStorage.getItem('mirror_openai_key') || '';
      document.getElementById('setting-gemini-key').value = localStorage.getItem('mirror_gemini_key') || '';
      toggleApiKeyFields();
      updateModelBadge();
    }

    function toggleApiKeyFields() {
      const provider = document.getElementById('setting-provider').value;
      document.getElementById('openai-key-group').classList.toggle('hidden', provider !== 'openai');
      document.getElementById('gemini-key-group').classList.toggle('hidden', provider !== 'gemini');
    }

    function updateModelBadge() {
      const provider = localStorage.getItem('mirror_provider') || 'demo';
      const label = provider === 'openai' ? 'OpenAI GPT' : provider === 'gemini' ? 'Google Gemini' : 'Local Neural Engine';
      document.getElementById('card-active-model').innerText = label;
    }

    function openSettingsModal() {
      document.getElementById('settings-modal').classList.remove('hidden');
    }

    function closeSettingsModal() {
      document.getElementById('settings-modal').classList.add('hidden');
    }

    async function saveSettings() {
      const provider = document.getElementById('setting-provider').value;
      const openaiKey = document.getElementById('setting-openai-key').value;
      const geminiKey = document.getElementById('setting-gemini-key').value;
      
      localStorage.setItem('mirror_provider', provider);
      localStorage.setItem('mirror_openai_key', openaiKey);
      localStorage.setItem('mirror_gemini_key', geminiKey);

      const name = document.getElementById('setting-name').value;
      const title = document.getElementById('setting-title').value;
      const bio = document.getElementById('setting-bio').value;

      await fetch('/api/profile', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          subject_name: name,
          title: title,
          bio: bio,
          speaking_style: 'First person, articulate, direct.'
        })
      });

      closeSettingsModal();
      refreshProfile();
      updateModelBadge();
    }

    async function refreshProfile() {
      const res = await fetch('/api/profile');
      const data = await res.json();
      document.getElementById('card-subject-name').innerText = data.subject_name;
      document.getElementById('avatar-container').innerText = data.subject_name.charAt(0);
      document.getElementById('card-subject-title').innerText = data.title;
      document.getElementById('card-subject-bio').innerText = data.bio;
      document.getElementById('card-fact-count').innerText = data.total_facts;
      document.getElementById('tab-fact-badge').innerText = data.total_facts;
    }

    async function refreshFacts() {
      const res = await fetch('/api/facts');
      const facts = await res.json();
      const container = document.getElementById('facts-container');
      container.innerHTML = '';
      
      if (facts.length === 0) {
        container.innerHTML = '<p class="text-xs text-slate-500 text-center py-6">No facts stored yet. Ingest documents or add one above!</p>';
        return;
      }

      facts.forEach(f => {
        const div = document.createElement('div');
        div.className = 'p-2.5 rounded-xl bg-slate-900/70 border border-slate-800/80 flex items-start justify-between text-xs group hover:border-slate-700 transition';
        div.innerHTML = `
          <div class="flex-1 pr-2">
            <p class="text-slate-200 leading-relaxed">${escapeHtml(f.fact)}</p>
            <div class="mt-1 flex items-center space-x-2 text-[10px] text-slate-500">
              <span class="bg-indigo-500/10 text-indigo-400 px-1.5 py-0.5 rounded border border-indigo-500/20 font-medium">${f.category || 'general'}</span>
              <span>Source: ${f.source || 'manual'}</span>
            </div>
          </div>
          <button onclick="deleteFact('${f.id}')" class="text-slate-500 hover:text-rose-400 p-1 opacity-0 group-hover:opacity-100 transition">
            <i data-lucide="trash" class="w-3.5 h-3.5"></i>
          </button>
        `;
        container.appendChild(div);
      });
      lucide.createIcons();
    }

    async function addManualFact() {
      const input = document.getElementById('manual-fact-input');
      const val = input.value.trim();
      if (!val) return;
      await fetch('/api/facts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fact: val })
      });
      input.value = '';
      refreshProfile();
      refreshFacts();
    }

    async function deleteFact(id) {
      await fetch(`/api/facts/${id}`, { method: 'DELETE' });
      refreshProfile();
      refreshFacts();
    }

    async function clearAllFacts() {
      if (!confirm('Are you sure you want to clear all memory facts?')) return;
      await fetch('/api/facts/clear', { method: 'POST' });
      refreshProfile();
      refreshFacts();
    }

    async function exportFactsJson() {
      const res = await fetch('/api/facts');
      const facts = await res.json();
      const blob = new Blob([JSON.stringify(facts, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `mirror_memory_${Date.now()}.json`;
      a.click();
    }

    // Switch Persona Presets
    async function switchPreset(preset) {
      currentPreset = preset;
      document.getElementById('btn-preset-piyush').className = preset === 'piyush' ? 'px-3 py-1 text-xs font-semibold rounded-lg bg-indigo-600 text-white shadow-sm transition' : 'px-3 py-1 text-xs font-semibold rounded-lg text-slate-400 hover:text-white transition';
      document.getElementById('btn-preset-thomas').className = preset === 'thomas' ? 'px-3 py-1 text-xs font-semibold rounded-lg bg-indigo-600 text-white shadow-sm transition' : 'px-3 py-1 text-xs font-semibold rounded-lg text-slate-400 hover:text-white transition';
      await fetch(`/api/preset/${preset}`, { method: 'POST' });
      await refreshProfile();
      await refreshFacts();
      
      // Post system announcement in chat
      appendMessage('🪞', `Switched active mirror persona to ${preset === 'piyush' ? 'Piyush (AI/ML Engineer)' : 'Thomas (Software Engineering Manager)'}. All answers will now reflect this persona!`, []);
    }

    // Tab Switching
    function switchTab(tab) {
      ['chat', 'teach', 'memory'].forEach(t => {
        document.getElementById(`view-${t}`).classList.toggle('hidden', t !== tab);
        const btn = document.getElementById(`tab-btn-${t}`);
        if (t === tab) {
          btn.className = 'flex-1 py-2 px-3 text-xs font-semibold rounded-xl bg-indigo-600 text-white shadow-sm flex items-center justify-center space-x-2 transition';
        } else {
          btn.className = 'flex-1 py-2 px-3 text-xs font-semibold rounded-xl text-slate-400 hover:text-white flex items-center justify-center space-x-2 transition';
        }
      });
      if (tab === 'memory') refreshFacts();
      if (tab === 'teach') loadInterviewStatus();
    }

    // Send Quick Prompt
    function sendQuickPrompt(text) {
      document.getElementById('chat-input').value = text;
      switchTab('chat');
      sendMessage();
    }

    // Send Chat Message
    async function sendMessage() {
      const input = document.getElementById('chat-input');
      const text = input.value.trim();
      if (!text) return;
      input.value = '';

      // User Message
      appendUserMessage(text);

      // Loading bubble
      const loadingId = appendLoadingBubble();

      const provider = localStorage.getItem('mirror_provider') || 'demo';
      const apiKey = provider === 'openai' ? localStorage.getItem('mirror_openai_key') : provider === 'gemini' ? localStorage.getItem('mirror_gemini_key') : null;

      try {
        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            message: text,
            provider: provider,
            api_key: apiKey
          })
        });
        const data = await res.json();
        removeLoadingBubble(loadingId);
        appendMessage('🪞', data.response, data.cited_facts || []);

        // Auto speak if checked
        if (document.getElementById('auto-speak-toggle').checked) {
          speakText(data.response);
        }
      } catch (err) {
        removeLoadingBubble(loadingId);
        appendMessage('⚠️', 'Error generating response: ' + err.message, []);
      }
    }

    function appendUserMessage(text) {
      const container = document.getElementById('chat-messages');
      const div = document.createElement('div');
      div.className = 'flex items-start justify-end space-x-3';
      div.innerHTML = `
        <div class="bg-indigo-600/90 text-white p-3.5 rounded-2xl rounded-tr-sm max-w-[80%] text-xs shadow-md shadow-indigo-600/10">
          <p class="leading-relaxed">${escapeHtml(text)}</p>
        </div>
      `;
      container.appendChild(div);
      container.scrollTop = container.scrollHeight;
    }

    function appendLoadingBubble() {
      const id = 'loading-' + Date.now();
      const container = document.getElementById('chat-messages');
      const div = document.createElement('div');
      div.id = id;
      div.className = 'flex items-start space-x-3';
      div.innerHTML = `
        <div class="w-8 h-8 rounded-xl bg-indigo-600 flex items-center justify-center text-xs font-bold text-white flex-shrink-0">
          🪞
        </div>
        <div class="bg-slate-900 border border-slate-800 p-3.5 rounded-2xl rounded-tl-sm flex items-center space-x-1.5">
          <div class="w-2 h-2 rounded-full bg-indigo-400 animate-bounce"></div>
          <div class="w-2 h-2 rounded-full bg-indigo-400 animate-bounce [animation-delay:0.2s]"></div>
          <div class="w-2 h-2 rounded-full bg-indigo-400 animate-bounce [animation-delay:0.4s]"></div>
        </div>
      `;
      container.appendChild(div);
      container.scrollTop = container.scrollHeight;
      return id;
    }

    function removeLoadingBubble(id) {
      const el = document.getElementById(id);
      if (el) el.remove();
    }

    function appendMessage(avatar, text, facts) {
      const container = document.getElementById('chat-messages');
      const div = document.createElement('div');
      div.className = 'flex items-start space-x-3';

      let factsHtml = '';
      if (facts && facts.length > 0) {
        const factList = facts.map(f => `<li class="text-[11px] text-slate-300">• ${escapeHtml(f.fact)}</li>`).join('');
        factsHtml = `
          <details class="mt-2.5 text-[11px] text-indigo-400/90 cursor-pointer">
            <summary class="hover:underline flex items-center space-x-1">
              <i data-lucide="shield" class="w-3 h-3"></i>
              <span>Cited ${facts.length} memory records</span>
            </summary>
            <ul class="mt-1.5 pl-2 space-y-1 bg-slate-950/60 p-2 rounded-lg border border-slate-800/80">
              ${factList}
            </ul>
          </details>
        `;
      }

      div.innerHTML = `
        <div class="w-8 h-8 rounded-xl bg-indigo-600 flex items-center justify-center text-xs font-bold text-white flex-shrink-0 mt-0.5">
          ${avatar}
        </div>
        <div class="bg-slate-900 border border-slate-800 p-3.5 rounded-2xl rounded-tl-sm max-w-[85%] text-xs">
          <p class="text-slate-200 leading-relaxed">${escapeHtml(text)}</p>
          ${factsHtml}
          <div class="mt-2.5 flex items-center space-x-2">
            <button onclick="speakText(this.closest('.bg-slate-900').querySelector('p').innerText)" class="px-2 py-1 rounded-md text-[11px] bg-slate-800 hover:bg-slate-700 text-indigo-300 flex items-center space-x-1 transition">
              <i data-lucide="volume-2" class="w-3 h-3"></i>
              <span>Listen</span>
            </button>
          </div>
        </div>
      `;
      container.appendChild(div);
      container.scrollTop = container.scrollHeight;
      lucide.createIcons();
    }

    // Voice Synthesis (Browser Web Speech API)
    function speakText(text) {
      if (!('speechSynthesis' in window)) {
        alert('Speech synthesis is not supported in this browser.');
        return;
      }
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1.05;
      utterance.pitch = 1.0;
      window.speechSynthesis.speak(utterance);
    }

    // Voice Input (Web Speech Recognition)
    function toggleSpeechRecognition() {
      const micBtn = document.getElementById('mic-btn');
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      
      if (!SpeechRecognition) {
        alert('Voice recognition not supported in this browser. Please use Google Chrome or Edge.');
        return;
      }

      if (isRecording && recognition) {
        recognition.stop();
        isRecording = false;
        micBtn.classList.remove('pulse-mic', 'bg-rose-600', 'text-white');
        return;
      }

      recognition = new SpeechRecognition();
      recognition.continuous = false;
      recognition.interimResults = false;
      recognition.lang = 'en-US';

      recognition.onstart = () => {
        isRecording = true;
        micBtn.classList.add('pulse-mic', 'bg-rose-600', 'text-white');
      };

      recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        document.getElementById('chat-input').value = transcript;
        sendMessage();
      };

      recognition.onerror = () => {
        isRecording = false;
        micBtn.classList.remove('pulse-mic', 'bg-rose-600', 'text-white');
      };

      recognition.onend = () => {
        isRecording = false;
        micBtn.classList.remove('pulse-mic', 'bg-rose-600', 'text-white');
      };

      recognition.start();
    }

    // File Upload Ingestion
    async function handleFileUpload(event) {
      const file = event.target.files[0];
      if (!file) return;

      const statusEl = document.getElementById('upload-status');
      statusEl.classList.remove('hidden');
      statusEl.innerText = `Extracting facts from ${file.name}...`;

      const provider = localStorage.getItem('mirror_provider') || 'demo';
      const apiKey = provider === 'openai' ? localStorage.getItem('mirror_openai_key') : null;

      const formData = new FormData();
      formData.append('file', file);
      formData.append('provider', provider);
      if (apiKey) formData.append('api_key', apiKey);

      try {
        const res = await fetch('/api/upload', {
          method: 'POST',
          body: formData
        });
        const data = await res.json();
        statusEl.innerText = `✅ Successfully extracted ${data.facts_extracted} facts!`;
        refreshProfile();
      } catch (err) {
        statusEl.innerText = `❌ Error: ${err.message}`;
      }
    }

    // Paste Text Extraction
    async function extractFromPastedText() {
      const val = document.getElementById('paste-text-input').value.trim();
      if (!val) return;
      const provider = localStorage.getItem('mirror_provider') || 'demo';
      const apiKey = provider === 'openai' ? localStorage.getItem('mirror_openai_key') : null;

      const res = await fetch('/api/extract_text', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: val, provider: provider, api_key: apiKey })
      });
      const data = await res.json();
      alert(`Extracted and stored ${data.facts_extracted} facts into knowledge bank!`);
      document.getElementById('paste-text-input').value = '';
      refreshProfile();
    }

    // Interactive Interview Mode
    async function loadInterviewStatus() {
      const res = await fetch('/api/interview');
      const data = await res.json();
      document.getElementById('interview-step-badge').innerText = `Step ${data.current_step} of ${data.total_steps}`;
      if (data.completed) {
        document.getElementById('interview-question').innerText = '🎉 Interview complete! All answers have been indexed into your personal mirror.';
        document.getElementById('interview-answer-input').disabled = true;
      } else {
        document.getElementById('interview-question').innerText = data.question;
        document.getElementById('interview-answer-input').disabled = false;
      }
    }

    async function submitInterviewAnswer() {
      const input = document.getElementById('interview-answer-input');
      const ans = input.value.trim();
      if (!ans) return;
      input.value = '';

      await fetch('/api/interview/answer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ answer: ans })
      });

      refreshProfile();
      loadInterviewStatus();
    }

    function escapeHtml(text) {
      if (!text) return '';
      return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }

    // Init
    loadSavedKeys();
    refreshProfile();
  </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    return HTMLResponse(content=HTML_PAGE, status_code=200)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting MirrorGPT on http://localhost:{port}")
    uvicorn.run("web_app:app", host="0.0.0.0", port=port, reload=False)
