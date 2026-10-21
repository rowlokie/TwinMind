import os
import re
import uuid
import json
import warnings
from typing import List, Dict, Any, Optional
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Optional imports with graceful fallbacks
try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class MirrorEngine:
    """
    Core AI engine for MirrorGPT: manages persona, personal knowledge base,
    semantic search/retrieval, fact extraction, and first-person conversational reasoning.
    """
    def __init__(self, subject_name: str = "Piyush", title: str = "AI/ML Engineer"):
        self.subject_name = subject_name
        self.title = title
        self.bio = f"AI and Machine Learning Engineer passionate about autonomous agents, LLMs, and intelligent systems."
        self.speaking_style = "Confident, technical yet accessible, direct, and conversational in the first person."
        
        # Knowledge Base: list of dicts: {"id": str, "fact": str, "category": str, "source": str}
        self.facts: List[Dict[str, Any]] = []
        
        # Embedding cache
        self.embed_model = None
        self.fact_embeddings: Optional[np.ndarray] = None
        
        # Interactive interview state
        self.interview_questions = [
            "What is your current role or university background, and what are you primarily working on?",
            "What are your top technical skills, programming languages, and AI/ML frameworks you specialize in?",
            "Can you tell me about a major project, hackathon, or achievement you are proud of?",
            "How would you describe your communication style, leadership approach, or personal interests outside of coding?"
        ]
        self.current_interview_idx = 0
        
        # Load default persona pre-seeds
        self.load_preset("piyush")

    def init_embed_model(self):
        """Lazy load sentence transformer if available for offline semantic search"""
        if self.embed_model is None and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception as e:
                print(f"Warning: Could not load sentence-transformers: {e}")
                self.embed_model = None

    def recompute_embeddings(self):
        """Recompute vector embeddings for all facts in memory"""
        if not self.facts:
            self.fact_embeddings = None
            return

        texts = [f["fact"] for f in self.facts]
        self.init_embed_model()
        
        if self.embed_model is not None:
            try:
                self.fact_embeddings = self.embed_model.encode(texts, convert_to_numpy=True)
                return
            except Exception as e:
                print(f"Embedding computation error: {e}")

        # Fallback to simple TF-IDF / term frequency vectorizer for 100% offline reliability
        self.fact_embeddings = self._compute_tfidf_embeddings(texts)

    def _compute_tfidf_embeddings(self, texts: List[str]) -> np.ndarray:
        """Lightweight bag-of-words / frequency embedding for fast offline fallback"""
        vocab = {}
        for text in texts:
            for word in re.findall(r'\b\w+\b', text.lower()):
                if word not in vocab:
                    vocab[word] = len(vocab)
        
        if not vocab:
            return np.zeros((len(texts), 1))

        vectors = np.zeros((len(texts), len(vocab)), dtype=np.float32)
        for i, text in enumerate(texts):
            words = re.findall(r'\b\w+\b', text.lower())
            for word in words:
                if word in vocab:
                    vectors[i, vocab[word]] += 1.0
            norm = np.linalg.norm(vectors[i])
            if norm > 0:
                vectors[i] /= norm
        return vectors

    def add_fact(self, fact_text: str, category: str = "general", source: str = "manual") -> Dict[str, Any]:
        """Add a single fact to memory and update embeddings"""
        fact_text = fact_text.strip()
        if not fact_text:
            return {}
            
        fact_obj = {
            "id": str(uuid.uuid4())[:8],
            "fact": fact_text,
            "category": category,
            "source": source
        }
        self.facts.append(fact_obj)
        self.recompute_embeddings()
        return fact_obj

    def delete_fact(self, fact_id: str) -> bool:
        """Remove a fact by ID"""
        initial_len = len(self.facts)
        self.facts = [f for f in self.facts if f["id"] != fact_id]
        if len(self.facts) != initial_len:
            self.recompute_embeddings()
            return True
        return False

    def clear_facts(self):
        """Clear all stored facts"""
        self.facts = []
        self.fact_embeddings = None

    def search_facts(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve the top_k most relevant facts for a given query"""
        if not self.facts:
            return []

        # If we have precomputed embeddings
        if self.fact_embeddings is not None and self.embed_model is not None:
            try:
                query_vec = self.embed_model.encode([query], convert_to_numpy=True)[0]
                # Cosine similarity
                scores = np.dot(self.fact_embeddings, query_vec) / (
                    np.linalg.norm(self.fact_embeddings, axis=1) * np.linalg.norm(query_vec) + 1e-9
                )
                top_indices = np.argsort(scores)[::-1][:top_k]
                results = []
                for idx in top_indices:
                    results.append({
                        **self.facts[idx],
                        "score": float(scores[idx])
                    })
                return results
            except Exception as e:
                print(f"Error in vector search: {e}")

        # Fallback keyword overlap / term relevance
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        scored = []
        for f in self.facts:
            fact_words = set(re.findall(r'\b\w+\b', f["fact"].lower()))
            overlap = len(query_words.intersection(fact_words))
            score = overlap / max(len(query_words), 1)
            scored.append((score, f))
            
        scored.sort(key=lambda x: x[0], reverse=True)
        return [{**f, "score": s} for s, f in scored[:top_k]]

    def extract_facts_from_text(self, text: str, source: str = "document", api_key: Optional[str] = None, provider: str = "openai") -> List[str]:
        """
        Splits and converts long-form text (resume, bio, notes) into atomic facts.
        Uses LLM if API key provided, otherwise uses intelligent heuristic sentence segmenter.
        """
        extracted = []
        
        # If API key provided, attempt LLM extraction
        if api_key and provider == "openai" and OPENAI_AVAILABLE:
            try:
                client = OpenAI(api_key=api_key)
                prompt = (
                    f"You are an AI personal data extractor. Convert the following text about {self.subject_name} "
                    f"into a JSON list of single-sentence, atomic, first-person or factual statements. "
                    f"Example: [\"Graduated with a degree in Computer Science\", \"Built a real-time computer vision system\"]\n\n"
                    f"Text:\n{text[:4000]}\n\n"
                    f"Return ONLY valid JSON array of strings."
                )
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                )
                raw_json = response.choices[0].message.content.strip()
                if "```json" in raw_json:
                    raw_json = raw_json.split("```json")[1].split("```")[0].strip()
                elif "```" in raw_json:
                    raw_json = raw_json.split("```")[1].split("```")[0].strip()
                parsed = json.loads(raw_json)
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, str) and item.strip():
                            self.add_fact(item.strip(), category="document", source=source)
                            extracted.append(item.strip())
                    return extracted
            except Exception as e:
                print(f"LLM fact extraction failed, falling back to heuristic: {e}")

        # Heuristic fallback: clean sentences & bullet points
        lines = text.split("\n")
        for line in lines:
            line = line.strip().lstrip("•-*0123456789. ")
            if len(line) > 20 and not line.lower().startswith(("page ", "http", "email:", "phone:")):
                # Split multiple sentences if any
                sentences = re.split(r'(?<=[.!?])\s+', line)
                for s in sentences:
                    s = s.strip()
                    if len(s) > 15:
                        self.add_fact(s, category="document", source=source)
                        extracted.append(s)
        return extracted

    def parse_pdf_file(self, file_path: str) -> str:
        """Extract raw text from PDF"""
        if not PYPDF_AVAILABLE:
            raise ImportError("pypdf is required to parse PDF documents.")
        
        reader = PdfReader(file_path)
        full_text = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text.append(text)
        return "\n".join(full_text)

    def generate_response(
        self,
        user_message: str,
        api_key: Optional[str] = None,
        provider: str = "demo",
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generates a first-person mirror response based on retrieved facts.
        Supports OpenAI, Google Gemini, and a high-quality local Demo Mode.
        """
        relevant_facts = self.search_facts(user_message, top_k=5)
        context_str = "\n".join([f"- {f['fact']}" for f in relevant_facts]) if relevant_facts else "No specific records found."

        system_instruction = (
            f"You are the autonomous AI Mirror of {self.subject_name}.\n"
            f"Title: {self.title}\n"
            f"Bio: {self.bio}\n"
            f"Speaking Style: {self.speaking_style}\n\n"
            f"CRITICAL RULES:\n"
            f"1. You MUST ALWAYS speak in the FIRST PERSON ('I', 'me', 'my', 'myself'). Never refer to {self.subject_name} in third person.\n"
            f"2. Use the retrieved memory facts below to answer accurately about your experience, skills, thoughts, and background.\n"
            f"3. If the retrieved facts do not contain enough information to answer a factual question about your life, politely state: "
            f"'I don't have that specific detail in my memory yet. Feel free to ask me about my background, or ask the real {self.subject_name} directly!'\n"
            f"4. Be natural, authentic, articulate, and engaging.\n\n"
            f"RETRIEVED PERSONAL FACTS:\n{context_str}"
        )

        # 1. OpenAI Integration
        if provider == "openai" and api_key and OPENAI_AVAILABLE:
            try:
                client = OpenAI(api_key=api_key)
                selected_model = model_name or "gpt-4o-mini"
                resp = client.chat.completions.create(
                    model=selected_model,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=0.7,
                )
                answer = resp.choices[0].message.content
                return {
                    "response": answer,
                    "cited_facts": relevant_facts,
                    "provider": f"OpenAI ({selected_model})",
                    "subject": self.subject_name
                }
            except Exception as e:
                return {
                    "response": f"⚠️ Error calling OpenAI API: {str(e)}",
                    "cited_facts": relevant_facts,
                    "provider": "OpenAI (Error)",
                    "subject": self.subject_name
                }

        # 2. Google Gemini Integration
        if provider == "gemini" and api_key and GEMINI_AVAILABLE:
            try:
                genai.configure(api_key=api_key)
                gemini_model = genai.GenerativeModel(
                    model_name=model_name or "gemini-1.5-flash",
                    system_instruction=system_instruction
                )
                resp = gemini_model.generate_content(user_message)
                return {
                    "response": resp.text,
                    "cited_facts": relevant_facts,
                    "provider": f"Google Gemini ({model_name or 'gemini-1.5-flash'})",
                    "subject": self.subject_name
                }
            except Exception as e:
                return {
                    "response": f"⚠️ Error calling Gemini API: {str(e)}",
                    "cited_facts": relevant_facts,
                    "provider": "Gemini (Error)",
                    "subject": self.subject_name
                }

        # 3. Intelligent Local Demo Mode (Zero API keys required!)
        return self._generate_demo_mode_response(user_message, relevant_facts)

    def _generate_demo_mode_response(self, user_message: str, relevant_facts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Provides high-quality, instant first-person responses without needing an external paid API key"""
        msg_lower = user_message.lower()

        # Greetings
        if any(w in msg_lower for w in ["hi", "hello", "hey", "who are you", "what is your name", "introduce"]):
            resp = (
                f"Hey there! I am the AI Mirror of {self.subject_name}. "
                f"I'm a {self.title}. I've been trained on my background, technical work, and experiences so I can chat with you directly. "
                f"Feel free to ask me about my projects, skills, work experience, or background!"
            )
            return {
                "response": resp,
                "cited_facts": relevant_facts[:2],
                "provider": "Local Mirror Engine (Demo Mode)",
                "subject": self.subject_name
            }

        # If facts are relevant
        if relevant_facts and relevant_facts[0].get("score", 0) > 0.05:
            top_facts = [f["fact"] for f in relevant_facts[:3]]
            fact_summary = " Additionally, ".join(top_facts)
            resp = f"Regarding that: {fact_summary}. Let me know if you'd like to dive deeper into any of that!"
            return {
                "response": resp,
                "cited_facts": relevant_facts[:3],
                "provider": "Local Mirror Engine (Demo Mode)",
                "subject": self.subject_name
            }

        # Fallback
        resp = (
            f"I don't have that exact detail in my current memory bank yet. "
            f"You can ask me about my education, projects, skills, or add more data about me in the 'Knowledge Base' tab!"
        )
        return {
            "response": resp,
            "cited_facts": [],
            "provider": "Local Mirror Engine (Demo Mode)",
            "subject": self.subject_name
        }

    def process_interview_answer(self, answer: str) -> Dict[str, Any]:
        """Interactive teaching interview loop: extract facts from user answer and provide next question"""
        if answer.strip():
            # Add fact
            self.add_fact(answer.strip(), category="interview", source="teach_interview")
            
        self.current_interview_idx += 1
        has_more = self.current_interview_idx < len(self.interview_questions)
        next_q = self.interview_questions[self.current_interview_idx] if has_more else None
        
        return {
            "recorded": answer.strip(),
            "next_question": next_q,
            "completed": not has_more,
            "total_facts": len(self.facts)
        }

    def reset_interview(self):
        self.current_interview_idx = 0

    def load_preset(self, preset_name: str):
        """Loads ready-made rich persona profiles for instant testing & hackathon demonstration"""
        self.clear_facts()
        
        if preset_name.lower() == "piyush":
            self.subject_name = "Piyush"
            self.title = "AI/ML Engineer & Agentic Systems Developer"
            self.bio = "Specializing in autonomous LLM agents, RAG architectures, neural pipelines, and full-stack AI development."
            self.speaking_style = "Insightful, technically precise, energetic, first-person, and passionate about generative AI."
            
            sample_facts = [
                "I am an AI and Machine Learning developer focused on building production-ready autonomous agent systems and RAG pipelines.",
                "I specialize in Python, PyTorch, LangChain, Transformers, FastAPI, and modern vector databases.",
                "I built MirrorGPT — a personal AI digital twin that mimics personal facts, tone, and voice synthesis.",
                "I actively participate in AI hackathons, developing high-impact applications with generative AI and agentic workflows.",
                "I graduated with strong foundations in Computer Science, Data Structures, and Applied Machine Learning.",
                "I enjoy exploring modern multi-agent systems, multimodal models, and low-latency voice AI integrations.",
                "My work philosophy centers on building practical, user-centric AI tools that automate complex workflows and augment human productivity."
            ]
            for f in sample_facts:
                self.add_fact(f, category="profile", source="preset_piyush")

        elif preset_name.lower() == "thomas":
            self.subject_name = "Thomas Crosley"
            self.title = "Software Engineering Manager & Computer Vision Leader"
            self.bio = "Engineering leader bridging machine learning, production software, and customer-centric product development."
            self.speaking_style = "Strategic, managerial, experienced, first-person."
            
            sample_facts = [
                "I worked as a Software Development Manager at AWS leading computer vision software engineers for Just Walk Out Technology (Amazon Go and Amazon Fresh).",
                "I led the accuracy program for VP-level audiences and launched algorithms on custom edge compute platforms.",
                "I was a Software Engineering Manager at Axon, scaling a speech-to-text compute cluster and transcription-based products.",
                "I studied Computer Science at the University of Washington.",
                "I have experience across FAANG companies, academic labs, and Seattle startups."
            ]
            for f in sample_facts:
                self.add_fact(f, category="experience", source="crosleythomas_linkedin.txt")
