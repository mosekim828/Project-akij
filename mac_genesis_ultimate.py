import os
import time
import json
import re
import subprocess
import glob
import shutil
import random
import requests
from datetime import datetime
from collections import Counter
from typing import List, Dict, TypedDict, Any, Tuple, Union
from mlx_lm import load, generate

# ★ RAG 검색용 라이브러리 추가
try:
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError:
    print("⚠️ [System] LangChain 라이브러리가 없습니다. 'pip install langchain-community chromadb' 필요.")

import fitz  # PyMuPDF

# ==============================================================================
# [0] 핵심 설정 (USER CONFIGURATION)
# ==============================================================================

# 1. PC 화가 서버 (제공해주신 코드대로 61번으로 유지함)
PC_FLUX_SERVER_URL = "http://192.168.0.61:8000/draw"

# 2. 기본 삽화 빈도
DEFAULT_ILLUSTRATION_FREQ = 3 

# 3. 경로 설정
USER_HOME = "/Users/juson"
MODEL_PATH = f"{USER_HOME}/.cache/huggingface/hub/models--mlx-community--Llama-4-Maverick-17B-16E-Instruct-6bit/snapshots/542ea389fcd614c665c4306bd60ad053d9da8d03"

FACTORY_DIR = f"{USER_HOME}/Desktop/factory_input"
DIR_RESULT = os.path.join(FACTORY_DIR, "1_Result")
DIR_REFERENCE = os.path.join(FACTORY_DIR, "2_Reference_Style")

GENESIS_PATH = f"{USER_HOME}/Desktop/Genesis_Project"
DIR_TEMPLATE = os.path.join(GENESIS_PATH, "D0_template")
DIR_FONTS = os.path.join(GENESIS_PATH, "D5_Fonts")

# ★ [V12] Vector DB 경로 & 임베딩 모델 (Builder와 일치해야 함)
DB_PERSIST_DIR = f"{USER_HOME}/Desktop/Genesis_Project/99_VectorDB"
EMBEDDING_MODEL_ID = "BAAI/bge-m3"

# 4. 폰트/기타
DEFAULT_FONT_TITLE = "AppleSDGothicNeo-Bold"
DEFAULT_FONT_BODY = "AppleMyungjo"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 5. 맞춤법 검사기
try:
    from hanspell import spell_checker
    HANSPELL_AVAILABLE = True
except ImportError:
    HANSPELL_AVAILABLE = False
    print("⚠️ [System] py-hanspell 미설치. AI 교정만 수행합니다.")


# ==============================================================================
# [1] 도구 클래스 (TOOLKIT - V12 RAG Integrated)
# ==============================================================================

class StyleReplicator:
    @staticmethod
    def analyze_pdf(pdf_path):
        try:
            doc = fitz.open(pdf_path)
            page = doc[0]
            blocks = page.get_text("dict")["blocks"]
            font_sizes = []
            for b in blocks:
                if "lines" in b:
                    for l in b["lines"]:
                        for s in l["spans"]:
                            font_sizes.append(s["size"])
            if not font_sizes: return None
            
            body_size = Counter(font_sizes).most_common(1)[0][0]
            title_size = max(font_sizes)
            rect = page.rect
            margin_x = (rect.width - (blocks[0]["bbox"][2] - blocks[0]["bbox"][0])) / 2 if blocks else 72
            
            return {
                "filename": os.path.basename(pdf_path),
                "page_width": rect.width, "page_height": rect.height,
                "body_size": f"{body_size:.1f}pt", "title_size": f"{title_size:.1f}pt",
                "margin": f"{margin_x / 2.83:.1f}mm"
            }
        except: return None

class TermGuard:
    CORRECTION_DICT = {
        "노가": "노아", "노아의 방주": "노아의 방주", 
        "여호와": "여호와", "다윗": "다윗", "바울": "바울",
        "세신자": "새신자", "세례교인": "세례교인",
        "예수님": "예수님", "하나님": "하나님",
        "그리스인": "그리스도인", "기를 축복합니다": "주님의 이름으로 축복합니다"
    }
    @staticmethod
    def enforce(text: str) -> str:
        for wrong, right in TermGuard.CORRECTION_DICT.items():
            if wrong in text: text = text.replace(wrong, right)
        text = text.replace("<", "").replace(">", "")
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text

    @staticmethod
    def run_spell_check(text: str) -> str:
        if not HANSPELL_AVAILABLE: return text
        try:
            corrected = ""
            for line in text.split('\n'):
                if not line.strip(): corrected += "\n"
                elif len(line) < 500: corrected += spell_checker.check(line).checked + "\n"
                else: corrected += line + "\n"
            return corrected.strip()
        except: return text

class TextManager:
    @staticmethod
    def split_text_clean(text, chunk_size=3000):
        chunks = []
        start = 0
        text_len = len(text)
        while start < text_len:
            end = min(start + chunk_size, text_len)
            if end < text_len:
                last_period = text.rfind('.', start, end)
                if last_period != -1 and last_period > start + (chunk_size // 2): end = last_period + 1
            chunk = text[start:end].strip()
            if chunk: chunks.append(chunk)
            start = end
        return chunks

class KnowledgeManager:
    """
    [V12 Librarian] Vector DB 검색 및 폰트/템플릿 관리
    """
    @staticmethod
    def scan_fonts(font_dir):
        found = []
        if os.path.exists(font_dir):
            for root, _, files in os.walk(font_dir):
                for file in files:
                    if file.lower().endswith(('.ttf', '.otf')):
                        name = os.path.splitext(file)[0]
                        name = re.sub(r"-(Bold|Regular|Light|Medium|Black|Thin|ExtraBold|SemiBold)", "", name, flags=re.IGNORECASE)
                        name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name) 
                        if "Noto Sans" in name: name = "Noto Sans KR"
                        if "Noto Serif" in name: name = "Noto Serif KR"
                        name = name.strip()
                        if name not in found: found.append(name)
        return found if found else [DEFAULT_FONT_TITLE, DEFAULT_FONT_BODY]

    @staticmethod
    def load_templates():
        """디자인 템플릿은 DB가 아닌 파일에서 직접 로드 (정확성)"""
        buf = ""
        if os.path.exists(DIR_TEMPLATE):
            for file in glob.glob(os.path.join(DIR_TEMPLATE, "*.typ")):
                try: 
                    with open(file, 'r', encoding='utf-8') as f:
                        buf += f"\n[Template Code: {os.path.basename(file)}]\n{f.read()}\n"
                except: pass
        return buf

    @staticmethod
    def search_vector_db(query: str, k: int = 5) -> str:
        """
        ★ [V12 RAG Core] Vector DB에서 관련 지식 검색
        """
        if not os.path.exists(DB_PERSIST_DIR):
            log("Librarian", "⚠️ Vector DB가 없습니다. 'build_knowledge_base.py'를 실행해주세요.")
            return ""

        log("Librarian", f"🔍 지식 도서관 검색 중: '{query[:30]}...'")
        try:
            # 임베딩 모델 (Mac mps 가속)
            embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_ID,
                model_kwargs={'device': 'mps'},
                encode_kwargs={'normalize_embeddings': True}
            )
            # DB 연결
            vectordb = Chroma(
                persist_directory=DB_PERSIST_DIR, 
                embedding_function=embeddings,
                collection_name="genesis_knowledge"
            )
            # 검색
            docs = vectordb.similarity_search(query, k=k)
            
            # 결과 정리
            context_text = ""
            for i, doc in enumerate(docs):
                src = doc.metadata.get("source", "Unknown")
                cat = doc.metadata.get("category", "Unknown")
                context_text += f"\n[Reference {i+1} | {cat}/{src}]\n{doc.page_content}\n"
            
            log("Librarian", f"✅ 관련 문서 {len(docs)}건 확보 완료.")
            return context_text
            
        except Exception as e:
            log("Librarian", f"❌ 검색 오류: {e}")
            return ""

    @staticmethod
    def fix_typst_syntax(code: str) -> str:
        """
        [V12.5 수정] Typst 0.11+ 호환성 패치 (loc -> context)
        특히 'unknown variable: loc' 오류를 잡기 위해 query 함수 내부를 정밀 타격함.
        """
        # 1. locate(loc => ...) 패턴을 #context로 변경
        if "locate(" in code:
            log("System", "🛠️ Typst 문법 수선: locate -> context")
            code = re.sub(r"#?locate\s*\(\s*\w+\s*=>", "#context", code) # locate(loc => 지움
            code = code.replace("locate(loc =>", "#context") # 혹시 몰라 단순 치환도 유지

        # 2. [핵심 수정] query 함수 내부에 있는 ', loc' 인자를 제거
        # 예: query(heading, loc) -> query(heading)
        # 이 부분이 'unknown variable: loc' 에러의 주범임
        code = re.sub(r"query\(([^)]+),\s*loc\)", r"query(\1)", code)

        # 3. counter(page).at(loc) -> counter(page).get()
        code = code.replace(".at(loc)", ".get()")
        
        # 4. 닫히지 않은 괄호 정리
        open_sq = code.count('[')
        close_sq = code.count(']')
        if close_sq > open_sq:
            diff = close_sq - open_sq
            code = code[::-1].replace(']', '', diff)[::-1]
        return code

def log(agent, msg):
    icons = {"Director": "🎬", "Librarian": "📚", "Editor": "✍️", "Designer": "🎨", "System": "⚙️", "Replicator": "🧬", "Illustrator": "🖌️"}
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {icons.get(agent, '🤖')} [{agent}] {msg}")

def cleanup_on_start():
    for d in [FACTORY_DIR, DIR_RESULT, DIR_REFERENCE, DIR_TEMPLATE]:
        if not os.path.exists(d): os.makedirs(d)
    target = os.path.join(FACTORY_DIR, "pc_output.json")
    if os.path.exists(target):
        try:
            timestamp = int(time.time())
            os.rename(target, os.path.join(FACTORY_DIR, f"ignored_{timestamp}.json"))
        except: pass

# ==============================================================================
# [2] 상태 관리
# ==============================================================================
class BookState(TypedDict):
    user_instruction: str; raw_material: str; img_snap: str; img_flux: str
    learned_style: str; knowledge_context: str; available_fonts: List[str]
    book_title: str; text_chunks: List[str]
    polished_chunks: List[Tuple[str, str]]
    current_chunk_idx: int; layout_config: Dict[str, str]
    replicated_template_name: str; selected_style_name: str
    illustration_freq: int

# ==============================================================================
# [3] AI 에이전트
# ==============================================================================

def load_model_once():
    try:
        log("System", "Maverick(Llama-4) 엔진 예열 중...")
        return load(MODEL_PATH)
    except: return None, None

def agent_librarian(model, tokenizer, state: BookState) -> BookState:
    """
    [V12 Librarian] RAG 검색 실행
    """
    log("Librarian", "지식 도서관 접속 중...")
    state['available_fonts'] = KnowledgeManager.scan_fonts(DIR_FONTS)
    
    # 1. 템플릿 로드 (디자인용)
    tpl_context = KnowledgeManager.load_templates()
    
    # 2. ★ RAG 검색: 사용자 지시 + 원문 일부를 쿼리로 사용
    search_query = f"{state['user_instruction']} {state['raw_material'][:500]}"
    retrieved_knowledge = KnowledgeManager.search_vector_db(search_query, k=6)
    
    # 3. 최종 컨텍스트 합체
    state['knowledge_context'] = tpl_context + "\n" + retrieved_knowledge
    return state

def agent_replicator(model, tokenizer, state: BookState) -> BookState:
    state['replicated_template_name'] = ""
    state['selected_style_name'] = "Default"
    
    pdf_files = glob.glob(os.path.join(DIR_REFERENCE, "*.pdf"))
    if not pdf_files:
        log("Replicator", "⚠️ 스타일 참고용 PDF가 없습니다. 기본값 사용.")
        return state

    target_pdf = None
    instruction = state['user_instruction'].lower()
    for pdf in pdf_files:
        if os.path.splitext(os.path.basename(pdf))[0].lower() in instruction:
            target_pdf = pdf; break
    if not target_pdf: target_pdf = random.choice(pdf_files)
    
    state['selected_style_name'] = os.path.basename(target_pdf)
    analysis = StyleReplicator.analyze_pdf(target_pdf)
    
    if analysis:
        log("Replicator", f"스타일 분석 완료: {analysis['filename']}")
        fonts_str = ", ".join(state['available_fonts'])
        
        prompt = f"""<|system|>Typst 0.11+ 버전 전문가입니다.
[데이터]: {analysis}
[폰트]: {fonts_str}
[주의]: Typst 0.11부터 `locate` 함수가 삭제되었습니다. 반드시 `context` 키워드를 사용하십시오.
[지시]: 위 폰트 중 적절한 것을 골라 사용하고, 오직 Typst 코드만 출력.<|user|>작성<|assistant|>"""
        
        typ_code = generate(model, tokenizer, prompt=prompt, max_tokens=1500, verbose=False)
        typ_code = typ_code.replace("```typst", "").replace("```", "").strip()
        typ_code = KnowledgeManager.fix_typst_syntax(typ_code)
        
        template_name = f"replicated_{int(time.time())}.typ"
        with open(os.path.join(DIR_TEMPLATE, template_name), "w", encoding="utf-8") as f: f.write(typ_code)
        state['replicated_template_name'] = template_name

    return state

def agent_director(model, tokenizer, state: BookState) -> BookState:
    if not state['raw_material']: return state
    log("Director", "기획 및 전처리 중...")
    
    prompt = f"""<|system|>베스트셀러 기획자입니다. 제목 결정. '제목:' 제거.
[지시]: "{state['user_instruction']}"
[내용]: {state['raw_material'][:3000]}<|user|>제목 결정<|assistant|>"""
    
    title_raw = generate(model, tokenizer, prompt=prompt, max_tokens=60, verbose=False).strip()
    state['book_title'] = re.sub(r"^(제목|책\s*제목|Title)\s*[:：]\s*", "", title_raw, flags=re.IGNORECASE).strip('"\' ')
    log("Director", f"책 제목 확정: {state['book_title']}")
    
    state['text_chunks'] = TextManager.split_text_clean(state['raw_material'])
    return state

def agent_hybrid_editor(model, tokenizer, state: BookState) -> BookState:
    idx = state['current_chunk_idx']
    if idx >= len(state['text_chunks']): return state
    
    current_text = state['text_chunks'][idx]
    log("Editor", f"윤문 작업 중... [{idx+1}/{len(state['text_chunks'])}]")
    
    prev_text = ""
    if state['polished_chunks']:
        for t, c in reversed(state['polished_chunks']):
            if t == 'text':
                prev_text = c[-300:]
                break
    
    # RAG로 검색된 지식은 model의 system prompt나 context에 자동 포함됨 (knowledge_context)
    prompt = f"""<|system|>수석 편집장입니다. 문맥 복원 및 윤문.
[참고 지식]: {state['knowledge_context'][:2000]}
[원칙] 문맥 복원, 문어체, 문단 구분.
<|user|>[이전]:...{prev_text}\n[원문]:{current_text}\n[지시]:윤문하라.<|assistant|>"""
    
    draft = generate(model, tokenizer, prompt=prompt, max_tokens=4000, verbose=False)
    draft = re.sub(r"^(네|물론|알겠|확인|수정|윤문|제시|따라서).+?(\n|$)", "", draft, flags=re.MULTILINE).strip()
    
    if HANSPELL_AVAILABLE: 
        draft = TermGuard.run_spell_check(draft)
    
    final_text = TermGuard.enforce(draft)
    state['polished_chunks'].append(('text', final_text))
    state['current_chunk_idx'] += 1
    return state

def agent_illustrator(model, tokenizer, state: BookState) -> BookState:
    """
    ★ [V12.4 최종 수정] 한글 원천 봉쇄 & 잠꼬대 방지
    """
    processed_count = len(state['polished_chunks'])
    if not state['polished_chunks']: return state

    last_type, last_content = state['polished_chunks'][-1]
    freq = state.get('illustration_freq', DEFAULT_ILLUSTRATION_FREQ)

    if last_type == 'text' and (processed_count % freq == 0):
        log("Illustrator", f"🎨 문맥 분석 및 삽화 의뢰 (설정 빈도: {freq})...")
        
        # 1. AI에게 지시 (영어만 쓰라고 강력히 요구)
        prompt_desc = f"""<|system|>You are a Visual Director.
Create a text-to-image prompt based on the context.
[Context]: {last_content[:500]}

[STRICT RULES]:
1. Output ONLY the raw English prompt.
2. DO NOT include introductory phrases.
3. ABSOLUTELY NO KOREAN.
4. Style: Biblical oil painting, solemn, cinematic lighting.
<|user|>Write prompt<|assistant|>"""
        
        # 2. 생성 시도 (만약 모델이 덜 로딩되었으면 여기서 멈칫할 수 있음)
        try:
            visual_prompt = generate(model, tokenizer, prompt=prompt_desc, max_tokens=150, verbose=False).strip()
        except Exception as e:
            log("Illustrator", f"⚠️ 모델 생성 오류: {e}")
            visual_prompt = "Error"

        # ======================================================================
        # ★ [철통 보안] 한글 감지 시 강제 교체 (Iron Wall)
        # ======================================================================
        has_korean = bool(re.search(r'[가-힣]', visual_prompt))
        is_too_long = len(visual_prompt) > 400
        is_error = "Error" in visual_prompt or not visual_prompt

        if has_korean or is_too_long or is_error:
            log("Illustrator", f"⚠️ [차단] AI가 한글/잠꼬대를 했습니다. (내용: {visual_prompt[:30]}...)")
            log("Illustrator", "🛡️ [방어] '기본 안전 프롬프트'로 강제 교체하여 전송합니다.")
            
            # 안전한 기본 영어 프롬프트로 바꿔치기
            visual_prompt = "A holy biblical scene, oil painting style, cinematic lighting, solemn atmosphere, 8k resolution, detailed texture"
        
        # 사족 제거 (Here is... 등)
        visual_prompt = re.sub(r'^(Here is|Sure|Certainly|The prompt|Prompt:).*?[\:\n]', '', visual_prompt, flags=re.IGNORECASE | re.DOTALL).strip()
        
        log("Illustrator", f"의뢰서 전송(최종): {visual_prompt}")

        # 3. PC로 전송
        try:
            res = requests.post(PC_FLUX_SERVER_URL, json={"prompt": visual_prompt}, timeout=60)
            if res.status_code == 200:
                fname = res.json().get("filename")
                if fname:
                    log("Illustrator", f"✅ PC 화가로부터 그림 도착: {fname}")
                    state['polished_chunks'].append(('image', fname))
            else: 
                log("Illustrator", f"⚠️ PC 서버 오류: {res.status_code}")
        except Exception as e: 
            log("Illustrator", f"❌ PC 연결 실패: {e}")
            
    return state

def agent_designer(model, tokenizer, state: BookState) -> bool:
    log("Designer", "📚 최종 조판 작업 시작 (텍스트+삽화).")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(DIR_RESULT, f"GenesisBook_{timestamp}")
    
    fonts = state['available_fonts']
    f_title = fonts[0] if fonts else DEFAULT_FONT_TITLE
    f_body = fonts[1] if len(fonts) > 1 else DEFAULT_FONT_BODY
    log("Designer", f"사용 폰트: 제목='{f_title}', 본문='{f_body}'")

    body_code = ""
    for c_type, content in state['polished_chunks']:
        if c_type == 'text':
            clean = content
            for char in ["*", "_", "`", "$", "#", "[", "]", "<", ">", "@"]:
                clean = clean.replace(char, "\\" + char)
            body_code += f"{clean}\n\n"
            
        elif c_type == 'image':
            full_img_path = os.path.join(FACTORY_DIR, content)
            if os.path.exists(full_img_path):
                body_code += f"#v(1em)\n#figure(image(\"../{content}\", width: 90%), gap: 0.5em)\n#v(1em)\n\n"
            else:
                log("Designer", f"⚠️ 이미지 파일 누락됨 (건너뜀): {content}")

    target_tpl = state.get('replicated_template_name', "")
    ref_code = ""
    if target_tpl:
        marker = f"[Template Code: {target_tpl}]"
        idx = state['knowledge_context'].find(marker)
        if idx != -1: ref_code = state['knowledge_context'][idx:idx+8000]
    
    # 템플릿 코드 못 찾으면 검색된 지식 일부 사용
    if not ref_code: ref_code = state['knowledge_context'][:5000]

    prompt = f"""<|system|>Typst 0.11+ 버전 전문가입니다.
[참고 템플릿]: {ref_code}
[정보]: 제목="{state['book_title']}", 폰트="{f_title}"
[주의]: Typst 0.11+ 문법을 엄수하십시오. 'locate(loc => ...)' 대신 'context'를 사용하고, query 함수에 'loc' 인자를 넣지 마십시오.
[지시]: 표지와 목차 코드만 작성. 오직 Typst 코드만 출력.<|user|>작성<|assistant|>"""

    gen_code = generate(model, tokenizer, prompt=prompt, max_tokens=2000, verbose=False)
    gen_code = gen_code.replace("```typst", "").replace("```", "").strip()
    
    # [V12.5] 문법 수선 및 괄호 정리
    gen_code = KnowledgeManager.fix_typst_syntax(gen_code)

    full_typst = f"""
    // Genesis V12 RAG Ultimate Edition
    #set text(font: "{f_body}", size: 10.5pt, lang: "ko")
    
    {gen_code}
    
    #pagebreak()
    {body_code}
    """
    
    with open(f"{out_file}.typ", "w", encoding="utf-8") as f: f.write(full_typst)
    try:
        subprocess.run(["typst", "compile", f"{out_file}.typ", f"{out_file}.pdf", "--root", FACTORY_DIR, "--font-path", DIR_FONTS], check=True)
        log("Designer", f"🎉 성공: {os.path.basename(out_file)}.pdf")
        return True
    except Exception as e:
        log("Designer", f"❌ 실패: {e}")
        return False

# ==============================================================================
# [5] 메인 루프 (V11 워크플로우)
# ==============================================================================
def run_genesis_architect(model, tokenizer, input_json):
    # [V11.7] 빈도 추출
    custom_freq = DEFAULT_ILLUSTRATION_FREQ
    if "frequency" in input_json:
        try: custom_freq = int(input_json["frequency"])
        except: pass
    else:
        script_text = input_json.get("script", "")
        match = re.search(r"(?:빈도|frequency|freq)\s*[:=]\s*(\d+)", script_text, flags=re.IGNORECASE)
        if match:
            custom_freq = int(match.group(1))
            log("System", f"📋 작업지시서에서 삽화 빈도를 발견했습니다: {custom_freq}")

    state: BookState = {
        "user_instruction": input_json.get("script", ""), 
        "raw_material": input_json.get("script_ko", ""),  
        "img_snap": input_json.get("image_source", ""),
        "img_flux": input_json.get("flux_source", ""),
        "learned_style": "", "knowledge_context": "", "available_fonts": [],
        "book_title": "", "text_chunks": [], "polished_chunks": [],
        "current_chunk_idx": 0, "layout_config": {}, 
        "replicated_template_name": "", "selected_style_name": "",
        "illustration_freq": custom_freq
    }
    
    # 1. 기획 단계 (Librarian이 RAG 검색 수행)
    state = agent_librarian(model, tokenizer, state) 
    state = agent_replicator(model, tokenizer, state)
    state = agent_director(model, tokenizer, state)
    
    # 2. 제작 루프
    total = len(state['text_chunks'])
    while state['current_chunk_idx'] < total:
        state = agent_hybrid_editor(model, tokenizer, state)
        state = agent_illustrator(model, tokenizer, state)
    
    # 3. 최종 조판
    return agent_designer(model, tokenizer, state)

def main():
    print("\n" + "="*80)
    print(" 🏛️  [GENESIS WRITER V12.4: FINAL SAFETY]")
    print("     Prompt Guard Activated (Anti-Parrot)")
    print(f"     Monitoring: {FACTORY_DIR}")
    print("="*80)
    
    model, tokenizer = load_model_once()
    if not model: return
    cleanup_on_start()
    
    while True:
        try: _ = os.listdir(FACTORY_DIR)
        except: pass
        target = os.path.join(FACTORY_DIR, "pc_output.json")
        if os.path.exists(target):
            time.sleep(1)
            try:
                with open(target, 'r') as f: data = json.load(f)
                log("System", "🚀 작업 시작.")
                if run_genesis_architect(model, tokenizer, data):
                    os.rename(target, os.path.join(FACTORY_DIR, f"done_{int(time.time())}.json"))
                else: os.rename(target, os.path.join(FACTORY_DIR, target + ".err"))
            except Exception as e:
                log("System", f"⚠️ 오류: {e}")
                if os.path.exists(target): os.rename(target, target + ".err")
        time.sleep(1)

if __name__ == "__main__":
    main()
