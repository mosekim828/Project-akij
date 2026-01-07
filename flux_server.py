import os
import time
import json
import gc
import torch
import whisper
import threading
import uvicorn
import shutil
import paramiko
from datetime import datetime
from fastapi import FastAPI
from diffusers import FluxPipeline

# ==============================================================================
# [1] 시스템 설정 (SFTP 모드 - 끊김 없는 전송)
# ==============================================================================
# 1. Mac 연결 정보 (공유기 IP 사용 권장)
MAC_IP = "192.168.0.62"
MAC_USER = "JUSON"       # Mac 아이디
MAC_PASS = "2350QQ"        # Mac 비밀번호

# 2. ★ Mac에서 감시할 폴더 경로 (Mac상에서의 절대 경로)
# (Finder에서 폴더 우클릭 -> Option키 누름 -> '경로 이름 복사'로 확인)
REMOTE_WATCH_DIR = "/Users/juson/Genesis_Project"  

# 3. 윈도우 작업 공간 (여기로 다운로드해서 작업함)
LOCAL_TEMP_DIR = "C:/genesis_factory"

# 모델 ID
FLUX_MODEL_ID = "black-forest-labs/FLUX.1-schnell"
pipe = None

def log(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] �️ {msg}")

def flush_gpu(): 
    gc.collect()
    torch.cuda.empty_cache()

# ==============================================================================
# [2] SFTP 매니저 (배달부 설정)
# ==============================================================================
app = FastAPI()

def create_sftp_client():
    """Mac에 SSH로 접속하여 SFTP 클라이언트를 생성합니다."""
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        # 타임아웃을 넉넉히 줍니다
        ssh.connect(MAC_IP, username=MAC_USER, password=MAC_PASS, timeout=10)
        sftp = ssh.open_sftp()
        return ssh, sftp
    except Exception as e:
        log(f"⚠️ SFTP 접속 실패 (Mac 켜져 있나요?): {e}")
        return None, None

@app.on_event("startup")
def load_flux_model():
    # 로컬 작업 폴더 생성
    if not os.path.exists(LOCAL_TEMP_DIR):
        os.makedirs(LOCAL_TEMP_DIR, exist_ok=True)
        
    global pipe
    log(f"� [System] SFTP 모드 가동 (No Z-Drive)")
    log(f"� [Target] {MAC_USER}@{MAC_IP}:{REMOTE_WATCH_DIR}")
    log("� [System] RTX 5090 엔진 로딩 (Whisper + Flux)")
    
    try:
        log("� Flux 모델은 대기(CPU) 상태로 로드합니다.")
        pipe = FluxPipeline.from_pretrained(FLUX_MODEL_ID, torch_dtype=torch.bfloat16)
        log("✅ [Ready] 준비 완료")
    except Exception as e:
        log(f"❌ 모델 로딩 실패: {e}")

# ==============================================================================
# [3] 핵심 기능 (Whisper + Flux) - 기존 기능 100% 유지
# ==============================================================================

def run_whisper_job(local_file_path):
    # 로컬로 다운로드된 파일을 처리합니다.
    filename = os.path.basename(local_file_path)
    log(f"� [Whisper] 듣기 작업 시작: {filename}")
    
    global pipe
    if pipe is not None: pipe.to("cpu")
    flush_gpu()
    
    try:
        model = whisper.load_model("large-v3", device="cuda")
        result = model.transcribe(local_file_path)
        text = result["text"]
        del model; flush_gpu()
        log(f"✅ [Whisper] 완료 (글자수: {len(text)})")
        return text
    except Exception as e:
        log(f"❌ Whisper 오류: {e}")
        return None

def run_flux_job(prompt):
    global pipe
    log(f"� [Flux] 그리기 작업 시작: {prompt[:30]}...")
    timestamp = int(time.time())
    filename = f"flux_{timestamp}.png"
    
    # 로컬 임시 폴더에 저장
    local_save_path = os.path.join(LOCAL_TEMP_DIR, filename)
    
    try:
        pipe.to("cuda"); flush_gpu()
        image = pipe(
            prompt, 
            guidance_scale=0.0, 
            num_inference_steps=4, 
            max_sequence_length=512, 
            generator=torch.Generator("cuda").manual_seed(timestamp)
        ).images[0]
        
        image.save(local_save_path)
        log(f"✅ [Flux] 생성 완료: {filename}")
        
        pipe.to("cpu"); flush_gpu()
        # 파일명과 로컬 경로를 둘 다 반환 (업로드용)
        return filename, local_save_path
    except Exception as e:
        log(f"❌ Flux 오류: {e}")
        try: pipe.to("cpu")
        except: pass
        return None, None

# ==============================================================================
# [4] 작업 처리 로직 (다운로드 -> 처리 -> 업로드)
# ==============================================================================

def process_remote_task(ssh, sftp, remote_json_path, filename):
    # 윈도우 로컬 경로
    local_json_path = os.path.join(LOCAL_TEMP_DIR, filename)
    
    try:
        # 1. JSON 파일 다운로드 (Mac -> Windows)
        sftp.get(remote_json_path, local_json_path)
        
        with open(local_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        target_file = data.get("target_path", "")
        prompt = data.get("prompt", "")
        result_data = {}

        # 2. 작업 수행
        if target_file:
            # 영상 파일 처리
            video_name = os.path.basename(target_file)
            
            # Mac 경로 보정
            remote_video_path = target_file
            if not target_file.startswith("/"):
                remote_video_path = f"{REMOTE_WATCH_DIR}/{target_file}"
            
            local_video_path = os.path.join(LOCAL_TEMP_DIR, video_name)
            
            log(f"� [Download] 영상 가져오는 중: {video_name}")
            try:
                sftp.get(remote_video_path, local_video_path)
                
                # 로컬 파일로 Whisper 실행
                text = run_whisper_job(local_video_path)
                result_data = {"status": "success", "task": "transcribe", "text": text, "source": target_file}
                
                # 다 쓴 로컬 영상 삭제 (용량 관리)
                os.remove(local_video_path)
                
            except Exception as e:
                log(f"⚠️ 영상 다운로드/처리 실패: {e}")
                result_data = {"status": "error", "message": str(e)}

        elif prompt:
            # 이미지 생성 처리
            image_name, local_img_path = run_flux_job(prompt)
            
            if image_name:
                log(f"� [Upload] 결과 이미지 전송 중: {image_name}")
                remote_img_path = f"{REMOTE_WATCH_DIR}/{image_name}"
                
                # 생성된 이미지를 Mac으로 업로드
                sftp.put(local_img_path, remote_img_path)
                
                # 로컬 이미지 삭제
                os.remove(local_img_path)
                result_data = {"status": "success", "task": "image", "image_filename": image_name, "prompt": prompt}
            else:
                result_data = {"status": "error", "message": "Image generation failed"}
        
        else:
            log("⚠️ 알 수 없는 작업 요청")
            result_data = {"status": "error", "message": "Unknown task"}

        # 3. 결과 보고서 전송 (Mac으로 업로드)
        out_name = f"pc_output.json" # 파일명 고정하거나 타임스탬프 사용 가능
        local_out_path = os.path.join(LOCAL_TEMP_DIR, out_name)
        remote_out_path = f"{REMOTE_WATCH_DIR}/{out_name}"
        
        with open(local_out_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=4)
            
        sftp.put(local_out_path, remote_out_path)
        log(f"� 결과 보고서 전송 완료!")
        
        # 4. 처리된 Mac의 작업지시서 이름 변경 (.old)
        try:
            old_path = remote_json_path + ".old"
            # 기존 .old 파일이 있으면 삭제 (덮어쓰기 위해)
            try: sftp.remove(old_path)
            except: pass
            
            sftp.rename(remote_json_path, old_path)
        except Exception as e:
            log(f"⚠️ 파일명 변경 실패 (.old): {e}")
        
        # 로컬 찌꺼기 청소
        if os.path.exists(local_json_path): os.remove(local_json_path)
        if os.path.exists(local_out_path): os.remove(local_out_path)

    except Exception as e:
        log(f"❌ 작업 처리 중 에러: {e}")
        # 에러 난 파일 표시 (.err)
        try: sftp.rename(remote_json_path, remote_json_path + ".err")
        except: pass

# ==============================================================================
# [5] 감시 루프 (Watch Tower)
# ==============================================================================

def watcher_loop():
    time.sleep(3) # 서버 부팅 대기
    print("\n" + "="*60)
    print(" � [GENESIS FACTORY] SFTP LINK ACTIVATED")
    print(f" � Mac 감시 경로: {REMOTE_WATCH_DIR}")
    print("="*60)
    
    while True:
        # 매번 연결을 새로 맺어 안정성 확보 (끊김 자동 복구)
        ssh, sftp = create_sftp_client()
        
        if ssh and sftp:
            try:
                # Mac 폴더의 파일 목록 조회
                files = sftp.listdir(REMOTE_WATCH_DIR)
                
                for filename in files:
                    # JSON 파일 필터링
                    if filename.lower().endswith('.json') and \
                       "output" not in filename and \
                       not filename.endswith(".old") and \
                       not filename.endswith(".err"):
                        
                        log(f"⚡ [감지] 작업지시서 발견: {filename}")
                        remote_full_path = f"{REMOTE_WATCH_DIR}/{filename}"
                        
                        # 작업 시작
                        process_remote_task(ssh, sftp, remote_full_path, filename)
                        
            except Exception as e:
                # 연결이 끊겨도 조용히 넘어가고 다시 시도
                pass
            
            # 연결 종료 (리소스 반환)
            try: 
                sftp.close()
                ssh.close()
            except: pass
            
        # 1초 대기 후 다시 확인
        time.sleep(1)

if __name__ == "__main__":
    t = threading.Thread(target=watcher_loop, daemon=True)
    t.start()
    uvicorn.run(app, host="0.0.0.0", port=8000)
