import os
import json
import time
import tkinter as tk
from tkinter import messagebox, filedialog

# ==============================================================================
# [설정] 경로 매핑 (Mac -> HP Omen 변환용)
# ==============================================================================
# 1. 지시서가 저장될 위치 (Mac의 factory_input)
FACTORY_INPUT_DIR = "/Users/juson/Desktop/factory_input"

# 2. 경로 변환 규칙 (Mac에서 선택하면 -> Windows 경로로 자동 변경)
# Mac에서 이 경로를 포함하는 파일을 선택하면...
MAC_BASE_PATH = "/Users/juson/Desktop/Genesis_Project" 
# Windows(HP Omen)에서는 이 경로로 바꿔줍니다.
WIN_BASE_PATH = "Y:"

# ==============================================================================
# [GUI] 커맨더 프로그램 로직
# ==============================================================================
class GenesisCommanderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GENESIS COMMANDER (Auto Path Converter)")
        self.root.geometry("650x550")
        self.root.configure(bg="#2c3e50")

        # 스타일 설정
        label_style = {"bg": "#2c3e50", "fg": "white", "font": ("Arial", 12, "bold")}
        entry_style = {"bg": "#ecf0f1", "fg": "black", "font": ("Arial", 11)}

        # 1. 헤더
        tk.Label(root, text="🏭 GENESIS PROJECT: COMMAND CENTER", bg="#2c3e50", fg="#f1c40f", font=("Arial", 16, "bold")).pack(pady=20)

        # 2. 타겟 영상 선택 (파일 선택 기능 추가)
        tk.Label(root, text="[Target Video] (Mac에서 파일을 선택하세요)", **label_style).pack(pady=(10, 5))
        
        # 입력창과 버튼을 가로로 배치하기 위한 프레임
        file_frame = tk.Frame(root, bg="#2c3e50")
        file_frame.pack(pady=5)
        
        self.target_entry = tk.Entry(file_frame, width=50, **entry_style)
        self.target_entry.pack(side="left", ipady=5, padx=5)
        
        # ★ 파일 선택 버튼
        tk.Button(file_frame, text="📂 파일 선택", command=self.select_file, 
                  bg="#3498db", fg="black", font=("Arial", 11, "bold")).pack(side="left")

        # 3. 자동 변환 안내 문구
        self.path_info = tk.Label(root, text="※ 파일을 선택하면 자동으로 'Y:/...' 경로로 변환됩니다.", bg="#2c3e50", fg="gray", font=("Arial", 10))
        self.path_info.pack(pady=(0, 15))

        # 4. 작업 지시 사항 입력
        tk.Label(root, text="[Instruction] (AI에게 내릴 명령)", **label_style).pack(pady=(10, 5))
        self.instruction_text = tk.Text(root, height=8, width=70, **entry_style)
        self.instruction_text.pack(pady=5)
        self.instruction_text.insert("1.0", "이 영상을 바탕으로 300페이지 분량의 심층 서적을 집필해줘.")

        # 5. 명령 버튼
        btn_frame = tk.Frame(root, bg="#2c3e50")
        btn_frame.pack(pady=30)

        tk.Button(btn_frame, text="🚀 작전 개시 (Launch)", command=self.create_order, 
                  bg="#e74c3c", fg="black", font=("Arial", 14, "bold"), width=20, height=2).pack()

        # 하단 상태바
        self.status_label = tk.Label(root, text=f"Output: {FACTORY_INPUT_DIR}", bg="#2c3e50", fg="gray")
        self.status_label.pack(side="bottom", pady=10)

    def select_file(self):
        """파일 탐색기를 열고, 선택된 경로를 Windows 포맷으로 변환"""
        file_path = filedialog.askopenfilename(
            initialdir=MAC_BASE_PATH,
            title="Genesis Project 영상 선택",
            filetypes=[("Video files", "*.mp4 *.mov *.mkv *.avi"), ("All files", "*.*")]
        )
        
        if file_path:
            # 경로 변환 로직 (Mac -> Win)
            # 만약 선택한 파일이 Genesis_Project 폴더 안에 있다면?
            if MAC_BASE_PATH in file_path:
                # 1. 앞부분(/Users/.../Genesis_Project)을 떼어내고 뒷부분만 남김
                relative_path = file_path.replace(MAC_BASE_PATH, "")
                # 2. Y드라이브 주소(Y:/Genesis_Project)를 앞에 붙임
                final_path = WIN_BASE_PATH + relative_path
                self.path_info.config(text=f"✅ 변환됨: {final_path}", fg="#2ecc71")
            else:
                # 밖에서 선택했다면 경고하고 그냥 원래 경로 넣음 (Omen이 못 읽을 수 있음)
                final_path = file_path
                self.path_info.config(text="⚠️ 주의: Genesis_Project 외부 파일입니다. HP Omen이 못 읽을 수 있습니다.", fg="#e74c3c")
            
            # 입력창에 채워넣기
            self.target_entry.delete(0, tk.END)
            self.target_entry.insert(0, final_path)

    def create_order(self):
        target_path = self.target_entry.get().strip()
        instruction = self.instruction_text.get("1.0", tk.END).strip()

        if not instruction:
            messagebox.showwarning("경고", "지시 사항을 입력해주세요!")
            return

        # JSON 데이터 생성
        order_data = {
            "instruction": instruction,
            "target_path": target_path,
            "timestamp": time.time()
        }

        # 파일명 생성
        filename = f"command_{int(time.time())}.json"
        save_path = os.path.join(FACTORY_INPUT_DIR, filename)

        # 폴더 확인 및 생성
        if not os.path.exists(FACTORY_INPUT_DIR):
            try:
                os.makedirs(FACTORY_INPUT_DIR)
            except Exception as e:
                messagebox.showerror("에러", f"폴더 생성 실패: {e}")
                return

        # JSON 파일 쓰기
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(order_data, f, ensure_ascii=False, indent=4)
            
            messagebox.showinfo("성공", f"✅ 명령 하달 완료!\nHP Omen이 곧 작업을 시작합니다.")
            
        except Exception as e:
            messagebox.showerror("실패", f"파일 저장 중 오류 발생: {e}")

# ==============================================================================
# [메인 실행]
# ==============================================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = GenesisCommanderApp(root)
    root.mainloop()
