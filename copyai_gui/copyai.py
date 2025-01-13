import os
import google.generativeai as genai
from dotenv import load_dotenv
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QVBoxLayout,
                           QLabel, QFileDialog, QTextEdit, QHBoxLayout, QMessageBox,
                           QFrame, QSpacerItem, QSizePolicy)
from PyQt5.QtGui import QPixmap, QFont, QPalette, QColor
from PyQt5.QtCore import Qt, QSize
from crewai import LLM, Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

def upload_to_gemini(file_path, mime_type=None):
    """Upload file to Gemini API"""
    try:
        file = genai.upload_file(file_path, mime_type=mime_type)
        return file
    except Exception as e:
        QMessageBox.critical(None, "Error", f"Failed to upload file to Gemini: {e}")
        return None

def get_gemini_response(file_path):
    """Get description from Gemini for an image"""
    generation_config = {
        "temperature": 0.5,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash-thinking-exp-1219",
        generation_config=generation_config,
    )

    files = [
        upload_to_gemini(file_path, mime_type="image/png"),
    ]

    if not files[0]:
        return None

    chat_session = model.start_chat(
        history=[
            {
                "role": "user",
                "parts": [
                    files[0],
                    "صف",
                ],
            }
        ]
    )

    response = chat_session.send_message("""أريدك أن تصف صورة ويب وصفًا تفصيليًا ودقيقًا بحيث يشمل كافة التفاصيل التي يحتاجها المبرمج لإعادة إنشاء الصورة بدقة. يجب أن يحتوي الوصف على:

توزيع الأقسام في الصورة:
    عدد الأقسام الرئيسية والفرعية.
    موقع كل قسم بالنسبة للصفحة.
    إذا كان القسم داخل شكل أو يتمركز بشكل حر.
    نسب وأبعاد كل قسم.
    المسافات بين الأقسام والعناصر.

الألوان:
    لون خلفية كل قسم.
    ألوان النصوص.
    ألوان الأيقونات والرموز.
    تدرجات الألوان والتأثيرات.

النصوص:
    استخراج النصوص بدقة.
    نوع الخط وحجمه وسماكته ولونه.
    تمركز النصوص وتنسيقها.

الأيقونات والإيموجي:
    وصف وموقع الأيقونات.
    ألوان وأحجام الأيقونات.

الخلفيات والأشكال:
    وصف دقيق للخلفيات والأشكال الهندسية.
    الأحجام النسبية والألوان.
    تفاصيل الحواف والتأثيرات.

التفاعلات:
    أماكن الأزرار والعناصر التفاعلية.
    وصف تأثيرات التفاعل.

عناصر أخرى:
    استخراج النصوص من الصور.
    وصف الصور والرموز الإضافية.""")
    return response

# Configure LLM
llm_gemini = LLM(
    model="gemini/gemini-2.0-flash-thinking-exp-1219",
    verbose=False,
    temperature=0.5,
    api_key=os.getenv("GEMINI_API_KEY"),
)

# Create Programmer Agent
programer = Agent(
    role='senior programmer',
    goal='Create a web page based on a description.',
    verbose=True,
    memory=True,
    backstory=(
        "Skilled programmer who does professional programming on demand. "
        "Very accurate, writes the code through the description with all professionalism."
    ),
    llm=llm_gemini,
    allow_delegation=False
)

def create_programmer_task(description):
    """Create a task for the programmer agent"""
    return Task(
        description=(
            f"Using this description: {description}\n"
            "Create accurate HTML code that matches the description. "
            "Ensure the HTML is well-structured and semantic, and save the output to the specified file."
        ),
        expected_output='One well-structured HTML file named index.html that matches the description exactly.',
        agent=programer,
        async_execution=False,
        output_file='index.html'
    )

def run_crew(description):
    """Run the crew with the programmer task"""
    programmer_task = create_programmer_task(description)
    crew = Crew(
        agents=[programer],
        tasks=[programmer_task],
        process=Process.sequential,
        verbose=True
    )
    return crew.kickoff()

class ModernButton(QPushButton):
    """Custom modern button class"""
    def __init__(self, text, primary=False):
        super().__init__(text)
        self.primary = primary
        self.setup_style()
    
    def setup_style(self):
        self.setMinimumHeight(40)
        self.setFont(QFont('Arial', 10))
        
        if self.primary:
            self.setStyleSheet("""
                QPushButton {
                    background-color: #2196F3;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    padding: 8px 16px;
                }
                QPushButton:hover {
                    background-color: #1976D2;
                }
                QPushButton:pressed {
                    background-color: #0D47A1;
                }
                QPushButton:disabled {
                    background-color: #BDBDBD;
                }
            """)
        else:
            self.setStyleSheet("""
                QPushButton {
                    background-color: #FFFFFF;
                    color: #2196F3;
                    border: 2px solid #2196F3;
                    border-radius: 4px;
                    padding: 8px 16px;
                }
                QPushButton:hover {
                    background-color: #E3F2FD;
                }
                QPushButton:pressed {
                    background-color: #BBDEFB;
                }
                QPushButton:disabled {
                    border-color: #BDBDBD;
                    color: #BDBDBD;
                }
            """)

class ModernImageFrame(QFrame):
    """Custom modern image frame class"""
    def __init__(self):
        super().__init__()
        self.setup_style()
        
    def setup_style(self):
        self.setStyleSheet("""
            QFrame {
                background-color: #F5F5F5;
                border: 2px dashed #BDBDBD;
                border-radius: 8px;
            }
        """)

class CodeGeneratorApp(QWidget):
    """Main application window"""
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("مولد كود الويب")
        self.setGeometry(100, 100, 1000, 700)
        self.setup_theme()
        
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(30, 30, 30, 30)
        
        # Header
        header_label = QLabel("مولد كود الويب")
        header_label.setFont(QFont('Arial', 24, QFont.Bold))
        header_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(header_label)
        
        # Image section
        image_section = QVBoxLayout()
        
        # Image frame
        self.image_frame = ModernImageFrame()
        self.image_frame.setFixedSize(300, 300)
        
        # Image label
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(280, 280)
        
        # Add image label to frame
        image_frame_layout = QVBoxLayout()
        image_frame_layout.addWidget(self.image_label)
        self.image_frame.setLayout(image_frame_layout)
        
        # Buttons
        self.select_button = ModernButton("اختر صورة")
        self.select_button.clicked.connect(self.select_image)
        
        self.process_button = ModernButton("ابدأ العملية", primary=True)
        self.process_button.clicked.connect(self.process_image)
        self.process_button.setEnabled(False)
        
        # Button layout
        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.select_button)
        buttons_layout.addWidget(self.process_button)
        
        # Add to image section
        image_section.addWidget(self.image_frame, alignment=Qt.AlignCenter)
        image_section.addLayout(buttons_layout)
        
        # Code section
        code_section = QVBoxLayout()
        
        code_label = QLabel("الكود الناتج:")
        code_label.setFont(QFont('Arial', 12, QFont.Bold))
        
        self.code_display = QTextEdit()
        self.code_display.setReadOnly(True)
        self.code_display.setMinimumHeight(300)
        self.code_display.setStyleSheet("""
            QTextEdit {
                background-color: #F5F5F5;
                border: none;
                border-radius: 8px;
                padding: 10px;
                font-family: 'Courier New';
                font-size: 12px;
            }
        """)
        
        code_section.addWidget(code_label)
        code_section.addWidget(self.code_display)
        
        # Add sections to main layout
        main_layout.addLayout(image_section)
        main_layout.addLayout(code_section)
        
        self.setLayout(main_layout)

    def setup_theme(self):
        """Set up the application theme"""
        self.setStyleSheet("""
            QWidget {
                background-color: white;
                color: #333333;
            }
        """)

    def select_image(self):
        """Handle image selection"""
        file_dialog = QFileDialog()
        file_dialog.setStyleSheet(self.styleSheet())
        file_path, _ = file_dialog.getOpenFileName(
            self,
            "اختر صورة",
            "",
            "Image files (*.jpg *.jpeg *.png)"
        )
        
        if file_path:
            self.image_path = file_path
            pixmap = QPixmap(self.image_path)
            scaled_pixmap = pixmap.scaled(
                280, 280,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
            self.process_button.setEnabled(True)
        else:
            self.process_button.setEnabled(False)

    def process_image(self):
        """Process the selected image"""
        if hasattr(self, 'image_path'):
            self.code_display.clear()
            self.process_button.setEnabled(False)
            self.process_button.setText("جارٍ المعالجة...")
            QApplication.processEvents()

            gemini_response = get_gemini_response(self.image_path)
            if gemini_response and gemini_response.text:
                description = gemini_response.text
                self.programmer_task = create_programmer_task(description)
                generated_code = run_crew(description)
                
                if generated_code:
                    try:
                        with open(self.programmer_task.output_file, 'r', encoding='utf-8') as f:
                            self.code_display.setText(f.read())
                    except Exception as e:
                        self.code_display.setText(f"خطأ في قراءة ملف الإخراج: {e}")
                else:
                    self.code_display.setText("فشل في إنشاء الكود.")
            else:
                self.code_display.setText("فشل في الحصول على وصف من Gemini.")

            self.process_button.setText("ابدأ العملية")
            self.process_button.setEnabled(True)
        else:
            QMessageBox.warning(self, "تحذير", "الرجاء اختيار صورة أولاً.")

if __name__ == "__main__":
    app = QApplication([])
    window = CodeGeneratorApp()
    window.show()
    app.exec_()
