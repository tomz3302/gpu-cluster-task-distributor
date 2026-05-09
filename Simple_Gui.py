import sys
import os
import re
import time
import requests
import json
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QTextEdit, QLabel, QScrollArea, QFrame,
    QSizePolicy
)
from PySide6.QtCore import QThread, Signal, Qt, QTimer
from PySide6.QtGui import QFont, QColor, QTextCursor

HAPROXY_URL = "http://localhost:8080/generate"


class RequestWorker(QThread):
    response_received = Signal(str, dict)
    error_occurred = Signal(str)
    
    def __init__(self, query):
        super().__init__()
        self.query = query
    
    def run(self):
        try:
            payload = {
                "id": 1,
                "query": self.query,
                "max_tokens": 512,
                "temperature": 0.2,
                "top_k": 3,
                "use_rag": True
            }
            
            start_time = time.time()
            response = requests.post(HAPROXY_URL, json=payload, timeout=180)
            end_time = time.time()
            response.raise_for_status()
            
            data = response.json()
            answer = data.get("result", "No answer provided")
            
            # Prefer the HAProxy injected header, fallback to the node's internal environment variable
            worker_name = response.headers.get("X-Worker-Name", data.get("worker_name", "Unknown Node"))
            
            # Calculate simple metrics
            time_taken = end_time - start_time
            approx_tokens = len(answer.split())
            speed = approx_tokens / time_taken if time_taken > 0 else 0
            
            metrics = {
                "worker": worker_name,
                "time": round(time_taken, 2),
                "tokens": approx_tokens,
                "speed": round(speed, 2)
            }
            
            self.response_received.emit(answer, metrics)
            
        except Exception as e:
            self.error_occurred.emit(str(e))


class TypingAnimation(QThread):
    char_typed = Signal(str)
    finished = Signal()
    
    def __init__(self, text, speed=0.015):
        super().__init__()
        self.text = text
        self.speed = speed
    
    def run(self):
        for i in range(len(self.text)):
            self.char_typed.emit(self.text[:i+1])
            self.msleep(int(self.speed * 1000))
        self.finished.emit()


class ChatBubble(QWidget):
    def __init__(self, text, is_user=False):
        super().__init__()
        self.is_user = is_user
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        
        # Message label
        self.label = QLabel(text)
        self.label.setWordWrap(True)
        self.label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        
        # Add Assistant name to response bubbles
        if not self.is_user:
            self.assistant_name = QLabel("<b>Assistant</b>")
            self.assistant_name.setStyleSheet("color: #e0e0e0; font-size: 14px; font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;")
            
            # Container for assistant name and text
            assistant_container = QWidget()
            assistant_layout = QVBoxLayout(assistant_container)
            assistant_layout.setContentsMargins(0, 0, 0, 0)
            assistant_layout.setSpacing(5)
            assistant_layout.addWidget(self.assistant_name)
            assistant_layout.addWidget(self.label)
            
            self.label_widget = assistant_container
        else:
            self.label_widget = self.label
        
        if self.is_user:
            # User styling (right aligned, dark bubble)
            layout.addStretch()
            self.label.setStyleSheet("""
                background-color: #2d3748;
                color: #ffffff;
                padding: 12px 16px;
                border-radius: 18px;
                font-size: 14px;
                font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
            """)
            layout.addWidget(self.label_widget)
        else:
            # Assistant styling (left aligned, plain text, full width)
            # Set size policy to expanding to occupy full width
            self.label_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            self.label.setStyleSheet("""
                color: #e0e0e0;
                font-size: 14px;
                line-height: 1.6;
                font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
            """)
            layout.addWidget(self.label_widget)
            # Removed layout.addStretch() to let it occupy the whole page

    def update_text(self, text):
        if not self.is_user:
            import re
            html = text
            # Escape HTML first
            html = html.replace('<', '&lt;').replace('>', '&gt;')
            # Bold
            html = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', html)
            html = re.sub(r'__(.*?)__', r'<b>\1</b>', html)
            # Italic
            html = re.sub(r'\*(.*?)\*', r'<i>\1</i>', html)
            html = re.sub(r'_(.*?)_', r'<i>\1</i>', html)
            # Code blocks
            html = re.sub(r'```[\s\S]*?\n([\s\S]*?)```', r'<pre style="background-color: #2d3748; padding: 10px; border-radius: 6px; border: 1px solid #4a5568; font-family: monospace; color: #e0e0e0;"><code>\1</code></pre>', html)
            html = re.sub(r'```([\s\S]*?)```', r'<pre style="background-color: #2d3748; padding: 10px; border-radius: 6px; border: 1px solid #4a5568; font-family: monospace; color: #e0e0e0;"><code>\1</code></pre>', html)
            # Inline code
            html = re.sub(r'`(.*?)`', r'<code style="background-color: #2d3748; padding: 2px 4px; border-radius: 4px; color: #f56565; font-family: monospace;">\1</code>', html)
            
            # Simple list items
            html = re.sub(r'^\s*[\-\*]\s+(.*)$', r'• \1', html, flags=re.MULTILINE)
            
            # Highlight list titles if they have a colon (e.g. "1. Distributed parallelism: ...")
            html = re.sub(r'^(\s*\d+\.\s+[^\:]+:)(.*)$', r'<b>\1</b>\2', html, flags=re.MULTILINE)
            html = re.sub(r'^(\s*[•\-\*]\s+[^\:]+:)(.*)$', r'<b>\1</b>\2', html, flags=re.MULTILINE)
            
            # Line breaks
            html = html.replace('\n', '<br>')
            
            self.label.setTextFormat(Qt.RichText)
            self.label.setText(html)
        else:
            self.label.setText(text)


class ChatWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chat")
        self.setGeometry(100, 100, 900, 700)
        self.worker = None
        self.typing_thread = None
        self.current_assistant_bubble = None
        
        self.setup_ui()
    
    def setup_ui(self):
        # Main widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Apply generic stylesheet
        central.setStyleSheet("""
            QWidget {
                background-color: #0f1419;
                color: #e0e0e0;
                font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
            }
            QLineEdit {
                background-color: #1a1f2b;
                color: #e0e0e0;
                border: 1px solid #2d3748;
                border-radius: 18px;
                padding: 12px 18px;
                font-size: 14px;
            }
            QLineEdit:focus {
                border: 1px solid #4a5568;
                background-color: #0f1419;
                outline: none;
            }
            QPushButton {
                background-color: #3182ce;
                color: #ffffff;
                border: none;
                border-radius: 18px;
                padding: 10px 20px;
                font-weight: 500;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #2b6cb0;
            }
            QPushButton:pressed {
                background-color: #2c5282;
            }
            QPushButton:disabled {
                background-color: #2d3748;
                color: #4a5568;
            }
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                background-color: transparent;
                width: 8px;
                margin: 0px 0px 0px 0px;
            }
            QScrollBar::handle:vertical {
                background-color: #4a5568;
                border-radius: 4px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #718096;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        
        # Header
        header = QFrame()
        header.setStyleSheet("background-color: #0f1419; border-bottom: 1px solid #2d3748;")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(20, 16, 20, 16)
        
        title = QLabel("Chat")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        header_layout.addWidget(title, alignment=Qt.AlignCenter)
        
        main_layout.addWidget(header)
        
        # Chat area (Scroll view)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        
        self.chat_container = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_container)
        self.chat_layout.setContentsMargins(20, 20, 20, 20)
        self.chat_layout.setSpacing(15)
        self.chat_layout.addStretch()
        
        self.scroll_area.setWidget(self.chat_container)
        main_layout.addWidget(self.scroll_area, 1)
        
        # Input area
        input_widget = QWidget()
        # Ensure it resizing cleanly with window width
        input_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        input_layout = QHBoxLayout(input_widget)
        # Use responsive margins (less hardcoded px) 
        input_layout.setContentsMargins(20, 16, 20, 24)
        input_layout.setSpacing(10)
        
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Message...")
        self.input_field.returnPressed.connect(self.send_message)
        self.input_field.setMinimumHeight(44)
        self.input_field.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)
        self.send_button.setMinimumHeight(44)
        self.send_button.setCursor(Qt.PointingHandCursor)
        self.send_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        
        input_layout.addWidget(self.input_field)
        input_layout.addWidget(self.send_button)
        
        main_layout.addWidget(input_widget)
        
        # Add welcome message
        self.add_message("Hi! Ask me anything about distributed computing and I'll help you out.", is_user=False)
    
    def add_message(self, text, is_user=False):
        """Add a ChatBubble to the display"""
        bubble = ChatBubble(text, is_user=is_user)
        
        # Insert before the stretch at the end
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, bubble)
        
        # Scroll to bottom
        QTimer.singleShot(100, self.scroll_to_bottom)
        
        return bubble
    
    def scroll_to_bottom(self):
        vbar = self.scroll_area.verticalScrollBar()
        vbar.setValue(vbar.maximum())
    
    def send_message(self):
        """Send message"""
        query = self.input_field.text().strip()
        
        if not query:
            return
        
        # Add user message
        self.add_message(query, is_user=True)
        self.input_field.clear()
        self.send_button.setEnabled(False)
        
        # Placeholder for assistant response
        self.current_assistant_bubble = self.add_message("Thinking...", is_user=False)
        
        # Send request
        self.worker = RequestWorker(query)
        self.worker.response_received.connect(self.on_response)
        self.worker.error_occurred.connect(self.on_error)
        self.worker.start()
    
    def on_response(self, response_text, metrics):
        """Handle response"""
        self.current_assistant_bubble.update_text("")
        
        # Append metrics string to the end of the text
        metrics_str = f"\n\n_🖥️ Worker: {metrics['worker']} | ⏱️ Time: {metrics['time']}s | 📝 Approx Tokens: {metrics['tokens']} | ⚡ Speed: {metrics['speed']} w/s_"
        full_text = response_text + metrics_str
        
        self.start_typing_animation(full_text)
    
    def start_typing_animation(self, text):
        self.typing_thread = TypingAnimation(text, speed=0.0015)
        self.typing_thread.char_typed.connect(self.on_char_typed)
        self.typing_thread.finished.connect(self.on_typing_finished)
        self.typing_thread.start()
        
    def on_char_typed(self, current_text):
        if self.current_assistant_bubble:
            self.current_assistant_bubble.update_text(current_text)
            # Optionally scroll to bottom during long generation
            QTimer.singleShot(10, self.scroll_to_bottom)

    def on_typing_finished(self):
        self.send_button.setEnabled(True)
    
    def on_error(self, error_msg):
        """Handle error"""
        error_text = "I encountered an error. Make sure HAProxy is running on http://localhost:8080"
        self.current_assistant_bubble.update_text("")
        self.start_typing_animation(error_text)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChatWindow()
    window.show()
    sys.exit(app.exec())