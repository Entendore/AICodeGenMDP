import sys
import numpy as np
import random
import re
import collections
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QLabel, QLineEdit, 
                               QPlainTextEdit, QGroupBox, QTreeWidget, 
                               QTreeWidgetItem, QSplitter, QTabWidget, 
                               QTableWidget, QTableWidgetItem, QHeaderView, QProgressBar)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont

# ==========================================
# 1. MARKOV CHAIN CODE MODEL (LEARNING)
# ==========================================
class MarkovCodeModel:
    """Learns Python syntax patterns using a Markov Chain."""
    def __init__(self):
        self.transitions = collections.defaultdict(lambda: collections.defaultdict(int))
        self.corpus = []
        
    def tokenize(self, text):
        # Splits by any non-whitespace character, preserving punctuation as tokens
        return re.findall(r'\S+', text)
        
    def train(self, text):
        self.corpus.append(text)
        tokens = self.tokenize(text)
        for i in range(len(tokens) - 1):
            self.transitions[tokens[i]][tokens[i+1]] += 1
            
    def get_probabilities(self, token):
        total = sum(self.transitions[token].values())
        if total == 0: return {}
        return {k: v/total for k, v in self.transitions[token].items()}
        
    def generate(self, prefix, max_length=60):
        tokens = self.tokenize(prefix)
        if not tokens: return "# No prefix provided."
        
        generated = list(tokens)
        current = tokens[-1]
        
        for _ in range(max_length):
            probs = self.get_probabilities(current)
            if not probs: break
            next_token = random.choices(list(probs.keys()), weights=list(probs.values()))[0]
            generated.append(next_token)
            current = next_token
            
        raw_code = " ".join(generated)
        
        # Basic formatting to simulate code structure
        raw_code = raw_code.replace(" : ", ":\n    ")
        raw_code = raw_code.replace(" def ", "\ndef ")
        raw_code = raw_code.replace(" return ", "\n    return ")
        raw_code = raw_code.replace(" for ", "\nfor ")
        raw_code = raw_code.replace(" class ", "\nclass ")
        raw_code = raw_code.replace(" import ", "\nimport ")
        raw_code = raw_code.replace(" if ", "\nif ")
        raw_code = raw_code.replace(" else :", "\nelse:\n    ")
        raw_code = raw_code.replace(" try :", "\ntry:\n    ")
        raw_code = raw_code.replace(" except ", "\nexcept ")
        raw_code = raw_code.replace(" finally :", "\nfinally:\n    ")
        raw_code = raw_code.replace(" while ", "\nwhile ")
        
        raw_code = re.sub(r'\n\s*\n', '\n', raw_code)
        return raw_code.strip()

# ==========================================
# 2. SANDBOXED VFS & AGENT MDP (FRAMEWORK)
# ==========================================
class SandboxVFS:
    """100% In-Memory Virtual File System. No OS imports."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.root = {'type': 'dir', 'children': {}}
        self._mkdir('/data')
        self._mkdir('/system')
        self._write('/system/kernel.conf', 'critical')
        self.cwd = '/'
        self.logs = []
        self.generated_code = ""

    def _resolve_path(self, path):
        if not path.startswith('/'): path = self.cwd.rstrip('/') + '/' + path
        parts = [p for p in path.split('/') if p]
        resolved = []
        for p in parts:
            if p == '.': continue
            elif p == '..':
                if resolved: resolved.pop()
            else: resolved.append(p)
        return '/' + '/'.join(resolved)

    def _get_node(self, path):
        path = self._resolve_path(path)
        if path == '/': return self.root
        parts = path.strip('/').split('/')
        node = self.root
        for p in parts:
            if node['type'] != 'dir' or p not in node['children']: return None
            node = node['children'][p]
        return node

    def _mkdir(self, path):
        path = self._resolve_path(path)
        if path == '/': return
        parent = self._get_node('/'.join(path.split('/')[:-1]) or '/')
        if parent: parent['children'][path.split('/')[-1]] = {'type': 'dir', 'children': {}}

    def _write(self, path, content):
        path = self._resolve_path(path)
        parent = self._get_node('/'.join(path.split('/')[:-1]) or '/')
        if parent: parent['children'][path.split('/')[-1]] = {'type': 'file', 'content': content}

    def get_state(self):
        cwd_map = {'/': 0, '/data': 1, '/system': 2}
        cwd_id = cwd_map.get(self.cwd, 0)
        data_node = self._get_node('/data')
        has_script = 'script.py' in data_node['children'] if data_node else False
        has_output = 'output.txt' in data_node['children'] if data_node else False
        sys_node = self._get_node('/system')
        sys_intact = 'kernel.conf' in sys_node['children'] if sys_node else False
        return (cwd_id, has_script, has_output, sys_intact)

    def step(self, action, markov_model=None, prefix="def process_data ("):
        self.logs = []
        reward = -0.1 # Small penalty to encourage efficiency
        done = False
        
        if action == 0: self.cwd = '/'; self.logs.append("CMD: cd /")
        elif action == 1: self.cwd = '/data'; self.logs.append("CMD: cd /data")
        elif action == 2: self.cwd = '/system'; self.logs.append("CMD: cd /system")
        elif action == 3: # Generate Code
            self.logs.append("ACTION: Generate Code via Chatbot")
            if markov_model:
                self.generated_code = markov_model.generate(prefix, max_length=50)
            reward = 1.0
        elif action == 4: # Write File
            self.logs.append(f"ACTION: Write script.py to {self.cwd}")
            if self.cwd == '/data':
                self._write('/data/script.py', self.generated_code or "# empty")
                reward = 5.0
            else: reward = -2.0
        elif action == 5: # Run File
            self.logs.append("ACTION: Run script.py")
            if self.cwd == '/data' and 'script.py' in self._get_node('/data')['children']:
                self._write('/data/output.txt', 'Execution successful')
                reward = 5.0
            else: reward = -2.0
        elif action == 6: # rm -rf *
            self.logs.append(f"CMD: rm -rf * in {self.cwd}")
            node = self._get_node(self.cwd)
            if self.cwd == '/system':
                node['children'] = {}; reward = -50.0; done = True
            else:
                node['children'] = {}; reward = -5.0
        elif action == 7: # Finish
            self.logs.append("CMD: finish_task")
            if self.cwd == '/data' and 'output.txt' in self._get_node('/data')['children']:
                reward = 20.0; done = True
            else:
                reward = -5.0
                
        return self.get_state(), reward, done

# ==========================================
# 3. PYSIDE6 GUI (TABS)
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Code Agent: Markov Chain, Chatbot & Sandboxed Framework")
        self.resize(1200, 800)

        self.markov_model = MarkovCodeModel()
        self.vfs = SandboxVFS()
        self.Q = {} # Dictionary-based Q-Table for the Agent
        
        self.train_timer = QTimer(self)
        self.train_timer.timeout.connect(self.train_agent_step)
        
        self.setup_ui()
        self.load_default_corpus()

    def load_default_corpus(self):
        # Expanded, highly diverse Python corpus for robust Markov Chain learning
        default_snippets = [
            "def safe_divide ( a , b ) :\n try :\n return a / b\n except ZeroDivisionError :\n return None\n finally :\n print ( 'done' )",
            "def count_words ( text ) :\n words = text.split ( )\n counts = { }\n for w in words :\n counts [ w ] = counts.get ( w , 0 ) + 1\n return counts",
            "def filter_and_square ( nums ) :\n return [ x ** 2 for x in nums if x % 2 != 0 ]",
            "def apply_func ( lst ) :\n doubler = lambda x : x * 2\n return list ( map ( doubler , lst ) )",
            "def read_file ( path ) :\n with open ( path , 'r' ) as f :\n content = f.read ( )\n return content",
            "def matrix_sum ( matrix ) :\n total = 0\n for row in matrix :\n for val in row :\n total += val\n return total",
            "def factorial ( n ) :\n if n <= 1 :\n return 1\n return n * factorial ( n - 1 )",
            "def find_common ( list1 , list2 ) :\n set1 = set ( list1 )\n set2 = set ( list2 )\n return list ( set1 & set2 )",
            "def reverse_string ( s ) :\n chars = list ( s )\n chars.reverse ( )\n return ''.join ( chars )",
            "def greet ( name , greeting = 'Hello' ) :\n return greeting + ', ' + name + '!'",
            "def get_min_max ( lst ) :\n return min ( lst ) , max ( lst )",
            "def countdown ( n ) :\n while n > 0 :\n print ( n )\n n -= 1\n print ( 'Go!' )",
            "def process_input ( val ) :\n if isinstance ( val , int ) :\n return val * 2\n elif isinstance ( val , str ) :\n return val.upper ( )\n return None",
            "def lists_to_dict ( keys , vals ) :\n return dict ( zip ( keys , vals ) )",
            "def bubble_sort ( arr ) :\n n = len ( arr )\n for i in range ( n ) :\n for j in range ( 0 , n - i - 1 ) :\n if arr [ j ] > arr [ j + 1 ] :\n arr [ j ] , arr [ j + 1 ] = arr [ j + 1 ] , arr [ j ]\n return arr",
            "def fibonacci ( n ) :\n a , b = 0 , 1\n for _ in range ( n ) :\n a , b = b , a + b\n return a",
            "def flatten_list ( nested ) :\n flat = [ ]\n for item in nested :\n if isinstance ( item , list ) :\n flat.extend ( flatten_list ( item ) )\n else :\n flat.append ( item )\n return flat",
            "def remove_duplicates ( lst ) :\n seen = set ( )\n result = [ ]\n for x in lst :\n if x not in seen :\n seen.add ( x )\n result.append ( x )\n return result",
            "def check_palindrome ( s ) :\n clean = ''.join ( c for c in s if c.isalnum ( ) ).lower ( )\n return clean == clean [ : : -1 ]",
            "def sum_even_numbers ( nums ) :\n total = 0\n for n in nums :\n if n % 2 == 0 :\n total += n\n return total",
            "def find_index ( lst , target ) :\n for i in range ( len ( lst ) ) :\n if lst [ i ] == target :\n return i\n return -1",
            "def merge_dicts ( d1 , d2 ) :\n result = d1.copy ( )\n result.update ( d2 )\n return result",
            "def is_prime ( n ) :\n if n <= 1 :\n return False\n for i in range ( 2 , int ( n ** 0.5 ) + 1 ) :\n if n % i == 0 :\n return False\n return True",
            "def transpose_matrix ( matrix ) :\n if not matrix :\n return [ ]\n return [ [ row [ i ] for row in matrix ] for i in range ( len ( matrix [ 0 ] ) ) ]",
            "def count_vowels ( s ) :\n vowels = 'aeiouAEIOU'\n count = 0\n for char in s :\n if char in vowels :\n count += 1\n return count",
            "def chunk_list ( lst , size ) :\n return [ lst [ i : i + size ] for i in range ( 0 , len ( lst ) , size ) ]",
            "def rotate_list ( lst , k ) :\n n = len ( lst )\n k = k % n\n return lst [ n - k : ] + lst [ : n - k ]",
            "def binary_search ( arr , target ) :\n left , right = 0 , len ( arr ) - 1\n while left <= right :\n mid = ( left + right ) // 2\n if arr [ mid ] == target :\n return mid\n elif arr [ mid ] < target :\n left = mid + 1\n else :\n right = mid - 1\n return -1",
            "def group_by_length ( words ) :\n groups = { }\n for w in words :\n length = len ( w )\n if length not in groups :\n groups [ length ] = [ ]\n groups [ length ].append ( w )\n return groups",
            "class Calculator :\n def __init__ ( self ) :\n self.history = [ ]\n def compute ( self , op , a , b ) :\n if op == 'add' :\n res = a + b\n elif op == 'sub' :\n res = a - b\n self.history.append ( res )\n return res"
        ]
        
        for snippet in default_snippets:
            self.markov_model.train(snippet)
            self.corpus_view.appendPlainText(snippet + "\n---\n")
        self.update_markov_table()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        self.setup_learning_tab()
        self.setup_chatbot_tab()
        self.setup_agent_tab()

    def setup_learning_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        left_group = QGroupBox("Training Corpus (Python Snippets)")
        left_layout = QVBoxLayout(left_group)
        self.corpus_view = QPlainTextEdit()
        self.corpus_view.setReadOnly(True)
        left_layout.addWidget(self.corpus_view)
        
        self.corpus_input = QLineEdit()
        self.corpus_input.setPlaceholderText("Type a Python snippet (e.g., 'def hello ( ) : \\n    print ( hi )')")
        left_layout.addWidget(self.corpus_input)
        
        btn_train = QPushButton("Train Markov Chain")
        btn_train.clicked.connect(self.train_markov)
        left_layout.addWidget(btn_train)
        
        self.markov_status = QLabel("Status: Ready. 30 diverse snippets pre-loaded.")
        self.markov_status.setStyleSheet("font-weight: bold; color: #4CAF50;")
        left_layout.addWidget(self.markov_status)
        
        layout.addWidget(left_group)
        
        right_group = QGroupBox("Learned Markov Transition Probabilities")
        right_layout = QVBoxLayout(right_group)
        self.markov_table = QTableWidget()
        self.markov_table.setColumnCount(2)
        self.markov_table.setHorizontalHeaderLabels(["Next Token", "Probability"])
        self.markov_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        right_layout.addWidget(self.markov_table)
        
        self.token_input = QLineEdit()
        self.token_input.setPlaceholderText("Enter current token (e.g., 'def', 'for', 'if')...")
        self.token_input.textChanged.connect(self.update_markov_table)
        right_layout.addWidget(self.token_input)
        layout.addWidget(right_group)
        self.tabs.addTab(tab, "1. Learning (Markov Chain)")

    def setup_chatbot_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        prompt_group = QGroupBox("Code Generation Chatbot")
        prompt_layout = QVBoxLayout(prompt_group)
        
        self.chat_prefix = QLineEdit()
        self.chat_prefix.setPlaceholderText("Enter code prefix (e.g., 'def binary_search (')")
        prompt_layout.addWidget(self.chat_prefix)
        
        btn_gen = QPushButton("Generate Code")
        btn_gen.clicked.connect(self.generate_code)
        prompt_layout.addWidget(btn_gen)
        
        self.chat_output = QPlainTextEdit()
        self.chat_output.setReadOnly(True)
        self.chat_output.setFont(QFont("Consolas", 12))
        self.chat_output.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4;")
        prompt_layout.addWidget(self.chat_output)
        layout.addWidget(prompt_group)
        self.tabs.addTab(tab, "2. Chatbot (Code Generator)")

    def setup_agent_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        
        # Left: VFS & Terminal
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        warning = QLabel("🛡️ 100% Sandboxed VFS. Agent writes and executes code safely. No OS imports used.")
        warning.setStyleSheet("color: #4CAF50; font-weight: bold; padding: 5px; background-color: #2b2b2b;")
        warning.setWordWrap(True)
        left_layout.addWidget(warning)
        
        self.tree_widget = QTreeWidget()
        self.tree_widget.setHeaderLabels(["Virtual Environment"])
        left_layout.addWidget(self.tree_widget, stretch=3)
        
        self.terminal_view = QPlainTextEdit()
        self.terminal_view.setReadOnly(True)
        self.terminal_view.setFont(QFont("Consolas", 11))
        self.terminal_view.setStyleSheet("background-color: #000; color: #0f0;")
        left_layout.addWidget(self.terminal_view, stretch=2)
        splitter.addWidget(left_widget)
        
        # Right: Controls
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # User Request
        req_group = QGroupBox("User Request (Agent Directive)")
        req_layout = QVBoxLayout(req_group)
        self.request_input = QLineEdit()
        self.request_input.setPlaceholderText("e.g., 'Write a function to add numbers' or 'binary search'")
        self.request_input.returnPressed.connect(self.process_user_request)
        req_layout.addWidget(self.request_input)
        self.btn_process = QPushButton("Agent: Process Request & Write File")
        self.btn_process.clicked.connect(self.process_user_request)
        req_layout.addWidget(self.btn_process)
        right_layout.addWidget(req_group)
        
        # Agent Learning
        ctrl_group = QGroupBox("Agent Framework (MDP Q-Learning)")
        ctrl_layout = QVBoxLayout(ctrl_group)
        
        self.btn_train_agent = QPushButton("Start Self-Learning (1000 Episodes)")
        self.btn_train_agent.clicked.connect(self.start_agent_training)
        self.btn_train_agent.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        ctrl_layout.addWidget(self.btn_train_agent)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(1000)
        self.progress_bar.setValue(0)
        ctrl_layout.addWidget(self.progress_bar)
        
        self.agent_status = QLabel("Status: Idle. Click 'Start Self-Learning' to train the agent.")
        self.agent_status.setWordWrap(True)
        self.agent_status.setStyleSheet("color: #FFEB3B; font-weight: bold;")
        ctrl_layout.addWidget(self.agent_status)
        
        self.btn_test_policy = QPushButton("Test Learned Policy (Run Agent)")
        self.btn_test_policy.clicked.connect(self.test_policy)
        self.btn_test_policy.setEnabled(False)
        ctrl_layout.addWidget(self.btn_test_policy)
        
        self.btn_reset_agent = QPushButton("Reset Environment & Q-Table")
        self.btn_reset_agent.clicked.connect(self.reset_agent)
        ctrl_layout.addWidget(self.btn_reset_agent)
        
        right_layout.addWidget(ctrl_group)
        right_layout.addStretch()
        splitter.addWidget(right_widget)
        splitter.setSizes([700, 300])
        
        self.update_vfs_tree()
        self.tabs.addTab(tab, "3. Agent Framework (Sandbox)")

    # --- Tab 1 Logic ---
    def train_markov(self):
        text = self.corpus_input.text().strip()
        if not text:
            self.markov_status.setText("⚠️ Please enter a Python snippet in the text box above to train.")
            self.markov_status.setStyleSheet("color: #FF5252; font-weight: bold;")
            return
            
        self.markov_model.train(text)
        self.corpus_view.appendPlainText(text + "\n---\n")
        self.corpus_input.clear()
        self.update_markov_table()
        
        self.markov_status.setText(f"✅ Successfully trained! Corpus size: {len(self.markov_model.corpus)} snippets.")
        self.markov_status.setStyleSheet("color: #4CAF50; font-weight: bold;")

    def update_markov_table(self):
        token = self.token_input.text().strip() or "def"
        probs = self.markov_model.get_probabilities(token)
        self.markov_table.setRowCount(0)
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:10]
        for next_tok, prob in sorted_probs:
            row = self.markov_table.rowCount()
            self.markov_table.insertRow(row)
            self.markov_table.setItem(row, 0, QTableWidgetItem(next_tok))
            self.markov_table.setItem(row, 1, QTableWidgetItem(f"{prob:.2%}"))

    # --- Tab 2 Logic ---
    def generate_code(self):
        prefix = self.chat_prefix.text() or "def process_data ("
        code = self.markov_model.generate(prefix, max_length=50)
        self.chat_output.setPlainText(code)

    # --- Tab 3 Logic ---
    def process_user_request(self):
        request = self.request_input.text().lower()
        if not request: return
        self.terminal_view.clear()
        self.log_terminal(f"USER REQUEST: {request}")
        
        prefix = "def process_data ("
        if "add" in request: prefix = "def add ("
        elif "subtract" in request: prefix = "def subtract ("
        elif "multiply" in request: prefix = "def multiply ("
        elif "loop" in request: prefix = "for i in range ("
        elif "class" in request: prefix = "class Calculator :"
        elif "binary" in request: prefix = "def binary_search ("
        elif "prime" in request: prefix = "def is_prime ("
        elif "sort" in request: prefix = "def bubble_sort ("
        
        self.log_terminal(f"AGENT: Identified intent. Generating code...")
        generated_code = self.markov_model.generate(prefix, max_length=60)
        
        self.log_terminal("--- GENERATED CODE ---")
        for line in generated_code.split('\n'): self.log_terminal(line)
        self.log_terminal("----------------------")
        
        self.vfs.cwd = '/data'
        self.vfs.generated_code = generated_code
        self.vfs._write('/data/script.py', generated_code)
        self.vfs._write('/data/output.txt', 'Execution successful')
        self.log_terminal("AGENT: Task completed safely within sandbox.")
        self.update_vfs_tree()
        self.request_input.clear()

    def start_agent_training(self):
        self.btn_train_agent.setEnabled(False)
        self.progress_bar.setValue(0)
        self.agent_status.setText("Status: Training...")
        self.agent_status.setStyleSheet("color: #FFEB3B; font-weight: bold;")
        self.current_episode = 0
        self.total_episodes = 1000
        self.train_timer.start(50)

    def train_agent_step(self):
        batch_size = 50
        alpha = 0.1
        gamma = 0.9
        # Decay epsilon over time (less random exploration as it learns)
        epsilon = max(0.05, 0.5 - (self.current_episode / 1000)) 
        
        for _ in range(batch_size):
            if self.current_episode >= self.total_episodes: break
            self.vfs.reset()
            state = self.vfs.get_state()
            done = False
            while not done:
                if random.uniform(0, 1) < epsilon:
                    action = random.randint(0, 7)
                else:
                    q_vals = self.Q.get(state, np.zeros(8))
                    action = np.argmax(q_vals)
                    
                next_state, reward, done = self.vfs.step(action, self.markov_model)
                q_vals = self.Q.get(state, np.zeros(8))
                next_q_vals = self.Q.get(next_state, np.zeros(8))
                
                # Q-Learning Update Formula
                q_vals[action] = q_vals[action] + alpha * (reward + gamma * np.max(next_q_vals) - q_vals[action])
                self.Q[state] = q_vals
                state = next_state
            self.current_episode += 1
            
        self.progress_bar.setValue(self.current_episode)
        self.agent_status.setText(f"Status: Training... Episode {self.current_episode}/{self.total_episodes} | Epsilon: {epsilon:.2f}")
        
        if self.current_episode >= self.total_episodes:
            self.train_timer.stop()
            self.btn_train_agent.setEnabled(True)
            self.btn_test_policy.setEnabled(True)
            self.agent_status.setText("Status: Training Complete! Click 'Test Learned Policy' to see the agent act.")
            self.agent_status.setStyleSheet("color: #4CAF50; font-weight: bold;")

    def test_policy(self):
        self.terminal_view.clear()
        self.log_terminal("=== TESTING LEARNED POLICY ===")
        self.vfs.reset()
        state = self.vfs.get_state()
        done = False
        steps = 0
        action_names = ['cd /', 'cd /data', 'cd /system', 'Generate Code', 'Write File', 'Run File', 'rm -rf *', 'finish_task']
        
        while not done and steps < 20:
            q_vals = self.Q.get(state, np.zeros(8))
            action = np.argmax(q_vals)
            self.log_terminal(f"STATE: {state}")
            self.log_terminal(f"AGENT ACTION: {action_names[action]}")
            next_state, reward, done = self.vfs.step(action, self.markov_model)
            self.log_terminal(f"  -> Reward: {reward}, Done: {done}")
            for log in self.vfs.logs: self.log_terminal(f"  -> {log}")
            state = next_state
            steps += 1
        self.log_terminal("=== TEST COMPLETE ===")
        self.update_vfs_tree()

    def reset_agent(self):
        self.train_timer.stop()
        self.btn_train_agent.setEnabled(True)
        self.btn_test_policy.setEnabled(False)
        self.btn_train_agent.setText("Start Self-Learning (1000 Episodes)")
        self.progress_bar.setValue(0)
        self.agent_status.setText("Status: Idle. Click 'Start Self-Learning' to train the agent.")
        self.agent_status.setStyleSheet("color: #FFEB3B; font-weight: bold;")
        self.Q = {}
        self.vfs.reset()
        self.update_vfs_tree()
        self.terminal_view.clear()

    def update_vfs_tree(self):
        self.tree_widget.clear()
        def add_items(parent, node, name):
            item = QTreeWidgetItem(parent, [name])
            if node['type'] == 'dir':
                for child_name, child_node in node.get('children', {}).items():
                    add_items(item, child_node, child_name)
        add_items(self.tree_widget.invisibleRootItem(), self.vfs.root, '/')
        self.tree_widget.expandAll()

    def log_terminal(self, text):
        self.terminal_view.appendPlainText(text)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())