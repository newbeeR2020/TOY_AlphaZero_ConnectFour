import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
from collections import deque

# ==============================================================================
# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
# 提供されたAIとゲームのコード（変更なし）
# ==============================================================================

# Cell 2: Connect Four Environment
class ConnectFour:
    ROWS, COLS = 6, 7
    def __init__(self):
        self.board = np.zeros((self.ROWS, self.COLS), dtype=np.int8)
        self.current_player = 1
        self.last_move = None

    def clone(self):
        env = ConnectFour()
        env.board = self.board.copy()
        env.current_player = self.current_player
        env.last_move=self.last_move
        return env

    def valid_moves(self):
        return [c for c in range(self.COLS) if self.board[0, c] == 0]

    def play(self, col):
        for r in range(self.ROWS - 1, -1, -1):
            if self.board[r, col] == 0:
                self.board[r, col] = self.current_player
                self.current_player = - self.current_player
                self.last_move=(r, col)
                return
        raise ValueError("Invalid move")

    def _check_winner(self):
        if self.last_move is None:
            return 0

        r, c = self.last_move
        player = self.board[r, c]

        for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            count = 1
            for i in range(1, 4):
                nr, nc = r + i * dr, c + i * dc
                if 0 <= nr < self.ROWS and 0 <= nc < self.COLS and self.board[nr, nc] == player:
                    count += 1
                else:
                    break
            for i in range(1, 4):
                nr, nc = r - i * dr, c - i * dc
                if 0 <= nr < self.ROWS and 0 <= nc < self.COLS and self.board[nr, nc] == player:
                    count += 1
                else:
                    break
            
            if count >= 4:
                return player
        
        return 0

    def game_over(self):
        if self.last_move is None:
            return False
        return self._check_winner() != 0 or np.all(self.board!=0)

    def winner(self):
        if self.last_move is None:
            return 0
        return self._check_winner()

    def state(self):
        canonical_board = self.board.copy().astype(np.float32)
        opponent = - self.current_player
        canonical_board[self.board == self.current_player] = 1.0
        canonical_board[self.board == opponent] = -1.0
        canonical_board[self.board == 0] = 0.0
        return torch.tensor(canonical_board, dtype=torch.float32).unsqueeze(0)

# Cell 3: Neural Network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.policy_head = nn.Conv2d(64, 2, 1)
        self.policy_fc = nn.Linear(2 * 6 * 7, 7)
        self.value_head = nn.Conv2d(64, 1, 1)
        self.value_fc1 = nn.Linear(6 * 7, 64)
        self.value_fc2 = nn.Linear(64, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        p = F.relu(self.policy_head(x))
        p = self.policy_fc(p.view(-1, 2 * 6 * 7))
        v = F.relu(self.value_head(x))
        v = F.relu(self.value_fc1(v.view(-1, 6 * 7)))
        v = torch.tanh(self.value_fc2(v))
        return F.log_softmax(p, dim=1), v

# Cell 4: MCTS
class Node:
    def __init__(self, env, parent=None):
        self.env = env
        self.parent = parent
        self.children = {}
        self.N = 0
        self.W = 0
        self.Q = 0
        self.P = None

class MCTS:
    def __init__(self, net, sims=50, c_puct=1.0):
        self.net = net
        self.sims = sims
        self.c_puct = c_puct

    def search(self, node):
        if node.env.game_over():
            winner = node.env.winner()
            v = -1.0 if winner != 0 else 0.0
            node.N += 1
            node.W += v
            node.Q = node.W / node.N
            return -v

        if node.P is None:
            with torch.no_grad():
                p_log, v_tensor = self.net(node.env.state())
            probs = torch.exp(p_log).detach().cpu().numpy()[0]
            valid = node.env.valid_moves()
            mask = np.zeros_like(probs, dtype=np.float32)
            mask[valid] = 1.0
            probs = probs * mask
            ps = probs.sum()
            if ps > 0:
                probs /= ps
            else:
                probs = mask / mask.sum()
            node.P = probs
            v = v_tensor.item()
            node.N += 1
            node.W += v
            node.Q = node.W / node.N
            return -v

        best_score = -float('inf')
        best_action = -1
        sqrtN = math.sqrt(node.N + 1e-8)
        for action in node.env.valid_moves():
            if action in node.children:
                child = node.children[action]
                q_value = -child.Q
                u_value = self.c_puct * node.P[action] * sqrtN / (1 + child.N)
                score = q_value + u_value
            else:
                score = self.c_puct * node.P[action] * sqrtN
            if score > best_score:
                best_score = score
                best_action = action
        
        action = best_action
        
        if action not in node.children:
            env2 = node.env.clone()
            env2.play(action)
            node.children[action] = Node(env2, parent=node)

        v_parent = self.search(node.children[action])
        node.W += v_parent
        node.N += 1
        node.Q = node.W / node.N
        return -v_parent

    def get_policy(self, env, temp=1):
        root = Node(env.clone())
        was_training = self.net.training
        self.net.eval()
        try:
            for _ in range(self.sims):
                self.search(root)
        finally:
            if was_training:
                self.net.train()
        
        counts = np.array([root.children[a].N if a in root.children else 0 for a in range(env.COLS)])
        if counts.sum() == 0:
            valid = env.valid_moves()
            return np.array([1.0/len(valid) if i in valid else 0 for i in range(env.COLS)])
        if temp == 0:
            probs = np.zeros(env.COLS)
            probs[np.argmax(counts)] = 1.0
            return probs
        counts = counts ** (1.0 / temp)
        return counts / counts.sum()

# ==============================================================================
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
# ここまで
# ==============================================================================


# --- Streamlit アプリケーション部分 ---

# 定数
MODEL_PATH = 'my_model_state2.pth'

# 駒の色を決定する関数を修正
def get_piece_color(player):
    if 'human_player' in st.session_state:
        if player == st.session_state.human_player:
            return "red"
        elif player == -st.session_state.human_player:
            return "yellow"
    return "white"

def draw_board(board):
    st.write("")
    board_html = '<div style="background-color: blue; padding: 10px; border-radius: 10px; display: grid; grid-template-columns: repeat(7, 1fr); grid-gap: 5px;">'
    for r in range(ConnectFour.ROWS):
        for c in range(ConnectFour.COLS):
            color = get_piece_color(board[r, c])
            board_html += f'<div style="width: 50px; height: 50px; background-color: {color}; border-radius: 50%; border: 2px solid black;"></div>'
    board_html += '</div>'
    st.markdown(board_html, unsafe_allow_html=True)
    st.write("")


@st.cache_resource
def load_ai_model():
    net = Net()
    try:
        net.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        net.eval()
    except FileNotFoundError:
        st.error(f"モデルファイル '{MODEL_PATH}' が見つかりません。")
        st.stop()
    return net

# ゲームの初期化関数を修正
def initialize_game(human_turn):
    st.session_state.env = ConnectFour()
    st.session_state.game_over = False
    st.session_state.winner = None
    st.session_state.game_started = True
    st.session_state.human_player = human_turn # プレイヤーが1か-1かを保存
    
    # 最初のメッセージを設定
    if human_turn == 1:
        st.session_state.message = "あなたの番です。列を選択してください。"
    else:
        st.session_state.message = "AIの番です。"


def handle_ai_turn(mcts):
    env = st.session_state.env
    if not env.game_over():
        with st.spinner("AIが考えています... 🤔"):
            policy = mcts.get_policy(env, temp=0) 
            ai_move = np.argmax(policy)
            env.play(ai_move)

        if env.game_over():
            st.session_state.game_over = True
            winner = env.winner()
            # AIが勝ったかどうかの判定を修正
            if winner == -st.session_state.human_player:
                st.session_state.message = "残念、AIの勝ちです！🤖"
                st.session_state.winner = "AI"
            elif winner == 0: # 引き分け
                st.session_state.message = "引き分けです。良い勝負でした！🤝"
                st.session_state.winner = "Draw"
        else:
            st.session_state.message = "あなたの番です。列を選択してください。"


# --- メインの実行部分 ---

st.title("🤖 コネクトフォーAI対戦 🔴🟡")

# モデルとMCTSのインスタンスを準備
net = load_ai_model()
mcts = MCTS(net, sims=1600)

# ゲームが開始されたかどうかの状態を管理
if 'game_started' not in st.session_state:
    st.session_state.game_started = False

# --- ゲーム開始前の設定画面 ---
if not st.session_state.game_started:
    st.subheader("ゲーム設定")
    turn_choice = st.radio(
        "どちらの番でプレイしますか？",
        ('先手 (あなたは赤色)', '後手 (あなたは赤色)')
    )
    
    if st.button("ゲーム開始", type="primary"):
        # プレイヤーの選択に応じて初期化
        human_turn = 1 if '先手' in turn_choice else -1
        initialize_game(human_turn)
        
        # プレイヤーが後手を選んだ場合、AIが最初に手を打つ
        if st.session_state.human_player == -1:
            handle_ai_turn(mcts)
        
        st.rerun() # 画面を再描画してゲーム画面に遷移

# --- ゲーム中の画面 ---
else:
    # 盤面を描画
    draw_board(st.session_state.env.board)

    # メッセージ表示エリア
    message_placeholder = st.empty()
    message_placeholder.info(st.session_state.message)

    # プレイヤーの入力（ボタン）
    is_human_turn = (st.session_state.env.current_player == st.session_state.human_player) and not st.session_state.game_over
    
    cols = st.columns(ConnectFour.COLS)
    valid_moves = st.session_state.env.valid_moves()

    for i, col in enumerate(cols):
        # 人間のターンでない、または無効な手の場合はボタンを無効化
        is_disabled = not is_human_turn or i not in valid_moves
        if col.button(f"列 {i+1}", key=f"col_{i}", disabled=is_disabled):
            env = st.session_state.env
            env.play(i) # 人間の手を反映

            winner = env.winner()
            if env.game_over():
                st.session_state.game_over = True
                # 人間が勝ったかどうかの判定を修正
                if winner == st.session_state.human_player:
                    st.session_state.message = "おめでとうございます！あなたの勝ちです！🎉"
                    st.session_state.winner = "You"
                else: # 引き分け
                    st.session_state.message = "引き分けです。良い勝負でした！🤝"
                    st.session_state.winner = "Draw"
            else:
                # AIのターンへ
                draw_board(st.session_state.env.board) 
                handle_ai_turn(mcts)

            st.rerun() # 画面を再描画

    # ゲーム終了時のメッセージ
    if st.session_state.game_over:
        if st.session_state.winner == "You":
            st.balloons()
            message_placeholder.success(st.session_state.message)
        elif st.session_state.winner == "AI":
            message_placeholder.error(st.session_state.message)
        else: # Draw
            message_placeholder.warning(st.session_state.message)
    
    # ゲームをリセットして設定画面に戻るボタン
    if st.button("設定に戻る"):
        st.session_state.game_started = False
        st.rerun()
