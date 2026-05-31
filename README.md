# AI Arena — Web Game Platform

클래식 아케이드 게임 6종을 AI와 실시간으로 대결하는 웹 게임 플랫폼입니다.  
Pong·Boxing·Space Invaders 세 게임은 **PPO(Proximal Policy Optimization) 강화학습 에이전트**가 탑재되어 있으며, 나머지는 검증된 고전 AI 알고리즘을 사용합니다.

---

## 목차

- [주요 기능](#주요-기능)
- [게임 목록](#게임-목록)
- [AI 설계](#ai-설계)
- [프로젝트 구조](#프로젝트-구조)
- [빠른 시작](#빠른-시작)
- [RL 모델 재학습](#rl-모델-재학습)
- [기술 스택](#기술-스택)
- [크레딧](#크레딧)

---

## 주요 기능

| 기능 | 설명 |
|------|------|
| **6개 게임** | Pong, Gomoku, Chess, Boxing, Space Invaders, Tetris |
| **3단계 난이도** | Easy(랜덤/약한 AI) → Medium(중간 체크포인트) → Hell(최강 모델) |
| **PPO 강화학습** | Pong·Boxing·Space Invaders에 학습된 신경망 AI 내장 |
| **듀얼스크린 대결** | Space Invaders·Tetris는 인간(좌)과 AI(우)가 나란히 동시 플레이 |
| **브라우저 추론** | RL 모델을 JSON으로 변환해 Python 서버 없이 브라우저에서 직접 추론 |
| **리더보드** | 게임별·난이도별 점수 기록 (JSON DB) |
| **커뮤니티** | 게시글, 좋아요, 댓글 |
| **게스트 로그인** | 회원가입 없이 즉시 플레이 |

---

## 게임 목록

### 1. Pong (Atari, 1972)
클래식 탁구 게임. 플레이어가 왼쪽 패들, AI가 오른쪽 패들을 조작합니다.

- **조작**: `W`/`S` 또는 `↑`/`↓`
- **승리 조건**: 11점 먼저 달성
- **AI**: PPO 신경망 (obs 6차원, act 3개)
  - Easy: 랜덤 정책
  - Medium: `pong_mid.json` (step 501,760 체크포인트)
  - Hell: `pong_best.json` (best reward, step 665,600)

---

### 2. Gomoku (오목, 15×15)
15×15 바둑판에서 5목을 먼저 만들면 승리합니다.

- **조작**: 클릭으로 돌 배치
- **승리 조건**: 가로·세로·대각선 5개 연속
- **AI**: 비트보드 기반 Alpha-Beta Minimax (Web Worker)
  - Easy: 그리디 휴리스틱 (즉시 응답)
  - Medium: 6-ply 미니맥스
  - Hell: 8-ply 미니맥스

---

### 3. Chess
Stockfish 18 WASM 체스 엔진을 사용합니다. 플레이어는 흰색(선수)입니다.

- **조작**: 기물 드래그 앤 드롭
- **AI**: [Stockfish 18](https://github.com/nmrugg/stockfish.js) (Lite Single-threaded WASM, 7 MB)
  - Easy: Skill Level 3, depth 2
  - Medium: Skill Level 10, depth 8
  - Hell: Skill Level 20, depth 15

---

### 4. Boxing (탑뷰 캔버스)
링 위에서 두 복서가 이동하고 펀치를 날리는 게임. 플레이어는 흰색(좌측)입니다.

- **조작**: `WASD`/화살표(이동), `K`/`L`/`Space`(펀치)
- **승리 조건**: 100점 먼저 달성 또는 2분 후 높은 점수
- **AI**: PPO 신경망 (obs 14차원, act 18개)
  - Easy: 랜덤 정책
  - Medium: `boxing_mid.json` (step 501,760)
  - Hell: `boxing_best.json` (best reward, step 342,016)

---

### 5. Space Invaders (Taito, 1978) — **듀얼스크린**
두 화면이 나란히 표시됩니다. 왼쪽은 플레이어, 오른쪽은 AI가 조종합니다.

```
┌──────────────────┬────┬──────────────────┐
│ 🎮 You (Player 1)│ VS │ 🤖 AI (Player 2) │
│   키보드 조작     │    │   PPO 에이전트    │
└──────────────────┴────┴──────────────────┘
```

- **조작**: `←`/`→`(이동), `Space`(발사) — 왼쪽 화면만
- **AI**: PPO 신경망 (obs 20차원, act 6개)
  - Easy: 랜덤 정책
  - Medium: `si_mid.json` (step 751,616)
  - Hell: `si_best.json` (best reward, step 1,433,600)

---

### 6. Tetris — **듀얼스크린**
두 화면이 나란히 표시됩니다. 왼쪽은 플레이어, 오른쪽은 AI가 자동으로 플레이합니다.

```
┌──────────────────┬────┬──────────────────┐
│ 🎮 You (Player 1)│ VS │ 🤖 AI (Player 2) │
│   키보드 조작     │    │  휴리스틱 AI      │
└──────────────────┴────┴──────────────────┘
```

- **조작**: `←`/`→`(이동), `↑`/`Z`(회전), `↓`(소프트드롭), `Space`(하드드롭) — 왼쪽 화면만
- **AI**: 4-가중치 휴리스틱 평가 함수 ([tetrisai](https://github.com/LeeYiyuan/tetrisai) 포팅)
  - Easy: 랜덤 배치
  - Medium: 1-피스 룩어헤드
  - Hell: 2-피스 룩어헤드 (유전 알고리즘으로 튜닝된 가중치)

---

## AI 설계

### PPO 강화학습 (Pong · Boxing · Space Invaders)

세 게임의 AI는 동일한 **ActorCritic PPO** 아키텍처를 공유합니다.

#### 신경망 구조

```
입력 (obs_dim)
    │
Linear → Tanh           256 노드
    │
Linear → Tanh           256 노드
    │
Actor Head              act_dim 노드 (Categorical 분포)
Critic Head             1 노드 (상태가치)
```

#### 게임별 관측·행동 공간

| 게임 | obs_dim | act_dim | 관측 내용 | 행동 |
|------|---------|---------|-----------|------|
| Pong | 6 | 3 | 공 위치·속도, 양쪽 패들 위치 | stay / up / down |
| Boxing | 14 | 18 | 두 복서 위치·점수·펀치·다운 상태, 남은 시간 | 9방향이동 × 펀치여부 |
| Space Invaders | 20 | 6 | 플레이어 위치·총알, 에일리언 대형, 폭탄 3개, 타깃 힌트 | stay·left·right × 발사여부 |

#### 학습 결과

| 게임 | 총 스텝 | 최고 avg-50 리워드 | mid 체크포인트 |
|------|---------|-------------------|----------------|
| Pong | 1,000,000 | +8.04 | step 501,760 |
| Boxing | 1,000,000 | +114.156 | step 501,760 |
| Space Invaders | 1,500,000 | +50.47 | step 751,616 |

#### 브라우저 추론 방식

Python·PyTorch 없이 브라우저에서 바로 실행됩니다.

```
학습 (Python)                     배포 (Browser)
─────────────────────────────     ──────────────────────────────
PPO → ppo_agent.py             →  export_weights.py
      ActorCritic (PyTorch)    →  model.json  (가중치 배열)
                                →  pong-rl-agent.js (JS 추론)
                                →  predict(obs) → action
```

`pong-rl-agent.js`의 `RLAgent` 클래스가 Pong·Boxing·Space Invaders 세 게임 모두에서 재사용됩니다.

---

### Gomoku — Alpha-Beta Minimax (Web Worker)

```
메인 스레드: 보드 → 비트보드 변환 → Worker 메시지 전송
Web Worker:  우선순위 탐색 → Alpha-Beta Pruning → 최적 수 반환
```

`gomoku-ai-worker.js`는 `gomoku/` 레퍼런스 프로젝트를 자체 포함형(self-contained)으로 포팅한 파일입니다.  
Medium은 6-ply, Hell은 8-ply로 탐색합니다. AI 계산 중에는 "AI thinking…" 애니메이션이 표시됩니다.

---

### Chess — Stockfish 18 WASM (UCI 프로토콜)

```javascript
engine.postMessage('uci');
engine.postMessage('setoption name Skill Level value 20');
engine.postMessage('position startpos moves e2e4 ...');
engine.postMessage('go depth 15');
// → bestmove d7d5
```

`stockfish-18-lite-single.js` + `.wasm` (총 ~7.5 MB)을 `/public/` 에 포함해 서빙합니다.  
CORS 헤더 불필요, 단일 스레드, 모든 모던 브라우저 지원.

---

### Tetris — 4-가중치 휴리스틱 (tetrisai 포팅)

```
점수 = -h × aggregateHeight + l × lines - o × holes - b × bumpiness
```

| 파라미터 | Medium | Hell (유전 알고리즘 튜닝) |
|----------|--------|--------------------------|
| h (높이 페널티) | 0.510 | 0.510 |
| l (라인 보너스) | 0.600 | 0.761 |
| o (구멍 페널티) | 0.450 | 0.357 |
| b (울퉁불퉁 페널티) | 0.300 | 0.184 |

Medium은 현재 피스만(1-piece lookahead), Hell은 현재+다음(2-piece lookahead)으로 탐색합니다.

---

## 프로젝트 구조

```
Game_web/
├── server.js                    # Node.js HTTP 서버 (API + 정적 파일)
├── boxing-bridge.js             # Node ↔ Python 브릿지 (구 Boxing용, 레거시)
├── package.json
│
├── public/                      # 웹 프론트엔드
│   ├── index.html               # 단일 페이지 앱 (SPA)
│   ├── css/
│   │   ├── styles.css           # 전체 UI 스타일
│   │   └── game-icons.css       # 게임 카드 아이콘
│   ├── js/
│   │   ├── app.js               # SPA 라우터, 인증, 리더보드
│   │   ├── ai-config.js         # 난이도 프리셋
│   │   ├── game-icons.js        # 게임 카드 렌더러
│   │   ├── pong-rl-agent.js     # 범용 PPO 추론 엔진 (Pong·Boxing·SI 공용)
│   │   ├── gomoku-ai-worker.js  # Gomoku Alpha-Beta Worker
│   │   └── games/
│   │       ├── canvas-arena.js  # 캔버스 게임 마운트 헬퍼
│   │       ├── pong-core.js     # Pong 게임 엔진
│   │       ├── pong.js          # Pong AI 통합 (PPO 3모드)
│   │       ├── gomoku.js        # Gomoku 게임 + Worker AI
│   │       ├── chess.js         # Chess + Stockfish UCI 통합
│   │       ├── boxing.js        # Boxing 게임 엔진 + PPO 통합
│   │       ├── space-invaders-core.js  # SI 게임 엔진
│   │       ├── space-invaders.js       # SI 듀얼스크린 + PPO 통합
│   │       └── tetris.js        # Tetris 듀얼스크린 + 휴리스틱 AI (자체 완결)
│   ├── models/                  # PPO 가중치 JSON (브라우저 직접 로드)
│   │   ├── pong_mid.json        # Pong mid  (501,760 steps, 1.5 MB)
│   │   ├── pong_best.json       # Pong best (665,600 steps, 1.5 MB)
│   │   ├── boxing_mid.json      # Boxing mid  (501,760 steps, 1.6 MB)
│   │   ├── boxing_best.json     # Boxing best (342,016 steps, 1.6 MB)
│   │   ├── si_mid.json          # SI mid  (751,616 steps, 1.5 MB)
│   │   └── si_best.json         # SI best (1,433,600 steps, 1.5 MB)
│   ├── stockfish.js             # Stockfish 18 Lite Single (WASM 래퍼)
│   ├── stockfish.wasm           # Stockfish WASM 바이너리 (7 MB)
│   └── img/chesspieces/wikipedia/  # 체스 기물 이미지 (로컬, CDN 불필요)
│
├── pong_rl/                     # Pong RL 학습 패키지
│   ├── pong_env.py              # 게임 환경 (JS 물리 1:1 재현)
│   ├── ppo_agent.py             # ActorCritic PPO 에이전트
│   ├── train.py                 # 학습 스크립트
│   ├── export_weights.py        # PyTorch → JSON 변환
│   ├── play.py                  # 학습된 모델 테스트
│   ├── models/
│   │   ├── mid_model.pt
│   │   └── best_model.pt
│   └── requirements.txt
│
├── boxing_rl/                   # Boxing RL 학습 패키지
│   ├── boxing_env.py            # 게임 환경 (JS 물리 1:1 재현)
│   ├── ppo_agent.py
│   ├── train.py
│   ├── export_weights.py
│   └── models/
│       ├── mid_model.pt
│       └── best_model.pt
│
├── space_invaders_rl/           # Space Invaders RL 학습 패키지
│   ├── space_invaders_env.py    # 게임 환경 (JS 물리 1:1 재현)
│   ├── ppo_agent.py
│   ├── train.py
│   ├── export_weights.py
│   └── models/
│       ├── mid_model.pt
│       └── best_model.pt
│
├── services/boxing/             # (레거시) PettingZoo Boxing 서비스
│   ├── boxing_web_service.py
│   └── requirements.txt
│
├── data/
│   └── db.json                  # 리더보드 + 커뮤니티 데이터 (JSON)
│
└── jstetris/                    # (레거시) iframe Tetris 소스 — 현재 미사용
    └── ...
```

---

## 빠른 시작

### 요구사항

| 항목 | 버전 |
|------|------|
| Node.js | 18+ |
| Python | 3.10+ (RL 재학습 시에만 필요) |
| CUDA (선택) | GPU 학습 가속 |

### 설치 및 실행

```bash
# 1. 의존성 설치
npm install

# 2. 서버 시작
npm start
# 또는: node server.js
```

브라우저에서 [http://localhost:3000](http://localhost:3000) 접속

> **참고**: 모든 게임은 서버 시작 즉시 플레이 가능합니다.  
> 체스는 Stockfish WASM(~7 MB)을 첫 플레이 시 로드합니다(약 2~3초).

---

## RL 모델 재학습

PPO 모델을 처음부터 다시 학습시키거나 하이퍼파라미터를 조정할 수 있습니다.

### Python 환경 설치

```bash
pip install torch numpy
```

GPU가 있으면 자동으로 CUDA를 사용합니다.

### Pong

```bash
cd pong_rl
python train.py --steps 1000000 --save-dir models
python export_weights.py --save-dir models --out-dir ../public/models
```

### Boxing

```bash
cd boxing_rl
python train.py --steps 1000000 --save-dir models
python export_weights.py --save-dir models --out-dir ../public/models
```

### Space Invaders

```bash
cd space_invaders_rl
python train.py --steps 1500000 --save-dir models
python export_weights.py --save-dir models --out-dir ../public/models
```

학습이 완료되면 서버를 재시작할 필요 없이 브라우저를 새로고침하면 새 모델이 반영됩니다.

#### 저장 체크포인트

| 파일 | 저장 시점 |
|------|-----------|
| `mid_model.pt` | 전체 스텝의 50% 시점 |
| `best_model.pt` | 최근 50 에피소드 평균 리워드가 역대 최고일 때마다 갱신 |

#### 모델 파일 구조 (JSON)

```json
{
  "shared_0_weight": [[...], ...],   // Linear(obs, 256) 가중치
  "shared_0_bias":   [...],
  "shared_2_weight": [[...], ...],   // Linear(256, 256) 가중치
  "shared_2_bias":   [...],
  "actor_weight":    [[...], ...],   // Linear(256, act) 가중치
  "actor_bias":      [...],
  "meta": { "total_steps": 665600, "updates": 325, "obs_dim": 6, "act_dim": 3 }
}
```

---

## 기술 스택

### 프론트엔드

| 기술 | 용도 |
|------|------|
| Vanilla JS (ES2020+) | SPA 라우터, 게임 로직 전체 |
| HTML5 Canvas | Pong, Boxing, Space Invaders, Tetris, Gomoku 렌더링 |
| Web Workers | Gomoku AI 비동기 탐색 |
| Fetch API | PPO 모델 JSON 로드, REST API 통신 |
| chessboard.js + chess.js | 체스 UI 및 룰 검증 (CDN) |
| p5.js | Gomoku 캔버스 렌더링 (CDN) |

### 백엔드

| 기술 | 용도 |
|------|------|
| Node.js (built-in `http`) | HTTP 서버, 정적 파일 서빙 |
| JSON 파일 (`data/db.json`) | 리더보드, 커뮤니티 데이터 저장 |

### AI / ML

| 기술 | 용도 |
|------|------|
| PyTorch | PPO 에이전트 학습 |
| NumPy | 학습 환경 시뮬레이션 |
| Stockfish 18 WASM | 체스 엔진 (브라우저 내 실행) |

### API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| POST | `/api/auth/signup` | 회원가입 |
| POST | `/api/auth/login` | 로그인 |
| POST | `/api/auth/guest` | 게스트 로그인 |
| GET | `/api/leaderboard?game=&difficulty=` | 리더보드 조회 |
| POST | `/api/leaderboard` | 점수 등록 |
| DELETE | `/api/leaderboard` | 리더보드 초기화 |
| GET | `/api/community/posts` | 게시글 목록 |
| POST | `/api/community/posts` | 게시글 작성 |
| POST | `/api/community/posts/:id/like` | 좋아요 토글 |
| POST | `/api/community/posts/:id/comments` | 댓글 작성 |

---

## 크레딧

| 구성요소 | 출처 | 라이선스 |
|----------|------|---------|
| Pong 게임 엔진 | [juliensimon/browser-games](https://github.com/juliensimon/browser-games) | MIT |
| Space Invaders 게임 엔진 | [juliensimon/browser-games](https://github.com/juliensimon/browser-games) | MIT |
| Tetris AI 가중치·알고리즘 | [LeeYiyuan/tetrisai](https://github.com/LeeYiyuan/tetrisai) | MIT |
| Gomoku AI Worker | [gomoku/](https://github.com/...) (내부 참조) | — |
| Stockfish.js | [nmrugg/stockfish.js](https://github.com/nmrugg/stockfish.js) | GPL-3.0 |
| chess.js | [jhlywa/chess.js](https://github.com/jhlywa/chess.js) | BSD |
| chessboard.js | [oakmac/chessboardjs](https://github.com/oakmac/chessboardjs) | MIT |
| p5.js | [processing/p5.js](https://p5js.org/) | LGPL |

원작 아케이드 게임(Pong, Space Invaders 등)의 트레이드마크는 각 권리자에 귀속됩니다.
