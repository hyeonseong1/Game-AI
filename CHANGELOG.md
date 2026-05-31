# Changelog

모든 주요 변경사항을 기록합니다.  
형식은 [Keep a Changelog](https://keepachangelog.com/ko/1.0.0/)를 따릅니다.

---

## [2.0.0] — 2026-05-31

### 🚀 대규모 AI 업그레이드 릴리스

이번 버전의 핵심은 **PPO 강화학습 에이전트 도입**과 **듀얼스크린 대결 구조** 전환입니다.  
6개 게임 모두 AI가 교체·강화되었으며, 브라우저 내 신경망 추론 파이프라인이 구축되었습니다.

---

### Pong — PPO 강화학습 AI 적용

**변경 이전**: `speedMult`·`mistakeRate` 파라미터로 패들 속도를 조절하는 단순 룰 기반 AI

**변경 이후**: 학습된 PPO 신경망이 매 프레임 관측 → 행동 결정

#### 추가
- `pong_rl/` 패키지 신규 생성
  - `pong_env.py`: JS 게임 물리(900×600, `PADDLE_SPEED=7`, `BALL_R=8`)를 Python으로 1:1 재현한 시뮬레이션 환경
  - `ppo_agent.py`: `ActorCritic` 네트워크 + `RolloutBuffer` + PPO 업데이트 루프
  - `train.py`: 1,000,000 스텝 학습, mid(50%)/best(최고 리워드) 자동 저장
  - `export_weights.py`: PyTorch `.pt` → JSON 변환 (브라우저 서빙용)
- `public/js/pong-rl-agent.js`: 범용 ActorCritic JS 추론 엔진 (`RLAgent` 클래스)
  - 모델 JSON을 `fetch`로 로드, `predict(obs) → action` 실행
  - Linear + Tanh forward pass를 순수 JS로 구현 (외부 의존성 없음)
- `public/models/pong_mid.json` (1.5 MB, step 501,760)
- `public/models/pong_best.json` (1.5 MB, step 665,600)

#### 수정
- `public/js/games/pong-core.js`: `AIController` 클래스에 3가지 모드 추가
  - `mode: 'random'` — Easy: 매 프레임 랜덤 방향
  - `mode: 'ppo'` — Medium/Hell: JS 추론 엔진 호출
  - `mode: 'rule'` — 기존 룰 기반 폴백 (모델 로드 실패 시)
  - `_buildObs()` 메서드: 게임 좌표를 `[-1, 1]` 범위로 정규화해 관측 벡터 생성
  - `update(ball, paddle2, paddle1)` 시그니처에 `paddle1` 추가 (관측에 상대 패들 포함)
- `public/js/games/pong.js`: 난이도별 비동기 모델 로드 + 게임 인스턴스 생성
  - `window._pongAgents` 캐시로 중복 로드 방지

#### 관측 공간 (6차원)

| 인덱스 | 의미 | 범위 |
|--------|------|------|
| 0 | 공 X (정규화) | [-1, 1] |
| 1 | 공 Y (정규화) | [-1, 1] |
| 2 | 공 VX / max_speed | [-1, 1] |
| 3 | 공 VY / max_speed | [-1, 1] |
| 4 | AI 패들 Y (정규화) | [-1, 1] |
| 5 | 플레이어 패들 Y (정규화) | [-1, 1] |

#### 행동 공간 (3개)
`0=stay`, `1=up`, `2=down`

---

### Gomoku — Web Worker Alpha-Beta Minimax AI 적용

**변경 이전**: `scoreCell()` 기반 단순 그리디 휴리스틱 (메인 스레드 인라인 실행)

**변경 이후**: 비트보드 표현 + Alpha-Beta Pruning + Web Worker 비동기 처리

#### 추가
- `public/js/gomoku-ai-worker.js`: `gomoku/` 레퍼런스 프로젝트의 AI Worker를 자체 포함형(self-contained)으로 포팅
  - 비트보드(8개 32비트 정수, 총 256비트)로 15×15 바둑판 표현
  - 즉각 위협 탐지 (`checkImmediateThreat`), 열린 4 패턴(`checkOpen4Threats`), 열린 3 패턴(`checkSimpleOpen3Threats`)
  - Alpha-Beta Minimax: Medium 6-ply, Hell 8-ply
  - `FIND_BEST_MOVE` / `BEST_MOVE_FOUND` / `NEW_GAME` 메시지 프로토콜

#### 수정
- `public/js/games/gomoku.js`: 전면 재작성
  - `board2Bitboards()` 인라인 구현: `board[r][c]` (0/1/2) → `{blackBitboard, whiteBitboard}`
  - Worker 비동기 통신: 플레이어 수 완료 → Worker 메시지 전송 → `BEST_MOVE_FOUND` 수신 → AI 돌 배치
  - AI 계산 중 "AI thinking…" 애니메이션 점 3개 (p5.js `frameCount` 기반)
  - star points(귀목) 5개 표시, 흑돌 하이라이트·백돌 하이라이트 렌더링 개선
  - Easy는 인-스레드 그리디 (즉시 응답), Medium/Hell만 Worker 사용

---

### Chess — Stockfish 18 WASM AI 적용

**변경 이전**: `game.moves()`에서 랜덤 선택 또는 포획 우선 선택하는 랜덤 AI

**변경 이후**: Stockfish 18 체스 엔진 (UCI 프로토콜, WASM)

#### 추가
- `public/stockfish.js` (21 KB) — Stockfish 18 Lite Single-threaded WASM 래퍼
- `public/stockfish.wasm` (7.0 MB) — Stockfish WASM 바이너리
- `public/img/chesspieces/wikipedia/*.png` (12개) — 체스 기물 이미지 로컬 복사 (CDN 의존 제거)

#### 수정
- `server.js`: `.wasm` MIME 타입 추가 (`application/wasm`)
- `public/js/games/chess.js`: 전면 재작성
  - `new Worker('/stockfish.js')`로 Stockfish UCI 엔진 초기화
  - `uci` → `readyok` 핸드셰이크 후 플레이 시작
  - UCI 명령 시퀀스: `setoption name Skill Level value {skill}` → `position startpos moves {history}` → `go depth {depth}` → `bestmove {move}` 파싱
  - `enginegame.js` 패턴(Skill Level·Maximum Error·Probability 연동) 적용
  - 체크메이트·드로우·스테일메이트 감지 및 게임 종료 처리
  - 기물 이미지 경로 `/img/chesspieces/wikipedia/{piece}.png`로 로컬 서빙

| 난이도 | Skill Level | Depth |
|--------|-------------|-------|
| Easy   | 3 | 2 |
| Medium | 10 | 8 |
| Hell   | 20 | 15 |

---

### Boxing — PPO 강화학습 AI 적용

**변경 이전**: PettingZoo `boxing_v2` Atari 환경 기반 Python 백엔드 서비스 (`boxing_web_service.py` + `boxing-bridge.js`). 실제로는 순수 Canvas 게임으로 교체되어 있었음.

**변경 이후**: 순수 Canvas 게임 + PPO 신경망 AI (브라우저 추론)

#### 추가
- `boxing_rl/` 패키지 신규 생성
  - `boxing_env.py`: JS `boxing.js` 물리를 Python 60fps 시뮬레이션으로 재현
    - Agent(오른쪽 복서) vs 룰 기반 상대(왼쪽 복서)
    - 반지름 16px, 펀치 범위 60px, 넉다운 1800ms
    - 관측: 14차원 (위치·속도·점수·펀치/다운 타이머·남은 시간)
    - 행동: 18개 (9방향이동 × 펀치여부)
  - `ppo_agent.py`, `train.py`, `export_weights.py`
- `public/models/boxing_mid.json` (1.6 MB, step 501,760)
- `public/models/boxing_best.json` (1.6 MB, step 342,016, best avg reward +114.156)

#### 수정
- `public/js/games/boxing.js`: PPO 통합
  - `loadBoxingAgent(key, url)`: `window.PongRLAgent` (범용 추론 엔진) 재사용
  - `buildObs(ai, player, timeLeft)`: 14차원 관측 벡터 생성
  - `decodeAction(action)`: 행동 → `{vx, vy, punch}` 변환
  - `aiInputPPO()`: 매 프레임 관측 → 추론 → 이동·펀치 실행
  - `aiInputRandom()`: Easy 난이도 랜덤 정책
  - Medium/Hell 시 "Loading PPO model…" 화면 표시 후 게임 시작
  - `HitEffect.draw()` 버그 수정: `alpha = Math.max(0, life/maxLife)` (음수 arc 반지름 방지)

#### 관측 공간 (14차원)

| 인덱스 | 의미 |
|--------|------|
| 0-1 | Agent X·Y (링 중심 기준 정규화) |
| 2-3 | Opponent X·Y |
| 4-5 | 상대와의 상대 거리 (dx/RING_W, dy/RING_H) |
| 6 | 두 복서 간 유클리드 거리 / 최대거리 |
| 7-8 | Agent 점수 / 100, Opponent 점수 / 100 |
| 9-10 | Agent·Opponent 펀치 타이머 / MAX |
| 11-12 | Agent·Opponent 넉다운 타이머 / MAX |
| 13 | 남은 시간 / 총 게임 시간 |

---

### Space Invaders — PPO AI + 듀얼스크린 대결 구조

**변경 이전**: 단일 캔버스, 인간 플레이어만 조작, 점수 벤치마크(Easy:500, Medium:1500, Hell:3500)와 비교

**변경 이후**: 듀얼스크린 (인간 Player 1 vs PPO AI Player 2), 실시간 동시 플레이

#### 추가
- `space_invaders_rl/` 패키지 신규 생성
  - `space_invaders_env.py`: `space-invaders-core.js` 물리를 Python 60fps 시뮬레이션으로 재현
    - 에일리언 포메이션 행진(속도 가변), 폭탄 발사(최대 3개), 플레이어 총알(1개)
    - 관측: 20차원 (플레이어 위치·총알, 포메이션 경계·방향, 폭탄 3개, 조준 힌트, 웨이브·생명)
    - 행동: 6개 (Stay·Left·Right × 발사여부)
  - `ppo_agent.py`, `train.py` (1,500,000 스텝), `export_weights.py`
- `public/models/si_mid.json` (1.5 MB, step 751,616)
- `public/models/si_best.json` (1.5 MB, step 1,433,600, best avg reward +50.47)

#### 수정
- `public/js/games/space-invaders-core.js`: `_aiInput` 훅 추가 (3줄)
  - `updatePlaying()` 내 키보드 입력 앞에 `ctrl = this._aiInput?.(this.getState())` 인터셉트
  - `ctrl.left·right·fire`로 플레이어 패들을 AI가 직접 제어
- `public/js/games/space-invaders.js`: 전면 재작성
  - `mountDualSI(container, ctx, aiController)`: 두 `Game` 인스턴스를 나란히 생성
  - `buildSIObs(gameState, game)`: 20차원 관측 벡터 (포메이션·폭탄·타깃 힌트 계산)
  - `decodeSIAction(action)`: 행동 → `{left, right, fire}` 변환
  - `makeRandomCtrl()`: Easy 난이도 랜덤 정책 컨트롤러
  - 양쪽 게임 종료 시 점수 비교 후 `ctx.onEnd()` 호출

#### 관측 공간 (20차원)

| 인덱스 | 의미 |
|--------|------|
| 0 | 플레이어 중심 X / W |
| 1 | 총알 활성 여부 (0/1) |
| 2-3 | 총알 X·Y (비활성 시 0) |
| 4 | 생존 에일리언 수 / 55 |
| 5-6 | 포메이션 좌측·우측 경계 X / W |
| 7 | 포메이션 최하단 Y / H |
| 8 | 포메이션 중심 X / W |
| 9 | 포메이션 이동 방향 (0=좌, 1=우) |
| 10-15 | 폭탄 3개의 [dx/W, y/H] (없으면 0,0) |
| 16 | 가장 가까운 에일리언 열까지 dx / W |
| 17 | 현재 웨이브 / 10 |
| 18 | 남은 생명 / 3 |
| 19 | 미스터리 쉽 X / W (없으면 0) |

---

### Tetris — 휴리스틱 AI + 듀얼스크린 대결 구조

**변경 이전**: jstetris embed (`public/jstetris/`) 를 `<iframe>`으로 삽입하는 단일 플레이어 방식

**변경 이후**: iframe 완전 제거, 단일 Canvas에 두 게임 인스턴스를 나란히 렌더링

#### 추가
- `public/js/games/tetris.js`: 전면 재작성 (2,400 → 새 파일, 자체 완결형)
  - `Grid` 클래스: `tetrisai/js/grid.js` 포팅 (aggregateHeight·lines·holes·bumpiness 평가 함수)
  - `Piece` 클래스: 7종 테트로미노, `rotateCW()`, wall kick, `moveLeft/Right/Down()`
  - `RPG` (Random Piece Generator): 7-bag 셔플 랜더마이저
  - `aiBest()`: 재귀 탐색으로 최적 배치 위치 계산 (`tetrisai/js/ai.js` 포팅)
  - `TetrisEngine`: 게임 상태 관리 (중력·잠금·줄 클리어·레벨업)
  - `renderBoard()` / `renderSidePanel()` / `_drawCell()`: 단일 Canvas 렌더러 (3D 하이라이트 셀)
  - `mountDualTetris()`: 두 엔진 인스턴스를 하나의 Canvas(600×480px)에 병렬 렌더링
  - Ghost piece(낙하 예측선), DAS/ARR 키 자동 반복 처리

| 난이도 | 방식 | 룩어헤드 | 가중치 (h/l/o/b) |
|--------|------|----------|-----------------|
| Easy | 랜덤 배치 | — | — |
| Medium | 휴리스틱 | 1-piece | 0.510/0.600/0.450/0.300 |
| Hell | 휴리스틱 | 2-piece | 0.510/0.761/0.357/0.184 |

---

### 공통 인프라 변경

#### 추가
- `server.js`: `.wasm` MIME 타입 등록 (`application/wasm`)
- `public/index.html`: `<script src="/js/pong-rl-agent.js" defer>` 추가

#### 버그 수정
- Boxing `HitEffect.draw()`: `alpha = this.life / this.maxLife`가 `life < 0` 일 때 음수가 되어 `ctx.arc(r < 0)` 에러 발생 → `Math.max(0, ...)` 클램프로 수정

---

## [1.0.0] — 초기 릴리스

### 추가

- **6개 클래식 게임** 초기 구현
  - **Pong**: `pong-core.js` 기반 Canvas 게임 (룰 기반 AI)
  - **Space Invaders**: `space-invaders-core.js` 기반 Canvas 게임 (단일 플레이어, 점수 벤치마크)
  - **Tetris**: jstetris iframe embed (`public/jstetris/`)
  - **Boxing**: 순수 Canvas 탑뷰 복싱 게임 (룰 기반 AI, speedMult·mistakeRate·reactionMs 파라미터)
  - **Chess**: chess.js + chessboard.js, 랜덤/포획 우선 AI
  - **Gomoku**: p5.js Canvas, 단순 그리디 휴리스틱 AI

- **SPA 인프라** (`public/js/app.js`)
  - 뷰 라우터: 로그인 → 로비 → 난이도 선택 → 게임 → 결과
  - 회원가입·로그인·게스트 세션

- **리더보드** (`GET/POST/DELETE /api/leaderboard`)
  - 게임별·난이도별 필터링
  - AI 대비 점수 퍼센트 계산

- **커뮤니티** (`/api/community/posts`)
  - 게시글 작성·조회, 좋아요 토글, 댓글

- **데이터 저장소**: `data/db.json` (JSON 파일 기반)

- **서버**: `server.js` — Node.js 순수 `http` 모듈 (외부 프레임워크 없음)

- **AI 난이도 시스템** (`public/js/ai-config.js`)
  - Easy/Medium/Hell 3단계별 파라미터 테이블
  - `getDifficulty(key)`, `shouldAiMistake(difficulty)` 헬퍼

- **레거시 Boxing 서비스** (`services/boxing/`)
  - PettingZoo `boxing_v2` (Atari ROM) Python 환경
  - JSON-lines RPC (`boxing_web_service.py` ↔ `boxing-bridge.js`)
  - 현재 버전에서는 미사용 (Canvas 게임으로 대체됨)
