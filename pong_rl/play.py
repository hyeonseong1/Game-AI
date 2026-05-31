"""
Pong RL — Play against AI
=========================
Difficulty:
  Easy   — Random policy (opponent acts randomly)
  Medium — Mid-epoch PPO model
  Hell   — Best-reward PPO model

Controls: W / ↑ = up    S / ↓ = down    ESC = back to menu
First to 7 points wins.
"""

import os
import sys

import numpy as np
import pygame

from pong_env import PongEnv
from ppo_agent import PPOAgent

# ── display ──────────────────────────────────────────────────────────
W, H = 900, 600
FPS = 60

# ── palette ──────────────────────────────────────────────────────────
BG          = (6, 6, 16)
WHITE       = (240, 240, 255)
GREY        = (70, 70, 95)
CYAN        = (40, 210, 255)
GREEN_NEON  = (50, 255, 110)
YELLOW_NEON = (255, 215, 30)
RED_NEON    = (255, 55, 55)
DARK_PANEL  = (18, 18, 36)

DIFF_COLOR = {'easy': GREEN_NEON, 'medium': YELLOW_NEON, 'hell': RED_NEON}
DIFF_LABEL = {'easy': 'EASY', 'medium': 'MEDIUM', 'hell': 'HELL'}
MODEL_PATH = {
    'medium': os.path.join('models', 'mid_model.pt'),
    'hell':   os.path.join('models', 'best_model.pt'),
}
WIN_SCORE = 7


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def center_text(surface, text, font, color, cx, cy):
    s = font.render(text, True, color)
    surface.blit(s, s.get_rect(center=(cx, cy)))
    return s.get_rect(center=(cx, cy))


def draw_glow(surface, pos, radius, color, alpha=45):
    sz = radius * 6
    gs = pygame.Surface((sz, sz), pygame.SRCALPHA)
    r, g, b = color
    pygame.draw.circle(gs, (r, g, b, alpha), (sz // 2, sz // 2), sz // 2)
    surface.blit(gs, (pos[0] - sz // 2, pos[1] - sz // 2))


def draw_paddle_glow(surface, rect, color, alpha=40):
    pad = 10
    gs = pygame.Surface((rect.w + pad * 2, rect.h + pad * 2), pygame.SRCALPHA)
    r, g, b = color
    pygame.draw.rect(gs, (r, g, b, alpha),
                     (0, 0, rect.w + pad * 2, rect.h + pad * 2),
                     border_radius=8)
    surface.blit(gs, (rect.x - pad, rect.y - pad))


# ──────────────────────────────────────────────────────────────────────
# Menu screen
# ──────────────────────────────────────────────────────────────────────

def menu_screen(screen: pygame.Surface, clock: pygame.time.Clock):
    f_title = pygame.font.Font(None, 110)
    f_sub   = pygame.font.Font(None, 34)
    f_btn   = pygame.font.Font(None, 52)
    f_hint  = pygame.font.Font(None, 28)

    BUTTONS = [
        ('easy',   'EASY',   'Random Policy',               GREEN_NEON,  H // 2 - 85),
        ('medium', 'MEDIUM', 'Mid-epoch PPO Model',         YELLOW_NEON, H // 2 +  5),
        ('hell',   'HELL',   'Best Reward PPO Model',       RED_NEON,    H // 2 + 95),
    ]
    BTN_W, BTN_H = 380, 62

    # Pre-render static title glow
    while True:
        mx, my = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.MOUSEBUTTONDOWN:
                for diff, label, desc, color, by in BUTTONS:
                    r = pygame.Rect(W // 2 - BTN_W // 2, by, BTN_W, BTN_H)
                    if r.collidepoint(mx, my):
                        return diff

        screen.fill(BG)

        # Decorative center line
        for yi in range(0, H, 22):
            if (yi // 22) % 2 == 0:
                pygame.draw.rect(screen, (25, 25, 45), (W // 2 - 2, yi, 4, 13))

        # Title
        draw_glow(screen, (W // 2, H // 4 - 10), 60, CYAN, 20)
        center_text(screen, 'PONG  RL', f_title, CYAN, W // 2, H // 4 - 10)
        center_text(screen, 'Human  vs  AI', f_sub, GREY, W // 2, H // 4 + 55)

        # Buttons
        for diff, label, desc, color, by in BUTTONS:
            rect = pygame.Rect(W // 2 - BTN_W // 2, by, BTN_W, BTN_H)
            hover = rect.collidepoint(mx, my)

            if hover:
                draw_paddle_glow(screen, rect, color, 30)

            bg = tuple(min(255, c + 25) for c in color) if hover else DARK_PANEL
            pygame.draw.rect(screen, bg, rect, border_radius=10)
            pygame.draw.rect(screen, color, rect, 2, border_radius=10)

            center_text(screen, label, f_btn, color, W // 2, by + BTN_H // 2 - 6)
            center_text(screen, desc,  f_hint, GREY,  W // 2, by + BTN_H // 2 + 18)

        # Controls hint
        center_text(screen,
                    'W / ↑ = up     S / ↓ = down     ESC = menu',
                    f_hint, GREY, W // 2, H - 22)

        pygame.display.flip()
        clock.tick(60)


# ──────────────────────────────────────────────────────────────────────
# Game screen
# ──────────────────────────────────────────────────────────────────────

def game_screen(screen: pygame.Surface, clock: pygame.time.Clock,
                difficulty: str) -> str:
    f_score = pygame.font.Font(None, 80)
    f_ui    = pygame.font.Font(None, 30)
    f_big   = pygame.font.Font(None, 96)

    env = PongEnv(W, H)
    ai_color = DIFF_COLOR[difficulty]

    # Load model if needed
    agent = None
    if difficulty in ('medium', 'hell'):
        path = MODEL_PATH[difficulty]
        if not os.path.exists(path):
            # Fall back gracefully
            missing = os.path.basename(path)
            screen.fill(BG)
            center_text(screen,
                        f'Model not found: {missing}',
                        f_ui, RED_NEON, W // 2, H // 2 - 20)
            center_text(screen,
                        'Run  python train.py  first, then retry.',
                        f_ui, GREY, W // 2, H // 2 + 20)
            pygame.display.flip()
            pygame.time.wait(3000)
            return 'menu'
        agent = PPOAgent()
        agent.load(path, eval_mode=True)

    player_score = 0
    ai_score = 0

    # Ball trail
    trail: list[tuple[int, int]] = []
    TRAIL_LEN = 8

    state = env.reset()

    def get_ai_action(s):
        if difficulty == 'easy':
            return np.random.randint(3)
        return agent.predict(s)

    def get_player_action():
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            return 1
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            return 2
        return 0

    def draw(show_score=True):
        screen.fill(BG)

        # Center dashed line
        for yi in range(0, H, 22):
            if (yi // 22) % 2 == 0:
                pygame.draw.rect(screen, GREY, (W // 2 - 2, yi, 4, 13))

        # Ball trail
        for i, (tx, ty) in enumerate(trail):
            alpha = int(180 * (i + 1) / TRAIL_LEN)
            sz = max(2, env.BALL_R * (i + 1) // TRAIL_LEN)
            ts = pygame.Surface((sz * 2, sz * 2), pygame.SRCALPHA)
            pygame.draw.circle(ts, (255, 255, 255, alpha), (sz, sz), sz)
            screen.blit(ts, (tx - sz, ty - sz))

        # Paddles
        lp_rect = pygame.Rect(env.LEFT_X,
                              int(env.left_y - env.PADDLE_H // 2),
                              env.PADDLE_W, env.PADDLE_H)
        rp_rect = pygame.Rect(env.right_x,
                              int(env.right_y - env.PADDLE_H // 2),
                              env.PADDLE_W, env.PADDLE_H)

        draw_paddle_glow(screen, lp_rect, CYAN, 40)
        draw_paddle_glow(screen, rp_rect, ai_color, 40)
        pygame.draw.rect(screen, CYAN,     lp_rect, border_radius=6)
        pygame.draw.rect(screen, ai_color, rp_rect, border_radius=6)

        # Ball
        bx, by = int(env.ball[0]), int(env.ball[1])
        draw_glow(screen, (bx, by), env.BALL_R, WHITE, 40)
        pygame.draw.circle(screen, WHITE, (bx, by), env.BALL_R)

        if not show_score:
            return

        # Score board
        score_surf = f_score.render(
            f"{player_score}  —  {ai_score}", True, WHITE)
        screen.blit(score_surf,
                    score_surf.get_rect(center=(W // 2, 34)))

        # Side labels
        you_s = f_ui.render('YOU', True, CYAN)
        ai_s  = f_ui.render(DIFF_LABEL[difficulty], True, ai_color)
        screen.blit(you_s, (env.LEFT_X + 2, 8))
        screen.blit(ai_s, (W - env.RIGHT_X_OFFSET - ai_s.get_width() - 2, 8))

        # ESC hint
        esc_s = f_ui.render('ESC: menu', True, GREY)
        screen.blit(esc_s, (8, H - 24))

        pygame.display.flip()

    # Brief "get ready" pause
    draw()
    pygame.time.wait(600)

    # ── main loop ────────────────────────────────────────────────────
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return 'quit'
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return 'menu'

        p_action  = get_player_action()
        ai_action = get_ai_action(state)

        next_state, _, done, info = env.step(ai_action, p_action)
        state = next_state

        # Update trail
        trail.append((int(env.ball[0]), int(env.ball[1])))
        if len(trail) > TRAIL_LEN:
            trail.pop(0)

        if done:
            scorer = info.get('scorer', '')
            if scorer == 'ai':
                ai_score += 1
            else:
                player_score += 1

            trail.clear()
            draw()
            pygame.time.wait(700)

            # Check win
            if player_score >= WIN_SCORE or ai_score >= WIN_SCORE:
                winner   = 'YOU WIN!' if player_score > ai_score else 'AI WINS!'
                w_color  = CYAN if player_score > ai_score else ai_color
                screen.fill(BG)
                draw_glow(screen, (W // 2, H // 2 - 40), 80, w_color, 18)
                center_text(screen, winner,
                            f_big, w_color, W // 2, H // 2 - 40)
                center_text(screen,
                            f'{player_score}  —  {ai_score}',
                            f_score, WHITE, W // 2, H // 2 + 50)
                center_text(screen, 'Press any key …',
                            f_ui, GREY, W // 2, H // 2 + 115)
                pygame.display.flip()

                waiting = True
                while waiting:
                    for ev in pygame.event.get():
                        if ev.type == pygame.QUIT:
                            return 'quit'
                        if ev.type == pygame.KEYDOWN:
                            waiting = False
                return 'menu'

            state = env.reset()
            draw()
            pygame.time.wait(400)

        draw()
        clock.tick(FPS)


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────

def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption('Pong RL')
    clock = pygame.time.Clock()

    while True:
        diff = menu_screen(screen, clock)
        if diff is None:
            break
        result = game_screen(screen, clock, diff)
        if result == 'quit':
            break

    pygame.quit()
    sys.exit()


if __name__ == '__main__':
    main()
