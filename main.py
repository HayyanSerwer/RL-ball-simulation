import pygame
import math
import random
import numpy as np

WIDTH, HEIGHT = 800, 800
FPS = 30
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 100, 255)
GREEN = (0, 128, 0)


CIRCLE_CENTER = (WIDTH // 2, HEIGHT // 2)
CIRCLE_RADIUS = 200


ball_radius = 15
ball_rect = pygame.Rect(WIDTH // 2 - ball_radius, HEIGHT // 2 - 100 - ball_radius,
                        ball_radius * 2, ball_radius * 2)
ball_velocity = [5, 4]


score = 0


# Rectangle class
class FloatingRect:
    SPEED = 2.0  # constant speed for all rectangles

    def __init__(self, ball_pos=None):
        self.width = 15
        self.height = 15
        # Spawning within the circle but away from the ball
        max_attempts = 100
        for _ in range(max_attempts):
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(0, CIRCLE_RADIUS - 50)
            self.x = CIRCLE_CENTER[0] + distance * math.cos(angle)
            self.y = CIRCLE_CENTER[1] + distance * math.sin(angle)

            if ball_pos is not None:
                dx = self.x - ball_pos[0]
                dy = self.y - ball_pos[1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist > 50:
                    break
            else:
                break

        # Random direction but fixed speed
        angle = random.uniform(0, 2 * math.pi)
        self.vx = self.SPEED * math.cos(angle)
        self.vy = self.SPEED * math.sin(angle)

    def update(self):
        self.x += self.vx
        self.y += self.vy

        # Keep within circle boundary
        dx = self.x - CIRCLE_CENTER[0]
        dy = self.y - CIRCLE_CENTER[1]
        distance = math.sqrt(dx ** 2 + dy ** 2)

        if distance + self.width / 2 >= CIRCLE_RADIUS:
            # Bounce off circle boundary
            nx = dx / distance
            ny = dy / distance

            dot_product = self.vx * nx + self.vy * ny
            self.vx -= 2 * dot_product * nx
            self.vy -= 2 * dot_product * ny

    def draw(self, screen):
        pygame.draw.rect(screen, WHITE, self.get_rect())

    def get_rect(self):
        return pygame.Rect(int(self.x - self.width / 2),
                           int(self.y - self.height / 2),
                           self.width, self.height)

    def collides_with_ball(self, ball_rect):
        return self.get_rect().colliderect(ball_rect)


class AvoidBallEnv():
    MAX_VEL = 7
    def __init__(self):
        pygame.init()

        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()

        self.ball_radius = 15
        self.speed = [5, 4]

        self.reset()

    def reset(self):
        self.ball_rect = pygame.Rect(CIRCLE_CENTER[0] - self.ball_radius,
                                     CIRCLE_CENTER[1] - 100 - self.ball_radius,
                                     self.ball_radius * 2,
                                     self.ball_radius * 2)

        ball_pos = self.ball_rect.center
        self.ball_velocity = [5, 4]
        self.rectangles = [FloatingRect(ball_pos) for _ in range(8)]  # Pass ball position
        self.score = 0
        self.done = False
        self.prev_min_distance = None

        return self._get_state()

    def _get_state(self):
        bx = self.ball_rect.centerx
        by = self.ball_rect.centery

        dx_c = (bx - CIRCLE_CENTER[0]) / CIRCLE_RADIUS
        dy_c = (by - CIRCLE_CENTER[1]) / CIRCLE_RADIUS
        dist_from_wall = (CIRCLE_RADIUS - math.hypot(bx - CIRCLE_CENTER[0], by - CIRCLE_CENTER[1])) / CIRCLE_RADIUS

        state = [bx / WIDTH, by / HEIGHT, dist_from_wall, dx_c, dy_c]

        # compute distances to rectangles, sort
        rect_info = []
        for rect in self.rectangles:
            dx = (rect.x - bx) / CIRCLE_RADIUS
            dy = (rect.y - by) / CIRCLE_RADIUS
            dist = math.hypot(dx, dy)
            rect_info.append((dist, dx, dy))

        rect_info.sort(key=lambda x: x[0])

        # keep only 3 nearest, and only position deltas (no velocities)
        for dist, dx, dy in rect_info[:3]:
            state.extend([dx, dy, dist])

        # pad to fixed length (for your network), e.g. target length 16
        while len(state) < 16:
            state.append(0.0)

        return np.array(state, dtype=np.float32)

    def step(self, action):
        if self.done:
            return self._get_state(), 0, True, {}

        # --- APPLY ACTION ---
        ACCEL = 1.0  # lower for smoother control
        MAX_VEL = 7  # lower max velocity
        MOVE_STEP = 8  # adjust as needed

        # Direct movement
        if action == 1:  # RIGHT
            self.ball_rect.x += MOVE_STEP
        elif action == 2:  # LEFT
            self.ball_rect.x -= MOVE_STEP
        elif action == 3:  # DOWN
            self.ball_rect.y += MOVE_STEP
        elif action == 4:  # UP
            self.ball_rect.y -= MOVE_STEP

        dx = self.ball_rect.centerx - CIRCLE_CENTER[0]
        dy = self.ball_rect.centery - CIRCLE_CENTER[1]
        distance = math.hypot(dx, dy)

        if distance + self.ball_radius > CIRCLE_RADIUS:
            nx = dx / distance
            ny = dy / distance
            overlap = (distance + self.ball_radius) - CIRCLE_RADIUS

            self.ball_rect.x -= int(nx * overlap)
            self.ball_rect.y -= int(ny * overlap)

        # --- UPDATE RECTANGLES & COLLISION CHECK ---
        min_distance = float('inf')
        collision = False
        for rect in self.rectangles:
            rect.update()
            dx = rect.x - self.ball_rect.centerx
            dy = rect.y - self.ball_rect.centery
            dist = math.hypot(dx, dy)
            min_distance = min(min_distance, dist)
            if rect.collides_with_ball(self.ball_rect):
                collision = True

        if self.prev_min_distance is None:
            self.prev_min_distance = min_distance

        reward = 1.0

        # big negative on collision
        if collision:
            reward = -200.0
            self.done = True
            return self._get_state(), reward, self.done, {}

        # proximity penalty (soft): only if very close
        min_distance = min(
            math.hypot(rect.x - self.ball_rect.centerx, rect.y - self.ball_rect.centery) for rect in self.rectangles)
        if min_distance < 40:  # danger zone
            reward -= (40 - min_distance) * 0.02  # small shaping penalty

        # optional: tiny penalty for moving outside center (discourage hugging wall)
        dx_c = self.ball_rect.centerx - CIRCLE_CENTER[0]
        dy_c = self.ball_rect.centery - CIRCLE_CENTER[1]
        distance_to_center = math.hypot(dx_c, dy_c)
        # reward bonus for staying more central (optional)
        reward += (CIRCLE_RADIUS - distance_to_center) / CIRCLE_RADIUS * 0.1

        return self._get_state(), reward, self.done, {}

    def render(self):
        self.screen.fill(BLACK)

        pygame.draw.circle(self.screen, WHITE, CIRCLE_CENTER, CIRCLE_RADIUS, 3)
        pygame.draw.circle(self.screen, GREEN, self.ball_rect.center, self.ball_radius)

        for r in self.rectangles:
            r.draw(self.screen)

        pygame.display.flip()
        self.clock.tick(FPS)

    def quit(self):
        pygame.quit()




