import pygame
import math
import random
import numpy as np

WIDTH, HEIGHT = 800, 800
FPS = 60
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
ball_velocity = [10, 10]


score = 0


# Rectangle class
class FloatingRect:
    def __init__(self):
        self.width = 15
        self.height = 15
        # Spawning within the circle boundary
        angle = random.uniform(0, 2 * math.pi)
        distance = random.uniform(0, CIRCLE_RADIUS - 50)
        self.x = CIRCLE_CENTER[0] + distance * math.cos(angle)
        self.y = CIRCLE_CENTER[1] + distance * math.sin(angle)
        self.vx = random.uniform(-2, 2)
        self.vy = random.uniform(-2, 2)

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

        self.ball_velocity = [5, 4]
        self.rectangles = [FloatingRect() for _ in range(10)]
        self.score = 0
        self.done = False

        return self._get_state()

    def _get_state(self):
        # --- Ball info ---
        bx = self.ball_rect.centerx
        by = self.ball_rect.centery
        bvx = self.ball_velocity[0]
        bvy = self.ball_velocity[1]

        state = [
            bx / WIDTH,
            by / HEIGHT,
            bvx / 10,
            bvy / 10,
        ]


        rect_info = []
        for rect in self.rectangles:
            dx = rect.x - bx
            dy = rect.y - by
            dist = math.sqrt(dx*dx + dy*dy)
            rect_info.append((dist, rect))

        # Sort by nearest
        rect_info.sort(key=lambda x: x[0])


        MAX_SPEED = 3  # max rect velocity for normalization

        for dist, rect in rect_info:

            dx = (rect.x - bx) / CIRCLE_RADIUS
            dy = (rect.y - by) / CIRCLE_RADIUS

            # Normalize velocities
            dvx = rect.vx / MAX_SPEED
            dvy = rect.vy / MAX_SPEED

            state.extend([dx, dy, dvx, dvy])

        return np.array(state, dtype=np.float32)

    def step(self, action):
        if self.done:
            return self._get_state(), 0, True, {}

        # Control
        if action == 1: self.ball_velocity[0] -= 1
        if action == 2: self.ball_velocity[0] += 1
        if action == 3: self.ball_velocity[1] -= 1
        if action == 4: self.ball_velocity[1] += 1

        # Move ball
        MAX_VEL = 15
        self.ball_velocity[0] = float(np.clip(self.ball_velocity[0], -MAX_VEL, MAX_VEL))
        self.ball_velocity[1] = float(np.clip(self.ball_velocity[1], -MAX_VEL, MAX_VEL))

        self.ball_rect.x += self.ball_velocity[0]
        self.ball_rect.y += self.ball_velocity[1]

        # Ball boundary collision
        dx = self.ball_rect.centerx - CIRCLE_CENTER[0]
        dy = self.ball_rect.centery - CIRCLE_CENTER[1]
        distance = math.sqrt(dx * dx + dy * dy)

        if distance + self.ball_radius >= CIRCLE_RADIUS:
            nx = dx / distance
            ny = dy / distance

            overlap = (distance + self.ball_radius) - CIRCLE_RADIUS
            self.ball_rect.x -= nx * overlap
            self.ball_rect.y -= ny * overlap

            dot = self.ball_velocity[0] * nx + self.ball_velocity[1] * ny
            self.ball_velocity[0] -= 2 * dot * nx
            self.ball_velocity[1] -= 2 * dot * ny

            self.ball_velocity[0] *= 0.98
            self.ball_velocity[1] *= 0.98


        for rect in self.rectangles:
            rect.update()
            if rect.collides_with_ball(self.ball_rect):
                self.done = True
                return self._get_state(), -20, True, {}

        return self._get_state(), 1, False, {}

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




