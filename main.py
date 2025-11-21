import pygame
import math
import random


pygame.init()


WIDTH, HEIGHT = 800, 800
FPS = 60
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 100, 255)
GREEN = (0, 128, 0)

CIRCLE_CENTER = (WIDTH // 2, HEIGHT // 2)
CIRCLE_RADIUS = 300


ball_radius = 15
ball_rect = pygame.Rect(WIDTH // 2 - ball_radius, HEIGHT // 2 - 100 - ball_radius,
                        ball_radius * 2, ball_radius * 2)
ball_velocity = [4, 3]


score = 0


# Rectangle class
class FloatingRect:
    def __init__(self):
        self.width = 20
        self.height = 20
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


rectangles = [FloatingRect() for _ in range(10)]

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Reinforcement Learning Simulation")
clock = pygame.time.Clock()

font = pygame.font.Font(None, 48)

running = True
count = 0
gCount = 0
bCount = 255
dynamic_white = (count, gCount, bCount)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    ball_rect.x += ball_velocity[0]
    ball_rect.y += ball_velocity[1]

    ball_center_x = ball_rect.centerx
    ball_center_y = ball_rect.centery
    dx = ball_center_x - CIRCLE_CENTER[0]
    dy = ball_center_y - CIRCLE_CENTER[1]
    distance = math.sqrt(dx ** 2 + dy ** 2)

    if distance + ball_radius >= CIRCLE_RADIUS:
        nx = dx / distance
        ny = dy / distance

        overlap = (distance + ball_radius) - CIRCLE_RADIUS
        ball_rect.x -= nx * overlap
        ball_rect.y -= ny * overlap

        dot_product = ball_velocity[0] * nx + ball_velocity[1] * ny
        ball_velocity[0] -= 2 * dot_product * nx
        ball_velocity[1] -= 2 * dot_product * ny

        ball_velocity[0] *= 0.98
        ball_velocity[1] *= 0.98

        if count <= 254:
            count += 50
        elif gCount <= 254:
            gCount += 50
        elif bCount <= 254:
            bCount += 50

        count = min(count, 255)
        gCount = min(gCount, 255)
        bCount = min(bCount, 255)

        dynamic_white = (count, gCount, bCount)

    rectangles_to_remove = []
    for rect in rectangles:
        rect.update()
        if rect.collides_with_ball(ball_rect):
            rectangles_to_remove.append(rect)
            score += 1

    for rect in rectangles_to_remove:
        rectangles.remove(rect)

    screen.fill(BLACK)

    pygame.draw.circle(screen, dynamic_white, CIRCLE_CENTER, CIRCLE_RADIUS, 3)

    for rect in rectangles:
        rect.draw(screen)

    pygame.draw.circle(screen, GREEN, ball_rect.center, ball_radius)

    score_text = font.render(f"Score: {score}", True, WHITE)
    screen.blit(score_text, (20, 20))

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()