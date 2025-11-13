import pygame
import math

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 800
FPS = 60
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 100, 255)
GREEN = ((0, 128, 0))

# Circle boundary
CIRCLE_CENTER = (WIDTH // 2, HEIGHT // 2)
CIRCLE_RADIUS = 300

# Ball properties
ball_pos = [WIDTH // 2, HEIGHT // 2 - 100]
ball_radius = 15
ball_velocity = [4, 3]

# Create display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Reinforcement Learning Simulation")
clock = pygame.time.Clock()

running = True
count = 0
gCount = 0
bCount = 255
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update ball position
    ball_pos[0] += ball_velocity[0]
    ball_pos[1] += ball_velocity[1]

    # Calculate distance from ball center to circle center
    dx = ball_pos[0] - CIRCLE_CENTER[0]
    dy = ball_pos[1] - CIRCLE_CENTER[1]
    distance = math.sqrt(dx ** 2 + dy ** 2)

    if distance + ball_radius >= CIRCLE_RADIUS:
        nx = dx / distance
        ny = dy / distance


        overlap = (distance + ball_radius) - CIRCLE_RADIUS
        ball_pos[0] -= nx * overlap
        ball_pos[1] -= ny * overlap


        dot_product = ball_velocity[0] * nx + ball_velocity[1] * ny
        ball_velocity[0] -= 2 * dot_product * nx
        ball_velocity[1] -= 2 * dot_product * ny

        ball_velocity[0] *= 0.98
        ball_velocity[1] *= 0.98

        WHITE = (count, gCount, bCount)
        if count <= 254:
            count += 50
        elif gCount <= 254:
            gCount += 50
        elif bCount <= 254:
            bCount += 50

    screen.fill(BLACK)


    pygame.draw.circle(screen, WHITE, CIRCLE_CENTER, CIRCLE_RADIUS, 3)


    pygame.draw.circle(screen, GREEN, (int(ball_pos[0]), int(ball_pos[1])), ball_radius)


    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()