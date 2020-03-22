import pygame
import math
import numpy as np


class Agent:
    def  __init__(self):
        self.position = (0, 0)
        self.direction = 45
        self.size = 8
        self.speed = 0.6
        self.id = 0

    def __eq__(self, other):
        if isinstance(other,Agent):
            return other.id == self.id
        else:
            return False

    def update_pos(self):
        dx = self.speed * math.cos(self.direction*math.pi/180.0)
        dy = self.speed * math.sin(self.direction*math.pi/180.0)
        return np.array([[self.position[0] + dx], [self.position[1] + dy]])

    def get_pos(self):
        return np.array([[self.position[0]], [self.position[1]]])

    def clip_direction(self, angle=None):
        if angle is None:
            if self.direction > 360.0:
                self.direction -= 360.0
            elif self.direction < 0.0:
                self.direction = 360.0 + self.direction
            return None
        else:
            angle_n = angle
            if angle > 360.0:
                angle_n = angle - 360.0
            elif angle < 0.0:
                angle_n = 360.0 + angle
            return angle_n


    def set_position(self,pos):
        self.position = (pos[0][0], pos[1][0])

    def collides(self, other):
        d = np.array([[other.position[0] - self.position[0]],[other.position[1] -self.position[1]]])
        return np.linalg.norm(d) < other.size + self.size

    def collides_pos(self, new_pos, other):
        d = np.array([[other.position[0] - new_pos[0][0]],[other.position[1] - new_pos[1][0]]])
        return np.linalg.norm(d) < other.size + self.size

    def bounce(self, new_pos, other):

        if self.collides_pos(new_pos, other):
            p_abs=self.get_pos()
            d_new = self.clip_direction(self.direction + 180.0)
            return p_abs, d_new
        else:
            return None

    def draw(self, bg):
        pygame.draw.circle(bg, (128, 128, 128), (int(self.position[0]), int(self.position[1])),self.size)

class Wall:
    def __init__(self,p1, p2):
        self.p1 = p1
        self.p2 = p2
        l = self.get_length()
        self.mat_to_glob = np.transpose (np.array([[self.p2[0] - self.p1[0], self.p2[1]-self.p1[1]],[self.p1[1]-self.p2[1], self.p2[0]-self.p1[0]]], dtype=float))/l
        self.mat_to_rel = np.linalg.inv(self.mat_to_glob)

    def get_length(self):
        return np.sqrt((self.p1[0] - self.p2[0])*(self.p1[0] - self.p2[0]) + (self.p1[1] - self.p2[1])*(self.p1[1] - self.p2[1]))

    def collides(self,a: Agent):
        l = self.get_length()
        mat = np.array([[self.p2[0]-self.p1[0], self.p1[1]-self.p2[1]],[self.p2[1]-self.p1[1], self.p2[0] - self.p1[0]]], dtype=float)/l
        dp = np.array([[a.position[0] - self.p1[0]], [a.position[1] - self.p1[1]]], dtype=float)
        minv = np.linalg.inv(mat)
        r = np.dot(minv,dp)
        return math.fabs(r[1][0]) < a.size and r[0][0]/l >= 0.0 and r[0][0]/l <= 1.0

    def bounce_direction(self,a: Agent):
        angle = math.atan2(self.p2[1]-self.p1[1], self.p2[0]-self.p1[0])*180/math.pi
        angle_res = angle - (a.direction - angle)
        if angle_res > 360.0:
            angle_res -= 360.0
        elif angle_res < 0.0:
            angle_res = 360.0 + angle_res
        return angle_res

    def rel_direction(self,a: Agent):
        l = self.get_length()
        mat = np.array([[self.p2[0]-self.p1[0], self.p1[1]-self.p2[1]],[self.p2[1]-self.p1[1], self.p2[0] - self.p1[0]]], dtype=float)/l
        minv = np.linalg.inv(mat)
        d = np.array([[math.sin(a.direction*math.pi/180.0)],[math.cos(a.direction*math.pi/180.0)]],dtype=float)
        rel_coords = np.dot(minv,d)
        return rel_coords

    def rel_pos(self,pos):
        d0 = np.array([[self.p1[0]], [self.p1[1]]], dtype=float)
        rel_coords = np.dot(self.mat_to_rel,pos-d0)
        return rel_coords

    def glob_pos(self, pos):
        p0 = np.array([[self.p1[0]], [self.p1[1]]])
        return p0 + np.dot(self.mat_to_glob, pos)

    def bounce_calc(self, old_pos, new_pos, size):
        d_r = self.rel_pos(new_pos) - self.rel_pos(old_pos)
        p_r_0 = self.rel_pos(old_pos)
        p_r_1 = self.rel_pos(new_pos)
        l = self.get_length()
        bounced = False
        if (((d_r[1] < 0 and p_r_0[1] > size) and
                (p_r_1[1] < size and p_r_1[0] + size > 0 and p_r_1[0]-size < l)) or
                ((d_r[1] > 0 and p_r_0[1] < -size) and
                (p_r_1[1] > -size and p_r_1[0] + size > 0 and p_r_1[0] - size < l))):  # Agent has bumped into wall
            q = (a.size - p_r_0[1][0])/d_r[1]  # compute the length of the distance vector to the bounce point
            p_r_bounce = p_r_0 + q*d_r # position where the agent hits the wall
            d_r_mirror = d_r
            d_r_mirror[1][0] = -d_r_mirror[1][0] # relative direction after bounce
            p_r_mirror = p_r_bounce + (1-q)*d_r_mirror # the final position after bounce
            p_abs = self.glob_pos(p_r_mirror)

            d_new = np.dot(self.mat_to_glob,(p_r_mirror - p_r_bounce))
            d_new = math.atan2(d_new[1][0], d_new[0][0])*180.0/math.pi
            p_bounce = self.glob_pos(p_r_bounce)
            bounced = True
        else:
            p_abs = new_pos
            d_new = None
            p_bounce = None
        return bounced, p_abs, d_new, p_bounce


    def draw(self, bg):
        pygame.draw.line(bg,(0,0,0), self.p1, self.p2)

if __name__ == "__main__":
    FPS = 60
    screensize = (720,480)
    pygame.init()
    screen = pygame.display.set_mode(screensize)
    bg = pygame.Surface(screensize)
    dbg_font = pygame.font.SysFont('Comic Sans MS', 30)
    clock = pygame.time.Clock()


    agents = []
    a = Agent()
    a.position=(20, 30)
    a.direction = 2
    a.id = 1
    agents.append(a)

    for cnt in range(20):
        a = Agent()
        collided = True
        while collided is True:
            a.position = (np.random.randint(20, screensize[0]-20), np.random.randint(20, screensize[1]-20))
            a.direction = np.random.randint(0, 360)
            a.id = cnt + 10
            collided = False
            for ao in agents:
                if ao.collides(a):
                    collided = True
        agents.append(a)

    OFFSET = 10
    walls = []
    w = Wall((OFFSET, OFFSET), (screensize[0]-OFFSET, OFFSET))
    walls.append(w)
    w = Wall((screensize[0]-OFFSET, OFFSET), (screensize[0]-OFFSET, screensize[1]-OFFSET))
    walls.append(w)
    w = Wall((screensize[0]-OFFSET, screensize[1]-OFFSET), (OFFSET,screensize[1]-OFFSET))
    walls.append(w)
    w = Wall((OFFSET, screensize[1]-OFFSET), (OFFSET, OFFSET))
    walls.append(w)

    running = True
    while running:
        clock.tick(FPS)

        bg.fill((255, 255, 255))
        for w in walls:
            w.draw(bg)
        for a in agents:
            a.draw(bg)
            new_pos = a.update_pos()
            old_pos = a.get_pos()
            p_new = None
            for w in walls:
                (bounced, p_new, d_new, p_bounce) = w.bounce_calc(old_pos, new_pos, a.size)

                if bounced is True:
                    # do some special actions here if needed, like a second round of collision detection
                    a.direction = d_new
                    a.clip_direction()
                    for w in walls:
                        p_i = p_new
                        (bounced, p_new2, d_new2, p_bounce2) = w.bounce_calc(p_bounce, p_i, a.size)
                        if bounced is True:
                            p_new = p_new2
                            break
            for other in agents:
                if other != a:
                    b_res = a.bounce(new_pos, other)
                    if b_res is not None:
                        a.direction = b_res[1]
                        p_new = b_res[0]


            a.set_position(p_new)

        bg.blit(dbg_font.render("direction: {:.2f}\nspeed: {:.2f}".format(agents[0].direction,agents[0].speed),True,(255, 0, 0)), (10, screensize[1]-50-10))
        screen.blit(bg, (0, 0))
        pygame.display.flip()

        for evt in pygame.event.get():
            if evt.type == pygame.QUIT:
                running = False
                quit()

            elif evt.type == pygame.KEYDOWN:
                if evt.key == pygame.K_w:
                    agents[0].speed += 0.1
                    agents[0].clip_direction()
                elif evt.key == pygame.K_s:
                    agents[0].speed -= 0.1
                    agents[0].clip_direction()

