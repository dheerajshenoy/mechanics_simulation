import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import random as rand
from itertools import combinations

WIDTH = 5
HEIGHT = 5
COLORS = ['gray', '#FF5000', '#FF2201', '#FA1122']

class Particle:
    def __init__(self, x, y, r, vx, vy, m=10):
        self.radius = r
        self.r = np.array([x, y])
        self.v = np.array([vx, vy])
        self._m = m

    @property
    def x(self):
        return self.r[0]

    @x.setter
    def x(self, value):
        self.r[0] = value

    @property
    def y(self):
        return self.r[1]

    @y.setter
    def y(self, value):
        self.r[1] = value

    @property
    def vx(self):
        return self.v[0]

    @vx.setter
    def vx(self, value):
        self.v[0] = value

    @property
    def vy(self):
        return self.v[1]

    @vy.setter
    def vy(self, value):
        self.v[1] = value

    @property
    def m(self):
        return self._m

    @m.setter
    def m(self, value):
        self._m = value

    def advance(self, dt):
        self.r += self.v * dt

        if self.x - self.radius <= 0:
            self.x = self.radius
            self.vx = -self.vx

        if self.y - self.radius <= 0:
            self.y = self.radius
            self.vy = -self.vy

        if self.x + self.radius >= WIDTH:
            self.x = WIDTH-self.radius
            self.vx = -self.vx

        if self.y + self.radius >= HEIGHT:
            self.y = HEIGHT-self.radius
            self.vy = -self.vy
        
    def draw(self, ax):
        circle = Circle(radius=self.radius, xy=self.r, edgecolor=rand.choice(COLORS), fill=True)
        ax.add_patch(circle)
        return circle

    def overlaps(self, other):
        return (np.hypot(*(self.r - other.r)) < self.radius + other.radius)

class Simulation:
    def __init__(self, n, radius=0.01):
        self.init_particles(n, radius)

    def init_particles(self, n, radius):
        try:
            iterator = iter(radius)
            assert n == len(radius)
        except TypeError:
            def r_gen(n, radius):
                for i in range(n):
                    yield radius
            radius = r_gen(n, radius)

        self.n = n
        self.particles = []
        for i, rad in enumerate(radius):
            while True:
                x, y = rad + (WIDTH-rad**2) * np.random.random(2)
                vr = 2
                vphi = 2*np.pi*np.random.random()
                vx, vy = vr * np.cos(vphi), vr * np.sin(vphi)
                m = 10
                p = Particle(x, y, rad, vx, vy, m)
                for p2 in self.particles:
                    if p2.overlaps(p):
                        break
                else:
                    self.particles.append(p)
                    break
    
    def handle_collision(self):

        def change_velocities(p1, p2):
            m1, m2 = p1.m, p2.m
            M = m1 + m2
            r1, r2 = p1.r, p2.r
            d = np.linalg.norm(r1 - r2)**2
            v1, v2 = p1.v, p2.v
            u1 = v1 - 2*m2/M * np.dot(v1-v2, r1-r2)/d * (r1-r2)
            u2 = v2 - 2*m1/M * np.dot(v2-v1, r2-r1)/d * (r2-r1)
            p1.v = u1
            p2.v = u2

        pairs = combinations(range(self.n), 2)
        for i, j in pairs:
            if self.particles[i].overlaps(self.particles[j]):
                change_velocities(self.particles[i], self.particles[j])

    def advance_animation(self, dt):

        for i, p in enumerate(self.particles):
            p.advance(dt)
            self.circles[i].center = p.r
        self.handle_collision()
        return self.circles

    def advance(self, dt):
        for i,p in enumerate(self.particles):
            p.advance(dt)
        self.handle_collision()

    def init(self):
        self.circles = []
        for particle in self.particles:
            self.circles.append(particle.draw(self.ax))
        return self.circles

    def animate(self, i):
        self.advance_animation(0.01)
        return self.circles
    
    def do_animation(self, save=False):
        fig, self.ax = plt.subplots()
        self.ax.xaxis.set_ticks([])
        self.ax.yaxis.set_ticks([])
        self.ax.set_xlim([0, WIDTH])
        self.ax.set_ylim([0, HEIGHT])
        self.ax.set_aspect('equal', 'box')
        anim = FuncAnimation(fig, self.animate, init_func=self.init, frames=800, interval=2, blit=True)
        
        if save:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=100, bitrate=1000)
            anim.save('collision.mp4', writer=writer)
        else:
            plt.show()

if __name__ == "__main__":
    nparticles = 5
    radii = 0.5
    sim = Simulation(nparticles, radii)
    sim.do_animation(save=False)
