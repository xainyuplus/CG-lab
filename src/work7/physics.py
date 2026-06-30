import taichi as ti

from .config import *


positions = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)
velocities = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)
forces = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)
fixed = ti.field(dtype=ti.i32, shape=NUM_PARTICLES)

x_next = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)
v_next = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)
f_next = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)

spring_pairs = ti.Vector.field(2, dtype=ti.i32, shape=MAX_SPRINGS)
spring_lengths = ti.field(dtype=ti.f32, shape=MAX_SPRINGS)
spring_indices = ti.field(dtype=ti.i32, shape=MAX_SPRINGS * 2)
num_springs = ti.field(dtype=ti.i32, shape=())

GRAVITY_VEC = ti.Vector(GRAVITY)


@ti.func
def pid(i, j):
    return i * CLOTH_N + j


@ti.kernel
def reset_spring_count():
    num_springs[None] = 0


@ti.kernel
def init_positions():
    for i, j in ti.ndrange(CLOTH_N, CLOTH_N):
        idx = pid(i, j)
        positions[idx] = ti.Vector([i * 0.05 - 0.5, 0.8, j * 0.05 - 0.5])
        velocities[idx] = ti.Vector([0.0, 0.0, 0.0])
        forces[idx] = ti.Vector([0.0, 0.0, 0.0])

        fixed[idx] = 0
        if j == 0 and (i == 0 or i == CLOTH_N - 1):
            fixed[idx] = 1


@ti.kernel
def init_springs():
    for i, j in ti.ndrange(CLOTH_N, CLOTH_N):
        idx = pid(i, j)

        if i < CLOTH_N - 1:
            idx_right = pid(i + 1, j)
            s = ti.atomic_add(num_springs[None], 1)
            spring_pairs[s] = ti.Vector([idx, idx_right])
            spring_lengths[s] = (positions[idx] - positions[idx_right]).norm()

        if j < CLOTH_N - 1:
            idx_down = pid(i, j + 1)
            s = ti.atomic_add(num_springs[None], 1)
            spring_pairs[s] = ti.Vector([idx, idx_down])
            spring_lengths[s] = (positions[idx] - positions[idx_down]).norm()


@ti.kernel
def init_spring_indices():
    for s in range(num_springs[None]):
        spring_indices[s * 2] = spring_pairs[s][0]
        spring_indices[s * 2 + 1] = spring_pairs[s][1]


@ti.func
def compute_forces_on(pos: ti.template(), vel: ti.template(), force: ti.template()):
    for i in range(NUM_PARTICLES):
        force[i] = PARTICLE_MASS * GRAVITY_VEC - DAMPING * vel[i]

    for s in range(num_springs[None]):
        a = spring_pairs[s][0]
        b = spring_pairs[s][1]
        delta = pos[a] - pos[b]
        dist = delta.norm()
        if dist > 1e-6:
            direction = delta / dist
            spring_force = -SPRING_K * (dist - spring_lengths[s]) * direction
            ti.atomic_add(force[a], spring_force)
            ti.atomic_add(force[b], -spring_force)


@ti.func
def clamp_velocity(vel: ti.template(), idx: ti.i32):
    speed = vel[idx].norm()
    if speed > MAX_SPEED:
        vel[idx] = vel[idx] / speed * MAX_SPEED


@ti.kernel
def step_explicit():
    compute_forces_on(positions, velocities, forces)
    for i in range(NUM_PARTICLES):
        if fixed[i] == 0:
            positions[i] += velocities[i] * TIME_STEP
            velocities[i] += forces[i] / PARTICLE_MASS * TIME_STEP
            clamp_velocity(velocities, i)
        else:
            velocities[i] = ti.Vector([0.0, 0.0, 0.0])


@ti.kernel
def step_semi_implicit():
    compute_forces_on(positions, velocities, forces)
    for i in range(NUM_PARTICLES):
        if fixed[i] == 0:
            velocities[i] += forces[i] / PARTICLE_MASS * TIME_STEP
            clamp_velocity(velocities, i)
            positions[i] += velocities[i] * TIME_STEP
        else:
            velocities[i] = ti.Vector([0.0, 0.0, 0.0])


@ti.kernel
def step_implicit_iter():
    for i in range(NUM_PARTICLES):
        v_next[i] = velocities[i]
        x_next[i] = positions[i]

    for _ in ti.static(range(IMPLICIT_ITERS)):
        compute_forces_on(x_next, v_next, f_next)
        for i in range(NUM_PARTICLES):
            if fixed[i] == 0:
                v_next[i] = velocities[i] + f_next[i] / PARTICLE_MASS * TIME_STEP
                clamp_velocity(v_next, i)
                x_next[i] = positions[i] + v_next[i] * TIME_STEP
            else:
                v_next[i] = ti.Vector([0.0, 0.0, 0.0])
                x_next[i] = positions[i]

    for i in range(NUM_PARTICLES):
        velocities[i] = v_next[i]
        positions[i] = x_next[i]


def initialize():
    reset_spring_count()
    init_positions()
    init_springs()
    init_spring_indices()


def step(method):
    if method == 0:
        step_explicit()
    elif method == 1:
        step_semi_implicit()
    else:
        step_implicit_iter()
