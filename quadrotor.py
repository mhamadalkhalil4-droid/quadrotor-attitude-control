"""Quadrotor 6-DoF simulation with a cascaded attitude + altitude controller.

Rigid-body dynamics (Euler angles, body rates, Newton–Euler), rotor thrust/torque mixing,
motor first-order lag and saturation. Controller: PID altitude -> PD attitude (outer angle,
inner rate) -> mixer -> four motor commands. Test: 3 s hover, then a 20° roll step and a 1 m
altitude step with a wind gust torque disturbance.
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

m, g, L = 1.2, 9.81, 0.22
J = np.diag([0.0123, 0.0123, 0.0224]); Jinv = np.linalg.inv(J)
kT, kQ, w_max, tau_m = 2.5e-5, 4.5e-7, 900.0, 0.03      # thrust/torque coeffs, max rotor speed, motor lag
dt, T = 0.001, 10.0; N = int(T / dt); t = np.arange(N) * dt

def rot(phi, th, psi):
    c, s = np.cos, np.sin
    return np.array([[c(th)*c(psi), s(phi)*s(th)*c(psi)-c(phi)*s(psi), c(phi)*s(th)*c(psi)+s(phi)*s(psi)],
                     [c(th)*s(psi), s(phi)*s(th)*s(psi)+c(phi)*c(psi), c(phi)*s(th)*s(psi)-s(phi)*c(psi)],
                     [-s(th),       s(phi)*c(th),                      c(phi)*c(th)]])

# mixer: [T, tau_x, tau_y, tau_z]  ->  w^2 of 4 rotors (X configuration)
Mix = np.array([[kT, kT, kT, kT],
                [-kT*L/np.sqrt(2),  kT*L/np.sqrt(2),  kT*L/np.sqrt(2), -kT*L/np.sqrt(2)],
                [ kT*L/np.sqrt(2),  kT*L/np.sqrt(2), -kT*L/np.sqrt(2), -kT*L/np.sqrt(2)],
                [ kQ, -kQ, kQ, -kQ]])
Mix_inv = np.linalg.inv(Mix)

# gains
Kp_z, Ki_z, Kd_z = 8.0, 2.0, 5.0
Kp_ang = np.array([6.0, 6.0, 4.0]); Kp_rate = np.array([0.18, 0.18, 0.05]); Kd_rate = np.array([0.002, 0.002, 0.0])

# state
pos, vel = np.zeros(3), np.zeros(3); eul = np.zeros(3); omega = np.zeros(3); w_rot = np.full(4, np.sqrt(m*g/4/kT))
iz = 0.0; rate_prev = np.zeros(3)
log = np.zeros((N, 10))
for i in range(N):
    z_cmd = 1.0 if t[i] > 6 else 0.0
    ang_cmd = np.array([np.deg2rad(20) if 3 < t[i] < 5 else 0.0, 0.0, 0.0])
    # --- controller ---
    ez = z_cmd - pos[2]; iz += ez * dt
    T_cmd = m * (g + Kp_z * ez + Ki_z * iz - Kd_z * vel[2]) / (np.cos(eul[0]) * np.cos(eul[1]))
    rate_cmd = Kp_ang * (ang_cmd - eul)
    er = rate_cmd - omega
    tau_cmd = Kp_rate * er + Kd_rate * (er - rate_prev) / dt; rate_prev = er
    w2 = np.clip(Mix_inv @ np.r_[T_cmd, tau_cmd], 0, w_max**2)
    w_cmd = np.sqrt(w2)
    # --- motors (first-order lag) ---
    w_rot += (w_cmd - w_rot) * dt / tau_m
    Fm = Mix @ w_rot**2
    thrust, tau = Fm[0], Fm[1:]
    # --- disturbance: gust torque around pitch at t = 8 s ---
    if 8.0 < t[i] < 8.2: tau = tau + np.array([0.0, 0.15, 0.0])
    # --- dynamics ---
    R = rot(*eul)
    acc = np.array([0, 0, -g]) + R @ np.array([0, 0, thrust]) / m - 0.1 * vel
    omega_dot = Jinv @ (tau - np.cross(omega, J @ omega))
    phi, th, _ = eul
    E = np.array([[1, np.sin(phi)*np.tan(th), np.cos(phi)*np.tan(th)],
                  [0, np.cos(phi), -np.sin(phi)],
                  [0, np.sin(phi)/np.cos(th), np.cos(phi)/np.cos(th)]])
    vel += acc * dt; pos += vel * dt; omega += omega_dot * dt; eul += E @ omega * dt
    log[i] = np.r_[pos, eul, thrust, w_rot[0], ang_cmd[0], z_cmd]

roll = np.rad2deg(log[:, 3]); z = log[:, 2]
i3, i5 = int(3/dt), int(5/dt)
ov = (roll[i3:i5].max() - 20) / 20 * 100
rise = t[i3 + np.argmax(roll[i3:i5] > 18)] - 3
print(f"Roll step 20°: rise time (90%) {rise*1000:.0f} ms, overshoot {ov:.1f} %")
i6 = int(6/dt); print(f"Altitude step 1 m: overshoot {(z[i6:].max()-1)*100:.1f} cm, settled within 5 cm after {t[i6 + np.argmax(np.abs(z[i6:]-1) < 0.05)] - 6:.2f} s")
print(f"Pitch excursion from 0.15 N·m gust: {np.rad2deg(np.abs(log[int(8/dt):int(9/dt), 4])).max():.1f}°")

fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
ax[0].plot(t, roll, label="roll"); ax[0].plot(t, np.rad2deg(log[:, 4]), label="pitch"); ax[0].plot(t, np.rad2deg(log[:, 8]), "k--", lw=1, label="roll cmd"); ax[0].set_ylabel("[°]"); ax[0].legend(); ax[0].set_title("Attitude: 20° roll step at 3 s, pitch gust at 8 s")
ax[1].plot(t, z, label="z"); ax[1].plot(t, log[:, 9], "k--", lw=1, label="z cmd"); ax[1].set_ylabel("altitude [m]"); ax[1].legend()
ax[2].plot(t, log[:, 7]); ax[2].axhline(w_max, c="r", ls="--", lw=1); ax[2].set_ylabel("rotor 1 speed [rad/s]"); ax[2].set_xlabel("t [s]")
for a in ax: a.grid(alpha=.3)
plt.tight_layout(); plt.savefig("figures/response.png", dpi=130)
