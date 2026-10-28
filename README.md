# Quadrotor 6-DoF simulation and cascaded attitude/altitude control

From scratch in NumPy: rigid-body dynamics (Newton–Euler, Euler-angle kinematics), X-configuration
rotor mixing with thrust and reaction-torque coefficients, first-order motor lag and rotor-speed
saturation, plus a cascade controller:

```
altitude PID ──► collective thrust ┐
                                   ├─► mixer ─► ω²_1..4 ─► motors (lag, saturation) ─► plant
angle P ─► rate PD ─► torques      ┘
```

## Result

| Test | Result |
|---|---|
| 20° roll step | rise time 295 ms, overshoot 1.5 % |
| +1 m altitude step | overshoot 13 cm, within 5 cm after 1.1 s |
| 0.15 N·m pitch gust for 0.2 s | 6° excursion, recovered in < 1 s |

![response](figures/response.png)

## Run

```bash
pip install numpy matplotlib
python quadrotor.py
```

## Notes

* Thrust is compensated by `1/(cos φ cos θ)` so a roll command does not lose altitude — visible at
  3 s where z barely moves during the 20° roll.
* The angle → rate cascade is the structure every real flight stack uses (PX4, ArduPilot, Betaflight);
  the inner rate loop runs on gyro data and is what keeps the vehicle controllable.
* Next: quaternion attitude (no gimbal lock), position control outer loop, a complementary filter
  on simulated IMU data in place of the true state, and a PX4 SITL comparison.
