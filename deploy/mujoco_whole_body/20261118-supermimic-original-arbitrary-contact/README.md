# Original SuperMimic in MuJoCo with arbitrary point interaction

This bundle runs the released frozen SuperMimic motion encoder and decoder directly, with no compliance student and no force tokens. It provides a stationary nominal-pose goal and lets MuJoCo physical forces perturb any represented robot body.

```bash
python scripts/mujoco_whole_body_student.py --policy supermimic-original-arbitrary-contact
```

Left-drag any visible represented G1 body surface. The point follows the body and receives an 80 N/m spring with 4 N s/m damping, capped at 30 N and slewed at 150 N/s. The force changes the simulated robot dynamics but is not an input to the original SuperMimic policy. This makes it a physical-interaction baseline against the learned surface-compliance model.

The base controller is the frozen `e11c_0220000` SuperMimic release. Its 10-frame proprioceptive history and stationary nominal-pose goal are sent through the original encoder/decoder path. There are no motion clips or training data loaded at runtime.
