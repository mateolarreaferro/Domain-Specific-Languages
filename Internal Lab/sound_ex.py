#!/usr/bin/env python3

from sound import Sound, Volume, Speed


g = Sound("./samples/guitar.mp3")
d = Sound("./samples/snare.mp3")
o = Sound("./samples/organ-riff.mp3")


# # a drum
# print("▶ Single snare hit")
# d.play()

# # eight times
# print("▶ Snare repeated 8 ×")
# (d * 8).play()

# # eight times, double speed
# print("▶ Snare 8 × at double speed (speed first, then repeat)")
# (d @ Speed(2) * 8).play()

# # eight times, double speed (a different way; notice the slight audio difference)
# print("▶ Snare 8 × then whole loop doubled in speed")
# ((d * 8) @ Speed(2)).play()

# # sequenced
# print("▶ Snare then guitar (sequenced with |)")
# (d | g).play()

# # parallel
# print("▶ Snare and guitar together (mixed with &)")
# (d & g).play()

# # slice
# print("▶ Guitar first second (g[:1])")
# g[:1].play()

# print("▶ Guitar full clip (g[:])")
# g[:].play()

# print("▶ Guitar from 1 s to end (g[1:])")
# g[1:].play()

# print("▶ Guitar between 1 s and 2 s (g[1:2])")
# g[1:2].play()

# # a few things
# print("▶ 4‑second groove: guitar + 32 quiet, double‑speed snares")
# (g & (d @ Volume(0.5) @ Speed(2) * 32))[:4.0].play()

# error: bad speed (any error is fine)
# print("▶ Expecting error: Speed(0)")
# (g @ Speed(0)).play()
# print("▶ Expecting error: Speed(-1)")
# (g @ Speed(-1.0)).play()

# error: bad volume (any error is fine)
# print("▶ Expecting error: Volume(-1)")
# (g @ Volume(-1.0)).play()

# write something yourself!
print("▶ Custom: Let's make some music!")
((g @ Volume(1)@ Speed(1) * 4) & (d @ Volume(0.25) @ Speed(5) * 82) & (o @Volume(0.1) @ Speed(0.5) * 2))[:6.8].play()