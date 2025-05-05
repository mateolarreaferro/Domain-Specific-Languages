#!/usr/bin/env python3

from sound import Sound, Volume, Speed


g = Sound("./samples/guitar.mp3")
d = Sound("./samples/snare.mp3")

# a drum
# d.play()

# eight times
# (d * 8).play()

# eight times, double speed
# (d @ Speed(2) * 8).play()

# eight times, double speed (a different way; notice the slight audio difference)
# ((d * 8) @ Speed(2)).play()

# sequenced
# (d | g).play()

# parallel
# (d & g).play()

# slice
# g[:1].play()
# g[:].play()
# g[1:].play()
# g[1:2].play()

# a few things
# (g & (d @ Volume(0.5) @ Speed(2) * 32))[:4.0].play()

# error: bad speed (any error is fine)
# (g @ Speed(0)).play()
# (g @ Speed(-1.0)).play()

# error: bad volume (any error is fine)
# (g @ Volume(-1.0)).play()

# write something yourself!
