# Homework 3 Written Solutions

## 1: What operators would you have chosen? (3+ sentences)
For sequencing I’d choose the right‑shift operator >> because it reads like “then” in plain English (kick >> snare). Overlaying feels additive, so the plus sign + is the clearest metaphor (kick + hiHat). I’d keep * for repetition since musicians already think “× 4.” For parameter tweaks I’d prefer percent % instead of the current @, hinting at “apply this amount” (snare % Volume(0.5)). Slicing with [] stays exactly the same—Python users already expect clip[1:3] to denote a time window.

## Bonus: Explain your GOTO implementation. (5-20 sentences)
(Not attempted — I skipped the bonus problem.)

## Feedback (at least 1 sentence each)

### What what your favorite problem on this assignment and why?

The sound DSL was my favorite because it became a tiny musical sandbox: a few operators turned into instant, audible feedback. I even whipped up something that sounds surprisingly musical:
((g @ Volume(1) @ Speed(1) * 4) & (d @ Volume(0.25) @ Speed(5) * 82) & (o @ Volume(0.1) @ Speed(0.5) * 2))[:6.8].play()
That one‑liner layers a guitar loop, rapid snares, and a slowed‑down pad—and I could iterate in seconds. The expressiveness‑to‑code ratio is very promising...


### What what your least favorite problem on this assignment and why?

The SAT‑solver DSL was the hardest. Before coding I had to teach myself NAND rewrites, Tseitin encoding, and CNF structures—concepts that were brand‑new to me—so progress felt slow until the theory clicked. Once the pieces fit, it was satisfying, but the learning curve made it the most challenging part of the p‑set. Overall, this pset felt a lot more manageable than the first one: the problems were better scoped (in my option), and once I understood the concepts, the implementation made more sense.

