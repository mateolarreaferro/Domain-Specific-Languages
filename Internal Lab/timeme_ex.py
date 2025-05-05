from timeme import timeme
from time import sleep

with timeme("sleep 0.5"):
    sleep(0.5)

with timeme("sleep 1"):
    sleep(1)
