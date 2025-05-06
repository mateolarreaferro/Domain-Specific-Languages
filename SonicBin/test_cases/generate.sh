#!/bin/bash
as -o /tmp/a.out $1
objcopy -O binary /tmp/a.out $2
