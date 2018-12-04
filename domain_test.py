import math

uh_oh = []
num = -3.0

while num <= 3.0:
    uh_oh.append(num)
    num += .1

for runs, i in enumerate(range(len(uh_oh))):
    if(runs != 0):
        print( uh_oh[runs], ((1.0/math.log(2)) * (10/runs*4)) % 8)