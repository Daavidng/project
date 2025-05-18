import random

# Counter for iterations
attempts = 0
cost = 0

while True:
    attempts += 1
    cost += 286.00
    your_numbers = set(random.sample(range(1, 58-1), 12)) 
    # draw = set(random.sample(range(1, 58-1), 6)) 
    draw = set([4, 24, 49, 51, 52, 55])
    
    if attempts % 1000 == 0:
        print(f"Checkpoint: {attempts} attempts...")

    if draw.issubset(your_numbers):
        print(f'{sorted(your_numbers)}')
        print(f"\nYour numbers are a subset of the draw after {attempts} attempts!")
        print(f"Draw was: {sorted(draw)}")
        print(f'cost: {cost}')
        break
