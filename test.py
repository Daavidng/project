import random

# Counter for iterations
attempts = 0
cost = 0

while True:
    your_numbers = set(random.sample(range(1, 58-1), 6)) 
    attempts += 1
    cost += 2.00
    draw = set(random.sample(range(1, 58-1), 6)) 
    
    if attempts % 1000000 == 0:
        print(f"Checkpoint: {attempts} attempts...")

    if your_numbers.issubset(draw):
        print(f'{sorted(your_numbers)}')
        print(f"\nYour numbers are a subset of the draw after {attempts} attempts!")
        print(f"Draw was: {sorted(draw)}")
        print(f'cost: {cost}')
        break
