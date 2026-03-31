from collections import deque

def water_jug(jug1, jug2, target):
    visited = set()
    queue = deque()
    queue.append((0, 0))   # initial state

    path = []

    while queue:
        x, y = queue.popleft()

        if (x, y) in visited:
            continue

        visited.add((x, y))
        path.append((x, y))

        # If target achieved
        if x == target or y == target:
            print("Solution Path:")
            for state in path:
                print(state)
            return

        # Possible operations
        states = [
            (jug1, y),          # Fill Jug1
            (x, jug2),          # Fill Jug2
            (0, y),             # Empty Jug1
            (x, 0),             # Empty Jug2
            # Pour Jug1 → Jug2
            (x - min(x, jug2 - y), y + min(x, jug2 - y)),
            # Pour Jug2 → Jug1
            (x + min(y, jug1 - x), y - min(y, jug1 - x))
        ]

        for state in states:
            if state not in visited:
                queue.append(state)

    print("Solution not possible")

if __name__ == "__main__":
    jug1 = int(input("Enter capacity of Jug1: "))
    jug2 = int(input("Enter capacity of Jug2: "))
    target = int(input("Enter target amount: "))

    water_jug(jug1, jug2, target)
