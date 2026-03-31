from collections import defaultdict

jug1, jug2, aim = 4, 3, 2
visited = defaultdict(lambda: False)

def waterJugSolver(amt1, amt2):

    # Goal condition
    if amt1 == aim or amt2 == aim:
        print(amt1, amt2)
        return True

    if not visited[(amt1, amt2)]:
        print(amt1, amt2)
        visited[(amt1, amt2)] = True

        return (waterJugSolver(0, amt2) or              # Empty Jug1
                waterJugSolver(amt1, 0) or              # Empty Jug2
                waterJugSolver(jug1, amt2) or           # Fill Jug1
                waterJugSolver(amt1, jug2) or           # Fill Jug2
                waterJugSolver(                          # Pour Jug2 → Jug1
                    amt1 + min(amt2, jug1 - amt1),
                    amt2 - min(amt2, jug1 - amt1)
                ) or
                waterJugSolver(                          # Pour Jug1 → Jug2
                    amt1 - min(amt1, jug2 - amt2),
                    amt2 + min(amt1, jug2 - amt2)
                ))
    else:
        return False

print("Steps:")
waterJugSolver(0, 0)
