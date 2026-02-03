class ConstraintSatisfactionProblem:
    def __init__(self, words):
        self.words = [w.strip() for w in words]
        self.variables = sorted(set("".join(self.words)))
        self.domains = {v: list(range(10)) for v in self.variables}
        self.constraints = {v: [] for v in self.variables}
        for v in self.variables:
            for o in self.variables:
                if v != o:
                    self.constraints[v].append(
                        lambda var, val, ass, o=o: o not in ass or ass[o] != val
                    )
        leads = {w[0] for w in self.words}
        for lead in leads:
            self.constraints[lead].append(lambda v, val, ass: val != 0)
    def is_consistent(self, var, val, ass):
        return all(c(var, val, ass) for c in self.constraints[var])
    def backtrack(self, ass):
        if len(ass) == len(self.variables):
            if self.check(ass):
                return ass
            return None
        var = next(v for v in self.variables if v not in ass)
        for val in self.domains[var]:
            if self.is_consistent(var, val, ass):
                ass[var] = val
                res = self.backtrack(ass)
                if res:
                    return res
                ass.pop(var)
        return None
    def word_value(self, word, a):
        return sum(a[ch] * (10 ** i) for i, ch in enumerate(reversed(word)))
    def check(self, a):
        total = sum(self.word_value(w, a) for w in self.words[:-1])
        return total == self.word_value(self.words[-1], a)
num_words = int(input("Enter the number of words (including the result): "))
words = []
for i in range(num_words):
    if i < num_words - 1:
        w = input(f"Enter word {i+1} to sum: ").strip()
    else:
        w = input(f"Enter result word: ").strip()
    words.append(w)
csp = ConstraintSatisfactionProblem(words)
solution = csp.backtrack({})
if solution:
    print("\nAssigned Values:")
    for k in sorted(solution):
        print(f"{k} = {solution[k]}")
    print("\nResult:")
    for w in words:
        print(f"{w} = {csp.word_value(w, solution)}")
else:
    print("\nNo solution found")
