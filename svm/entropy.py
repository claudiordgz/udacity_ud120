import scipy.stats
entropy = scipy.stats.entropy([2,1], base=2)
print(entropy)
print(1 - (entropy * 3/4))
