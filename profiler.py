import pstats
from pstats import SortKey

p = pstats.Stats('profile.txt')
p.sort_stats(SortKey.CUMULATIVE).print_stats(100)

print("Done")

