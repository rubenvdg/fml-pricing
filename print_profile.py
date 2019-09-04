import pstats
p = pstats.Stats('n10.txt')
p.sort_stats('cumulative').print_stats(30)