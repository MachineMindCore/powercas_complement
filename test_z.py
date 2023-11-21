from z_kron import GraphZ

Z_plus = GraphZ(4)
Z_plus.add(0.4j, (1,0))
Z_plus.add(0.45j, (1,2))
Z_plus.add(0.25j, (1,3))
Z_plus.add(0.35j, (2,3))
Z_plus.add(0.35j, (2,0))
Z_plus.add(0.15j, (3,4))


print()