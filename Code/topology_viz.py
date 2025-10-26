# topology_viz.py
import networkx as nx
import matplotlib.pyplot as plt

# Example topology: one base station at center and 4 users at corners/radii
G = nx.Graph()
G.add_node("BS", pos=(0,0), role="bs")
users = {"U1":(50,20), "U2":(-40,30), "U3":(-60,-20), "U4":(30,-50)}
for k,pos in users.items():
    G.add_node(k, pos=pos, role="user")

# Connect BS to users (star)
for u in users.keys():
    G.add_edge("BS", u)

pos = nx.get_node_attributes(G, 'pos')
roles = nx.get_node_attributes(G, 'role')
colors = []
sizes = []
labels = {}
for n in G.nodes():
    if roles[n]=="bs":
        colors.append('red')
        sizes.append(400)
        labels[n] = "BS"
    else:
        colors.append('blue')
        sizes.append(200)
        labels[n] = n

plt.figure(figsize=(6,6))
nx.draw(G, pos, with_labels=False, node_color=colors, node_size=sizes, edge_color='gray')
nx.draw_networkx_labels(G, pos, labels=labels)
# Example power allocations (for visualization) - fractions to each user (sum<=1)
alloc_example = {"U1":0.35, "U2":0.15, "U3":0.25, "U4":0.25}
# annotate allocation near each user
for u,frac in alloc_example.items():
    x,y = pos[u]
    plt.text(x+5, y+5, f"P={frac:.2f}", fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
plt.title("Example topology and per-user power fraction (snapshot)")
plt.axis('equal'); plt.show()
