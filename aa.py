import pandas as pd

def parse_obj(path):
    verts = 0
    faces = 0
    adj = {}
    with open(path, 'r') as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == 'v':
                verts += 1
                adj[verts - 1] = set()
            elif parts[0] == 'f':
                faces += 1
                idxs = []
                for tok in parts[1:]:
                    core = tok.split('/')[0]      # "6893.0" or "6893"
                    core = core.split('.')[0]     # drop ".0" if present
                    idx = int(core) - 1
                    idxs.append(idx)
                for i in range(len(idxs)):
                    a, b = idxs[i], idxs[(i + 1) % len(idxs)]
                    adj[a].add(b)
                    adj[b].add(a)
    valences = [len(neigh) for neigh in adj.values()]
    avg_valence = sum(valences) / len(valences) if valences else 0
    return verts, faces, avg_valence

if __name__ == "__main__":
    files = {
        "body_female.obj": r"C:\Users\User\Downloads\pant_female_meshes\body_female.obj",
        "pant_female.obj": r"C:\Users\User\Downloads\pant_female_meshes\garment_pant_female.obj"
    }

    rows = []
    for name, path in files.items():
        try:
            verts, faces, avg_val = parse_obj(path)
            rows.append({
                "object": name,
                "num_vertices": verts,
                "num_faces": faces,
                "avg_valence": round(avg_val, 2)
            })
        except FileNotFoundError:
            rows.append({
                "object": name,
                "num_vertices": None,
                "num_faces": None,
                "avg_valence": None
            })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
