import os

def write_box(f, x1, y1, z1, x2, y2, z2, v_idx):
    verts = [
        (x1, y1, z1), (x2, y1, z1), (x2, y2, z1), (x1, y2, z1),
        (x1, y1, z2), (x2, y1, z2), (x2, y2, z2), (x1, y2, z2)
    ]
    for v in verts:
        f.write(f"v {v[0]} {v[1]} {v[2]}\n")
    
    # Bottom (1,4,3,2)
    f.write(f"f {v_idx+1} {v_idx+4} {v_idx+3}\n")
    f.write(f"f {v_idx+1} {v_idx+3} {v_idx+2}\n")
    # Top (5,6,7,8)
    f.write(f"f {v_idx+5} {v_idx+6} {v_idx+7}\n")
    f.write(f"f {v_idx+5} {v_idx+7} {v_idx+8}\n")
    # Front (1,2,6,5)
    f.write(f"f {v_idx+1} {v_idx+2} {v_idx+6}\n")
    f.write(f"f {v_idx+1} {v_idx+6} {v_idx+5}\n")
    # Right (2,3,7,6)
    f.write(f"f {v_idx+2} {v_idx+3} {v_idx+7}\n")
    f.write(f"f {v_idx+2} {v_idx+7} {v_idx+6}\n")
    # Back (3,4,8,7)
    f.write(f"f {v_idx+3} {v_idx+4} {v_idx+8}\n")
    f.write(f"f {v_idx+3} {v_idx+8} {v_idx+7}\n")
    # Left (4,1,5,8)
    f.write(f"f {v_idx+4} {v_idx+1} {v_idx+5}\n")
    f.write(f"f {v_idx+4} {v_idx+5} {v_idx+8}\n")
    return v_idx + 8

def main():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "classroom", "classroom_map.obj"), "w") as f:
        f.write("# Generated L-Shape Corridor with Classroom and Door (Triangulated)\n")
        v_idx = 0
        
        # --- FLOORS ---
        v_idx = write_box(f, -0.2, -0.2, -0.1, 2.7, 15.2, 0, v_idx)
        v_idx = write_box(f, 2.7, 12.3, -0.1, 12.7, 15.2, 0, v_idx)
        v_idx = write_box(f, 4.3, 2.3, -0.1, 12.7, 12.3, 0, v_idx)
        
        # --- WALLS ---
        v_idx = write_box(f, -0.2, -0.2, 0, 0, 15.2, 3, v_idx)
        v_idx = write_box(f, 0, -0.2, 0, 2.5, 0, 3, v_idx)
        v_idx = write_box(f, 2.5, 0, 0, 2.7, 12.5, 3, v_idx)
        v_idx = write_box(f, -0.2, 15.0, 0, 12.7, 15.2, 3, v_idx)
        v_idx = write_box(f, 12.5, 12.3, 0, 12.7, 15.0, 3, v_idx)
        v_idx = write_box(f, 2.7, 12.3, 0, 4.5, 12.5, 3, v_idx)
        v_idx = write_box(f, 4.3, 2.3, 0, 4.5, 12.3, 3, v_idx)
        v_idx = write_box(f, 4.5, 2.3, 0, 12.7, 2.5, 3, v_idx)
        v_idx = write_box(f, 12.5, 2.5, 0, 12.7, 12.3, 3, v_idx)
        
        # --- DOORWAY AND DOOR ---
        v_idx = write_box(f, 4.5, 12.3, 0, 5.0, 12.5, 3, v_idx)
        v_idx = write_box(f, 6.2, 12.3, 0, 12.5, 12.5, 3, v_idx)
        v_idx = write_box(f, 5.0, 12.3, 2.1, 6.2, 12.5, 3, v_idx)
        v_idx = write_box(f, 5.0, 11.1, 0, 5.05, 12.3, 2.1, v_idx)

if __name__ == '__main__':
    main()
