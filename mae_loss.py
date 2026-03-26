import sys

def mae_loss(file1, file2):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()

    assert len(lines1) == len(lines2), "Files have different number of points"    

    total_error = 0
    n = len(lines1)

    for i in range(n):
        parts1 = lines1[i].strip().split()
        parts2 = lines2[i].strip().split()

        assert len(parts1) == 4 and len(parts2) == 4, f"Invalid format at line {i+1}"
        assert parts1[:3] == parts2[:3], f"Mismatch in coordinates at line {i+1}"

        I1 = int(parts1[3])
        I2 = int(parts2[3])

        total_error += abs(I1 - I2)

    mae = total_error / n
    return mae


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python mae.py knn.txt approx_knn.txt")
        sys.exit(1)

    file1 = sys.argv[1]
    file2 = sys.argv[2]

    mae = mae_loss(file1, file2)
    print(f"MAE: {mae:.6f}")