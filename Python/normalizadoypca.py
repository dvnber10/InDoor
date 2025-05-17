import csv
import random
import math

# Leer CSV
def load_csv(path):
    data = []
    with open(path, newline='') as f:
        reader = csv.reader(f)
        headers = next(reader)
        for row in reader:
            if row:
                data.append(row)
    return headers, data

# Guardar CSV
def save_csv(path, headers, rows):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

# Convertir a float menos la última columna (etiqueta)
def parse_data(data):
    X = [[float(v) for v in row[:-1]] for row in data]
    y = [row[-1] for row in data]
    return X, y

# Normalización MinMax
def normalize(X):
    min_vals = [min(col) for col in zip(*X)]
    max_vals = [max(col) for col in zip(*X)]
    return [[(x[i] - min_vals[i]) / (max_vals[i] - min_vals[i]) if max_vals[i] != min_vals[i] else 0 for i in range(len(x))] for x in X]

# Transponer matriz
def transpose(matrix):
    return list(map(list, zip(*matrix)))

# Media por columna
def mean_center(matrix):
    cols = transpose(matrix)
    means = [sum(col)/len(col) for col in cols]
    centered = [[x - means[i] for i, x in enumerate(row)] for row in matrix]
    return centered, means

# Multiplicación de matrices
def matmul(A, B):
    result = []
    for i in range(len(A)):
        row = []
        for j in range(len(B[0])):
            val = sum(A[i][k] * B[k][j] for k in range(len(B)))
            row.append(val)
        result.append(row)
    return result

# Obtener 100 muestras por clase
def subsample(X, y, n=100):
    class_dict = {}
    for xi, yi in zip(X, y):
        class_dict.setdefault(yi, []).append(xi)
    X_out, y_out = [], []
    for label, group in class_dict.items():
        selected = random.sample(group, min(n, len(group)))
        X_out.extend(selected)
        y_out.extend([label]*len(selected))
    return X_out, y_out

# Matriz covarianza
def covariance_matrix(X):
    X_t = transpose(X)
    size = len(X)
    cov = [[sum(X_t[i][k] * X_t[j][k] for k in range(size)) / (size - 1) for j in range(len(X_t))] for i in range(len(X_t))]
    return cov

# Obtener eigenvectores (PCA) con método de potencias
def power_iteration(matrix, num_components=3, num_iter=1000):
    vectors = []
    for _ in range(num_components):
        b_k = [random.random() for _ in range(len(matrix))]
        for _ in range(num_iter):
            # Multiplica A*b
            b_k1 = [sum(matrix[i][j] * b_k[j] for j in range(len(matrix))) for i in range(len(matrix))]
            # Normaliza
            norm = math.sqrt(sum(x**2 for x in b_k1))
            b_k = [x / norm for x in b_k1]
        vectors.append(b_k)
        # Método de deflación: elimina componente
        outer = [[b_k[i]*b_k[j] for j in range(len(b_k))] for i in range(len(b_k))]
        matrix = [[matrix[i][j] - outer[i][j] for j in range(len(matrix))] for i in range(len(matrix))]
    return vectors

# Proyectar datos en componentes
def project(X, components):
    return [[sum(x[i]*v[i] for i in range(len(x))) for v in components] for x in X]

# Main
def main():
    # Cargar datasets
    
    _, data1 = load_csv("dataset_combinado.csv")
    data = data1

    # Parseo
    X_raw, y = parse_data(data)

    # Submuestreo por clase
    X_sampled, y_sampled = subsample(X_raw, y, n=100)

    # Normalización
    X_norm = normalize(X_sampled)

    # Media centrada
    X_centered, _ = mean_center(X_norm)

    # PCA manual
    cov = covariance_matrix(X_centered)
    components = power_iteration(cov, num_components=3)
    X_pca = project(X_centered, components)

    # Guardar
    rows = [x + [label] for x, label in zip(X_pca, y_sampled)]
    save_csv("dataset_combinado_reducido_pca_manual.csv", ['pca_1', 'pca_2', 'pca_3', 'location'], rows)

    print("✅ Dataset optimizado guardado como 'dataset_combinado_reducido_pca_manual.csv'")

if __name__ == "__main__":
    main()
