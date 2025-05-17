import random
import csv

def print_progress(iteration, total, prefix='', suffix='', length=50):
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    if iteration == total:
        print()

class RandomForest:
    def __init__(self, n_trees=10, max_depth=5, min_size=2, test_size=0.2):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_size = min_size
        self.test_size = test_size
        self.trees = []

    def gini_index(self, grupos, clases, num_instancias):
        gini = 0.0
        for grupo in grupos:
            n_grupo = float(len(grupo))
            if n_grupo == 0:
                continue
            score = 0.0
            for valor_clase in clases:
                p = [fila[-1] for fila in grupo].count(valor_clase) / n_grupo
                score += p ** 2
            gini += (1.0 - score) * (n_grupo / num_instancias)
        return gini

    def test_split(self, index, value, dataset):
        left, right = [], []
        for row in dataset:
            if float(row[index]) < value:
                left.append(row)
            else:
                right.append(row)
        return left, right

    def get_split(self, dataset):
        class_values = list(set(row[-1] for row in dataset))
        best_index, best_value, best_score, best_groups = None, None, float('inf'), None
        for index in range(len(dataset[0]) - 1):
            for row in dataset:
                value = float(row[index])
                groups = self.test_split(index, value, dataset)
                gini = self.gini_index(groups, class_values, len(dataset))
                if gini < best_score:
                    best_index, best_value, best_score, best_groups = index, value, gini, groups
        return {'index': best_index, 'value': best_value, 'groups': best_groups}

    def to_terminal(self, group):
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key=outcomes.count)

    def split(self, node, depth):
        left, right = node['groups']
        del node['groups']
        if not left or not right:
            node['left'] = node['right'] = self.to_terminal(left + right)
            return
        if depth >= self.max_depth:
            node['left'], node['right'] = self.to_terminal(left), self.to_terminal(right)
            return
        if len(left) <= self.min_size:
            node['left'] = self.to_terminal(left)
        else:
            node['left'] = self.get_split(left)
            self.split(node['left'], depth + 1)
        if len(right) <= self.min_size:
            node['right'] = self.to_terminal(right)
        else:
            node['right'] = self.get_split(right)
            self.split(node['right'], depth + 1)

    def build_tree(self, train):
        root = self.get_split(train)
        self.split(root, 1)
        return root

    def predict(self, node, row):
        if float(row[node['index']]) < node['value']:
            if isinstance(node['left'], dict):
                return self.predict(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.predict(node['right'], row)
            else:
                return node['right']

    def bootstrap_sample(self, dataset):
        sample = []
        while len(sample) < len(dataset):
            index = random.randrange(len(dataset))
            sample.append(dataset[index])
        return sample

    def train_test_split(self, dataset):
        random.shuffle(dataset)
        test_size = int(len(dataset) * self.test_size)
        return dataset[test_size:], dataset[:test_size]

    def fit(self, dataset):
        self.trees = []
        train, _ = self.train_test_split(dataset)
        print("Entrenando modelo Random Forest...")
        for i in range(self.n_trees):
            sample = self.bootstrap_sample(train)
            tree = self.build_tree(sample)
            self.trees.append(tree)
            print_progress(i + 1, self.n_trees, prefix='Progreso', suffix='Completado')

    def bagging_predict(self, row):
        predictions = [self.predict(tree, row) for tree in self.trees]
        return max(set(predictions), key=predictions.count)

# Normalización manual
def normalize_dataset(dataset):
    features = list(zip(*dataset))[:-1]
    normalized = []

    for feature in features:
        feature = list(map(float, feature))
        min_val = min(feature)
        max_val = max(feature)
        if max_val - min_val == 0:
            norm_feature = [0.0] * len(feature)
        else:
            norm_feature = [(x - min_val) / (max_val - min_val) for x in feature]
        normalized.append(norm_feature)

    normalized = list(zip(*normalized))  # volver a formato filas
    for i in range(len(normalized)):
        normalized[i] = list(normalized[i]) + [dataset[i][-1]]
    return normalized

# Cargar datos
def load_csv(file_path):
    dataset = []
    with open(file_path, newline='') as f:
        reader = csv.reader(f)
        next(reader)  # encabezado
        for row in reader:
            if len(row) < 2: continue
            dataset.append(row)
    return dataset

# Entrenamiento y prueba
if __name__ == "__main__":
    data = load_csv("dataset_combinado_reducido_pca.csv")
    data = normalize_dataset(data)

    rf = RandomForest(n_trees=10, max_depth=5)
    rf.fit(data)

    # Guardar el modelo (simplemente los árboles serializados)
    import pickle
    with open("modelo_entrenado_rf.pkl", "wb") as f:
        pickle.dump(rf.trees, f)

    # Predicción de prueba
    _, test = rf.train_test_split(data)
    for row in test[:10]:
        pred = rf.bagging_predict(row)
        print(f"Predicho: {pred}, Real: {row[-1]}")
