from sklearn.metrics import precision_score, recall_score, f1_score

def compute_metrics(preds, targets):
    """
    preds: list of decoded prediction strings
    targets: list of decoded target strings
    """
    # Đưa về nhãn dạng role (giả sử format: "<Role> answer")
    def normalize(text):
        return text.strip().lower()

    preds_norm = [normalize(p) for p in preds]
    targets_norm = [normalize(t) for t in targets]

    # Vì output là chuỗi, ta so khớp exact match (1 nếu bằng nhau, 0 nếu khác)
    y_true = []
    y_pred = []
    for p, t in zip(preds_norm, targets_norm):
        y_true.append(t)
        y_pred.append(p)

    # Có thể dùng strict exact match: đúng = (p == t)
    labels = list(set(y_true + y_pred))  # lấy toàn bộ nhãn
    precision = precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)

    return precision, recall, f1
