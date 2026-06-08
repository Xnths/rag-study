import string

def lcs_length(a, b):
    a = a.split()
    b = b.split()
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]

def normalize(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [t for t in tokens if t not in {"a", "an", "the"}]
    return " ".join(tokens)

def rouge_l(y_pred, y_ref):
    y_pred = normalize(y_pred)
    y_ref = normalize(y_ref)

    pred_tokens = y_pred.split()
    ref_tokens = y_ref.split()

    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0

    lcs = lcs_length(y_pred, y_ref)
    P = lcs / len(pred_tokens)
    R = lcs / len(ref_tokens)

    if P + R == 0:
        return 0.0

    return (2 * P * R) / (P + R)


y_pred = "Wikipedia dump 2018"
y_ref  = "Wikipedia"
print(rouge_l(y_pred, y_ref))

y_pred = "BART"
y_ref  = "BART"
print(rouge_l(y_pred, y_ref))

y_pred = "I don't know"
y_ref  = "Maximum Inner Product Search"
print(rouge_l(y_pred, y_ref))