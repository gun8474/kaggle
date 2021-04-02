# MAP@& 평가 척도를 구하는 코드

import numpy as np

def apk(actual, predicted, k=7, default = 0.0):
    # MAP@7 이므로, 최대 7개만 사용한다.
    if len(predicted > k): # 금융 제품 예측이 7보다 크다면
        predicted = predicted[:7] # predicted list는 7까지로 제한한다.

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        # 예측 값이 정답에 있다 ('p in actual')
        # 예측 값이 중복이 아니면 ('p not in predicted[:i]')
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    # 정답 값이 공백일 경우, 무조건  0.0점을 반환한다
    if not actual:
        return default

    # 정답의 개수 (len(actual))로 average precicison을 구한다
    return score / min(len(actual), k)

def mapk(actual, predicted, k=7, default = 0.0):
    # list of list인 정답 값(actual)과 예측값(predicted)에서 고객별 Average Precision을 구하고, np.mean()을 통해 평균을 계산한다.
    return np.mean([apk(a,p,k,default) for a, p in zip(actual, predicted)])


