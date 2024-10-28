import pandas as pd
import numpy as np
from pmdarima import auto_arima
from pmdarima.arima import ndiffs
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# 데이터 전처리
df = pd.read_csv('data.csv')
df['date'] = pd.to_datetime(df['date'])

series = df.set_index('date')['average']

# 차분 횟수 자동 결정
kpss_diffs = ndiffs(series, alpha=0.05, test='kpss', max_d=6)
adf_diffs = ndiffs(series, alpha=0.05, test='adf', max_d=6)
n_diffs = max(adf_diffs, kpss_diffs)

# auto arima 모델 학습
model = auto_arima(series, d=n_diffs, seasonal=True, trace=True, stepwise=True)
model.fit(series)


def forecast_once():
    fc, conf_int = model.predict(n_periods=1, return_conf_int=True)
    return (int(fc.tolist()[0]), np.asarray(conf_int).tolist()[0])


prediction = []

for s in series:
    if len(prediction) == 365: # 1년치만 예측
        break

    fc, conf = forecast_once() # 예측값, 신뢰구간
    prediction.append(fc)

    # 모형 업데이트
    model.update(s)


forecast_index = pd.date_range(start='2024-01-01', periods=len(prediction), freq='D')
forecast_df = pd.DataFrame({'prediction': prediction}, index=forecast_index)
forecast_df.to_csv('forecast.csv', index=True, header=True)


# 폰트 설정(한글 깨짐)
font_path = 'C:/Windows/Fonts/NanumGothic.ttf'
font_prop = fm.FontProperties(fname=font_path)


plt.figure(figsize=(14, 7))
plt.plot(series, label='last', color='blue', alpha=0.5)
plt.plot(forecast_df, label = 'predicted', color='red')
plt.title('2024년 전력 수요량 예측', fontproperties=font_prop)
plt.xlabel('날짜', fontproperties=font_prop)
plt.ylabel('전력 수요량(MWh)', fontproperties=font_prop)    
plt.show()