import librosa
import numpy as np

filename = 'PATH'

# 어떤 오디오 파일에 대해 mel spectrogram을 구하게 되면,
# pip estimation을 통해 pitch 및 magnitude sequence를 얻을 수 있음

# 프레임별 pitch 및 magnitude 정보에서 가장 유력한 factor (F0)를 뽑음 (상대적인 값)
def pip_to_pitch(pitches):
    return [np.argmax(t) for t in pitches]

# 임의의 pitch/magnitude  sequence에서 stdev를 계산
def error_tendency(pit_seq,mean):
    return np.std(pit_seq/mean)

# 에너지의 기준 (magnitude의 mean - stdev; heuristic한 기준)을 하나 잡고,
# 그보다 작은 값을 가진 frame을 ineffective한 것으로 정의 (voice 존재성)
# 결과적으로 silence의 비율 및 silence 여부를 알려주는 index frame 반환
def find_delay(mag_seq, std):
    count = 0
    z = np.ones(len(mag_seq))
    for i in range(len(mag_seq)):
        if mag_seq[i] <  np.average(mag_seq) - std:
            count += 1
            z[i] = 0
    return count/len(mag_seq), z

# effective한 frame의 pitch들로만 이루어진 sequence (eff)를 계산하고
# 그로부터 평균(mean)과 표준편차(error)를 계산
def effective_analysis(pitches, z):
    eff = []
    for i in range(len(pitches)):
        if z[i]:
            eff.append(pitches[i])
    mean  = np.average(eff)
    error = error_tendency(eff,mean)
    return mean, error

# 파일 하나에 대해 통계치를 추출
def extract_features(filename):
    # 파일 로드 부분
    y, sr = librosa.load(filename)
    # mel spectrogram을 fmax = 8000으로 계산 후 차원 고려하여 transpose
    S = librosa.feature.melspectrogram(y=y,sr=sr,n_mels=128,fmax=8000)
    S = np.transpose(S)
    # piptrack 알고리즘
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitches = np.transpose(pitches)
    magnitudes = np.transpose(magnitudes)
    # 가장 유효한 값들을 나열한 일차원 sequence로 변환
    pit_seq = pip_to_pitch(pitches)
    mag_seq = pip_to_pitch(magnitudes)
    # 발화 전체 중 silence의 비율 (delay_p)
    delay_p, mag_index  = find_delay(mag_seq, np.std(mag_seq))
    # 유효 frame에 대해 pitch 의 mean 및 stdev
    mean, error = effective_analysis(pit_seq, mag_index)
    return delay_p, mean, error
