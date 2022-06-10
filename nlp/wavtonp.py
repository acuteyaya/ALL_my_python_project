import numpy as np
from pydub import AudioSegment
wav =np.frombuffer(AudioSegment.from_mp3(r"E:\window\tiaoshi\pycharm\ya\nlp\traints\wav\qj\ya1.wav").raw_data, dtype="int16")
wav = wav.astype("float64")
norms = np.linalg.norm(wav, axis=0)
#wav=wav/norms
print(wav.shape)