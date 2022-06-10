import pyaudio
import wave
import numpy as np
import queue

p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
   dev = p.get_device_info_by_index(i)
   print((i,dev['name'],dev['maxInputChannels']))

def Monitor():
    count=0
    CHUNK = 100
    FORMAT = pyaudio.paInt16
    CHANNELS =1
    RATE = 140000
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,#采样深度
                    channels=CHANNELS,#采样通道
                    rate=RATE,#采样率
                    input=True,
                    frames_per_buffer=CHUNK)#缓存

    frames = []
    yalen1 = 100
    yalen2 = 1200
    t = queue.Queue(yalen1)  # 如果不设置长度,默认为无限长
    for i in range(0,yalen1):
        data=stream.read(CHUNK)
        t.put(data)
    j=4
    print("开始缓存录音")
    while (True):
        #print(t.qsize())
        data = stream.read(CHUNK)
        t.get()
        t.put(data)
        audio_data = np.frombuffer(data, dtype=np.int16)
        temp = np.max(audio_data)
        #print(temp)
        if temp >2400 :
            print("检测到信号")
            q = queue.Queue(yalen2)  # 如果不设置长度,默认为无限长
            for i in range(0, yalen2):
                data = stream.read(CHUNK)
                q.put(data)
            for i in range(0, yalen1):
                frames.append(t.get())

            for i in range(0, yalen2):
                if (i <= yalen2 // 9.5):
                    q.get()
                else:
                    frames.append(q.get())
            j=j+1
            WAVE_OUTPUT_FILENAME = "1/ht/ya" + str(j) + ".wav"
            wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            frames = []
            for i in range(0, yalen1):
                data = stream.read(CHUNK)
                t.put(data)
            print("已录音" + str(j))
            if(j%10==0):
                break


    stream.stop_stream()
    stream.close()
    p.terminate()
if __name__ == '__main__':
    Monitor()
    print("over")