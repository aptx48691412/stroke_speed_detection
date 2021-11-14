from __future__ import division

import re
import sys

from google.cloud import speech

import pyaudio
from six.moves import queue

import time

import csv

import matplotlib.pyplot as plt
import cv2

import pykakasi

all_list=list()

# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms

class MicrophoneStream(object):
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk

        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            # The API currently only supports 1-channel (mono) audio
            # https://goo.gl/z757pE
            channels=1,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )

        self.closed = False

        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Continuously collect data from the audio stream, into the buffer."""
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b"".join(data)

def listen_print_loop(responses,all_start):
    lap_dict=dict()
    lap_dict_kks=dict()

    start=time.time()

    num_chars_printed = 0

    with open('./csv/aswsa_.csv','w',newline='') as f:
        writer=csv.writer(f)

        #start_speech=time.time()
        #print('start_speech={}'.format(start_speech-all_start))
        
        #print('start_speech_new_={}'.format(time.time()-all_start))

        try:

            for response in responses:
                
                #print('response={}'.format(response))

                if not response.results:
                    continue
            
                result = response.results[0]
                #print('result={}'.format(result))

                if not result.alternatives:
                    continue

                

            # Display the transcription of the top alternative.
            
                lap=time.time()

                kks=pykakasi.kakasi()
            

                transcript = result.alternatives[0].transcript
                transcript_kks_list=list()
                for kkksss in kks.convert(transcript):
                    transcript_kks_list.append(kkksss['hira'])

                lap_dict[transcript]=lap-start
                lap_dict_kks[transcript_kks_list[-1]]=lap-start

                all_list.append([lap_dict,transcript])
                #writer.writerow(list(lap_dict.items()))

                overwrite_chars = " " * (num_chars_printed - len(transcript))

                #print(num_chars_printed - len(transcript))
                #print('==={}==='.format(overwrite_chars))
                

                if not result.is_final:
                    sys.stdout.write(transcript +'-----' +overwrite_chars 
                    #+ "\r"


                    )

                    sys.stdout.flush()

                    num_chars_printed = len(transcript)

                
                    #print(transcript)
                    #print(lap_dict)
                    #print(lap_dict_kks)

                    #writer.writerow(all_list)
                
                #elif key==ord('t'):
                    #x__=list(lap_dict.values())
                    #figure=plt.figure()
                    #plt.plot(x__,np.sin(x__))
                    #fig.savefig('speech_text.png')
                    #plt.show()

                else:
                    print(transcript + overwrite_chars)

                    #print('====={}====='.format(transcript))
                    #print('====={}====='.format(lap_dict))
                    #print('====={}====='.format(lap_dict_kks))
                    
                    #writer.writerow(all_list)

                    # Exit recognition if any of the transcribed phrases could be
                    # one of our keywords.
                
                    if re.search(r"\b(exit|quit)\b", transcript, re.I):
                        print("Exiting..")
                        break

                    num_chars_printed = 0

        except:
            #with open ('itemitem.csv','w') as hjk:
                #writer_hjk=csv.writer(hjk)
            writer.writerow(lap_dict.values())
            writer.writerow(lap_dict)
            writer.writerow(lap_dict_kks)
            
            

def main(all_start_):
    # See http://g.co/cloud/speech/docs/languages
    # for a list of supported languages.
    language_code = "ja-JP"  # a BCP-47 language tag

    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=language_code,
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config, interim_results=True
    )

    with MicrophoneStream(RATE, CHUNK) as stream:
        audio_generator = stream.generator()
        requests = (
            speech.StreamingRecognizeRequest(audio_content=content)
            for content in audio_generator
        )

        

        start_speech=time.time()
        print('start_speech={}'.format(start_speech-all_start_))

        time.sleep(10-(start_speech-all_start_))

        print('start_speech_new={}'.format(time.time()-all_start_))

        responses = client.streaming_recognize(streaming_config, requests)

        #print('------{}------'.format(responses))

        # Now, put the transcription responses to use.
        listen_print_loop(responses,all_start_)

if __name__ == "__main__":
    main(all_start_)