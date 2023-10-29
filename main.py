from models import Detector, Midas, MobileCam
from setup import call_midas_model

from IPython.display import display, Audio, clear_output
import ipywidgets as widgets
import time

playing_audio = None  # Variable to store the currently playing audio object

def stop_audio():
    global playing_audio
    if playing_audio:
        print()
        #print('\x1b[2J\x1b[H')  # Clear the screen to stop the audio playback
        clear_output()
        playing_audio = None   # Reset the currently playing audio object


if __name__=="__main__":
  midas_model, transform, device = call_midas_model()
  mymodel = MobileCam(midas_model, transform, device, 'PS')

  while True:
    vw = mymodel.MultOut_RealTime(True)
    stop_audio()
    playing_audio = vw  # Update the currently playing audio object
    display(vw) # Stop any previously playing audio
    time.sleep(5)

    #mymodel = MobileCam(midas_model, transform, device, 'PS')
    #mymodel.MultOut_img("dog.jpg")  