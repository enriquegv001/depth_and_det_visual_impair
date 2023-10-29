from models import Detector, Midas, MobileCam
from setup import call_midas_model

if __name__=="__main__":
    #mymodel = MobileCam(midas_model, transform, 'PS')
    #mymodel.MultOut_RealTime(False)

    midas_model, transform, device = call_midas_model()
    mymodel = MobileCam(midas_model, transform, device, 'PS')
    mymodel.MultOut_img("dog.jpg")