from models import Detector, Midas, MobileCam

if __name__=="__main__":
    mymodel = MobileCam(midas_model, transform, 'PS')
    mymodel.MultOut_RealTime(False)

    mymodel = MobileCam(midas_model, transform, 'PS')
    mymodel.MultOut_img("city image.jfif")