import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print("Memory growth must be set before GPUs have been initialized")
        print(e)

# Frame size of training data
w = 480
h = 320
# Patch size and patch step for training
patchsize = 40
patchstep = 20

# Folders
testfolderRawYuv = './testrawyuv/'
testfolderRawPng = './testrawpng/'
testfolderCompYuv = './testcompyuv/'
testfolderCompPng = './testcomppng/'
trainfolderRawYuv = './trainrawyuv/'
trainfolderRawPng = './trainrawpng/'
trainfolderCompYuv = './traincompyuv/'
trainfolderCompPng = './traincomppng/'

def cal_psnr(img_orig, img_out):
    squared_error = np.square(img_orig - img_out)
    mse = np.mean(squared_error)
    psnr = 10 * np.log10(1.0 / (mse + 1e-10))
    return psnr

def yuv2rgb(Y, U, V, fw, fh):
    U_new = cv2.resize(U, (fw, fh), interpolation=cv2.INTER_CUBIC)
    V_new = cv2.resize(V, (fw, fh), interpolation=cv2.INTER_CUBIC)
    rf = Y + 1.4075 * (V_new - 128.0)
    gf = Y - 0.3455 * (U_new - 128.0) - 0.7169 * (V_new - 128.0)
    bf = Y + 1.7790 * (U_new - 128.0)
    rf = np.clip(rf, 0, 255)
    gf = np.clip(gf, 0, 255)
    bf = np.clip(bf, 0, 255)
    return rf, gf, bf

def FromFolderYuvToFolderPNG(folderyuv, folderpng, fw, fh):
    os.makedirs(folderpng, exist_ok=True)
    for name in os.listdir(folderpng):
        os.remove(os.path.join(folderpng, name))

    fwuv = fw // 2
    fhuv = fh // 2
    expected_frame_bytes = fw * fh + 2 * fwuv * fhuv  # YUV420

    numdx = (fw - patchsize) // patchstep
    numdy = (fh - patchsize) // patchstep
    dx = (np.arange(numdx) * patchstep).astype(int)
    dy = (np.arange(numdy) * patchstep).astype(int)

    Im = np.zeros((patchsize, patchsize, 3), dtype=np.uint8)
    dir_list = os.listdir(folderyuv)
    pngframenum = 0

    for name in dir_list:
        fullname = os.path.join(folderyuv, name)
        if not fullname.endswith('.yuv'):
            continue

        file_size = os.path.getsize(fullname)
        frames = min(100, file_size // expected_frame_bytes)
        if frames <= 0:
            print(f"Skip {fullname}: not enough data for one frame")
            continue

        with open(fullname, 'rb') as fp:
            print(f"{fullname}, frames: {frames}")
            for f in range(frames):
                try:
                    Y = np.frombuffer(fp.read(fw * fh), dtype=np.uint8).reshape((fh, fw))
                    U = np.frombuffer(fp.read(fwuv * fhuv), dtype=np.uint8).reshape((fhuv, fwuv))
                    V = np.frombuffer(fp.read(fwuv * fhuv), dtype=np.uint8).reshape((fhuv, fwuv))
                except ValueError:
                    print(f"Corrupted frame {f} in {fullname}")
                    break

                r, g, b = yuv2rgb(Y, U, V, fw, fh)
                for i in dx:
                    for j in dy:
                        Im[:, :, 0] = b[j:j+patchsize, i:i+patchsize]
                        Im[:, :, 1] = g[j:j+patchsize, i:i+patchsize]
                        Im[:, :, 2] = r[j:j+patchsize, i:i+patchsize]
                        pngfilename = os.path.join(folderpng, f"{pngframenum}.png")
                        cv2.imwrite(pngfilename, Im)
                        pngframenum += 1
    return pngframenum - 1

def LoadImagesFromFolder(foldername, max_samples=20000):
    files = [f for f in os.listdir(foldername) if f.endswith('.png')]
    files = files[:max_samples]  # ← ограничение памяти!
    N = len(files)
    print(f"Loading {N} images from {foldername}")
    x = np.zeros((N, patchsize, patchsize, 3), dtype=np.float32)
    for idx, name in enumerate(files):
        img = cv2.imread(os.path.join(foldername, name))
        if img is None:
            continue
        x[idx, :, :, 0] = img[:, :, 2]  # R
        x[idx, :, :, 1] = img[:, :, 1]  # G
        x[idx, :, :, 2] = img[:, :, 0]  # B
    return x

def psnr(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    return 10.0 * tf.math.log(1.0 / (mse + 1e-8)) / tf.math.log(10.0)

def EnhancerModel(fw, fh):
    comp_tensor = layers.Input(shape=(fh, fw, 3))
    conv_1 = layers.Conv2D(filters=128, kernel_size=9, padding="same")(comp_tensor)
    conv_1 = layers.PReLU(shared_axes=[1, 2])(conv_1)
    conv_2 = layers.Conv2D(filters=64, kernel_size=7, padding="same")(conv_1)
    conv_2 = layers.PReLU(shared_axes=[1, 2])(conv_2)
    conv_3 = layers.Conv2D(filters=64, kernel_size=3, padding="same")(conv_2)
    conv_3 = layers.PReLU(shared_axes=[1, 2])(conv_3)
    conv_4 = layers.Conv2D(filters=32, kernel_size=1, padding="same")(conv_3)
    conv_4 = layers.PReLU(shared_axes=[1, 2])(conv_4)

    conv_11 = layers.Conv2D(filters=128, kernel_size=9, padding="same")(comp_tensor)
    conv_11 = layers.PReLU(shared_axes=[1, 2])(conv_11)
    feat_11 = layers.concatenate([conv_1, conv_11], axis=-1)

    conv_22 = layers.Conv2D(filters=64, kernel_size=7, padding="same")(feat_11)
    conv_22 = layers.PReLU(shared_axes=[1, 2])(conv_22)
    feat_22 = layers.concatenate([conv_2, conv_22], axis=-1)

    conv_33 = layers.Conv2D(filters=64, kernel_size=3, padding="same")(feat_22)
    conv_33 = layers.PReLU(shared_axes=[1, 2])(conv_33)
    feat_33 = layers.concatenate([conv_3, conv_33], axis=-1)

    conv_44 = layers.Conv2D(filters=32, kernel_size=1, padding="same")(feat_33)
    conv_44 = layers.PReLU(shared_axes=[1, 2])(conv_44)
    feat_44 = layers.concatenate([conv_4, conv_44], axis=-1)

    conv_out = layers.Conv2D(filters=3, kernel_size=5, padding="same")(feat_44)
    output_tensor = layers.Add()([comp_tensor, conv_out])
    return Model(inputs=comp_tensor, outputs=output_tensor)

def TrainImageEnhancementModel(folderRaw, folderComp, folderRawVal, folderCompVal):
    print('Loading raw train images...')
    Xraw = LoadImagesFromFolder(folderRaw, max_samples=20000)
    print('Loading compressed train images...')
    Xcomp = LoadImagesFromFolder(folderComp, max_samples=20000)
    Xraw = Xraw / 255.0
    Xcomp = Xcomp / 255.0

    print('Loading raw validation images...')
    XrawVal = LoadImagesFromFolder(folderRawVal, max_samples=2000)
    print('Loading compressed validation images...')
    XcompVal = LoadImagesFromFolder(folderCompVal, max_samples=2000)
    XrawVal = XrawVal / 255.0
    XcompVal = XcompVal / 255.0

    print(f"Train shape: {Xcomp.shape}, Val shape: {XcompVal.shape}")

    enhancer = EnhancerModel(patchsize, patchsize)
    optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
    enhancer.compile(loss='mean_squared_error', optimizer=optimizer, metrics=[psnr])

    checkpoint_filepath = 'best_model2025.weights.h5'
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        mode='min',
        verbose=1
    )

    NumEpochs = 20
    hist = enhancer.fit(
        Xcomp, Xraw,
        epochs=NumEpochs,
        batch_size=128,
        verbose=1,
        validation_data=(XcompVal, XrawVal),
        callbacks=[checkpoint_callback]
    )
    enhancer.save_weights('enhancer2025.weights.h5')
    return enhancer

def InferenceImageEnhancementModel(fw, fh):
    enhancer = EnhancerModel(fw, fh)
    enhancer.load_weights('enhancer2025.weights.h5')
    return enhancer

def GetRGBFrame(folderyuv, VideoNumber, FrameNumber, fw, fh):
    fwuv = fw // 2
    fhuv = fh // 2
    dir_list = os.listdir(folderyuv)
    v = 0
    for name in dir_list:
        fullname = os.path.join(folderyuv, name)
        if v != VideoNumber:
            v += 1
            continue
        if not fullname.endswith('.yuv'):
            continue
        file_size = os.path.getsize(fullname)
        expected_frame_bytes = fw * fh + 2 * (fw//2) * (fh//2)
        frames = file_size // expected_frame_bytes
        with open(fullname, 'rb') as fp:
            for f in range(frames):
                Y = np.frombuffer(fp.read(fw * fh), dtype=np.uint8).reshape((fh, fw))
                U = np.frombuffer(fp.read((fw//2)*(fh//2)), dtype=np.uint8).reshape((fh//2, fw//2))
                V = np.frombuffer(fp.read((fw//2)*(fh//2)), dtype=np.uint8).reshape((fh//2, fw//2))
                if f == FrameNumber:
                    return yuv2rgb(Y, U, V, fw, fh)
    return None

def GetEngancedRGB(RGBin, fw, fh):
    RGBin = np.expand_dims(RGBin, axis=0).astype(np.float32) / 255.0
    pred = enhancer.predict(RGBin, verbose=0)
    return np.squeeze(pred, axis=0) * 255.0

def ShowOneFrameEnhancement(folderyuvraw, foldercomp, VideoIndex, FrameIndex):
    r1, g1, b1 = GetRGBFrame(folderyuvraw, VideoIndex, FrameIndex, w, h)
    RGBRAW = np.stack([r1, g1, b1], axis=-1)
    r2, g2, b2 = GetRGBFrame(foldercomp, VideoIndex, FrameIndex, w, h)
    RGBCOMP = np.stack([r2, g2, b2], axis=-1)
    RGBENH = GetEngancedRGB(RGBCOMP, w, h)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(RGBRAW / 255.0)
    plt.title("Raw\nPSNR: —")
    plt.axis('off')
    plt.subplot(1, 3, 2)
    psnr1 = cal_psnr(RGBRAW / 255.0, RGBCOMP / 255.0)
    plt.imshow(RGBCOMP / 255.0)
    plt.title(f"Compressed\nPSNR: {psnr1:.2f} dB")
    plt.axis('off')
    plt.subplot(1, 3, 3)
    psnr2 = cal_psnr(RGBRAW / 255.0, RGBENH / 255.0)
    plt.imshow(RGBENH / 255.0)
    plt.title(f"Enhanced\nPSNR: {psnr2:.2f} dB")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def ShowFramePSNRPerformance(folderyuv, foldercomp, VideoIndex, framesmax, fw, fh):
    dir_list = os.listdir(folderyuv)
    v = 0
    for name in dir_list:
        fullname = os.path.join(folderyuv, name)
        if v != VideoIndex:
            v += 1
            continue
        if not fullname.endswith('.yuv'):
            continue
        file_size = os.path.getsize(fullname)
        expected_frame_bytes = fw * fh + 2 * (fw//2) * (fh//2)
        frames = min(framesmax, file_size // expected_frame_bytes)
        PSNRCOMP = np.zeros(frames)
        PSNRENH = np.zeros(frames)
        for f in range(frames):
            r1, g1, b1 = GetRGBFrame(folderyuv, VideoIndex, f, w, h)
            RGBRAW = np.stack([r1, g1, b1], axis=-1)
            r2, g2, b2 = GetRGBFrame(foldercomp, VideoIndex, f, w, h)
            RGBCOMP = np.stack([r2, g2, b2], axis=-1)
            PSNRCOMP[f] = cal_psnr(RGBRAW / 255.0, RGBCOMP / 255.0)
            RGBENH = GetEngancedRGB(RGBCOMP, w, h)
            PSNRENH[f] = cal_psnr(RGBRAW / 255.0, RGBENH / 255.0)
        break
    sorted_indices = np.argsort(PSNRCOMP)
    PSNRCOMP_sorted = PSNRCOMP[sorted_indices]
    PSNRENH_sorted = PSNRENH[sorted_indices]

    plt.plot(PSNRCOMP_sorted, label='Compressed')
    plt.plot(PSNRENH_sorted, label='Enhanced')
    plt.xlabel('Frame (sorted by Compressed PSNR)')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.grid(True)
    plt.title(f"{name}: Compressed = {PSNRCOMP.mean():.2f} dB, Enhanced = {PSNRENH.mean():.2f} dB")
    plt.show()
    #plt.plot(PSNRCOMP, label='Compressed')
    #plt.plot(PSNRENH, label='Enhanced')
    #plt.xlabel('Frame')
    #plt.ylabel('PSNR (dB)')
    #plt.legend()
    #plt.grid(True)
    #plt.title(f"{name}: Compressed = {PSNRCOMP.mean():.2f} dB, Enhanced = {PSNRENH.mean():.2f} dB")
    #plt.show()

TrainMode = 0
PrepareDataSetFromYUV = 0

if TrainMode == 1:
    if PrepareDataSetFromYUV == 1:
        FromFolderYuvToFolderPNG(testfolderRawYuv, testfolderRawPng, w, h)
        FromFolderYuvToFolderPNG(testfolderCompYuv, testfolderCompPng, w, h)
        FromFolderYuvToFolderPNG(trainfolderRawYuv, trainfolderRawPng, w, h)
        FromFolderYuvToFolderPNG(trainfolderCompYuv, trainfolderCompPng, w, h)
    TrainImageEnhancementModel(trainfolderRawPng, trainfolderCompPng, testfolderRawPng, testfolderCompPng)

if True:
    enhancer = InferenceImageEnhancementModel(w, h)
    #ShowOneFrameEnhancement(trainfolderRawYuv, trainfolderCompYuv, 0, 0)
    #ShowOneFrameEnhancement(testfolderRawYuv, testfolderCompYuv, 0, 0)
    #ShowFramePSNRPerformance(trainfolderRawYuv, trainfolderCompYuv, 0, 20, w, h)
    #ShowFramePSNRPerformance(testfolderRawYuv, testfolderCompYuv, 0, 20, w, h)
    ShowFramePSNRPerformance(testfolderRawYuv, testfolderCompYuv, 0, 100, w, h)





