from keras.applications.vgg16 import ( VGG16, preprocess_input, decode_predictions )
from tensorflow.keras.preprocessing import image
from keras.layers.core import Lambda
from keras.models import Sequential
from tensorflow.python.framework import ops
import keras.backend as K
import tensorflow as tf
import keras

import sys
import cv2
import numpy as np
import numpy.ma as ma
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

from cnn_cmip import cnn_cmip

def normalize(x):
		# utility function to normalize a tensor by its L2 norm
		return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def load_image_self(path, index):
	img = np.load(path)['x_val'][index]
	y = np.load(path)['y_val'][index]
	x = np.expand_dims(img, axis=0)
	x = preprocess_input(x)
	x = ma.masked_where(x>999, x)
	return x, y

def grad_cam(input_model, image, y_val, layer_name):
    '''
    Parameters
    ----------
    input_model : model
        評価するkerasモデル
    image : tuple等
        入力画像(枚数、　縦、　横、チャンネル)
    y_val : float
        正解ラベルの値
    layer_name : str
        最後のconv層の後のactivation層のレイヤー名。
        最後のconv層でactivationを指定していればconv層のレイヤー名
        batch_normalizationを使う際などのようなconv層でactivationを指定していない場合は、
        そのあとのactivation層のレイヤー名

    Returns
    ----------
    cam: tuple
        Grad-camの画像

    Process
    ----------
    １．入力の予測値を計算
    ２．予測値のロスを計算
    ３．予測クラスのロスから最後の畳み込み層への逆伝搬勾配を計算
    ４．最後の畳み込み層のチャンネル毎に勾配を平均（Global Average Pooling）を計算して，各チャンネルの重要度とする
    ５．最後の畳み込み層の順伝搬の出力にチャンネルごとの重みをかけて，足し合わせる
    ６．入力画像とheatmapをかける->Grad-Camの計算ができた！
    '''
    # ------ 1. 入力画像の予測クラスを計算 -------
    y_val = y_val

    # -------2. 予測クラスのLossを計算 -------
    pred_val = input_model.output[0]
    y_val = tf.convert_to_tensor(y_val.astype(np.float32))
    # loss function(mean square error)を計算
    loss = K.mean(K.square(pred_val - y_val))
    # 引数のlayer_nameのレイヤー(最後のconv層)のoutputを取得する
    conv_output = input_model.get_layer(layer_name).output

    # ------3.予測クラスのLossから最後のconv層への逆伝搬(勾配)を計算 ----
    # 入力 : [判定したい画像.shape=(1, 24, 72, 6)],
    # 出力 : [最後のconv層の出力値.shape=(1, 6, 18, 30), 予測クラスの値から最後のconv層までの勾配.shape=(1, 6, 18, 30)]
    grads = normalize(K.gradients(loss, conv_output)[0])
    # imageを入力したときの最終convolutional layerの出力と勾配を計算
    output, grads_val = K.function([input_model.layers[0].input], [conv_output, grads])([image])
    # 整形後 output.shape=(14, 14, 512), grad_val.shape=(14, 14, 512)
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    # -----4.最後のconv層のチャンネル毎に勾配を平均を計算して、各チャンネルの重要度(重み)とする------
    # weights.shape=(512, )
    # cam.shape = (14, 14)
    weights = np.mean(grads_val, axis = (0, 1))

    # ------5.最後のconv層の順伝搬の出力にチャンネル毎の重みをかけて、足し合わせて、ReLUを通す------
    # 最後のconv層の順伝搬の出力にチャンネル毎の重みをかけて、足し合わせ
    cam = np.dot(output, weights)
    # 入力画像のサイズにリサイズ(14, 14) -> (72, 24)
    cam = cv2.resize(cam, (72, 24), cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    # 値を0~1に正規化
    heatmap = (cam - np.min(cam)) / (np.max(cam) - np.min(cam)) #自作モデルではこちら使用

    # ------6.入力画像とheatmapをかける --------
    # heatmapの値を0~255にしてカラーマップ化(3チャンネル化)
    #cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    # 入力画像とheatmapの足し合わせ
    #cam = np.float32(cam) + np.float32(image)
    # 値を0~255に正規化
    #cam = 255 * cam / np.max(cam)
    return np.uint8(cam), heatmap

def show(heatmap):
    proj = ccrs.PlateCarree(central_longitude=180)
    img_extent = (-180, 180, -55, 65)

    fig = plt.figure()
    ax = plt.subplot(projection =proj)

    ax.coastlines(resolution='50m', lw=0.5)
    ax.gridlines(xlocs = mticker.MultipleLocator(90),
                 ylocs = mticker.MultipleLocator(45),
                 linestyle = '-',
                 color = 'gray')

    mat = ax.matshow(heatmap,
                     cmap='BuPu',
                     extent=img_extent,
                     transform=proj)

    cbar = fig.colorbar(mat,
                        ax =ax,
                        orientation='horizontal')

def snapshot():
    # setting paths
    save_flag = False
    index = 50 # different index = different test data = different heatmap
    corr = '0.7064' # correlation of cnn_cmip run
    path = '/docker/home/hasegawa/docker-gpu/multi-layer-cnn/result_cnn_cmip/input/'+corr+'.npz'
    weights_path = '/docker/home/hasegawa/docker-gpu/multi-layer-cnn/result_cnn_cmip/weights/'+corr+'.h5'
    save_path = '/docker/home/hasegawa/docker-gpu/multi-layer-cnn/result_gradcam_cmip/fig/'+corr+'_'+np.str(index)+'.jpg'

    # 1. 入力画像の読み込み
    preprocessed_input, y_val = load_image_self(path, index)

    # 2. モデルの読み込み
    cnn = cnn_cmip(8, 'aug', 4)
    model = cnn.build_model()
    model.load_weights(weights_path)

    # 3.　入力画像の予測値の計算
    predictions = model.predict(preprocessed_input)

    # 4. Grad-Camの計算
    # 自作モデルの場合、引数の"block5_conv3"を自作モデルの最終conv層のレイヤー名に変更
    cam, heatmap = grad_cam(model, preprocessed_input, y_val, "conv2d_2")

    # 5. 画像の保存
    # 5.1 input image
    x_val = np.load(path)['x_val'][index]
    x_val = ma.masked_where(x_val>999, x_val)
    #show(x_val[:,:,:])

    # 5.2 heatmap image
    show(heatmap)

    if save_flag is True:
        plt.savefig(save_path)
        plt.show()

def saliency_mean():
    # setting paths
    save_flag = True

    corr = '0.7904' # correlation of cnn_cmip run
    path = '/docker/home/hasegawa/docker-gpu/multi-layer-cnn/result_cnn_cmip/input/'+corr+'.npz'
    weights_path = '/docker/home/hasegawa/docker-gpu/multi-layer-cnn/result_cnn_cmip/weights/'+corr+'.h5'
    save_path = '/docker/home/hasegawa/docker-gpu/multi-layer-cnn/result_gradcam_cmip/fig/'+corr+'_'+'saliency_mean'+'.jpg'

    x_val = np.load(path)['x_val']

    cnn = cnn_cmip(8, 'aug', 4)
    model = cnn.build_model()
    model.load_weights(weights_path)

    saliency = np.empty(x_val.shape[:3])

    for i in range(len(x_val)):
        preprocessed_input, y_val = load_image_self(path, i)
        predictions = model.predict(preprocessed_input)
        _, heatmap = grad_cam(model, preprocessed_input, y_val, "conv2d_2")
        saliency[i,:,:] = heatmap
        print(f'{i}/{len(x_val)}')

    saliency = saliency.mean(axis=0)

    show(saliency)

    if save_flag is True:
        plt.savefig(save_path)

    plt.show()

if __name__ == '__main__':
    saliency_mean()
