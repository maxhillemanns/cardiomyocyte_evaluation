# -*- coding: utf-8 -*-
"""
Created on Thu May 14 15:58:14 2020

@author: User
"""

#import keras
from matplotlib import image
import matplotlib.pyplot as plt
from keras.applications import vgg16
import numpy as np
from skimage.transform import resize
from keras import metrics, losses, optimizers
from keras_explain.integrated_gradients import IntegratedGrad
from keras_explain.grad_cam import GuidedGradCam
import matplotlib.gridspec as gridspec

my_loss = losses.mean_squared_error
my_metrics = [metrics.mae, metrics.categorical_accuracy]
my_opt = optimizers.adam(lr=0.00001)

cat_img = image.imread("cat.jpg")
cat_img = np.asarray(cat_img)
cat_img = resize(cat_img, (224,224,3))
cat_img_in = np.reshape(cat_img, (1,224,224,3))

model = vgg16.VGG16(include_top=True, weights="imagenet")

model.compile(optimizer=my_opt, loss=my_loss, metrics=my_metrics)

pred = model.predict(cat_img_in)

igrad_explainer = IntegratedGrad(model)
grad_cam_explainer = GuidedGradCam(model, layer=None)

igrad_exp = igrad_explainer.explain(cat_img, pred.argmax())
grad_cam_exp = grad_cam_explainer.explain(cat_img, pred.argmax())

gs = gridspec.GridSpec(2, 2)

fig = plt.figure()
ax1 = fig.add_subplot(gs[0, :])
ax1.imshow(cat_img)
ax1.axis('off')
ax1.set_title('original image')

ax2 = fig.add_subplot(gs[1,0])
ax2.imshow(igrad_exp[0], cmap='rainbow')
ax2.axis('off')
ax2.set_title('IGrad')

ax3 = fig.add_subplot(gs[1,1])
ax3.imshow(grad_cam_exp[0], cmap='rainbow')
ax3.axis('off')
ax3.set_title('Grad-CAM')

plt.savefig('explain_test_ver_1.png')

fig, a = plt.subplots(1,3)
a[0].imshow(cat_img)
a[0].axis('off')
a[0].set_title('original image')

a[1].imshow(igrad_exp[0], cmap='rainbow')
a[1].axis('off')
a[1].set_title('IGrad')

a[2].imshow(grad_cam_exp[0], cmap='rainbow')
a[2].axis('off')
a[2].set_title('Grad-CAM')

plt.savefig('explain_test_ver_2.png')

print(pred.argmax())
