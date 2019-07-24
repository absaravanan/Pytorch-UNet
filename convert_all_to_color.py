import cv2
import os

# outDir = "/home/ai/ai/data/coco/true_images/"

# for r, d, f in os.walk("/home/ai/ai/data/coco/images"):
#     for file in f:
#         img = cv2.imread(os.path.join(r, file),1)
#         print (img.shape)
#         cv2.imwrite(os.path.join(outDir, file), img)


img = cv2.imread("/home/ai/ai/data/coco/true_images/MMnRBURaC7HAlyb5oX25AjjIzvanvhvURmZFvWK0NyvV35bfq3.jpg")
print (img.shape)