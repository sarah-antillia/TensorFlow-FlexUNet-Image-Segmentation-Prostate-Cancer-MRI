import os
import cv2
import glob
import shutil
import traceback

class Generator:
  def __init__(self):
    pass

  def generate(self, images_dir, masks_dir, output_images_dir, output_masks_dir):
     mask_files = sorted(glob.glob(masks_dir + "/*.png"))
     num = len(mask_files)
     print("Number of mask_files", num)
     n = 0
     for mask_file in mask_files:
       mask = cv2.imread(mask_file)
       gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
       
       if gray.any() > 0:
         basename = os.path.basename(mask_file)
         image_file = os.path.join(images_dir, basename)
         shutil.copy2(mask_file, output_masks_dir)
         shutil.copy2(image_file, output_images_dir)
         n += 1
         print("Copied", n)
         
       else:
         print("Skipped")

if __name__ == "__main__":
  try:
    images_dir = "./Prostate-Cancer-2/images"
    masks_dir  = "./Prostate-Cancer-2/masks"
    output_dir = "./Prostate-Cancer-2-master"
    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
    output_images_dir = os.path.join(output_dir, "images")
    output_masks_dir = os.path.join(output_dir, "masks")
    os.makedirs(output_images_dir)
    os.makedirs(output_masks_dir)
    generator = Generator()
    generator.generate(images_dir, masks_dir, output_images_dir, output_masks_dir)
  except:
    traceback.print_exc()


                   
      

