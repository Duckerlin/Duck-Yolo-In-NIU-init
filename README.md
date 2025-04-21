這是NIUEE的大學生對於yolov8演算法改進之整理，專爲本研究系統製用，供各位學者參看和指教~


<使用指南>:


1.對大檔案運輸至github之方式


cd /home/b123/Duck-Yolo-In-NIU-init-/ #（自己github文件夾之位址）


  git lfs install


  git lfs pull


2.需先安裝相關的依賴包


  pip install ultralytics==8.2.79 numpy==1.24.4


  pip install tensorflow==2.17.0


3.再安裝相關pip包


  pip install timm


  pip install einops
