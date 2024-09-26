Pembagian tugas

Load dataset(jamal): menyesuaikan xml dan yml untuk proses training yolo + splitting train dan test

contoh:

train: path/to/train.txt

val: path/to/test.txt

nc: 3  # Number of classes

names: ['bacteria_class1', 'bacteria_class2', 'bacteria_class3']

isi txt

path/to/image1.jpg

path/to/image2.jpg

(Yolo punya cara sendiri buat load data training)


Preprocessing image+data augmentation(eldin)

train model(abed)

evaluasi model(abed+eldin)

hasil training untuk vidio deteksi bakteri(jamal)

Dataset + contoh load di pytorch(bukan yolo): 

https://www.kaggle.com/eldintarofarrandi/bacteria-in-water-detection

Contoh pytorch dan vidio:

https://www.kaggle.com/code/sudhanshu2198/mirorganism-detection-in-water\

Jurnal (bisa disamaain dulu):

https://github.com/eldntr/Detection-of-bacteria-in-water-using-Yolo-images
